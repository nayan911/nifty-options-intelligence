"""High-reliability Tick Normalization and Validation Engine.

Guarantees data integrity before database persistence and candle construction:
1. Calculates exact incremental traded volume from exchange cumulative volume.
2. Quarantines out-of-order ticks to prevent cumulative volume baseline corruption.
3. Drops duplicate network packets.
4. Strictly isolates spot index volume = 0.
5. Classifies market session state (VALID_LIVE vs VALID_SNAPSHOT) so holiday quotes
   do not trigger live trading breakout states.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple

import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import SessionState, Tick
from nifty_engine.utils.timezone import now_ist

IST = pytz.timezone("Asia/Kolkata")
INDEX_SYMBOLS = {
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "BSE:SENSEX-INDEX",
    "NSE:INDIAVIX-INDEX",
}


class TickStatus(str, Enum):
    VALID_LIVE = "VALID_LIVE"          # Valid active trading tick during open market
    VALID_SNAPSHOT = "VALID_SNAPSHOT"  # Quote snapshot outside regular trading hours (holiday/weekend)
    DUPLICATE = "DUPLICATE"            # Consecutive duplicate packet dropped
    OUT_OF_ORDER = "OUT_OF_ORDER"      # Stale / back-dated tick quarantined
    MALFORMED = "MALFORMED"            # Zero/negative price or invalid structure


@dataclass(frozen=True)
class NormalizationResult:
    tick: Optional[Tick]
    status: TickStatus
    reason: str


class TickNormalizer:
    """Stateful validator and volume normalizer for live streaming feeds."""

    def __init__(
        self,
        calendar: Optional[ExchangeCalendar] = None,
        max_freshness_seconds: float = 30.0,
    ):
        self.calendar = calendar or ExchangeCalendar()
        self.max_freshness_seconds = max_freshness_seconds
        # Per-symbol state tracking
        self._last_fingerprint: Dict[str, Tuple] = {}
        self._last_exchange_time: Dict[str, datetime] = {}
        self._last_cum_volume: Dict[str, int] = {}

    def normalize(
        self,
        raw_tick: Tick,
        current_time: Optional[datetime] = None,
    ) -> NormalizationResult:
        """Validate, sanitize, and compute true incremental volume for a tick."""
        # 1. Structure validation
        if raw_tick.ltp <= 0:
            return NormalizationResult(None, TickStatus.MALFORMED, f"Invalid LTP: {raw_tick.ltp}")

        sym = raw_tick.symbol
        ts = raw_tick.exchange_timestamp or raw_tick.local_timestamp
        is_index = sym in INDEX_SYMBOLS or "-INDEX" in sym

        # 2. Duplicate Detection
        fingerprint = (sym, ts, raw_tick.ltp, raw_tick.cumulative_volume, raw_tick.oi)
        if self._last_fingerprint.get(sym) == fingerprint:
            return NormalizationResult(raw_tick, TickStatus.DUPLICATE, "Identical consecutive tick dropped")

        # 3. Out-of-Order Check (if exchange timestamp is available)
        if raw_tick.exchange_timestamp is not None:
            last_ts = self._last_exchange_time.get(sym)
            if last_ts is not None and raw_tick.exchange_timestamp < last_ts:
                # Quarantined: do not update volume or timestamps
                return NormalizationResult(
                    raw_tick,
                    TickStatus.OUT_OF_ORDER,
                    f"Exchange timestamp {raw_tick.exchange_timestamp} is older than last seen {last_ts}",
                )
            self._last_exchange_time[sym] = raw_tick.exchange_timestamp

        self._last_fingerprint[sym] = fingerprint

        # 4. Volume Normalization
        incremental_vol = 0
        cum_vol = 0 if is_index else raw_tick.cumulative_volume

        if not is_index and cum_vol > 0:
            if sym not in self._last_cum_volume:
                # First tick for this symbol in this run: baseline established
                self._last_cum_volume[sym] = cum_vol
                incremental_vol = 0
            else:
                prev_cum = self._last_cum_volume[sym]
                if cum_vol >= prev_cum:
                    incremental_vol = cum_vol - prev_cum
                else:
                    # Daily or session reset: cum_vol dropped below previous
                    incremental_vol = cum_vol
                self._last_cum_volume[sym] = cum_vol

        # 5. Session & Live vs Snapshot Classification
        # Dual-gate validation:
        # A. Current real-world wall-clock time must be an open trading session
        current_wall_time = current_time or now_ist()
        current_session = self.calendar.get_session_state(current_wall_time, is_derivative=True)
        is_market_currently_open = current_session in (
            SessionState.REGULAR_CASH,
            SessionState.DERIVATIVES_EXT,
            SessionState.SPECIAL_SESSION,
        )

        is_live_trading = False
        if is_market_currently_open:
            if raw_tick.exchange_timestamp is not None:
                # Exchange timestamp must match current trading date and be fresh
                exch_dt = raw_tick.exchange_timestamp.astimezone(IST)
                same_trading_date = exch_dt.date() == current_wall_time.date()
                latency_sec = (current_wall_time - exch_dt).total_seconds()
                is_fresh = 0 <= latency_sec <= self.max_freshness_seconds
                is_live_trading = same_trading_date and is_fresh
            else:
                # Fallback to local arrival time if broker omitted exchange timestamp
                is_live_trading = True

        status = TickStatus.VALID_LIVE if is_live_trading else TickStatus.VALID_SNAPSHOT

        # Construct sanitized, normalized tick model
        normalized_tick = Tick(
            symbol=sym,
            exchange_timestamp=raw_tick.exchange_timestamp,
            local_timestamp=raw_tick.local_timestamp,
            ltp=raw_tick.ltp,
            cumulative_volume=cum_vol,
            incremental_volume=incremental_vol,
            is_index=is_index,
            bid=raw_tick.bid,
            ask=raw_tick.ask,
            bid_qty=raw_tick.bid_qty,
            ask_qty=raw_tick.ask_qty,
            oi=raw_tick.oi,
            source=raw_tick.source,
        )

        return NormalizationResult(
            tick=normalized_tick,
            status=status,
            reason="Normalized successfully" if status == TickStatus.VALID_LIVE else "Holiday/offline snapshot quote",
        )
