"""High-reliability Tick Normalization and Validation Engine.

Guarantees data integrity before database persistence and candle construction:
1. Calculates exact incremental traded volume from exchange cumulative volume.
2. Quarantines out-of-order ticks to prevent cumulative volume baseline corruption.
3. Drops duplicate network packets.
4. Strictly isolates spot index volume = 0.
5. Classifies market session state (VALID_LIVE vs VALID_SNAPSHOT) so holiday quotes
   do not trigger live trading breakout states.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
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


_DERIVATIVE_SUFFIX = re.compile(r"(FUT|CE|PE)$")


def is_derivative_symbol(symbol: str) -> bool:
    """True for exchange-traded futures/options (e.g. NSE:NIFTY26NOVFUT, NSE:NIFTY26NOV25400CE).

    Indices (-INDEX) and equities (-EQ) are cash-segment instruments.
    """
    s = symbol.upper()
    if s.endswith("-INDEX") or s.endswith("-EQ"):
        return False
    return bool(_DERIVATIVE_SUFFIX.search(s))


class TickStatus(str, Enum):
    VALID_LIVE = "VALID_LIVE"          # Valid active trading tick during open market
    VALID_SNAPSHOT = "VALID_SNAPSHOT"  # Quote snapshot outside regular trading hours (holiday/weekend)
    UNVERIFIED = "UNVERIFIED"          # Missing exchange timestamp during open market (quarantined from breakouts)
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
        max_clock_skew_seconds: float = 2.0,
    ):
        self.calendar = calendar or ExchangeCalendar()
        self.max_freshness_seconds = max_freshness_seconds
        # Local clock may run slightly behind the exchange clock (live logs showed
        # -37..-89 ms on ~11% of NIFTY ticks). Requiring latency >= 0 misclassified those
        # genuine live ticks as stale snapshots and dropped them from candle building.
        self.max_clock_skew_seconds = max_clock_skew_seconds
        # Per-symbol state tracking
        self._last_fingerprint: Dict[str, Tuple] = {}
        self._last_exchange_time: Dict[str, datetime] = {}
        # Session-isolated volume baselines
        self._live_session_date: Dict[str, date] = {}
        self._live_session_cum_volume: Dict[str, int] = {}

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

        # 4. Session & Live vs Snapshot Classification (Dual-Gate Validation)
        # Gate A: Current real-world wall-clock time must be an open trading session
        current_wall_time = current_time or now_ist()
        # Cash instruments (indices, equities) stop trading at 15:30; only F&O contracts
        # continue in the derivatives closing window. Using the derivatives session for
        # everything made the index/equities emit flat synthetic bars until 15:40.
        current_session = self.calendar.get_session_state(
            current_wall_time, is_derivative=is_derivative_symbol(sym)
        )
        is_market_currently_open = current_session in (
            SessionState.REGULAR_CASH,
            SessionState.DERIVATIVES_EXT,
            SessionState.SPECIAL_SESSION,
        )

        status = TickStatus.VALID_SNAPSHOT
        if is_market_currently_open:
            if raw_tick.exchange_timestamp is not None:
                # Exchange timestamp must match current trading date and be fresh
                exch_dt = raw_tick.exchange_timestamp.astimezone(IST)
                same_trading_date = exch_dt.date() == current_wall_time.date()
                latency_sec = (current_wall_time - exch_dt).total_seconds()
                is_fresh = -self.max_clock_skew_seconds <= latency_sec <= self.max_freshness_seconds
                status = TickStatus.VALID_LIVE if (same_trading_date and is_fresh) else TickStatus.VALID_SNAPSHOT
            else:
                # Missing exchange timestamp during open market: classify as UNVERIFIED (quarantined from breakouts)
                status = TickStatus.UNVERIFIED

        # 5. Session-Specific Volume Normalization
        incremental_vol = 0
        cum_vol = 0 if is_index else raw_tick.cumulative_volume

        if not is_index and cum_vol > 0:
            if status == TickStatus.VALID_LIVE:
                session_date = current_wall_time.date()
                last_session = self._live_session_date.get(sym)

                if last_session != session_date or sym not in self._live_session_cum_volume:
                    # First live observation of this trading session: establish baseline without false surge
                    self._live_session_cum_volume[sym] = cum_vol
                    self._live_session_date[sym] = session_date
                    incremental_vol = 0
                else:
                    prev_cum = self._live_session_cum_volume[sym]
                    if cum_vol >= prev_cum:
                        incremental_vol = cum_vol - prev_cum
                    else:
                        # Exchange/broker counter reset mid-session: establish new baseline safely
                        incremental_vol = 0
                    self._live_session_cum_volume[sym] = cum_vol
            else:
                # Off-market / holiday snapshots and unverified ticks never contaminate live session baselines
                incremental_vol = 0

        # Construct sanitized, normalized tick model
        if status == TickStatus.VALID_LIVE:
            event_class = "LIVE"
        elif status == TickStatus.UNVERIFIED:
            event_class = "UNVERIFIED"
        else:
            event_class = "SNAPSHOT"

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
            event_class=event_class,
        )

        reason_str = (
            "Normalized successfully"
            if status == TickStatus.VALID_LIVE
            else ("Unverified quote without exchange timestamp" if status == TickStatus.UNVERIFIED else "Holiday/offline snapshot quote")
        )

        return NormalizationResult(
            tick=normalized_tick,
            status=status,
            reason=reason_str,
        )
