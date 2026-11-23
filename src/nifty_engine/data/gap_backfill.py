"""Feed-gap detection and historical backfill for the signal instruments.

When the live feed drops (Wi-Fi loss, laptop sleep, broker stall) the candle builder
simply has no ticks for those minutes, so indicators would continue on a series with
holes and the bar spanning the outage would be built from partial ticks.

Lifecycle:
1. DETECT  - primary index silent > gap_seconds during the cash session.
2. HOLD    - while the gap is open, live candles for signal symbols are buffered
             instead of reaching the technical engine (no signals on a broken series).
3. RESUME  - feed is fresh again; wait until the next bar boundary + settle time so
             FYERS has finalized the bars around the outage.
4. REPAIR  - reconcile today's 1m/5m/15m bars from FYERS history (upserts and
             overwrites partial live bars), then replay every bar the engine missed,
             plus the held live bars, in bootstrap mode (indicators update, no setups).
5. RETRY   - if the broker call fails or coverage is still incomplete, retry later;
             readiness stays NOT_READY until repair succeeds (fail-closed).
"""

import logging
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pytz

from nifty_engine.core.interfaces import Candle

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)

BACKFILL_TIMEFRAMES = ["1m", "5m", "15m"]


class FeedGapBackfiller:
    def __init__(
        self,
        historical_manager,
        technical_engine,
        store,
        symbols_provider: Callable[[], List[str]],
        gap_seconds: float = 15.0,
        settle_seconds: float = 5.0,
        retry_seconds: float = 30.0,
    ) -> None:
        self.historical_manager = historical_manager
        self.technical_engine = technical_engine
        self.store = store
        self.symbols_provider = symbols_provider
        self.gap_seconds = gap_seconds
        self.settle_seconds = settle_seconds
        self.retry_seconds = retry_seconds

        self.gap_started_at: Optional[datetime] = None
        self.resumed_at: Optional[datetime] = None
        self.next_attempt_at: Optional[datetime] = None
        self.attempts: int = 0
        self.held: List[Candle] = []
        self.gaps_repaired: int = 0

    @property
    def active(self) -> bool:
        return self.gap_started_at is not None

    def hold(self, candle: Candle) -> bool:
        """Buffer a live signal-symbol candle while a gap is being repaired. Returns True if held."""
        if not self.active:
            return False
        self.held.append(candle)
        return True

    def observe(
        self,
        now: datetime,
        last_primary_tick_time: Optional[datetime],
        session_live: bool,
    ) -> Optional[str]:
        """Advance the state machine. Call about once per second. Returns a status message on transitions."""
        feed_age = (now - last_primary_tick_time).total_seconds() if last_primary_tick_time else None

        if not self.active:
            if session_live and feed_age is not None and feed_age > self.gap_seconds:
                self.gap_started_at = last_primary_tick_time
                self.resumed_at = None
                self.attempts = 0
                self.next_attempt_at = None
                return (
                    f"Feed gap detected (silent {feed_age:.0f}s since "
                    f"{last_primary_tick_time.strftime('%H:%M:%S')}); holding signals until backfill"
                )
            return None

        fresh = feed_age is not None and feed_age <= self.gap_seconds
        if self.resumed_at is None:
            if not fresh:
                return None
            self.resumed_at = now
            # The bar that contains the resume moment must close and be published by
            # the broker before reconciling, otherwise it stays partial.
            next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            self.next_attempt_at = next_boundary + timedelta(seconds=self.settle_seconds)
            return (
                f"Feed resumed after {(now - self.gap_started_at).total_seconds():.0f}s gap; "
                f"backfill scheduled at {self.next_attempt_at.strftime('%H:%M:%S')}"
            )

        if self.next_attempt_at is not None and now < self.next_attempt_at:
            return None
        return self._repair(now)

    # ------------------------------------------------------------------ internals
    def _repair(self, now: datetime) -> str:
        self.attempts += 1
        symbols = self.symbols_provider()
        session_date = now.date()
        try:
            self.historical_manager.reconcile_session_candles(
                symbols, target_date=session_date, timeframes=BACKFILL_TIMEFRAMES, as_of=now
            )
        except Exception as e:  # network / rate limit: keep gap open and retry later
            self.next_attempt_at = now + timedelta(seconds=self.retry_seconds)
            return f"Backfill attempt {self.attempts} failed: {e}; retry at {self.next_attempt_at.strftime('%H:%M:%S')}"

        incomplete = [
            s for s in symbols
            if not self.historical_manager.is_symbol_reconciled(s, session_date, BACKFILL_TIMEFRAMES)
        ]
        if incomplete:
            self.next_attempt_at = now + timedelta(seconds=self.retry_seconds)
            return (
                f"Backfill attempt {self.attempts}: broker coverage still incomplete for {incomplete}; "
                f"retry at {self.next_attempt_at.strftime('%H:%M:%S')}"
            )

        replay = self._collect_replay(symbols, now)
        replayed = self.technical_engine.bootstrap_candles(replay) if replay else 0

        gap_len = (self.resumed_at - self.gap_started_at).total_seconds() if self.resumed_at else 0.0
        self.gaps_repaired += 1
        self.gap_started_at = None
        self.resumed_at = None
        self.next_attempt_at = None
        self.held = []
        return f"Gap repaired ({gap_len:.0f}s outage): replayed {replayed} bars into indicators; signals re-enabled"

    def _collect_replay(self, symbols: List[str], now: datetime) -> List[Candle]:
        """Bars the engine has not seen yet: broker-reconciled bars first, held live bars as fallback."""
        session_start = IST.localize(datetime.combine(now.date(), time(9, 15)))
        by_key: Dict[Tuple[str, str, datetime], Candle] = {}

        for sym in symbols:
            state = self.technical_engine.states.get(sym)
            for tf in BACKFILL_TIMEFRAMES:
                last_ts = state.latest_candle_timestamps.get(tf) if state else None
                start = last_ts if last_ts is not None else session_start
                for c in self.store.get_candles(sym, tf, start_time=start, end_time=now):
                    if c.is_closed and (last_ts is None or c.timestamp > last_ts):
                        by_key[(c.symbol, c.timeframe, c.timestamp)] = c

        for c in self.held:
            if c.symbol not in symbols:
                continue
            state = self.technical_engine.states.get(c.symbol)
            last_ts = state.latest_candle_timestamps.get(c.timeframe) if state else None
            if last_ts is not None and c.timestamp <= last_ts:
                continue
            by_key.setdefault((c.symbol, c.timeframe, c.timestamp), c)

        return list(by_key.values())
