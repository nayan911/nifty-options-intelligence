"""Feed-gap detection, authoritative reconciliation and indicator rebuild for signal instruments.

When the live feed drops (Wi-Fi loss, laptop sleep, broker stall, or a single symbol's
stream stalling) the candle builder has no ticks for those minutes. Worse, the bar that
was open when the outage began closes with partial data and can reach the technical
engine before the outage is even detected.

Appending corrected bars cannot undo a partial bar already folded into EMAs, ATR,
VWAP, swings and breakout state. So repair = rebuild the signal symbols' technical
state from FYERS-verified bars with the SAME warmup routine a clean restart uses, then
swap it in. A repaired engine is therefore equivalent to a fresh restart.

Lifecycle:
1. DETECT    - any required signal instrument (NIFTY spot AND futures) has no valid live
               tick for > gap_seconds in the cash session. An instrument that has never
               produced a valid tick counts as stale once any other one is live.
2. HOLD      - live candles for signal symbols are kept out of the technical engine
               (still persisted to DuckDB); readiness is NOT_READY.
3. RESUME    - EVERY required instrument has a fresh valid tick; wait for the next bar
               boundary + settle so FYERS has published the bars spanning the outage.
4. WORKER    - off the maintenance thread: fetch 1m/5m/15m history from FYERS, require
               that FYERS itself returned every interval touched by the outage (stored
               partial live bars do not count), then warm a callback-free shadow engine
               from DuckDB. Nothing here can emit a signal or block the 1s loop.
5. SWAP      - on the maintenance thread, under the backfiller lock: atomically adopt the
               shadow states, replay any held bars newer than the rebuild (bootstrap mode,
               no signals), re-enable signals. Takes microseconds, not seconds.
6. RETRY     - any failure in 4 or 5 -> stay NOT_READY and retry later. Never crashes the
               runner, never leaves indicators half-built.
"""

import logging
import threading
import time as time_mod
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional

import pytz

from nifty_engine.core.interfaces import Candle

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)

BACKFILL_TIMEFRAMES = ["1m", "5m", "15m"]
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15}


def affected_intervals(tf: str, gap_start: datetime, as_of: datetime) -> List[datetime]:
    """Closed `tf` bars (start times) that overlap [gap_start, as_of): the bar that was open
    when the outage began through the last bar closed at as_of."""
    step = _TF_MINUTES[tf]
    session_start = IST.localize(datetime.combine(gap_start.astimezone(IST).date(), time(9, 15)))
    first = max(gap_start, session_start)
    k = int((first - session_start).total_seconds() // 60) // step
    cur = session_start + timedelta(minutes=k * step)
    out = []
    while cur + timedelta(minutes=step) <= as_of:
        out.append(cur)
        cur += timedelta(minutes=step)
    return out


class FeedGapBackfiller:
    IDLE = "IDLE"
    GAP_OPEN = "GAP_OPEN"                # a required feed is stale; waiting for all to resume
    AWAITING_SETTLE = "AWAITING_SETTLE"  # all fresh; waiting for bar boundary + settle (or retry)
    REPAIRING = "REPAIRING"              # worker: broker fetch + verification + shadow rebuild

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

        self.state: str = self.IDLE
        self.gap_started_at: Optional[datetime] = None
        self.stale_symbols: List[str] = []
        self.resumed_at: Optional[datetime] = None
        self.next_attempt_at: Optional[datetime] = None
        self.attempts: int = 0
        self.gaps_repaired: int = 0
        self.last_rebuild_seconds: Optional[float] = None
        self.held: List[Candle] = []
        self._first_live_seen_at: Optional[datetime] = None

        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._worker_done = threading.Event()
        self._worker_error: Optional[str] = None
        self._worker_shadow = None
        self._worker_symbols: List[str] = []

    @property
    def active(self) -> bool:
        return self.state != self.IDLE

    def hold(self, candle: Candle) -> bool:
        """Keep a live signal-symbol candle out of the technical engine during a gap.

        Held bars newer than the rebuilt state are replayed at swap time, so a bar that
        closes while the worker is rebuilding is never lost.
        """
        with self._lock:
            if not self.active:
                return False
            self.held.append(candle)
            return True

    # ------------------------------------------------------------------ freshness
    def _assess(self, now: datetime, last_tick_times: Dict[str, Optional[datetime]], session_live: bool):
        """Returns (stale_symbols, all_fresh, gap_start_estimate)."""
        seen = [ts for ts in last_tick_times.values() if ts is not None]
        if session_live and seen and self._first_live_seen_at is None:
            self._first_live_seen_at = min(seen)

        stale: Dict[str, datetime] = {}
        for sym, ts in last_tick_times.items():
            ref = ts if ts is not None else self._first_live_seen_at
            if ref is None:
                continue  # nothing live yet at all (pre-open / startup)
            if (now - ref).total_seconds() > self.gap_seconds:
                stale[sym] = ref
        all_fresh = bool(last_tick_times) and not stale and all(ts is not None for ts in last_tick_times.values())
        gap_start = min(stale.values()) if stale else None
        return sorted(stale), all_fresh, gap_start

    def observe(
        self,
        now: datetime,
        last_tick_times: Dict[str, Optional[datetime]],
        session_live: bool,
    ) -> Optional[str]:
        """Advance the state machine (call ~1/s from the maintenance loop). Never raises.

        last_tick_times: last VALID_LIVE tick time per required signal symbol (None if
        none yet). Returns a status message on transitions.
        """
        try:
            return self._observe(now, last_tick_times, session_live)
        except Exception as e:  # fail closed: keep/enter gap state, never kill the runner
            logger.exception("Feed-gap state machine error")
            with self._lock:
                if self.state == self.IDLE:
                    self.state = self.GAP_OPEN
                    self.gap_started_at = now
                else:
                    self.state = self.AWAITING_SETTLE
                    self.next_attempt_at = now + timedelta(seconds=self.retry_seconds)
            return f"Repair error ({e}); staying NOT_READY, retry in {self.retry_seconds:.0f}s"

    def _observe(self, now, last_tick_times, session_live) -> Optional[str]:
        stale, all_fresh, gap_start = self._assess(now, last_tick_times, session_live)

        with self._lock:
            if self.state == self.IDLE:
                if session_live and stale:
                    self.state = self.GAP_OPEN
                    self.gap_started_at = gap_start
                    self.stale_symbols = stale
                    self.resumed_at = None
                    self.attempts = 0
                    self.held = []
                    return (
                        f"Feed gap detected on {', '.join(stale)} (no valid live tick since "
                        f"{gap_start.strftime('%H:%M:%S')}); signals held until repair"
                    )
                return None

            if self.state == self.GAP_OPEN:
                if not all_fresh:
                    return None
                self.state = self.AWAITING_SETTLE
                self.resumed_at = now
                next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                self.next_attempt_at = next_boundary + timedelta(seconds=self.settle_seconds)
                return (
                    f"All signal feeds live again after {(now - self.gap_started_at).total_seconds():.0f}s; "
                    f"repair scheduled at {self.next_attempt_at.strftime('%H:%M:%S')}"
                )

            if self.state == self.AWAITING_SETTLE:
                if not all_fresh:
                    # Feed died again before repair: wait for it to come back first.
                    self.state = self.GAP_OPEN
                    return f"Feed stale again ({', '.join(stale) or 'missing'}) before repair; waiting"
                if self.next_attempt_at is not None and now < self.next_attempt_at:
                    return None
                return self._start_worker(now)

            if self.state == self.REPAIRING:
                if not self._worker_done.is_set():
                    return None
                if not all_fresh:
                    # Feed died again while the worker was rebuilding: do not adopt and re-enable
                    # signals on a feed that is already stale. Wait for it to resume, then redo
                    # the repair so it also covers the second outage.
                    self.state = self.GAP_OPEN
                    self.resumed_at = None
                    self._worker_shadow = None
                    return f"Feed stale again ({', '.join(stale) or 'missing'}) during repair; discarding rebuild, waiting"
                return self._finish(now)

        return None

    # ------------------------------------------------------------------ worker
    def _start_worker(self, now: datetime) -> str:
        self.attempts += 1
        symbols = list(self.symbols_provider())
        gap_start = self.gap_started_at
        self._worker_done.clear()
        self._worker_error = None
        self._worker_shadow = None
        self._worker_symbols = symbols

        def run() -> None:
            try:
                self._worker_shadow = self._repair_offthread(symbols, gap_start, now)
            except Exception as e:
                self._worker_error = str(e)
            finally:
                self._worker_done.set()

        self._worker = threading.Thread(target=run, name="gap-repair", daemon=True)
        self.state = self.REPAIRING
        self._worker.start()
        return f"Repairing {symbols} from FYERS history in background (attempt {self.attempts})"

    def _repair_offthread(self, symbols: List[str], gap_start: datetime, as_of: datetime):
        t0 = time_mod.perf_counter()
        session_date = as_of.date()
        hm = self.historical_manager
        hm.reconcile_session_candles(symbols, target_date=session_date, timeframes=BACKFILL_TIMEFRAMES, as_of=as_of)

        incomplete = [s for s in symbols if not hm.is_symbol_reconciled(s, session_date, BACKFILL_TIMEFRAMES)]
        if incomplete:
            raise RuntimeError(f"incomplete session coverage for {incomplete}")

        # Authoritative check: FYERS itself must have returned every bar the outage touched.
        returned = getattr(hm, "last_broker_returned", {})
        for sym in symbols:
            for tf in BACKFILL_TIMEFRAMES:
                need = affected_intervals(tf, gap_start, as_of)
                got = returned.get((sym, tf, session_date), set())
                missing = [ts for ts in need if ts not in got]
                if missing:
                    raise RuntimeError(
                        f"FYERS did not return {len(missing)} {tf} bar(s) for {sym} "
                        f"(first {missing[0].strftime('%H:%M')}); partial live bars not trusted"
                    )

        shadow = self.technical_engine.make_shadow()
        hm.warm_up_technical_engine(engine=shadow, symbols=symbols, as_of_date=session_date, reconcile_session=False)
        self.last_rebuild_seconds = time_mod.perf_counter() - t0
        return shadow

    def _retry_later(self, now: datetime, reason: str) -> str:
        self.state = self.AWAITING_SETTLE
        self.next_attempt_at = now + timedelta(seconds=self.retry_seconds)
        return f"Repair attempt {self.attempts} {reason}; staying NOT_READY, retry at {self.next_attempt_at.strftime('%H:%M:%S')}"

    def _finish(self, now: datetime) -> str:
        if self._worker_error is not None or self._worker_shadow is None:
            return self._retry_later(now, f"failed: {self._worker_error}")

        symbols = self._worker_symbols
        shadow = self._worker_shadow
        try:
            # Bars held during the rebuild that the shadow didn't see (closed after it read DuckDB)
            replay = []
            for c in self.held:
                st = shadow.states.get(c.symbol)
                last_ts = st.latest_candle_timestamps.get(c.timeframe) if st else None
                if c.symbol in symbols and (last_ts is None or c.timestamp > last_ts):
                    replay.append(c)
            if replay:
                shadow.bootstrap_candles(replay)
            self.technical_engine.adopt_states(shadow, symbols)
        except Exception as e:
            return self._retry_later(now, f"swap failed: {e}")

        outage = (self.resumed_at - self.gap_started_at).total_seconds() if self.resumed_at else 0.0
        self.gaps_repaired += 1
        self.state = self.IDLE
        self.gap_started_at = None
        self.stale_symbols = []
        self.resumed_at = None
        self.next_attempt_at = None
        self.held = []
        self._worker_shadow = None
        took = f"{self.last_rebuild_seconds:.2f}s" if self.last_rebuild_seconds is not None else "?"
        return (
            f"Gap repaired ({outage:.0f}s outage): FYERS-verified bars, indicators rebuilt off-thread "
            f"in {took} ({len(replay)} late bars replayed); signals re-enabled"
        )
