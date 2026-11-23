"""Feed-gap detection, historical reconciliation and indicator rebuild for signal instruments.

When the live feed drops (Wi-Fi loss, laptop sleep, broker stall, or a single symbol's
stream stalling) the candle builder has no ticks for those minutes. Worse, the bar that
was open when the outage began closes with partial data and can reach the technical
engine before the outage is even detected.

Appending corrected bars cannot undo a partial bar already folded into EMAs, ATR,
VWAP, swings and breakout state. So repair = discard the signal symbols' technical
state and rebuild it from FYERS-verified bars with the SAME warmup routine a clean
restart uses. A repaired engine is therefore equivalent to a fresh restart.

Lifecycle:
1. DETECT   - any signal symbol (NIFTY spot OR futures) silent > gap_seconds in the
              cash session.
2. HOLD     - live candles for signal symbols are kept out of the technical engine
              (they are still persisted to DuckDB), readiness is NOT_READY.
3. RESUME   - all signal symbols fresh again; wait for the next bar boundary + settle
              time so FYERS has published the bars spanning the outage.
4. RECONCILE- worker thread fetches 1m/5m/15m bars from FYERS history and upserts them
              (overwriting partial live bars). The 1s maintenance loop never blocks on it.
5. REBUILD  - on the main thread, under the backfiller lock: reset technical state and
              re-run warmup from DuckDB (no setups fire during rebuild). Candles that
              close while this runs wait on the lock, then flow live.
6. RETRY    - broker failure or incomplete coverage -> retry later, stay NOT_READY.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from nifty_engine.core.interfaces import Candle

logger = logging.getLogger(__name__)

BACKFILL_TIMEFRAMES = ["1m", "5m", "15m"]


class FeedGapBackfiller:
    IDLE = "IDLE"
    GAP_OPEN = "GAP_OPEN"            # feed stale, waiting for it to resume
    AWAITING_SETTLE = "AWAITING_SETTLE"  # feed back, waiting for bar boundary + settle
    RECONCILING = "RECONCILING"      # worker thread fetching broker bars

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
        self.held: List[Candle] = []  # diagnostic only: held bars are already in DuckDB

        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._worker_result: Optional[Exception] = None
        self._worker_done = threading.Event()

    @property
    def active(self) -> bool:
        return self.state != self.IDLE

    def hold(self, candle: Candle) -> bool:
        """Keep a live signal-symbol candle out of the technical engine during a gap.

        Blocks while a rebuild is in progress, so a candle closing mid-rebuild is either
        covered by the rebuild or processed live right after it - never lost.
        """
        with self._lock:
            if not self.active:
                return False
            self.held.append(candle)
            return True

    def observe(
        self,
        now: datetime,
        last_tick_times: Dict[str, Optional[datetime]],
        session_live: bool,
    ) -> Optional[str]:
        """Advance the state machine (call ~1/s). Returns a status message on transitions.

        last_tick_times: last live tick time per signal symbol. None = not seen yet this
        run (not treated as stale; startup readiness covers that case).
        """
        ages = {
            sym: (now - ts).total_seconds() for sym, ts in last_tick_times.items() if ts is not None
        }
        stale = sorted(sym for sym, age in ages.items() if age > self.gap_seconds)

        with self._lock:
            if self.state == self.IDLE:
                if session_live and stale:
                    oldest = min(last_tick_times[s] for s in stale)
                    self.state = self.GAP_OPEN
                    self.gap_started_at = oldest
                    self.stale_symbols = stale
                    self.resumed_at = None
                    self.attempts = 0
                    self.held = []
                    return (
                        f"Feed gap detected on {', '.join(stale)} (silent since "
                        f"{oldest.strftime('%H:%M:%S')}); signals held until repair"
                    )
                return None

            if self.state == self.GAP_OPEN:
                if stale or not ages:
                    return None
                self.state = self.AWAITING_SETTLE
                self.resumed_at = now
                next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                self.next_attempt_at = next_boundary + timedelta(seconds=self.settle_seconds)
                return (
                    f"Feed resumed after {(now - self.gap_started_at).total_seconds():.0f}s; "
                    f"reconcile scheduled at {self.next_attempt_at.strftime('%H:%M:%S')}"
                )

            if self.state == self.AWAITING_SETTLE:
                if self.next_attempt_at is not None and now < self.next_attempt_at:
                    return None
                return self._start_worker(now)

            if self.state == self.RECONCILING:
                if not self._worker_done.is_set():
                    return None
                return self._finish(now)

        return None

    # ------------------------------------------------------------------ internals
    def _start_worker(self, now: datetime) -> str:
        self.attempts += 1
        symbols = self.symbols_provider()
        self._worker_done.clear()
        self._worker_result = None
        self._worker_as_of = now

        def run() -> None:
            try:
                self.historical_manager.reconcile_session_candles(
                    symbols, target_date=now.date(), timeframes=BACKFILL_TIMEFRAMES, as_of=now
                )
            except Exception as e:  # network / rate limit
                self._worker_result = e
            finally:
                self._worker_done.set()

        self._worker = threading.Thread(target=run, name="gap-reconcile", daemon=True)
        self.state = self.RECONCILING
        self._worker.start()
        return f"Reconciling {symbols} from FYERS history (attempt {self.attempts})"

    def _retry_later(self, now: datetime, reason: str) -> str:
        self.state = self.AWAITING_SETTLE
        self.next_attempt_at = now + timedelta(seconds=self.retry_seconds)
        return f"Repair attempt {self.attempts} {reason}; retry at {self.next_attempt_at.strftime('%H:%M:%S')}"

    def _finish(self, now: datetime) -> str:
        if self._worker_result is not None:
            return self._retry_later(now, f"failed: {self._worker_result}")

        symbols = self.symbols_provider()
        session_date = self._worker_as_of.date()
        incomplete = [
            s for s in symbols
            if not self.historical_manager.is_symbol_reconciled(s, session_date, BACKFILL_TIMEFRAMES)
        ]
        if incomplete:
            return self._retry_later(now, f"incomplete broker coverage for {incomplete}")

        # Rebuild from verified bars exactly as a clean restart would (setups suppressed).
        self.technical_engine.reset_symbols(symbols)
        summary = self.historical_manager.warm_up_technical_engine(
            engine=self.technical_engine,
            symbols=symbols,
            as_of_date=session_date,
            reconcile_session=False,
        )
        rebuilt = sum(
            int(summary.get(s, {}).get("session_catchup_candles", 0) or 0) for s in symbols
        )
        outage = (self.resumed_at - self.gap_started_at).total_seconds() if self.resumed_at else 0.0
        held = len(self.held)

        self.gaps_repaired += 1
        self.state = self.IDLE
        self.gap_started_at = None
        self.stale_symbols = []
        self.resumed_at = None
        self.next_attempt_at = None
        self.held = []
        return (
            f"Gap repaired ({outage:.0f}s outage): indicators rebuilt from {rebuilt} verified "
            f"session bars ({held} live bars superseded); signals re-enabled"
        )
