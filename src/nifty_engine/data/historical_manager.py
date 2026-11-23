"""Historical Candle Manager and Pre-Market Warmup Orchestrator.

Manages incremental caching of multi-timeframe completed candles from FYERS into DuckDB:
1. Incremental Sync: Checks DuckDB for latest available bar per symbol/timeframe, fetching only the missing delta.
2. Standard Retention: Enforces clean storage boundaries (Daily, 15m, 5m, 1m).
3. Pre-Market Warmup: Seeds Prior Day High/Low/Close/Open into TechnicalAnalysisEngine.
4. Historical Bootstrap: Chronologically replays completed candles into TechnicalAnalysisEngine with zero signal generation,
   so moving averages, ATR, swings, and trend structure are 100% warmed up (is_warmed_up=True) before 09:15:00 AM.
"""

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle, SystemReadiness
from nifty_engine.data.fyers.rest_client import FyersRestClient
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.utils.timezone import now_ist

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class HistoricalCandleManager:
    """Orchestrates historical candle backfilling, caching, and pre-market technical warmup."""

    TIMEFRAME_TO_RESOLUTION = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1d": "D",
        "D": "D",
    }

    def __init__(
        self,
        rest_client: FyersRestClient,
        store: DuckDBStore,
        calendar: Optional[ExchangeCalendar] = None,
    ) -> None:
        self.rest_client = rest_client
        self.store = store
        self.calendar = calendar or ExchangeCalendar()
        self.reconciliation_status: Dict[Tuple[str, str, date], bool] = {}
        self.reconciliation_errors: Dict[Tuple[str, str, date], str] = {}
        self.reconciliation_counts: Dict[Tuple[str, str, date], int] = {}
        self.reconciliation_required_sessions: Set[date] = set()
        # Bar start timestamps FYERS actually returned in the most recent reconcile call.
        # Timestamp coverage in DuckDB alone can be satisfied by a stale/partial live bar,
        # so feed-gap repair checks the affected intervals against this authoritative set.
        self.last_broker_returned: Dict[Tuple[str, str, date], Set[datetime]] = {}

    @classmethod
    def get_expected_closed_intervals(
        cls,
        session_date: date,
        timeframe: str,
        as_of: Optional[datetime] = None,
    ) -> List[datetime]:
        """Calculates expected completed interval start timestamps for session_date up to as_of.

        Session trading window: 09:15 to 15:30 IST.
        An interval starting at T with duration M minutes is closed when T + M <= min(as_of, 15:30).
        """
        now = as_of or now_ist()
        session_start = IST.localize(datetime.combine(session_date, time(9, 15)))
        session_end = IST.localize(datetime.combine(session_date, time(15, 30)))

        interval_mins = 15 if timeframe == "15m" else (5 if timeframe == "5m" else 1)
        bar_delta = timedelta(minutes=interval_mins)

        # If as_of is before session start or before first bar completion
        if now < session_start + bar_delta:
            return []

        effective_cutoff = min(now, session_end)
        expected: List[datetime] = []
        curr = session_start
        while curr + bar_delta <= effective_cutoff:
            expected.append(curr)
            curr += bar_delta

        return expected

    def is_symbol_reconciled(
        self,
        symbol: str,
        session_date: Optional[date] = None,
        timeframes: Optional[List[str]] = None,
    ) -> bool:
        """Returns True if reconciliation succeeded and bar coverage is complete for all requested timeframes."""
        s_date = session_date or now_ist().date()
        tfs = timeframes or ["15m", "5m", "1m"]
        return all(self.reconciliation_status.get((symbol, tf, s_date), False) for tf in tfs)

    def get_reconciliation_errors(
        self,
        symbol: str,
        session_date: Optional[date] = None,
    ) -> Dict[str, str]:
        """Returns map of {timeframe: error_message} for any failed reconciliation calls for session_date."""
        s_date = session_date or now_ist().date()
        return {
            tf: err
            for (sym, tf, dt), err in self.reconciliation_errors.items()
            if sym == symbol and dt == s_date
        }

    def sync_candles(
        self,
        symbol: str,
        timeframe: str,
        days_back: int = 5,
        end_date: Optional[date] = None,
        trading_days_back: Optional[int] = None,
    ) -> int:
        """Incrementally syncs completed candles from FYERS to DuckDB."""
        to_d = end_date or now_ist().date()
        if trading_days_back is not None:
            from_d = self.calendar.get_prior_trading_date(to_d, trading_days_back=trading_days_back)
        else:
            from_d = to_d - timedelta(days=days_back)

        resolution = self.TIMEFRAME_TO_RESOLUTION.get(timeframe, timeframe)

        # Check existing latest stored candle in DuckDB
        latest_stored = self.store.get_latest_candle_timestamp(symbol, timeframe)
        if latest_stored is not None:
            now = now_ist()
            if timeframe == "D":
                if latest_stored.date() >= to_d:
                    # Daily candle for to_d already stored
                    return 0
            else:
                # Intraday intervals (1m, 5m, 15m)
                bar_min = 15 if timeframe == "15m" else (5 if timeframe == "5m" else 1)
                bar_delta = timedelta(minutes=bar_min)

                if to_d < now.date():
                    # Historical completed trading day: session closes at 15:30 IST
                    session_end_bar = IST.localize(datetime.combine(to_d, time(15, 30))) - bar_delta
                    if latest_stored >= session_end_bar:
                        return 0
                elif to_d == now.date():
                    if now.time() < time(9, 15):
                        # Pre-market: no candles exist for today yet
                        if latest_stored.date() >= to_d:
                            return 0
                    elif now.time() >= time(15, 30):
                        session_end_bar = IST.localize(datetime.combine(to_d, time(15, 30))) - bar_delta
                        if latest_stored >= session_end_bar:
                            return 0
                    else:
                        # Intraday during live market: up to date if latest stored is the last closed bar
                        if latest_stored >= now - bar_delta:
                            return 0

            # Overlapping fetch starting from the day of latest stored candle to fill any missing bars
            from_d = max(from_d, latest_stored.date())

        date_from_str = from_d.strftime("%Y-%m-%d")
        date_to_str = to_d.strftime("%Y-%m-%d")

        try:
            candles = self.rest_client.get_historical_candles_sync(
                symbol=symbol,
                resolution=resolution,
                date_from=date_from_str,
                date_to=date_to_str,
            )
            # Filter strictly to completed candles
            completed_candles = [c for c in candles if c.is_closed]
            if completed_candles:
                count = self.store.insert_candles_batch(completed_candles)
                logger.info(
                    "Synced %d %s candles for %s (%s to %s)",
                    count,
                    timeframe,
                    symbol,
                    date_from_str,
                    date_to_str,
                )
                return count
        except Exception as e:
            logger.warning("Historical sync failed for %s (%s): %s", symbol, timeframe, e)

        return 0

    def sync_pre_market_data(
        self,
        symbols: List[str],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Backfills historical Daily, 15m, 5m, and 1m bars for all tracked assets."""
        target_date = as_of_date or now_ist().date()
        results: Dict[str, Dict[str, int]] = {}

        for sym in symbols:
            results[sym] = {}
            # 1. Daily candles (last 20 completed trading sessions ~ 30 calendar days)
            results[sym]["D"] = self.sync_candles(sym, "D", trading_days_back=20, end_date=target_date)
            # 2. 15m candles (last 7 completed trading sessions ~ 10 calendar days)
            results[sym]["15m"] = self.sync_candles(sym, "15m", trading_days_back=7, end_date=target_date)
            # 3. 5m candles (last 4 completed trading sessions ~ 5 calendar days)
            results[sym]["5m"] = self.sync_candles(sym, "5m", trading_days_back=4, end_date=target_date)
            # 4. 1m candles (last 3 completed trading sessions to ensure prior full sessions are captured even on Mondays)
            results[sym]["1m"] = self.sync_candles(sym, "1m", trading_days_back=3, end_date=target_date)

        return results

    def fetch_session_opening_candle(
        self,
        symbol: str,
        session_date: date,
    ) -> Optional[Candle]:
        """Directly fetches and persists the confirmed 09:15:00 opening candle for session_date from FYERS,

        bypassing latest-stored incremental sync shortcuts.
        """
        date_str = session_date.strftime("%Y-%m-%d")
        try:
            candles = self.rest_client.get_historical_candles_sync(
                symbol=symbol,
                resolution="5",
                date_from=date_str,
                date_to=date_str,
            )
            completed_candles = [c for c in candles if c.is_closed]
            if completed_candles:
                self.store.insert_candles_batch(completed_candles)
            return self.store.get_session_opening_candle(symbol, session_date=session_date)
        except Exception as e:
            logger.warning("Failed direct 09:15 opening candle fetch for %s on %s: %s", symbol, date_str, e)
            return None

    def reconcile_session_candles(
        self,
        symbols: List[str],
        target_date: Optional[date] = None,
        timeframes: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Reconciles and overwrites intraday candles for target_date against FYERS historical API.

        Fetches verified broker candles (15m, 5m, 1m) directly for target_date, batch-upserts
        them into DuckDB, and validates that all expected closed intervals are present in storage.
        """
        t_date = target_date or now_ist().date()
        date_str = t_date.strftime("%Y-%m-%d")
        tfs = timeframes or ["15m", "5m", "1m"]
        as_of_dt = as_of or now_ist()
        results: Dict[str, Dict[str, int]] = {}

        # Mark this session date as requiring verified reconciliation
        self.reconciliation_required_sessions.add(t_date)

        session_start = IST.localize(datetime.combine(t_date, time(9, 15)))
        session_end = IST.localize(datetime.combine(t_date, time(15, 30)))

        for sym in symbols:
            results[sym] = {}
            for tf in tfs:
                res = self.TIMEFRAME_TO_RESOLUTION.get(tf, tf)
                try:
                    candles = self.rest_client.get_historical_candles_sync(
                        symbol=sym,
                        resolution=res,
                        date_from=date_str,
                        date_to=date_str,
                    )
                    completed = [c for c in candles if c.is_closed]
                    self.last_broker_returned[(sym, tf, t_date)] = {c.timestamp for c in completed}
                    count = 0
                    if completed:
                        count = self.store.insert_candles_batch(completed)
                        logger.info(
                            "Reconciled %d %s candles for %s on %s",
                            count,
                            tf,
                            sym,
                            date_str,
                        )
                    results[sym][tf] = count

                    # Validate expected bar coverage:
                    expected_intervals = self.get_expected_closed_intervals(t_date, tf, as_of=as_of_dt)
                    expected_count = len(expected_intervals)

                    # Check store's total completed candles for today
                    stored_candles = self.store.get_candles(
                        sym,
                        tf,
                        start_time=session_start,
                        end_time=session_end,
                    )
                    stored_closed = [c for c in stored_candles if c.is_closed]
                    stored_timestamps = {c.timestamp for c in stored_closed}

                    missing_intervals = [ts for ts in expected_intervals if ts not in stored_timestamps]

                    if expected_count > 0 and missing_intervals:
                        error_msg = (
                            f"Incomplete candle coverage for {sym} ({tf}) on {date_str}: "
                            f"missing {len(missing_intervals)}/{expected_count} expected intervals "
                            f"(first missing: {missing_intervals[0].strftime('%H:%M:%S')})"
                        )
                        logger.warning(error_msg)
                        self.reconciliation_status[(sym, tf, t_date)] = False
                        self.reconciliation_counts[(sym, tf, t_date)] = len(stored_closed)
                        self.reconciliation_errors[(sym, tf, t_date)] = error_msg
                    else:
                        self.reconciliation_status[(sym, tf, t_date)] = True
                        self.reconciliation_counts[(sym, tf, t_date)] = len(stored_closed)
                        self.reconciliation_errors.pop((sym, tf, t_date), None)

                except Exception as e:
                    logger.warning("Reconciliation failed for %s (%s) on %s: %s", sym, tf, date_str, e)
                    self.last_broker_returned[(sym, tf, t_date)] = set()
                    results[sym][tf] = 0
                    self.reconciliation_status[(sym, tf, t_date)] = False
                    self.reconciliation_counts[(sym, tf, t_date)] = 0
                    self.reconciliation_errors[(sym, tf, t_date)] = str(e)

        return results

    def get_daily_reference_levels(
        self,
        symbol: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Dict[str, float]]:
        """Queries the prior trading day's completed Daily candle to extract PDH, PDL, PDC."""
        target_date = as_of_date or now_ist().date()
        daily_candle = self.store.get_latest_completed_daily_candle(symbol, before_date=target_date)

        if daily_candle is None:
            # Fallback: query 15 days of daily candles to seed DuckDB
            try:
                self.sync_candles(symbol, "D", days_back=15, end_date=target_date)
                daily_candle = self.store.get_latest_completed_daily_candle(symbol, before_date=target_date)
            except Exception as e:
                logger.warning("Failed on-demand daily candle fetch for %s: %s", symbol, e)

        if daily_candle:
            cdo_val: Optional[float] = None
            now = now_ist()
            # If target_date is in the past, or target_date is today and session has already commenced (>= 09:15 IST)
            if target_date < now.date() or (target_date == now.date() and now.time() >= time(9, 15)):
                opening_candle = self.store.get_session_opening_candle(symbol, session_date=target_date)
                if opening_candle is None:
                    # Late startup fallback: directly query FYERS for target_date to retrieve confirmed 09:15 bar
                    opening_candle = self.fetch_session_opening_candle(symbol, session_date=target_date)

                if opening_candle is not None:
                    cdo_val = round(opening_candle.open, 2)

            return {
                "PDH": round(daily_candle.high, 2),
                "PDL": round(daily_candle.low, 2),
                "PDC": round(daily_candle.close, 2),
                "CDO": cdo_val,
            }

        return None

    def assess_readiness(
        self,
        engine: TechnicalAnalysisEngine,
        symbols: List[str],
        require_reconciliation: Optional[bool] = None,
        as_of_date: Optional[date] = None,
    ) -> Tuple[SystemReadiness, Dict[str, Any]]:
        """Assesses system operational readiness across all tracked symbols.

        Rules:
        - READY: All tracked symbols have complete daily reference levels (PDH, PDL, PDC, and verified CDO if session active),
          both 5m and 1m indicators are warmed up, and broker reconciliation succeeded with full bar coverage for target_date.
        - DEGRADED: All tracked symbols have daily levels, verified data completeness/reconciliation, and 5m indicators are warmed up,
          but 1m indicators are still cold (< 21 bars).
        - NOT_READY: Any tracked symbol is missing daily reference levels, has cold 5m indicators, missing opening price during active session,
          or failed broker reconciliation / missing expected interval bars when required.
        """
        target_date = as_of_date or now_ist().date()
        now = now_ist()
        is_active_session = target_date < now.date() or (target_date == now.date() and now.time() >= time(9, 15))

        enforce_recon = (
            require_reconciliation
            if require_reconciliation is not None
            else (target_date in self.reconciliation_required_sessions)
        )

        details: Dict[str, Any] = {}
        all_have_daily = True
        all_have_cdo = True
        all_reconciled = True
        all_5m_warmed = True
        all_1m_warmed = True

        for sym in symbols:
            state = engine.states.get(sym)
            daily_refs = (state.daily_reference_levels if state else {}) or {}
            has_daily = bool(
                daily_refs.get("PDH") is not None
                and daily_refs.get("PDL") is not None
                and daily_refs.get("PDC") is not None
            )

            has_cdo = bool(daily_refs.get("CDO") is not None) if is_active_session else True

            is_reconciled = self.is_symbol_reconciled(sym, session_date=target_date) if enforce_recon else True
            recon_errors = self.get_reconciliation_errors(sym, session_date=target_date) if enforce_recon else {}

            snap_5m = state.latest_snapshots.get("5m") if state else None
            is_5m = snap_5m.is_warmed_up if snap_5m else False

            snap_1m = state.latest_snapshots.get("1m") if state else None
            is_1m = snap_1m.is_warmed_up if snap_1m else False

            snap_15m = state.latest_snapshots.get("15m") if state else None
            has_15m = bool(snap_15m and snap_15m.trend_bias is not None)

            data_complete = has_daily and has_cdo and is_reconciled

            details[sym] = {
                "has_daily_levels": has_daily,
                "has_cdo": has_cdo,
                "is_reconciled": is_reconciled,
                "reconciliation_errors": recon_errors,
                "data_complete": data_complete,
                "is_5m_warmed": is_5m,
                "is_1m_warmed": is_1m,
                "has_15m_context": has_15m,
            }

            if not has_daily:
                all_have_daily = False
            if not has_cdo:
                all_have_cdo = False
            if not is_reconciled:
                all_reconciled = False
            if not is_5m:
                all_5m_warmed = False
            if not is_1m:
                all_1m_warmed = False

        if not all_have_daily or not all_have_cdo or not all_5m_warmed or (enforce_recon and not all_reconciled):
            readiness = SystemReadiness.NOT_READY
        elif not all_1m_warmed:
            readiness = SystemReadiness.DEGRADED
        else:
            readiness = SystemReadiness.READY

        return readiness, details

    def warm_up_technical_engine(
        self,
        engine: TechnicalAnalysisEngine,
        symbols: List[str],
        as_of_date: Optional[date] = None,
        candle_limit_1m: int = 100,
        candle_limit_5m: int = 100,
        candle_limit_15m: int = 50,
        reconcile_session: bool = False,
    ) -> Dict[str, Any]:
        """Pre-seeds daily references and bootstraps recent 15m, 5m & 1m historical candles into TechnicalAnalysisEngine."""
        target_date = as_of_date or now_ist().date()
        as_of_cutoff = IST.localize(datetime.combine(target_date, datetime.min.time()))
        warmup_summary: Dict[str, Any] = {}

        # Stage 1: Bootstrap historical candles for ALL symbols first
        # Doing this before session preparation prevents multi-symbol session rollover leaks
        for sym in symbols:
            recent_15m = self.store.get_recent_candles(sym, "15m", limit=candle_limit_15m, as_of=as_of_cutoff)
            recent_5m = self.store.get_recent_candles(sym, "5m", limit=candle_limit_5m, as_of=as_of_cutoff)
            recent_1m = self.store.get_recent_candles(sym, "1m", limit=candle_limit_1m, as_of=as_of_cutoff)
            combined_history = recent_15m + recent_5m + recent_1m
            bootstrapped_count = engine.bootstrap_candles(combined_history)
            warmup_summary[sym] = {
                "bootstrapped_candles": bootstrapped_count,
            }

        # Stage 2: Advance engine session to target_date once across all symbols
        engine.prepare_session(target_date)

        # Stage 3: Seed Prior Day Levels (PDH, PDL, PDC, CDO=None) for all symbols on the prepared session
        for sym in symbols:
            levels = self.get_daily_reference_levels(sym, as_of_date=target_date)
            if levels:
                engine.set_daily_references(
                    symbol=sym,
                    pdh=levels["PDH"],
                    pdl=levels["PDL"],
                    pdc=levels["PDC"],
                    cdo=levels.get("CDO"),
                )
                logger.info(
                    "Seeded daily references for %s: PDH=%.2f, PDL=%.2f, PDC=%.2f",
                    sym,
                    levels["PDH"],
                    levels["PDL"],
                    levels["PDC"],
                )

            # Verify warmup state
            state = engine.states.get(sym)
            snap_5m = state.latest_snapshots.get("5m") if state else None
            snap_1m = state.latest_snapshots.get("1m") if state else None
            is_5m_warmed = snap_5m.is_warmed_up if snap_5m else False
            is_1m_warmed = snap_1m.is_warmed_up if snap_1m else False

            warmup_summary[sym]["daily_levels"] = levels
            warmup_summary[sym]["is_5m_warmed_up"] = is_5m_warmed
            warmup_summary[sym]["is_1m_warmed_up"] = is_1m_warmed

        # Stage 4: Current-Session Catch-Up Replay (Late Startup Recovery)
        # If target_date has already commenced (past date or wall clock > 09:15 IST),
        # replay all completed candles of the current session up to the cutoff without generating trades.
        now = now_ist()
        if target_date < now.date() or (target_date == now.date() and now.time() > time(9, 15)):
            if reconcile_session:
                try:
                    self.reconcile_session_candles(symbols, target_date=target_date)
                except Exception as e:
                    logger.warning("Session reconciliation prior to catchup failed: %s", e)

            session_start = IST.localize(datetime.combine(target_date, time(9, 15)))
            session_end = now if target_date == now.date() else IST.localize(datetime.combine(target_date, time(15, 30)))

            for sym in symbols:
                # Query completed intraday candles on target_date from 09:15 onwards
                session_15m = self.store.get_candles(sym, "15m", start_time=session_start, end_time=session_end)
                session_5m = self.store.get_candles(sym, "5m", start_time=session_start, end_time=session_end)
                session_1m = self.store.get_candles(sym, "1m", start_time=session_start, end_time=session_end)

                session_candles = [c for c in (session_15m + session_5m + session_1m) if c.is_closed]
                if session_candles:
                    catchup_count = engine.bootstrap_candles(session_candles)
                    warmup_summary[sym]["session_catchup_candles"] = catchup_count
                    logger.info(
                        "Replayed %d session catch-up candles for %s on %s",
                        catchup_count,
                        sym,
                        target_date,
                    )

                # Re-verify warmup state after session catch-up
                state = engine.states.get(sym)
                snap_5m = state.latest_snapshots.get("5m") if state else None
                snap_1m = state.latest_snapshots.get("1m") if state else None
                if snap_5m:
                    warmup_summary[sym]["is_5m_warmed_up"] = snap_5m.is_warmed_up
                if snap_1m:
                    warmup_summary[sym]["is_1m_warmed_up"] = snap_1m.is_warmed_up

        # Stage 5: Evaluate system operational readiness across all tracked symbols
        overall_readiness, readiness_details = self.assess_readiness(
            engine,
            symbols,
            require_reconciliation=reconcile_session and (target_date < now.date() or (target_date == now.date() and now.time() > time(9, 15))),
            as_of_date=target_date,
        )
        warmup_summary["_overall_readiness"] = overall_readiness
        warmup_summary["_readiness_details"] = readiness_details

        return warmup_summary
