"""Historical Candle Manager and Pre-Market Warmup Orchestrator.

Manages incremental caching of multi-timeframe completed candles from FYERS into DuckDB:
1. Incremental Sync: Checks DuckDB for latest available bar per symbol/timeframe, fetching only the missing delta.
2. Standard Retention: Enforces clean storage boundaries (Daily, 15m, 5m, 1m).
3. Pre-Market Warmup: Seeds Prior Day High/Low/Close/Open into TechnicalAnalysisEngine.
4. Historical Bootstrap: Chronologically replays completed candles into TechnicalAnalysisEngine with zero signal generation,
   so moving averages, ATR, swings, and trend structure are 100% warmed up (is_warmed_up=True) before 09:15:00 AM.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.core.calendar import ExchangeCalendar
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

    def sync_candles(
        self,
        symbol: str,
        timeframe: str,
        days_back: int = 5,
        end_date: Optional[date] = None,
    ) -> int:
        """Incrementally syncs completed candles from FYERS to DuckDB."""
        to_d = end_date or now_ist().date()
        from_d = to_d - timedelta(days=days_back)

        resolution = self.TIMEFRAME_TO_RESOLUTION.get(timeframe, timeframe)

        # Check existing latest stored candle in DuckDB
        latest_stored = self.store.get_latest_candle_timestamp(symbol, timeframe)
        if latest_stored is not None:
            latest_d = latest_stored.date()
            if latest_d >= to_d:
                # Up to date
                return 0
            # Only fetch starting from the day of latest stored candle
            from_d = max(from_d, latest_d)

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
        """Backfills historical Daily, 15m, and 5m bars for all tracked assets."""
        target_date = as_of_date or now_ist().date()
        results: Dict[str, Dict[str, int]] = {}

        for sym in symbols:
            results[sym] = {}
            # 1. Daily candles (last 30 calendar days)
            results[sym]["D"] = self.sync_candles(sym, "D", days_back=30, end_date=target_date)
            # 2. 15m candles (last 10 calendar days)
            results[sym]["15m"] = self.sync_candles(sym, "15m", days_back=10, end_date=target_date)
            # 3. 5m candles (last 5 calendar days)
            results[sym]["5m"] = self.sync_candles(sym, "5m", days_back=5, end_date=target_date)

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
            return {
                "PDH": round(daily_candle.high, 2),
                "PDL": round(daily_candle.low, 2),
                "PDC": round(daily_candle.close, 2),
                "CDO": round(daily_candle.close, 2),
            }

        return None

    def warm_up_technical_engine(
        self,
        engine: TechnicalAnalysisEngine,
        symbols: List[str],
        as_of_date: Optional[date] = None,
        candle_limit_5m: int = 100,
        candle_limit_15m: int = 50,
    ) -> Dict[str, Any]:
        """Pre-seeds daily references and bootstraps recent 15m & 5m historical candles into TechnicalAnalysisEngine."""
        target_date = as_of_date or now_ist().date()
        warmup_summary: Dict[str, Any] = {}

        for sym in symbols:
            # 1. Query recent 15m and 5m candles up to start of target_date
            as_of_cutoff = IST.localize(datetime.combine(target_date, datetime.min.time()))
            recent_15m = self.store.get_recent_candles(sym, "15m", limit=candle_limit_15m, as_of=as_of_cutoff)
            recent_5m = self.store.get_recent_candles(sym, "5m", limit=candle_limit_5m, as_of=as_of_cutoff)

            # Interleave and bootstrap chronologically
            combined_history = recent_15m + recent_5m
            bootstrapped_count = engine.bootstrap_candles(combined_history)

            # 2. Advance session to target_date (resets intraday-specific state while preserving warmed indicators)
            engine.prepare_session(target_date)

            # 3. Seed Prior Day Levels (PDH, PDL, PDC, CDO) on the prepared target session
            levels = self.get_daily_reference_levels(sym, as_of_date=target_date)
            if levels:
                engine.set_daily_references(
                    symbol=sym,
                    pdh=levels["PDH"],
                    pdl=levels["PDL"],
                    pdc=levels["PDC"],
                    cdo=levels["CDO"],
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
            is_warmed = snap_5m.is_warmed_up if snap_5m else False

            warmup_summary[sym] = {
                "daily_levels": levels,
                "bootstrapped_candles": bootstrapped_count,
                "is_5m_warmed_up": is_warmed,
            }

        return warmup_summary
