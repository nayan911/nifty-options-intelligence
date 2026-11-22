"""Unit tests for HistoricalCandleManager, pre-market warmup, and DuckDB incremental caching."""

import threading
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.core.interfaces import Candle
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.data.storage.duckdb_store import DuckDBStore

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_hist.duckdb"
    return DuckDBStore(db_path=str(db_file))


def make_candle(symbol: str, timeframe: str, dt: datetime, o: float, h: float, l: float, c: float, v: int = 1000):
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=dt,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        oi=100000,
        is_closed=True,
    )


def test_duckdb_candle_batch_and_query_methods(store):
    """Verify DuckDBStore batch insertion, latest timestamp, daily query, and recent candles."""
    sym = "NSE:NIFTY50-INDEX"
    t0 = IST.localize(datetime(2026, 11, 20, 10, 0))

    # 1. Initially empty
    assert store.get_latest_candle_timestamp(sym, "5m") is None
    assert store.get_latest_completed_daily_candle(sym, before_date=date(2026, 11, 23)) is None
    assert store.get_recent_candles(sym, "5m", limit=10) == []

    # 2. Batch insert 5m candles
    c1 = make_candle(sym, "5m", t0, 25100.0, 25120.0, 25090.0, 25110.0)
    c2 = make_candle(sym, "5m", t0 + timedelta(minutes=5), 25110.0, 25130.0, 25105.0, 25125.0)
    c3 = make_candle(sym, "5m", t0 + timedelta(minutes=10), 25125.0, 25140.0, 25115.0, 25135.0)

    inserted = store.insert_candles_batch([c1, c2, c3])
    assert inserted == 3

    latest_ts = store.get_latest_candle_timestamp(sym, "5m")
    assert latest_ts == t0 + timedelta(minutes=10)

    # 3. Query recent candles returns chronologically ascending
    recent = store.get_recent_candles(sym, "5m", limit=2)
    assert len(recent) == 2
    assert recent[0].timestamp == t0 + timedelta(minutes=5)
    assert recent[1].timestamp == t0 + timedelta(minutes=10)

    # 4. Insert completed Daily candle
    d_candle = make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25150.0, 24980.0, 25110.0)
    store.insert_candle(d_candle)

    res_d = store.get_latest_completed_daily_candle(sym, before_date=date(2026, 11, 23))
    assert res_d is not None
    assert res_d.high == 25150.0
    assert res_d.low == 24980.0
    assert res_d.close == 25110.0


def test_rate_limiter_sync_acquire():
    """Verify AsyncTokenBucketRateLimiter acquire_sync allows thread-safe synchronous token consumption."""
    limiter = AsyncTokenBucketRateLimiter(rate_per_sec=10.0, rate_per_min=100, max_daily_requests=500)

    # Acquire 3 tokens synchronously
    limiter.acquire_sync()
    limiter.acquire_sync()
    limiter.acquire_sync()
    assert limiter.daily_requests_count == 3

    # Multi-threaded concurrent test
    def worker():
        for _ in range(5):
            limiter.acquire_sync()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert limiter.daily_requests_count == 18


def test_historical_manager_incremental_sync(store):
    """Verify HistoricalCandleManager only fetches missing delta from broker."""
    sym = "NSE:NIFTY50-INDEX"
    t_nov20 = IST.localize(datetime(2026, 11, 20, 15, 30))
    d_candle = make_candle(sym, "5m", t_nov20, 25100.0, 25120.0, 25090.0, 25110.0)
    store.insert_candle(d_candle)

    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # 1. When DB already has candles up to target date, should return 0 without broker call
    count = manager.sync_candles(sym, "5m", days_back=2, end_date=date(2026, 11, 20))
    assert count == 0
    mock_rest.get_historical_candles_sync.assert_not_called()

    # 2. When DB is behind (e.g. asking for Nov 23), broker is called for delta
    new_bar = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25130.0, 25110.0, 25125.0)
    mock_rest.get_historical_candles_sync.return_value = [new_bar]

    count_new = manager.sync_candles(sym, "5m", days_back=5, end_date=date(2026, 11, 23))
    assert count_new == 1
    mock_rest.get_historical_candles_sync.assert_called_once()
    assert store.get_latest_candle_timestamp(sym, "5m") == new_bar.timestamp


def test_historical_manager_get_daily_reference_levels(store):
    """Verify HistoricalCandleManager extracts PDH, PDL, PDC from prior completed daily candle."""
    sym = "NSE:NIFTY50-INDEX"
    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # Seed daily candle in store
    d_candle = make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25180.5, 24950.25, 25120.0)
    store.insert_candle(d_candle)

    levels = manager.get_daily_reference_levels(sym, as_of_date=date(2026, 11, 23))
    assert levels is not None
    assert levels["PDH"] == 25180.5
    assert levels["PDL"] == 24950.25
    assert levels["PDC"] == 25120.0
    assert levels["CDO"] == 25120.0


def test_pre_market_warmup_bootstraps_technical_engine_with_zero_trade_signals(store):
    """Verify that warming up TechnicalAnalysisEngine with historical candles sets is_warmed_up=True

    and seeds daily reference levels, without firing mock setup candidates or trades.
    """
    sym = "NSE:NIFTY50-INDEX"
    fut = "NSE:NIFTY26NOVFUT"

    # Seed daily candle for Friday Nov 20
    d_candle = make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25200.0, 24950.0, 25100.0)
    store.insert_candle(d_candle)

    # Seed 30 5m candles and 30 15m candles from Friday Nov 20
    t_start = IST.localize(datetime(2026, 11, 20, 9, 15))
    candles_5m = []
    for i in range(30):
        t = t_start + timedelta(minutes=5 * i)
        candles_5m.append(make_candle(sym, "5m", t, 25000.0 + i, 25010.0 + i, 24995.0 + i, 25005.0 + i))
    store.insert_candles_batch(candles_5m)

    candles_15m = []
    for i in range(25):
        t = t_start + timedelta(minutes=15 * i)
        candles_15m.append(make_candle(sym, "15m", t, 25000.0 + i, 25020.0 + i, 24990.0 + i, 25010.0 + i))
    store.insert_candles_batch(candles_15m)

    # Setup TechnicalAnalysisEngine with spy callbacks
    mock_setup_cb = MagicMock()
    mock_candidate_cb = MagicMock()

    engine = TechnicalAnalysisEngine(
        primary_index=sym,
        primary_futures=fut,
        on_setup_candidate=mock_setup_cb,
        on_technical_candidate=mock_candidate_cb,
    )

    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # Execute Pre-Market Warmup for Monday Nov 23
    summary = manager.warm_up_technical_engine(
        engine=engine,
        symbols=[sym],
        as_of_date=date(2026, 11, 23),
        candle_limit_5m=30,
        candle_limit_15m=25,
    )
    assert sym in summary
    assert summary[sym]["is_5m_warmed_up"] is True
    assert summary[sym]["bootstrapped_candles"] == 55

    # 1. Verify daily reference levels were seeded on the engine
    state = engine.states[sym]
    assert state.daily_reference_levels is not None
    assert state.daily_reference_levels["PDH"] == 25200.0
    assert state.daily_reference_levels["PDL"] == 24950.0
    assert state.daily_reference_levels["PDC"] == 25100.0

    # 2. Verify indicators are now fully warmed up!
    snap_5m = state.latest_snapshots.get("5m")
    assert snap_5m is not None
    assert snap_5m.is_warmed_up is True
    assert snap_5m.ema_9 is not None
    assert snap_5m.ema_21 is not None
    assert snap_5m.atr is not None

    # 3. CRITICAL SAFETY GUARD: Zero mock trade signals were generated during bootstrap
    mock_setup_cb.assert_not_called()
    mock_candidate_cb.assert_not_called()

    # 4. Monday opening session isolation: First live candle arrives at 09:15:00
    mon_1m_candle = make_candle(sym, "1m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25110.0, 25125.0, 25105.0, 25120.0)
    snap_live = engine.on_candle(mon_1m_candle)
    assert snap_live is not None

    # Daily references for Monday must still be intact
    assert state.daily_reference_levels["PDH"] == 25200.0
    # Prior 5m indicators remain warmed up
    assert state.latest_snapshots["5m"].is_warmed_up is True
