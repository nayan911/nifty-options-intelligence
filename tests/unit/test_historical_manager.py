"""Unit tests for HistoricalCandleManager, pre-market warmup, and DuckDB incremental caching."""

import threading
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.analysis.technical.support_resistance import LevelType
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
    assert levels["CDO"] is None


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

    # Daily references for Monday must still be intact and CDO must now be populated with actual live open
    assert state.daily_reference_levels["PDH"] == 25200.0
    assert state.daily_reference_levels["CDO"] == 25110.0

    # Dedicated CDO Support & Resistance zone must now be present with nominal price matching live opening price!
    cdo_zone = next((z for z in snap_live.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_zone is not None
    assert cdo_zone.nominal_price == 25110.0

    # Prior 5m indicators remain warmed up
    assert state.latest_snapshots["5m"].is_warmed_up is True


def test_multi_symbol_warmup_prevents_session_date_leak(store):
    """Verify that multi-symbol pre-market warmup bootstraps all symbols before session advance.

    Ensures that symbol 2 can bootstrap Friday candles without session date leakage from symbol 1,
    and both symbols have intact daily reference levels.
    """
    sym1 = "NSE:NIFTY50-INDEX"
    sym2 = "NSE:NIFTYBANK-INDEX"

    # Seed daily candle for Friday Nov 20 for both symbols
    store.insert_candle(make_candle(sym1, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25200.0, 24950.0, 25100.0))
    store.insert_candle(make_candle(sym2, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 52000.0, 52500.0, 51900.0, 52300.0))

    # Seed 30 5m candles for each
    t_start = IST.localize(datetime(2026, 11, 20, 9, 15))
    candles_s1 = [make_candle(sym1, "5m", t_start + timedelta(minutes=5 * i), 25000.0 + i, 25010.0 + i, 24995.0 + i, 25005.0 + i) for i in range(30)]
    candles_s2 = [make_candle(sym2, "5m", t_start + timedelta(minutes=5 * i), 52000.0 + i, 52020.0 + i, 51980.0 + i, 52010.0 + i) for i in range(30)]
    store.insert_candles_batch(candles_s1)
    store.insert_candles_batch(candles_s2)

    mock_setup_cb = MagicMock()
    mock_candidate_cb = MagicMock()
    engine = TechnicalAnalysisEngine(
        primary_index=sym1,
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=mock_setup_cb,
        on_technical_candidate=mock_candidate_cb,
    )

    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # Warmup both symbols for Monday Nov 23
    summary = manager.warm_up_technical_engine(
        engine=engine,
        symbols=[sym1, sym2],
        as_of_date=date(2026, 11, 23),
        candle_limit_5m=30,
        candle_limit_15m=0,
    )

    # Both symbols must be warmed up
    assert summary[sym1]["is_5m_warmed_up"] is True
    assert summary[sym2]["is_5m_warmed_up"] is True

    # Check state 1
    state1 = engine.states[sym1]
    assert state1.current_session_date == date(2026, 11, 23)
    assert state1.daily_reference_levels["PDH"] == 25200.0
    assert state1.daily_reference_levels["CDO"] is None

    # Check state 2 (was not corrupted or cleared by state 1's preparation)
    state2 = engine.states[sym2]
    assert state2.current_session_date == date(2026, 11, 23)
    assert state2.daily_reference_levels["PDH"] == 52500.0
    assert state2.daily_reference_levels["CDO"] is None

    # Zero trades generated
    mock_setup_cb.assert_not_called()
    mock_candidate_cb.assert_not_called()


def test_historical_manager_incremental_sync_fetches_intraday_afternoon_bars(store):
    """Verify that incremental sync does not skip afternoon bars when morning bars exist."""
    sym = "NSE:NIFTY50-INDEX"
    # Store only morning bar up to 10:00 AM on Friday Nov 20
    t_morning = IST.localize(datetime(2026, 11, 20, 10, 0))
    c_morning = make_candle(sym, "5m", t_morning, 25100.0, 25120.0, 25090.0, 25110.0)
    store.insert_candle(c_morning)

    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # Nov 20 is a completed historical day, but stored latest is 10:00 AM (earlier than 15:25 end of session)
    # The broker returns the full day including afternoon bars
    t_afternoon = IST.localize(datetime(2026, 11, 20, 15, 25))
    c_afternoon = make_candle(sym, "5m", t_afternoon, 25150.0, 25160.0, 25140.0, 25155.0)
    mock_rest.get_historical_candles_sync.return_value = [c_morning, c_afternoon]

    # Sync for Nov 20
    count = manager.sync_candles(sym, "5m", days_back=1, end_date=date(2026, 11, 20))

    # Must NOT skip! It must call broker and insert the afternoon bar
    mock_rest.get_historical_candles_sync.assert_called_once()
    assert count == 2
    assert store.get_latest_candle_timestamp(sym, "5m") == t_afternoon

    # Running sync a second time now that 15:25 is stored should return 0 without broker call
    mock_rest.reset_mock()
    count_again = manager.sync_candles(sym, "5m", days_back=1, end_date=date(2026, 11, 20))
    assert count_again == 0
    mock_rest.get_historical_candles_sync.assert_not_called()


def test_first_live_opening_candle_populates_cdo_sr_zones_across_timeframes():
    """Regression Test: Verify that first live opening candle populates CDO S/R zones across 1m, 5m, and 15m."""
    sym = "NSE:NIFTY50-INDEX"
    engine = TechnicalAnalysisEngine(
        primary_index=sym,
        primary_futures="NSE:NIFTY26NOVFUT",
    )

    # Pre-market warmup seeds session with CDO=None
    session_d = date(2026, 11, 23)
    engine.prepare_session(session_d)
    engine.set_daily_references(sym, pdh=25200.0, pdl=24950.0, pdc=25100.0, cdo=None)

    state = engine.states[sym]
    assert state.daily_reference_levels["CDO"] is None

    # First 1m live candle at 09:15 arrives with open = 25115.0
    c_1m_0915 = make_candle(sym, "1m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25130.0, 25110.0, 25125.0)
    snap_1m = engine.on_candle(c_1m_0915)
    assert snap_1m is not None

    # 1. State CDO must now be recorded as 25115.0
    assert state.daily_reference_levels["CDO"] == 25115.0
    assert state.session_open_price == 25115.0

    # 2. 1m S/R zones must include CDO at 25115.0
    cdo_1m = next((z for z in snap_1m.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_1m is not None
    assert cdo_1m.nominal_price == 25115.0

    # First 5m live candle arrives with open = 25115.0
    c_5m_0915 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25140.0, 25105.0, 25135.0)
    snap_5m = engine.on_candle(c_5m_0915)
    assert snap_5m is not None

    # 3. 5m S/R zones must include CDO at 25115.0
    cdo_5m = next((z for z in snap_5m.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_5m is not None
    assert cdo_5m.nominal_price == 25115.0

    # First 15m live candle arrives with open = 25115.0
    c_15m_0915 = make_candle(sym, "15m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25150.0, 25100.0, 25140.0)
    snap_15m = engine.on_candle(c_15m_0915)
    assert snap_15m is not None

    # 4. 15m S/R zones must include CDO at 25115.0
    cdo_15m = next((z for z in snap_15m.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_15m is not None
    assert cdo_15m.nominal_price == 25115.0

    # Second 1m live candle at 09:16 arrives with open = 25125.0
    c_1m_0916 = make_candle(sym, "1m", IST.localize(datetime(2026, 11, 23, 9, 16)), 25125.0, 25135.0, 25120.0, 25130.0)
    snap_1m_2 = engine.on_candle(c_1m_0916)
    assert snap_1m_2 is not None

    # 5. CDO must remain locked to the first opening price (25115.0), not second candle open
    assert state.daily_reference_levels["CDO"] == 25115.0
    cdo_1m_2 = next((z for z in snap_1m_2.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_1m_2 is not None
    assert cdo_1m_2.nominal_price == 25115.0


def test_late_startup_at_10am_retrieves_confirmed_915_opening_candle(store, monkeypatch):
    """Regression Test: Startup at 10:00 AM retrieves confirmed 09:15 candle and does not use 10:00 AM open."""
    sym = "NSE:NIFTY50-INDEX"

    # Seed daily candle for Nov 20
    store.insert_candle(make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25200.0, 24950.0, 25100.0))
    # Seed 09:15 AM opening candle for Monday Nov 23 (open = 25115.0)
    c_0915 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25130.0, 25110.0, 25125.0)
    store.insert_candle(c_0915)

    # Mock wall-clock time to 10:00 AM on Monday Nov 23
    mock_10am = IST.localize(datetime(2026, 11, 23, 10, 0))
    monkeypatch.setattr("nifty_engine.data.historical_manager.now_ist", lambda: mock_10am)

    mock_rest = MagicMock()
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    levels = manager.get_daily_reference_levels(sym, as_of_date=date(2026, 11, 23))
    assert levels is not None
    assert levels["PDH"] == 25200.0
    # CDO must be retrieved from the confirmed 09:15 candle!
    assert levels["CDO"] == 25115.0

    engine = TechnicalAnalysisEngine(
        primary_index=sym,
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    engine.prepare_session(date(2026, 11, 23))
    engine.set_daily_references(sym, pdh=levels["PDH"], pdl=levels["PDL"], pdc=levels["PDC"], cdo=levels["CDO"])

    state = engine.states[sym]
    assert state.daily_reference_levels["CDO"] == 25115.0
    assert state.session_open_price == 25115.0

    # First live candle seen by engine arrives from 10:00 AM with open = 25180.0
    c_1000 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 10, 0)), 25180.0, 25195.0, 25175.0, 25190.0)
    snap_1000 = engine.on_candle(c_1000)
    assert snap_1000 is not None

    # CRITICAL: CDO must remain 25115.0, NOT 25180.0!
    assert state.daily_reference_levels["CDO"] == 25115.0
    assert state.session_open_price == 25115.0
    cdo_zone = next((z for z in snap_1000.sr_zones if z.level_type == LevelType.CDO), None)
    assert cdo_zone is not None
    assert cdo_zone.nominal_price == 25115.0


def test_late_startup_at_10am_missing_opening_candle_does_not_use_later_candle(store, monkeypatch):
    """Regression Test: If 09:15 opening candle is completely missing, 10:00 AM candle does NOT become CDO."""
    sym = "NSE:NIFTY50-INDEX"

    # Seed daily candle for Nov 20, but DO NOT seed any 09:15 candle for Nov 23
    store.insert_candle(make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25200.0, 24950.0, 25100.0))

    # Mock wall-clock time to 10:00 AM on Monday Nov 23
    mock_10am = IST.localize(datetime(2026, 11, 23, 10, 0))
    monkeypatch.setattr("nifty_engine.data.historical_manager.now_ist", lambda: mock_10am)

    # Mock REST client history returns empty (unable to recover 09:15 candle)
    mock_rest = MagicMock()
    mock_rest.get_historical_candles_sync.return_value = []
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    levels = manager.get_daily_reference_levels(sym, as_of_date=date(2026, 11, 23))
    assert levels is not None
    assert levels["PDH"] == 25200.0
    assert levels["CDO"] is None

    engine = TechnicalAnalysisEngine(
        primary_index=sym,
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    engine.prepare_session(date(2026, 11, 23))
    engine.set_daily_references(sym, pdh=levels["PDH"], pdl=levels["PDL"], pdc=levels["PDC"], cdo=levels["CDO"])

    state = engine.states[sym]
    assert state.daily_reference_levels["CDO"] is None
    assert state.session_open_price is None

    # First live candle arrives at 10:00 AM with open = 25180.0
    c_1000 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 10, 0)), 25180.0, 25195.0, 25175.0, 25190.0)
    snap_1000 = engine.on_candle(c_1000)
    assert snap_1000 is not None

    # CRITICAL: Because candle timestamp is 10:00 AM and NOT 09:15 AM, CDO must remain None
    assert state.daily_reference_levels["CDO"] is None
    assert state.session_open_price is None
    assert not any(z.level_type == LevelType.CDO for z in snap_1000.sr_zones)


def test_late_startup_recovers_missing_915_opening_candle_even_when_later_bars_exist(store, monkeypatch):
    """Regression Test: Recover 09:15 opening candle from broker even if DuckDB already has a later bar from today (e.g. 09:55 AM)."""
    sym = "NSE:NIFTY50-INDEX"

    # Seed daily candle for Nov 20
    store.insert_candle(make_candle(sym, "D", IST.localize(datetime(2026, 11, 20, 15, 30)), 25000.0, 25200.0, 24950.0, 25100.0))

    # DuckDB already has 09:55 AM bar from today, but 09:15 AM opening bar is MISSING
    c_0955 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 9, 55)), 25170.0, 25185.0, 25165.0, 25180.0)
    store.insert_candle(c_0955)

    # Mock wall-clock time to 10:00 AM
    mock_10am = IST.localize(datetime(2026, 11, 23, 10, 0))
    monkeypatch.setattr("nifty_engine.data.historical_manager.now_ist", lambda: mock_10am)

    # Broker returns the confirmed 09:15 AM candle
    c_0915 = make_candle(sym, "5m", IST.localize(datetime(2026, 11, 23, 9, 15)), 25115.0, 25130.0, 25110.0, 25125.0)
    mock_rest = MagicMock()
    mock_rest.get_historical_candles_sync.return_value = [c_0915]
    manager = HistoricalCandleManager(rest_client=mock_rest, store=store)

    # Query daily references
    levels = manager.get_daily_reference_levels(sym, as_of_date=date(2026, 11, 23))

    # Must NOT skip! Must call broker and recover the 09:15 opening price
    mock_rest.get_historical_candles_sync.assert_called_once()
    assert levels is not None
    assert levels["CDO"] == 25115.0
