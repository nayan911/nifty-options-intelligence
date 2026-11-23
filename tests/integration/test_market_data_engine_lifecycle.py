"""Integration tests for MarketDataEngine lifecycle, exactly-once candle handling, and dynamic rotation."""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pytz

# Add scripts directory to path to import MarketDataEngine
scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_market_data import MarketDataEngine

from nifty_engine.core.interfaces import Candle, SessionState, SystemReadiness, Tick
from nifty_engine.data.fyers.instrument_master import NSE_FO_COLUMNS
from nifty_engine.data.storage.duckdb_store import DuckDBStore

IST = pytz.timezone("Asia/Kolkata")


def test_market_data_engine_exactly_once_candle_close():
    """Verify that clock-driven candle finalization executes exactly once:
    - Exactly one on_candle_close callback
    - Exactly one database row inserted
    - Exactly one increment of engine.candle_count
    """
    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    base_time = IST.localize(datetime(2026, 11, 23, 9, 15, 10))

    # Feed 2 ticks to start 1m candle for NIFTY futures
    t1 = Tick(
        symbol="NSE:NIFTY26NOVFUT",
        exchange_timestamp=base_time,
        local_timestamp=base_time,
        ltp=25000.0,
        cumulative_volume=1000,
        incremental_volume=0,
    )
    t2 = Tick(
        symbol="NSE:NIFTY26NOVFUT",
        exchange_timestamp=base_time + timedelta(seconds=20),
        local_timestamp=base_time + timedelta(seconds=20),
        ltp=25020.0,
        cumulative_volume=1100,
        incremental_volume=100,
    )

    # Process through engine during active session
    engine._on_tick_received(t1, current_time=base_time)
    engine._on_tick_received(t2, current_time=base_time + timedelta(seconds=20))

    assert engine.candle_count == 0  # Candle is still active, not closed yet

    # Advance clock past interval boundary (09:16:05) and flush
    expiry_time = IST.localize(datetime(2026, 11, 23, 9, 16, 5))
    engine.builder.flush_closed_bars(expiry_time)

    # Invariant: candle_count must be EXACTLY 1, NOT 2!
    assert engine.candle_count == 1

    # Invariant: Exactly 1 row in DuckDB candles table for 1m
    with store._lock:
        candles_count = store._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = 'NSE:NIFTY26NOVFUT' AND timeframe = '1m'"
        ).fetchone()[0]
    assert candles_count == 1

    store.close()


def test_market_data_engine_dynamic_option_rotation():
    """Verify that option contract watchlist rotates when spot price crosses strike boundary."""
    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    # Deterministic mock master fixture (zero network/disk dependency for CI)
    exp_date = date.today() + timedelta(days=5)
    exp_dt = datetime.combine(exp_date, time(15, 30), tzinfo=IST)
    expiry_epoch = int(exp_dt.timestamp())

    rows = []
    for s in [24950.0, 25000.0, 25050.0, 25100.0]:
        for opt in ["CE", "PE"]:
            rows.append([
                f"1011{int(s)}{opt}", f"NIFTY {int(s)} {opt}", 14, 65, 0.05, "",
                "0915-1540|1815-1915:", str(exp_date), expiry_epoch, f"NSE:NIFTY{int(s)}{opt}",
                10, 11, 15000, "NIFTY", 26000, s, opt, "101000000026000", "None", 0, 0.0
            ])

    df = pd.DataFrame(rows, columns=NSE_FO_COLUMNS)
    engine.master.load_from_dataframe(df)

    engine.active_atm_strike = 25000.0
    engine.subscribed_option_symbols = {
        "NSE:NIFTY24950CE",
        "NSE:NIFTY24950PE",
        "NSE:NIFTY25000CE",
        "NSE:NIFTY25000PE",
        "NSE:NIFTY25050CE",
        "NSE:NIFTY25050PE",
    }

    # Simulate spot moving +60 points to 25060 (ATM moves from 25000 -> 25050)
    engine.current_spot_price = 25060.0
    engine.check_and_rotate_options()

    assert engine.active_atm_strike == 25050.0
    # 24950 strikes should be rotated out and 25100 strikes rotated in
    assert "NSE:NIFTY24950CE" not in engine.subscribed_option_symbols
    assert "NSE:NIFTY24950PE" not in engine.subscribed_option_symbols
    assert "NSE:NIFTY25100CE" in engine.subscribed_option_symbols
    assert "NSE:NIFTY25100PE" in engine.subscribed_option_symbols

    store.close()


def test_market_data_engine_mixed_timeframe_handling():
    """Verify that MarketDataEngine handles 1m, 5m, and 15m candles cleanly:
    - 1m, 5m, and 15m bars are saved to DuckDB with exact timeframes.
    - TechnicalAnalysisEngine maintains isolated histories for 1m, 5m, and 15m.
    - Cross-timeframe pollution is completely eliminated.
    """
    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    sym = "NSE:NIFTY26NOVFUT"
    t1 = IST.localize(datetime(2026, 11, 23, 9, 15))

    c1m = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t1,
        open=25000.0,
        high=25010.0,
        low=24995.0,
        close=25005.0,
        volume=1000,
        is_closed=True,
    )
    c5m = Candle(
        symbol=sym,
        timeframe="5m",
        timestamp=t1,
        open=25000.0,
        high=25040.0,
        low=24990.0,
        close=25035.0,
        volume=5000,
        is_closed=True,
    )
    c15m = Candle(
        symbol=sym,
        timeframe="15m",
        timestamp=t1,
        open=25000.0,
        high=25060.0,
        low=24980.0,
        close=25050.0,
        volume=15000,
        is_closed=True,
    )

    # Process all three closed timeframes
    engine._on_candle_closed(c1m)
    engine._on_candle_closed(c5m)
    engine._on_candle_closed(c15m)

    # Verify DuckDB persistence has 1 row per timeframe
    with store._lock:
        cnt_1m = store._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = '1m'", [sym]
        ).fetchone()[0]
        cnt_5m = store._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = '5m'", [sym]
        ).fetchone()[0]
        cnt_15m = store._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = '15m'", [sym]
        ).fetchone()[0]

    assert cnt_1m == 1
    assert cnt_5m == 1
    assert cnt_15m == 1

    # Verify TechnicalAnalysisEngine state isolation
    state = engine.technical_engine.states[sym]
    assert len(state.candles_1m) == 1
    assert len(state.candles_5m) == 1
    assert len(state.candles_15m) == 1
    assert state.candles_1m[0].timeframe == "1m"
    assert state.candles_5m[0].timeframe == "5m"
    assert state.candles_15m[0].timeframe == "15m"

    store.close()




def test_market_data_engine_derivatives_integration():
    """Verify P0: MarketDataEngine initializes DerivativesIntelligenceEngine and connects live option chain polling."""
    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    # 1. Verify derivatives_engine was initialized on MarketDataEngine and wired to TechnicalAnalysisEngine
    assert engine.derivatives_engine is not None
    assert engine.technical_engine.derivatives_engine is engine.derivatives_engine

    # 2. Simulate option chain payload arrival
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    from nifty_engine.core.interfaces import OptionChainSnapshot, OptionChainStrike
    chain = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={"symbol": "NSE:NIFTY25100CE", "oi": 100000, "ltp": 80.0, "volume": 10000},
                put={"symbol": "NSE:NIFTY25100PE", "oi": 80000, "ltp": 70.0, "volume": 8000},
            )
        ],
        total_call_oi=100000,
        total_put_oi=80000,
        pcr=0.8,
    )

    # Dispatch to technical engine (as done in _poll_option_chain_if_due)
    engine.technical_engine.on_option_chain(chain, as_of=t0)

    # Verify OI profile and PCR metrics were populated in derivatives engine
    profile = engine.derivatives_engine.oi_analyzer.get_profile(underlying="NIFTY", as_of=t0)
    pcr = engine.derivatives_engine.pcr_engine.get_pcr(underlying="NIFTY", as_of=t0)

    assert profile is not None
    assert profile.spot_price == 25100.0
    assert pcr is not None
    assert pcr.aggregate_oi_pcr == 0.8

    store.close()


def test_market_data_engine_two_way_readiness_monitoring(monkeypatch):
    """Verify continuous two-way readiness re-evaluation:
    - Safe fail-closed default (NOT_READY)
    - Automatic upgrade on indicator warm-up (NOT_READY -> DEGRADED -> READY)
    - Safety downgrade during live trading when feed becomes stale or WS drops
    - Recovery upgrade when fresh tick arrives
    """
    from unittest.mock import MagicMock

    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    # 1. Initial State: Paper engine starts in fail-closed NOT_READY
    assert engine.paper_engine.system_readiness == SystemReadiness.NOT_READY

    # 2. Mock calendar to indicate regular trading session
    t_now = IST.localize(datetime(2026, 11, 23, 10, 0, 0))
    monkeypatch.setattr(engine.calendar, "get_session_state", lambda is_derivative=True: SessionState.REGULAR_CASH)
    monkeypatch.setattr("run_market_data.now_ist", lambda: t_now)

    # 3. Initially, indicators are cold -> evaluate_and_update_readiness returns NOT_READY
    r1 = engine.evaluate_and_update_readiness()
    assert r1 == SystemReadiness.NOT_READY
    assert engine.paper_engine.system_readiness == SystemReadiness.NOT_READY

    # 4. Seed daily levels and simulate 5m indicators warmed up, 1m cold -> DEGRADED
    from nifty_engine.core.config import settings
    sym_idx = settings.PRIMARY_INDEX
    sym_fut = engine.technical_engine.primary_futures

    engine.technical_engine.prepare_session(date(2026, 11, 23))
    engine.technical_engine.set_daily_references(sym_idx, pdh=25200.0, pdl=24900.0, pdc=25100.0)
    engine.technical_engine.set_daily_references(sym_fut, pdh=25250.0, pdl=24950.0, pdc=25150.0)

    snap_5m_mock = MagicMock()
    snap_5m_mock.is_warmed_up = True
    snap_5m_mock.trend_bias = "BULLISH"
    engine.technical_engine.states[sym_idx].latest_snapshots["5m"] = snap_5m_mock
    engine.technical_engine.states[sym_fut].latest_snapshots["5m"] = snap_5m_mock

    # WebSocket connected and feed fresh, but 1m indicators still cold -> DEGRADED
    engine.ws_client._is_connected = True
    engine.last_primary_tick_time = t_now

    r2 = engine.evaluate_and_update_readiness()
    assert r2 == SystemReadiness.DEGRADED
    assert engine.paper_engine.system_readiness == SystemReadiness.DEGRADED

    # 5. Simulate 1m indicators warming up -> UPGRADE to READY
    snap_1m_mock = MagicMock()
    snap_1m_mock.is_warmed_up = True
    snap_1m_mock.trend_bias = "BULLISH"
    engine.technical_engine.states[sym_idx].latest_snapshots["1m"] = snap_1m_mock
    engine.technical_engine.states[sym_fut].latest_snapshots["1m"] = snap_1m_mock

    r3 = engine.evaluate_and_update_readiness()
    assert r3 == SystemReadiness.READY
    assert engine.paper_engine.system_readiness == SystemReadiness.READY

    # 6. Safety Downgrade: Time advances by 70s without tick -> DOWNGRADE to DEGRADED
    monkeypatch.setattr("run_market_data.now_ist", lambda: t_now + timedelta(seconds=70))
    r4 = engine.evaluate_and_update_readiness()
    assert r4 == SystemReadiness.DEGRADED
    assert engine.paper_engine.system_readiness == SystemReadiness.DEGRADED

    # 7. Safety Downgrade: Time advances by 130s without tick -> DOWNGRADE to NOT_READY
    monkeypatch.setattr("run_market_data.now_ist", lambda: t_now + timedelta(seconds=130))
    r5 = engine.evaluate_and_update_readiness()
    assert r5 == SystemReadiness.NOT_READY
    assert engine.paper_engine.system_readiness == SystemReadiness.NOT_READY

    # 8. Recovery: Fresh tick arrives! -> UPGRADE back to READY
    t_recovered = t_now + timedelta(seconds=135)
    monkeypatch.setattr("run_market_data.now_ist", lambda: t_recovered)
    engine.last_primary_tick_time = t_recovered
    r6 = engine.evaluate_and_update_readiness()
    assert r6 == SystemReadiness.READY
    assert engine.paper_engine.system_readiness == SystemReadiness.READY

    # 9. Safety Downgrade: WebSocket disconnection -> DOWNGRADE to NOT_READY
    engine.ws_client._is_connected = False
    r7 = engine.evaluate_and_update_readiness()
    assert r7 == SystemReadiness.NOT_READY
    assert engine.paper_engine.system_readiness == SystemReadiness.NOT_READY

    store.close()


def test_runner_drops_incomplete_initial_15m_bar_and_prevents_duplicate_close():
    """Verify that when MarketDataEngine starts at 09:25:
    1. Incomplete 09:15–09:30 15m candle is dropped and NOT inserted into DuckDB or forwarded.
    2. A delayed tick after flush does not emit duplicate closed bars.
    """
    store = DuckDBStore(db_path=":memory:")
    engine = MarketDataEngine(store=store)

    sym = "NSE:NIFTY26NOVFUT"
    # First tick arrives at 09:25:10
    t_start = IST.localize(datetime(2026, 11, 23, 9, 25, 10))
    t1 = Tick(symbol=sym, exchange_timestamp=t_start, local_timestamp=t_start, ltp=25700.0, cumulative_volume=1000, incremental_volume=0)
    engine._on_tick_received(t1, current_time=t_start)

    # Ticks continue through 09:29:50
    t_end = IST.localize(datetime(2026, 11, 23, 9, 29, 50))
    t2 = Tick(symbol=sym, exchange_timestamp=t_end, local_timestamp=t_end, ltp=25668.50, cumulative_volume=101790, incremental_volume=100790)
    engine._on_tick_received(t2, current_time=t_end)

    # Boundary flush at 09:30:00
    t_0930 = IST.localize(datetime(2026, 11, 23, 9, 30, 0))
    engine.builder.flush_closed_bars(t_0930)

    # Check DuckDB: 5m candle for 09:25 exists
    candles_5m = store.get_recent_candles(sym, "5m", limit=10)
    assert len(candles_5m) == 1
    assert candles_5m[0].timestamp == IST.localize(datetime(2026, 11, 23, 9, 25, 0))
    assert candles_5m[0].close == 25668.50

    # Incomplete 15m candle (09:15) was NOT inserted into DuckDB!
    candles_15m = store.get_recent_candles(sym, "15m", limit=10)
    assert len(candles_15m) == 0

    # Delayed tick arrives for 09:29:59
    t_delayed_ts = IST.localize(datetime(2026, 11, 23, 9, 29, 59))
    t_delayed = Tick(symbol=sym, exchange_timestamp=t_delayed_ts, local_timestamp=t_delayed_ts, ltp=25670.0, cumulative_volume=101800, incremental_volume=10)
    engine._on_tick_received(t_delayed, current_time=t_0930 + timedelta(seconds=1))

    # Flush again at 09:30:02
    engine.builder.flush_closed_bars(t_0930 + timedelta(seconds=2))

    # Count of 5m candles must remain strictly 1 (NO duplicate!)
    candles_5m_after = store.get_recent_candles(sym, "5m", limit=10)
    assert len(candles_5m_after) == 1
    assert candles_5m_after[0].close == 25668.50

    store.close()


