"""Integration tests for MarketDataEngine lifecycle, exactly-once candle handling, and dynamic rotation."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# Add scripts directory to path to import MarketDataEngine
scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_market_data import MarketDataEngine

from nifty_engine.core.interfaces import Tick
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
    # Load cached instrument master fixture
    engine.master.load_master(auto_download=False)

    engine.active_atm_strike = 25000.0
    engine.subscribed_option_symbols = {
        "NSE:NIFTY26NOV24950CE",
        "NSE:NIFTY26NOV24950PE",
        "NSE:NIFTY26NOV25000CE",
        "NSE:NIFTY26NOV25000PE",
        "NSE:NIFTY26NOV25050CE",
        "NSE:NIFTY26NOV25050PE",
    }

    # Simulate spot moving +60 points to 25060 (ATM moves from 25000 -> 25050)
    engine.current_spot_price = 25060.0
    engine.check_and_rotate_options()

    assert engine.active_atm_strike == 25050.0

    store.close()
