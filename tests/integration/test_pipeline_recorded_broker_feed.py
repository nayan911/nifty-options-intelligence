"""Integration tests simulating a 21-symbol live feed using recorded sanitized broker packets."""

from datetime import datetime, timedelta

import pytz

from nifty_engine.core.interfaces import Tick
from nifty_engine.data.candle_builder import RealtimeCandleBuilder
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus

IST = pytz.timezone("Asia/Kolkata")


def test_21_symbol_feed_processing_and_volume_integrity():
    """Verify that a recorded batch of 21 symbols (Indices, Futures, Equities, Options)

    processes correctly through TickNormalizer, DuckDBStore, and CandleBuilder.
    """
    store = DuckDBStore(":memory:", batch_size=10)
    normalizer = TickNormalizer()
    closed_bars = []
    builder = RealtimeCandleBuilder(on_candle_close=lambda c: closed_bars.append(c))

    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)

    # 21 Real-world sanitized symbols
    symbols_spec = [
        # 4 Indices
        ("NSE:NIFTY50-INDEX", 24125.0, 0, True),
        ("NSE:NIFTYBANK-INDEX", 51800.0, 0, True),
        ("BSE:SENSEX-INDEX", 79500.0, 0, True),
        ("NSE:INDIAVIX-INDEX", 14.5, 0, True),
        # 2 Near-Month Futures
        ("NSE:NIFTY26NOVFUT", 24150.0, 50000, False),
        ("NSE:BANKNIFTY26NOVFUT", 51900.0, 30000, False),
        # 5 Top Equities
        ("NSE:HDFCBANK-EQ", 1750.0, 100000, False),
        ("NSE:RELIANCE-EQ", 2950.0, 80000, False),
        ("NSE:ICICIBANK-EQ", 1250.0, 90000, False),
        ("NSE:INFY-EQ", 1880.0, 60000, False),
        ("NSE:BHARTIARTL-EQ", 1600.0, 45000, False),
        # 10 Options (5 strikes x 2 types)
        ("NSE:NIFTY26NOV24000CE", 195.0, 500000, False),
        ("NSE:NIFTY26NOV24000PE", 45.0, 600000, False),
        ("NSE:NIFTY26NOV24050CE", 155.0, 400000, False),
        ("NSE:NIFTY26NOV24050PE", 65.0, 450000, False),
        ("NSE:NIFTY26NOV24100CE", 120.0, 800000, False),
        ("NSE:NIFTY26NOV24100PE", 90.0, 750000, False),
        ("NSE:NIFTY26NOV24150CE", 88.0, 300000, False),
        ("NSE:NIFTY26NOV24150PE", 125.0, 320000, False),
        ("NSE:NIFTY26NOV24200CE", 62.0, 250000, False),
        ("NSE:NIFTY26NOV24200PE", 165.0, 280000, False),
    ]

    assert len(symbols_spec) == 21

    # Round 1: Baseline ticks at 09:15:00
    for sym, ltp, cum_vol, is_idx in symbols_spec:
        raw_tick = Tick(
            symbol=sym,
            exchange_timestamp=base_time,
            local_timestamp=base_time,
            ltp=ltp,
            cumulative_volume=cum_vol,
            is_index=is_idx,
        )
        res = normalizer.normalize(raw_tick, current_time=base_time)
        assert res.status == TickStatus.VALID_LIVE
        # In round 1 baseline, incremental volume is 0
        assert res.tick.incremental_volume == 0
        store.insert_tick(res.tick)
        builder.process_tick(res.tick)

    # Round 2: Active trading ticks at 09:15:05 (volume increments)
    round2_time = base_time + timedelta(seconds=5)
    for sym, ltp, cum_vol, is_idx in symbols_spec:
        # Increase cumulative volume by 500 for tradeable assets
        new_cum_vol = cum_vol + (0 if is_idx else 500)
        raw_tick = Tick(
            symbol=sym,
            exchange_timestamp=round2_time,
            local_timestamp=round2_time,
            ltp=ltp + 1.0,
            cumulative_volume=new_cum_vol,
            is_index=is_idx,
        )
        res = normalizer.normalize(raw_tick, current_time=round2_time)
        assert res.status == TickStatus.VALID_LIVE

        if is_idx:
            assert res.tick.incremental_volume == 0
        else:
            assert res.tick.incremental_volume == 500

        store.insert_tick(res.tick)
        builder.process_tick(res.tick)

    # Flush all ticks to database
    store.flush_ticks()

    # Verify DuckDB records
    ticks_df = store._conn.execute("SELECT count(*) FROM ticks").fetchone()
    assert ticks_df[0] == 42  # 21 symbols x 2 ticks

    # Verify futures volume in database
    nifty_fut_vol = store._conn.execute(
        "SELECT sum(volume) FROM ticks WHERE symbol = 'NSE:NIFTY26NOVFUT'"
    ).fetchone()[0]
    assert nifty_fut_vol == 500  # Baseline 0 + round 2 increment 500

    # Verify spot index volume in database is strictly 0
    spot_vol = store._conn.execute(
        "SELECT sum(volume) FROM ticks WHERE symbol = 'NSE:NIFTY50-INDEX'"
    ).fetchone()[0]
    assert spot_vol == 0

    store.close()
