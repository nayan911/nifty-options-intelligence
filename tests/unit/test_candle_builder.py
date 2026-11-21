"""Unit tests for RealtimeCandleBuilder addressing volume, bar isolation, and duplicates."""

from datetime import datetime

import pytz

from nifty_engine.core.interfaces import Tick
from nifty_engine.data.candle_builder import RealtimeCandleBuilder

IST = pytz.timezone("Asia/Kolkata")


def test_spot_index_has_zero_volume():
    builder = RealtimeCandleBuilder()
    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 10))

    # Spot index with reported daily turnover in vol_traded_today
    t1 = Tick(symbol="NSE:NIFTY50-INDEX", local_timestamp=base_dt, ltp=25000.0, cumulative_volume=5000000, is_index=True)
    t2 = Tick(symbol="NSE:NIFTY50-INDEX", local_timestamp=base_dt.replace(second=20), ltp=25010.0, cumulative_volume=5010000, is_index=True)

    builder.process_tick(t1)
    builder.process_tick(t2)

    bar = builder.get_current_candle("NSE:NIFTY50-INDEX", "1m")
    assert bar is not None
    # Spot index candle volume must strictly be 0
    assert bar.volume == 0


def test_futures_incremental_volume_calculation():
    builder = RealtimeCandleBuilder()
    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 10))

    # Normalized ticks with incremental volume calculated by TickNormalizer:
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25100.0, cumulative_volume=1000000, incremental_volume=0, is_index=False)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=20), ltp=25105.0, cumulative_volume=1000050, incremental_volume=50, is_index=False)
    t3 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=40), ltp=25095.0, cumulative_volume=1000120, incremental_volume=70, is_index=False)

    builder.process_tick(t1)
    builder.process_tick(t2)
    builder.process_tick(t3)

    bar = builder.get_current_candle("NSE:NIFTY26NOVFUT", "1m")
    assert bar is not None
    # Incremental volume must be 0 + 50 + 70 = 120
    assert bar.volume == 120


def test_regression_no_double_volume_counting_when_cumulative_unchanged():
    """Regression test:
    Tick 1: cum = 1000, inc = 0 (baseline)
    Tick 2: cum = 1100, inc = 100
    Tick 3: cum = 1100, inc = 0 (price moves from 25050 to 25060, but no new contracts traded)
    Expected volume must remain strictly 100, NOT 200.
    """
    builder = RealtimeCandleBuilder()
    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 10))

    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25000.0, cumulative_volume=1000, incremental_volume=0, is_index=False)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=20), ltp=25050.0, cumulative_volume=1100, incremental_volume=100, is_index=False)
    t3 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=40), ltp=25060.0, cumulative_volume=1100, incremental_volume=0, is_index=False)

    builder.process_tick(t1)
    builder.process_tick(t2)
    builder.process_tick(t3)

    bar = builder.get_current_candle("NSE:NIFTY26NOVFUT", "1m")
    assert bar is not None
    assert bar.volume == 100  # Strictly 100, preventing the 200 double-count bug


def test_bar_high_low_strictly_isolated():
    builder = RealtimeCandleBuilder()
    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 0))

    # Ticks within this bar range between 25010 and 25040
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25020.0)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=20), ltp=25040.0)
    t3 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=40), ltp=25010.0)

    builder.process_tick(t1)
    builder.process_tick(t2)
    builder.process_tick(t3)

    bar = builder.get_current_candle("NSE:NIFTY26NOVFUT", "1m")
    assert bar.high == 25040.0
    assert bar.low == 25010.0
    assert bar.open == 25020.0
    assert bar.close == 25010.0


def test_duplicate_tick_rejection():
    builder = RealtimeCandleBuilder()
    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 10))

    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25000.0, cumulative_volume=1000)
    t2_dup = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25000.0, cumulative_volume=1000)

    assert builder.process_tick(t1) == []
    # Duplicate tick should be rejected
    assert builder.process_tick(t2_dup) == []

    active_bar = builder._active_bars.get(("NSE:NIFTY26NOVFUT", "1m"))
    assert active_bar is not None
    assert active_bar.tick_count == 1  # Only counted once


def test_time_based_candle_flush_without_new_ticks():
    closed_bars = []
    builder = RealtimeCandleBuilder(on_candle_close=lambda c: closed_bars.append(c))

    base_dt = IST.localize(datetime(2026, 11, 26, 9, 15, 10))
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25000.0, cumulative_volume=100)
    builder.process_tick(t1)

    # Clock moves to 09:16:01, but no new tick arrived
    current_clock = IST.localize(datetime(2026, 11, 26, 9, 16, 1))
    flushed = builder.flush_closed_bars(current_clock)

    assert len(flushed) >= 1
    assert flushed[0].symbol == "NSE:NIFTY26NOVFUT"
    assert flushed[0].is_closed is True
    assert flushed[0].timestamp.minute == 15
