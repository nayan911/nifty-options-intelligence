"""Unit tests for RealtimeCandleBuilder addressing volume, bar isolation, and duplicates."""

from datetime import datetime, timedelta

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


def test_duplicate_closed_candle_prevention_on_delayed_ticks():
    """Verify that late/delayed ticks arriving after a candle has been finalized

    can NEVER recreate an active candle or emit duplicate closed bars.
    """
    closed_bars = []
    builder = RealtimeCandleBuilder(on_candle_close=lambda c: closed_bars.append(c))

    base_dt = IST.localize(datetime(2026, 11, 23, 9, 25, 10))
    t1 = Tick(symbol="NSE:NIFTY50-INDEX", local_timestamp=base_dt, ltp=25708.70, cumulative_volume=0, is_index=True)
    t2 = Tick(symbol="NSE:NIFTY50-INDEX", local_timestamp=base_dt.replace(second=45), ltp=25708.70, cumulative_volume=0, is_index=True)

    builder.process_tick(t1)
    builder.process_tick(t2)

    # Flush at interval end boundary: 09:26:00
    t_boundary = IST.localize(datetime(2026, 11, 23, 9, 26, 0))
    builder.flush_closed_bars(t_boundary)

    # Exactly one 1m candle closed
    closed_1m = [c for c in closed_bars if c.timeframe == "1m"]
    assert len(closed_1m) == 1
    assert closed_1m[0].close == 25708.70
    assert closed_1m[0].timestamp == IST.localize(datetime(2026, 11, 23, 9, 25, 0))

    # A delayed tick timestamped 09:25:59 arrives with a different price (25,711.10)
    t_delayed = Tick(symbol="NSE:NIFTY50-INDEX", local_timestamp=base_dt.replace(second=59), ltp=25711.10, cumulative_volume=0, is_index=True)
    builder.process_tick(t_delayed)

    # Next flush at 09:26:01
    builder.flush_closed_bars(t_boundary + timedelta(seconds=1))

    # Invariant: NO duplicate candle for 09:25 must ever be emitted!
    closed_1m_after = [c for c in closed_bars if c.timeframe == "1m"]
    assert len(closed_1m_after) == 1
    assert closed_1m_after[0].close == 25708.70


def test_incomplete_initial_candle_dropped_on_late_startup():
    """Verify that when a symbol is subscribed or process starts mid-interval (e.g. 10m into 15m bar),

    the incomplete initial slice is dropped and not emitted as a full candle.
    """
    closed_bars = []
    builder = RealtimeCandleBuilder(on_candle_close=lambda c: closed_bars.append(c), drop_incomplete_initial_bars=True)

    # Runner starts at 09:25:10 (10 minutes and 10 seconds into the 09:15–09:30 15m interval)
    base_dt = IST.localize(datetime(2026, 11, 23, 9, 25, 10))
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25700.0, cumulative_volume=1000, incremental_volume=0, is_index=False)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(minute=29, second=50), ltp=25668.50, cumulative_volume=101790, incremental_volume=100790, is_index=False)

    builder.process_tick(t1)
    builder.process_tick(t2)

    # Flush at 09:30:00 (end of 09:15–09:30 15m interval, and 09:25–09:30 5m interval)
    t_0930 = IST.localize(datetime(2026, 11, 23, 9, 30, 0))
    builder.flush_closed_bars(t_0930)

    # 1. 5m candle started at 09:25:10 (10s offset <= 45s threshold) -> full 5m bar emitted!
    closed_5m = [c for c in closed_bars if c.timeframe == "5m"]
    assert len(closed_5m) == 1
    assert closed_5m[0].timestamp == IST.localize(datetime(2026, 11, 23, 9, 25, 0))
    assert closed_5m[0].close == 25668.50

    # 2. 15m candle started at 09:25:10 (610s offset > 120s threshold) -> INCOMPLETE! MUST BE DROPPED!
    closed_15m = [c for c in closed_bars if c.timeframe == "15m"]
    assert len(closed_15m) == 0  # Incomplete 15m candle was safely dropped!

    # 3. Subsequent 15m interval (09:30 to 09:45) is complete
    t3 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=t_0930 + timedelta(seconds=5), ltp=25670.0, cumulative_volume=102000, incremental_volume=210, is_index=False)
    t4 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=t_0930 + timedelta(minutes=14, seconds=50), ltp=25690.0, cumulative_volume=150000, incremental_volume=48000, is_index=False)
    builder.process_tick(t3)
    builder.process_tick(t4)

    t_0945 = IST.localize(datetime(2026, 11, 23, 9, 45, 0))
    builder.flush_closed_bars(t_0945)

    # Now the 09:30:00 15m candle is full, complete, and properly emitted
    closed_15m_final = [c for c in closed_bars if c.timeframe == "15m"]
    assert len(closed_15m_final) == 1
    assert closed_15m_final[0].timestamp == IST.localize(datetime(2026, 11, 23, 9, 30, 0))
    assert closed_15m_final[0].close == 25690.0


def test_complete_initial_candle_accepted_on_normal_startup():
    """Verify that when ticks start within initial offset tolerance (e.g. 5s into interval),

    all timeframes (1m, 5m, 15m) are accepted as complete.
    """
    closed_bars = []
    builder = RealtimeCandleBuilder(on_candle_close=lambda c: closed_bars.append(c), drop_incomplete_initial_bars=True)

    # Process starts at 09:15:05 (5 seconds into session open)
    base_dt = IST.localize(datetime(2026, 11, 23, 9, 15, 5))
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt, ltp=25800.0, cumulative_volume=1000, incremental_volume=0, is_index=False)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_dt.replace(second=55), ltp=25805.0, cumulative_volume=1500, incremental_volume=500, is_index=False)

    builder.process_tick(t1)
    builder.process_tick(t2)

    # Flush 1m at 09:16:00
    builder.flush_closed_bars(base_dt.replace(minute=16, second=0))

    closed_1m = [c for c in closed_bars if c.timeframe == "1m"]
    assert len(closed_1m) == 1
    assert closed_1m[0].timestamp == IST.localize(datetime(2026, 11, 23, 9, 15, 0))
    assert closed_1m[0].close == 25805.0

