"""Unit Tests and Micro-Benchmarks for Trendline Engine."""

import time
from datetime import datetime, timedelta

import pytest
import pytz

from nifty_engine.analysis.technical.swings import SwingDetector
from nifty_engine.analysis.technical.trendlines import TrendlineDetector
from nifty_engine.core.interfaces import Candle

IST = pytz.timezone("Asia/Kolkata")


def create_mock_candle(idx: int, open_: float, high: float, low: float, close: float, base_time: datetime) -> Candle:
    t = base_time + timedelta(minutes=5 * idx)
    return Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=t,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=2500,
        is_complete=True,
    )


def test_ascending_support_trendline():
    """Verify detection of an ascending support trendline connecting 3 swing lows."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    # Construct an uptrend where lows touch a line y = 2.0 * x + 24000
    candles = []
    for i in range(40):
        expected_low = 24000.0 + (2.0 * i)
        # Add small oscillation around the line
        if i in [5, 15, 25]:
            low = expected_low  # Exact touch
            close = low + 5.0
            high = low + 15.0
            open_ = low + 8.0
        else:
            low = expected_low + 8.0  # Floating above the line
            close = low + 10.0
            high = low + 20.0
            open_ = low + 5.0
        candles.append(create_mock_candle(i, open_, high, low, close, base_time))

    swing_detector = SwingDetector(timeframe="5m", left_bars=3, right_bars=3)
    for c in candles:
        swing_detector.on_candle(c)

    swings = swing_detector.get_visible_swings(candles[-1].timestamp)
    low_swings = [s for s in swings if s.is_low]
    assert len(low_swings) >= 2

    detector = TrendlineDetector(window_size=50, min_touches=2, min_touch_spacing=3)
    lines = detector.find_trendlines(candles, swings, atr_value=25.0)

    support_lines = [l for l in lines if l.line_type == "SUPPORT"]
    assert len(support_lines) >= 1
    best_support = support_lines[0]
    # Check slope is approximately 2.0
    assert pytest.approx(best_support.slope, rel=0.1) == 2.0
    assert best_support.num_touches >= 2


def test_trendline_invalidated_by_penetration():
    """Verify that a candidate trendline is rejected if price drops significantly below it."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    candles = []
    for i in range(30):
        line_val = 24000.0 + (1.0 * i)
        if i in [5, 12]:
            low = line_val
        elif i == 18:
            # Massive bearish dump breaking the trendline
            low = line_val - 50.0
        else:
            low = line_val + 5.0
        high = low + 15.0
        candles.append(create_mock_candle(i, low + 5, high, low, low + 10, base_time))

    swing_detector = SwingDetector(timeframe="5m", left_bars=2, right_bars=2)
    for c in candles:
        swing_detector.on_candle(c)

    swings = swing_detector.get_visible_swings(candles[-1].timestamp)
    detector = TrendlineDetector(window_size=50, min_touches=2, min_touch_spacing=3)
    lines = detector.find_trendlines(candles, swings, atr_value=20.0)

    # Any support line connecting touches prior to bar 18 must be rejected due to penetration at bar 18
    for line in lines:
        if line.line_type == "SUPPORT":
            assert line.value_at(18) - candles[18].close <= line.atr_tolerance


def test_trendline_micro_benchmark():
    """Ensure trendline calculation runs in < 50ms across a full 150-bar window."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    candles = []
    for i in range(150):
        base_p = 24000.0 + (i % 10) * 5
        candles.append(create_mock_candle(i, base_p, base_p + 15, base_p - 10, base_p + 5, base_time))

    swing_detector = SwingDetector(timeframe="5m", left_bars=3, right_bars=3)
    for c in candles:
        swing_detector.on_candle(c)

    swings = swing_detector.get_visible_swings(candles[-1].timestamp)
    detector = TrendlineDetector(window_size=150, min_touches=2)

    start_time = time.perf_counter()
    _ = detector.find_trendlines(candles, swings, atr_value=25.0)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[BENCHMARK] Trendline calculation for 150 bars took: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50.0, f"Trendline detection exceeded 50ms: {elapsed_ms:.2f} ms"


def test_trendline_as_of_time_isolation():
    """Verify that passing future candles with an as_of_time does not leak future data."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    candles = []
    for i in range(50):
        low = 24000.0 + (1.5 * i)
        candles.append(create_mock_candle(i, low + 5, low + 15, low, low + 10, base_time))

    swing_detector = SwingDetector(timeframe="5m", left_bars=2, right_bars=2)
    for c in candles:
        swing_detector.on_candle(c)

    # Set as_of_time at bar 25
    as_of = candles[25].timestamp
    swings = swing_detector.get_visible_swings(as_of)
    detector = TrendlineDetector(window_size=30, min_touches=2, min_touch_spacing=3)

    # Pass all 50 candles, but restrict with as_of_time
    lines = detector.find_trendlines(candles, swings, atr_value=20.0, as_of_time=as_of)
    for l in lines:
        # None of the trendlines should have end_index > 25
        assert l.end_index <= 25
        assert l.end_timestamp <= as_of
