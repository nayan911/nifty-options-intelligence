"""Unit Tests for Swing High/Low Detection Engine."""

from datetime import datetime, timedelta
import pandas as pd
import pytest
import pytz

from nifty_engine.analysis.technical.swings import SwingDetector, SwingPoint, SwingType
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
        volume=1000,
        is_complete=True,
    )


def test_swing_high_detection_no_lookahead():
    """Verify that a swing high at index 5 with right_bars=3 is confirmed ONLY at index 8."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    # Prices rising to peak at idx 5 (high=24100), then falling
    highs = [24000, 24020, 24050, 24070, 24080, 24100, 24080, 24060, 24040, 24020]
    lows = [h - 20 for h in highs]
    candles = [
        create_mock_candle(i, highs[i] - 10, highs[i], lows[i], highs[i] - 5, base_time)
        for i in range(len(highs))
    ]

    detector = SwingDetector(timeframe="5m", left_bars=3, right_bars=3)

    emitted_swings = []
    for i, c in enumerate(candles):
        swing = detector.on_candle(c)
        if swing is not None:
            emitted_swings.append((i, swing))

    # The peak is at index 5. With right_bars=3, it should be confirmed when index 5 + 3 = 8 is processed.
    assert len(emitted_swings) == 1
    conf_idx, swing = emitted_swings[0]
    assert conf_idx == 8
    assert swing.index == 5
    assert swing.price == 24100.0
    assert swing.type == SwingType.SWING_HIGH
    assert swing.timestamp == candles[5].timestamp
    assert swing.confirmation_timestamp == candles[8].timestamp

    # Check visible swings before index 8
    assert detector.get_visible_swings(candles[4].timestamp) == []
    assert detector.get_visible_swings(candles[5].timestamp) == []
    assert detector.get_visible_swings(candles[6].timestamp) == []
    assert detector.get_visible_swings(candles[7].timestamp) == []

    # At and after index 8, swing is visible
    visible_at_8 = detector.get_visible_swings(candles[8].timestamp)
    assert len(visible_at_8) == 1
    assert visible_at_8[0].price == 24100.0
    assert detector.get_latest_high(candles[8].timestamp).price == 24100.0


def test_swing_low_detection():
    """Verify swing low detection with right_bars=2."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    # Trough at idx 3 (low=23900)
    lows = [24000, 23950, 23920, 23900, 23930, 23960, 23980]
    highs = [l + 25 for l in lows]
    candles = [
        create_mock_candle(i, highs[i] - 10, highs[i], lows[i], lows[i] + 5, base_time)
        for i in range(len(lows))
    ]

    detector = SwingDetector(timeframe="5m", left_bars=2, right_bars=2)
    emitted = []
    for c in candles:
        s = detector.on_candle(c)
        if s is not None:
            emitted.append(s)

    assert len(emitted) == 1
    s = emitted[0]
    assert s.index == 3
    assert s.price == 23900.0
    assert s.type == SwingType.SWING_LOW
    assert s.confirmation_index == 5


def test_vectorized_and_streaming_equivalence():
    """Verify that detect_vectorized and streaming on_candle produce identical swings."""
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    # A wave with multiple peaks and troughs
    prices = [100, 105, 110, 108, 102, 95, 92, 98, 105, 112, 115, 108, 100, 104, 102]
    candles = [
        create_mock_candle(i, p - 2, p + 2, p - 4, p, base_time)
        for i, p in enumerate(prices)
    ]

    detector_stream = SwingDetector(timeframe="5m", left_bars=2, right_bars=2)
    stream_swings = []
    for c in candles:
        s = detector_stream.on_candle(c)
        if s is not None:
            stream_swings.append(s)

    # Vectorized
    df = pd.DataFrame([
        {
            "timestamp": c.timestamp,
            "high": c.high,
            "low": c.low,
            "close": c.close,
        }
        for c in candles
    ])

    detector_vec = SwingDetector(timeframe="5m", left_bars=2, right_bars=2)
    vec_swings = detector_vec.detect_vectorized(df)

    assert len(stream_swings) == len(vec_swings)
    for s_st, s_vc in zip(stream_swings, vec_swings):
        assert s_st.index == s_vc.index
        assert s_st.confirmation_index == s_vc.confirmation_index
        assert s_st.price == s_vc.price
        assert s_st.type == s_vc.type
