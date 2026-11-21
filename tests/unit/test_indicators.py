"""Unit tests for technical indicators engine."""

import pandas as pd
import pytest

from nifty_engine.analysis.technical.indicators import (
    compute_atr,
    compute_ema,
    compute_relative_volume,
    compute_rsi,
    compute_volatility_adjusted_move,
    compute_vwap,
    update_ema_streaming,
)


def test_ema_vector_and_streaming_equivalence():
    prices = [25000.0, 25050.0, 25020.0, 25080.0, 25100.0, 25070.0, 25120.0]
    period = 5

    # Vectorized
    ema_series = compute_ema(prices, period)

    # Streaming
    streaming_ema = prices[0]
    for p in prices[1:]:
        streaming_ema = update_ema_streaming(p, streaming_ema, period)

    # Check terminal values match within numerical tolerance
    assert abs(ema_series.iloc[-1] - streaming_ema) < 1e-4


def test_rsi_bounds_and_trending():
    # Strictly increasing prices -> RSI should approach 100
    increasing = pd.Series([100 + i * 5 for i in range(30)])
    rsi_up = compute_rsi(increasing, period=14)
    assert rsi_up.iloc[-1] > 95.0
    assert (rsi_up.dropna() >= 0.0).all() and (rsi_up.dropna() <= 100.0).all()

    # Strictly decreasing prices -> RSI should approach 0
    decreasing = pd.Series([200 - i * 5 for i in range(30)])
    rsi_down = compute_rsi(decreasing, period=14)
    assert rsi_down.iloc[-1] < 5.0


def test_atr_calculation():
    # 5 bars with constant True Range of 20
    df = pd.DataFrame({
        "high": [25020, 25030, 25040, 25050, 25060],
        "low": [25000, 25010, 25020, 25030, 25040],
        "close": [25010, 25020, 25030, 25040, 25050],
    })
    atr = compute_atr(df["high"], df["low"], df["close"], period=5)
    # True range for each step is 20 (high - low = 20, high - prev_close = 20)
    assert abs(atr.iloc[-1] - 20.0) < 1e-2


def test_vwap_rejects_zero_volume_spot_index():
    df = pd.DataFrame({
        "high": [25050, 25060],
        "low": [25000, 25010],
        "close": [25020, 25040],
        "volume": [0, 0],  # Spot index zero volume
    })
    with pytest.raises(ValueError) as exc:
        compute_vwap(df)
    assert "VWAP requires tradeable volume" in str(exc.value)


def test_vwap_with_volume_and_bands():
    df = pd.DataFrame({
        "high": [25050, 25060, 25070],
        "low": [25000, 25010, 25020],
        "close": [25020, 25040, 25060],
        "volume": [1000, 2000, 3000],
    })
    vwap, upper_1s, lower_1s = compute_vwap(df)
    assert len(vwap) == 3
    # VWAP must lie between low and high
    assert (vwap >= df["low"]).all() and (vwap <= df["high"]).all()
    # Bands should envelope VWAP
    assert (upper_1s >= vwap).all() and (lower_1s <= vwap).all()


def test_anchored_vwap_resets_variance_per_anchor():
    """Verify that anchored VWAP and standard deviation bands reset across sessions."""
    df = pd.DataFrame({
        "session": ["2026-11-20", "2026-11-20", "2026-11-23", "2026-11-23"],
        "high": [105.0, 110.0, 102.0, 106.0],
        "low": [95.0, 100.0, 98.0, 100.0],
        "close": [100.0, 105.0, 100.0, 104.0],
        "volume": [1000, 1000, 1000, 1000],
    })
    vwap, upper, lower = compute_vwap(df, anchor_col="session")
    assert len(vwap) == 4
    # On the first bar of the new session (idx 2), typical price is (102 + 98 + 100)/3 = 100.0
    # VWAP must be exactly 100.0 and variance must be 0
    assert pytest.approx(vwap.iloc[2], rel=1e-4) == 100.0
    assert pytest.approx(upper.iloc[2], rel=1e-4) == 100.0
    assert pytest.approx(lower.iloc[2], rel=1e-4) == 100.0


def test_relative_volume():
    volumes = pd.Series([100] * 19 + [250])  # Average 100, last bar is 250
    rvol = compute_relative_volume(volumes, period=20)
    # Last bar has volume 250 vs historical average ~107.5 -> RVOL > 2.0
    assert rvol.iloc[-1] > 2.0


def test_volatility_adjusted_move():
    closes = pd.Series([25000, 25040])  # +40 points move
    atr = pd.Series([20, 20])           # ATR = 20 points
    norm_move = compute_volatility_adjusted_move(closes, atr, periods=1)
    # Move is 40 / 20 = 2.0 ATR multiples
    assert norm_move.iloc[-1] == 2.0
