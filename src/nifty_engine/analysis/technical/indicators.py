"""Vectorized and Streaming Technical Indicators Engine.

Implements mathematically exact, un-repainting technical indicators:
1. EMA (Exponential Moving Average)
2. RSI (Wilder's Smoothed Relative Strength Index)
3. ATR (Average True Range via Wilder's smoothing)
4. VWAP & Anchored VWAP (Strictly requiring tradeable volume e.g. NIFTY Futures)
5. Relative Volume (RVOL)
6. Momentum (ROC & Price Velocity)
7. Volatility-Adjusted Price Movement (Normalized ATR Multiples)
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_ema(series: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """Vectorized Exponential Moving Average using standard alpha = 2 / (period + 1)."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.ewm(span=period, adjust=False).mean()


def update_ema_streaming(current_price: float, previous_ema: float, period: int) -> float:
    """Incremental O(1) single-tick EMA update."""
    alpha = 2.0 / (period + 1.0)
    return (current_price * alpha) + (previous_ema * (1.0 - alpha))


def compute_rsi(close_series: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Wilder's Smoothed Relative Strength Index (RSI)."""
    s = pd.Series(close_series) if not isinstance(close_series, pd.Series) else close_series
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing uses alpha = 1 / period (equivalent to com = period - 1)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # If loss is 0 and gain > 0 -> RSI = 100
    rsi = rsi.fillna(100.0).where(avg_loss != 0, 100.0)
    # If gain is 0 and loss > 0 -> RSI = 0
    rsi = rsi.where((avg_gain != 0) | (avg_loss != 0), 50.0)
    return rsi


def compute_atr(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 14
) -> pd.Series:
    """Average True Range (ATR) calculated using Wilder's smoothing."""
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    c_prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's exponential smoothing
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_vwap(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    anchor_col: Optional[str] = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Volume Weighted Average Price (VWAP) with Standard Deviation Bands.

    WARNING: Must be calculated using tradeable volume (e.g. NIFTY Futures).
    Do NOT apply directly to spot index data which has 0 volume.

    Returns: (vwap, upper_band_1sigma, lower_band_1sigma)
    """
    if df[volume_col].sum() == 0:
        raise ValueError(
            "Total volume is 0. VWAP requires tradeable volume (e.g., NIFTY Futures), "
            "not a spot index which has zero traded volume."
        )

    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    vol = df[volume_col]

    if anchor_col is not None and anchor_col in df.columns:
        # Grouped by anchor session (e.g., date)
        cum_tp_vol = (typical_price * vol).groupby(df[anchor_col]).cumsum()
        cum_vol = vol.groupby(df[anchor_col]).cumsum()
    else:
        cum_tp_vol = (typical_price * vol).cumsum()
        cum_vol = vol.cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Standard deviation bands
    price_variance = (typical_price - vwap) ** 2
    cum_var_vol = (price_variance * vol).cumsum()
    variance = cum_var_vol / cum_vol.replace(0, np.nan)
    std_dev = np.sqrt(variance.clip(lower=0))

    upper_1sigma = vwap + std_dev
    lower_1sigma = vwap - std_dev

    return vwap, upper_1sigma, lower_1sigma


def compute_relative_volume(
    volume_series: Union[pd.Series, np.ndarray],
    period: int = 20
) -> pd.Series:
    """Relative Volume (RVOL) = current volume / rolling average volume."""
    v = pd.Series(volume_series) if not isinstance(volume_series, pd.Series) else volume_series
    avg_vol = v.rolling(window=period, min_periods=1).mean()
    rvol = v / avg_vol.replace(0, np.nan)
    return rvol.fillna(1.0)


def compute_momentum_roc(
    close_series: Union[pd.Series, np.ndarray],
    periods: int = 10
) -> pd.Series:
    """Rate of Change (ROC) percentage momentum."""
    c = pd.Series(close_series) if not isinstance(close_series, pd.Series) else close_series
    return ((c - c.shift(periods)) / c.shift(periods).replace(0, np.nan)) * 100.0


def compute_volatility_adjusted_move(
    close_series: Union[pd.Series, np.ndarray],
    atr_series: Union[pd.Series, np.ndarray],
    periods: int = 1
) -> pd.Series:
    """Normalized Price Movement = (Close_t - Close_{t-k}) / ATR_t.

    Indicates how many ATR multiples the asset moved, standardizing breakout strength
    across low vs high volatility regimes.
    """
    c = pd.Series(close_series) if not isinstance(close_series, pd.Series) else close_series
    atr = pd.Series(atr_series) if not isinstance(atr_series, pd.Series) else atr_series
    price_change = c.diff(periods)
    norm_move = price_change / atr.replace(0, np.nan)
    return norm_move.fillna(0.0)
