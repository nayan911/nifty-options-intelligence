"""Unit Tests for Breakout State Machine Engine."""

from datetime import datetime, timedelta

import pytz

from nifty_engine.analysis.technical.breakouts import (
    BreakoutStateMachine,
)
from nifty_engine.analysis.technical.support_resistance import LevelType, PriceZone
from nifty_engine.core.interfaces import BreakoutState, Candle

IST = pytz.timezone("Asia/Kolkata")


def create_candle(idx: int, o: float, h: float, l: float, c: float, base_time: datetime) -> Candle:
    return Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=base_time + timedelta(minutes=5 * idx),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=5000,
        is_complete=True,
    )


def test_bullish_breakout_and_retest_to_setup_eligible():
    """Verify ideal progression: APPROACHING -> BREAKOUT_CONFIRMED -> RETEST -> SETUP_ELIGIBLE."""
    sm = BreakoutStateMachine(
        min_breakout_rvol=1.3,
        min_body_atr_ratio=0.35,
        approach_buffer_atr=0.2,
        invalidation_buffer_atr=0.2,
        max_retest_candles=5,
    )
    base_time = datetime(2026, 11, 23, 10, 0, tzinfo=IST)
    # Zone: Resistance at 24150 [24147, 24153] with ATR = 20.0
    zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=24150.0,
        upper_bound=24153.0,
        lower_bound=24147.0,
        description="PDH Resistance",
        weight=2.0,
    )

    # 1. Approaching bar
    c0 = create_candle(0, 24135, 24148, 24130, 24145, base_time)
    ctx = sm.update_candle(c0, 0, zone, atr_value=20.0, futures_rvol=1.0)
    assert ctx.current_state in (BreakoutState.APPROACHING_LEVEL, BreakoutState.BREAKOUT_DETECTED)

    # 2. Strong Breakout bar: Opens 24146, closes 24165 (body=19 > 0.35*20), RVOL=1.8x
    c1 = create_candle(1, 24146, 24168, 24145, 24165, base_time)
    ctx = sm.update_candle(c1, 1, zone, atr_value=20.0, futures_rvol=1.8, context=ctx)
    assert ctx.current_state == BreakoutState.BREAKOUT_CONFIRMED
    assert ctx.breakout_price == 24165.0

    # 3. Retest pullback bar: opens 24155, dips to 24152 (touching zone), buyers reject lower prices and close green at 24162
    c2 = create_candle(2, 24155, 24165, 24152, 24162, base_time)
    ctx = sm.update_candle(c2, 2, zone, atr_value=20.0, futures_rvol=1.1, context=ctx)
    assert ctx.current_state == BreakoutState.SETUP_ELIGIBLE
    assert ctx.stop_loss is not None
    assert ctx.target_price is not None
    assert ctx.target_price > ctx.breakout_price


def test_false_breakout_due_to_low_volume():
    """Verify that a breakout candle with RVOL < 1.3 is marked FALSE_BREAKOUT."""
    sm = BreakoutStateMachine(min_breakout_rvol=1.3, min_body_atr_ratio=0.35)
    base_time = datetime(2026, 11, 23, 10, 0, tzinfo=IST)
    zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=24150.0,
        upper_bound=24153.0,
        lower_bound=24147.0,
        description="PDH Resistance",
        weight=2.0,
    )

    c0 = create_candle(0, 24140, 24148, 24138, 24146, base_time)
    ctx = sm.update_candle(c0, 0, zone, atr_value=20.0, futures_rvol=0.8)

    # Closes above zone, but RVOL is only 0.7x (lacks volume confirmation)
    c1 = create_candle(1, 24147, 24162, 24146, 24160, base_time)
    ctx = sm.update_candle(c1, 1, zone, atr_value=20.0, futures_rvol=0.7, context=ctx)
    assert ctx.current_state == BreakoutState.FALSE_BREAKOUT


def test_retest_failed_when_price_collapses_back():
    """Verify RETEST_FAILED when candle closes below the broken level."""
    sm = BreakoutStateMachine(min_breakout_rvol=1.3, min_body_atr_ratio=0.35)
    base_time = datetime(2026, 11, 23, 10, 0, tzinfo=IST)
    zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=24150.0,
        upper_bound=24153.0,
        lower_bound=24147.0,
        description="PDH Resistance",
        weight=2.0,
    )

    c0 = create_candle(0, 24140, 24148, 24138, 24146, base_time)
    ctx = sm.update_candle(c0, 0, zone, atr_value=20.0, futures_rvol=1.0)

    # Valid breakout
    c1 = create_candle(1, 24147, 24168, 24146, 24165, base_time)
    ctx = sm.update_candle(c1, 1, zone, atr_value=20.0, futures_rvol=1.5, context=ctx)
    assert ctx.current_state == BreakoutState.BREAKOUT_CONFIRMED

    # Retest bar 1: dips towards zone, closes in retest range (RETEST_PENDING)
    c2 = create_candle(2, 24162, 24163, 24154, 24155, base_time)
    ctx = sm.update_candle(c2, 2, zone, atr_value=20.0, futures_rvol=1.0, context=ctx)
    assert ctx.current_state == BreakoutState.RETEST_PENDING

    # Retest bar 2: crashes deep below support at 24135 (RETEST_FAILED)
    c3 = create_candle(3, 24155, 24156, 24130, 24135, base_time)
    ctx = sm.update_candle(c3, 3, zone, atr_value=20.0, futures_rvol=1.2, context=ctx)
    assert ctx.current_state == BreakoutState.RETEST_FAILED
