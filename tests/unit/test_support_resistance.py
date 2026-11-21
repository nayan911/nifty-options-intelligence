"""Unit Tests for Dynamic Support and Resistance Zones Engine."""

from datetime import datetime, timedelta

import pytz

from nifty_engine.analysis.technical.support_resistance import (
    LevelType,
    SupportResistanceEngine,
)
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
        is_closed=True,
    )


def test_daily_levels_and_buffering():
    engine = SupportResistanceEngine(zone_atr_multiple=0.2, min_zone_width_points=3.0)
    # ATR is 25.0 -> Buffer is 0.2 * 25.0 = 5.0
    zones = engine.compute_daily_levels(
        pdh=24200.0,
        pdl=23950.0,
        pdc=24150.0,
        cdo=24120.0,
        atr_value=25.0,
    )
    assert len(zones) == 4
    pdh_zone = next(z for z in zones if z.level_type == LevelType.PDH)
    assert pdh_zone.nominal_price == 24200.0
    assert pdh_zone.upper_bound == 24205.0
    assert pdh_zone.lower_bound == 24195.0
    assert pdh_zone.contains(24202.0)
    assert not pdh_zone.contains(24210.0)


def test_opening_range_calculation():
    engine = SupportResistanceEngine(zone_atr_multiple=0.1, min_zone_width_points=2.0)
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    # 3 x 5m candles: 09:15, 09:20, 09:25
    c1 = create_mock_candle(0, 24100, 24140, 24090, 24120, base_time)
    c2 = create_mock_candle(1, 24120, 24160, 24110, 24150, base_time)
    c3 = create_mock_candle(2, 24150, 24155, 24080, 24100, base_time)
    # 4th candle at 09:30 (outside 15-min OR)
    c4 = create_mock_candle(3, 24100, 24200, 24050, 24180, base_time)

    zones = engine.compute_opening_range([c1, c2, c3, c4], atr_value=20.0)
    assert len(zones) == 2
    orh_zone = next(z for z in zones if z.level_type == LevelType.ORH)
    orl_zone = next(z for z in zones if z.level_type == LevelType.ORL)

    # Max high in [c1, c2, c3] is 24160.0
    assert orh_zone.nominal_price == 24160.0
    # Min low in [c1, c2, c3] is 24080.0
    assert orl_zone.nominal_price == 24080.0


def test_psychological_levels():
    engine = SupportResistanceEngine(zone_atr_multiple=0.1, min_zone_width_points=2.0)
    zones = engine.compute_psychological_levels(
        current_price=24135.0,
        atr_value=20.0,
        range_points=120.0,
    )
    strikes = [z.nominal_price for z in zones]
    # In range 24015 to 24255, 50-pt steps are 24050, 24100, 24150, 24200, 24250
    assert 24050.0 in strikes
    assert 24100.0 in strikes
    assert 24150.0 in strikes
    assert 24200.0 in strikes


def test_nearest_support_and_resistance():
    engine = SupportResistanceEngine(zone_atr_multiple=0.1, min_zone_width_points=2.0)
    z1 = engine.create_zone(LevelType.PDL, 24000.0, 20.0, "PDL")
    z2 = engine.create_zone(LevelType.SWING_LOW_CLUSTER, 24080.0, 20.0, "Swing Low")
    z3 = engine.create_zone(LevelType.ORH, 24150.0, 20.0, "ORH")
    z4 = engine.create_zone(LevelType.PDH, 24220.0, 20.0, "PDH")

    current_price = 24110.0
    supp, res = engine.find_nearest_zones(current_price, [z1, z2, z3, z4])

    assert supp is not None
    assert supp.nominal_price == 24080.0
    assert res is not None
    assert res.nominal_price == 24150.0


def test_opening_range_premature_rejection_and_completeness():
    """Verify Issue P1 Fix: Opening range is declared ready ONLY when all required intervals are present.
    - 1m chart requires all 15 candles (09:15-09:29). Having only 3 or 14 returns empty list.
    - 5m chart requires all 3 candles (09:15, 09:20, 09:25).
    - 15m chart requires 09:15 candle.
    """
    engine = SupportResistanceEngine(zone_atr_multiple=0.1, min_zone_width_points=2.0)
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)
    session_date = base_time.date()

    # 1. 1-minute chart: feed only 3 candles (09:15, 09:16, 09:17) -> MUST NOT be declared ready!
    c_1m_premature = [
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=0,
            is_closed=True,
        )
        for i in range(3)
    ]
    zones_premature = engine.compute_opening_range(c_1m_premature, atr_value=20.0, session_date=session_date)
    assert zones_premature == []

    # Feed 14 candles (missing 09:29) -> Still empty!
    c_1m_14 = [
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=0,
            is_closed=True,
        )
        for i in range(14)
    ]
    assert engine.compute_opening_range(c_1m_14, atr_value=20.0, session_date=session_date) == []

    # Complete 15 candles (09:15 through 09:29) -> Declared ready!
    c_1m_complete = [
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0 + i,
            low=24990.0 - i,
            close=25005.0,
            volume=0,
            is_closed=True,
        )
        for i in range(15)
    ]
    zones_1m = engine.compute_opening_range(c_1m_complete, atr_value=20.0, session_date=session_date)
    assert len(zones_1m) == 2
    orh = next(z for z in zones_1m if z.level_type == LevelType.ORH)
    orl = next(z for z in zones_1m if z.level_type == LevelType.ORL)
    assert orh.nominal_price == 25024.0  # 25010 + 14
    assert orl.nominal_price == 24976.0  # 24990 - 14

    # 2. 5-minute chart: 2 candles (09:15, 09:20) -> Premature, return empty
    c_5m_premature = [
        create_mock_candle(0, 25000, 25020, 24980, 25010, base_time),
        create_mock_candle(1, 25010, 25030, 24990, 25020, base_time),
    ]
    assert engine.compute_opening_range(c_5m_premature, atr_value=20.0, session_date=session_date) == []

    # 3 x 5m candles (09:15, 09:20, 09:25) -> Complete!
    c_5m_complete = c_5m_premature + [create_mock_candle(2, 25020, 25040, 25000, 25030, base_time)]
    zones_5m = engine.compute_opening_range(c_5m_complete, atr_value=20.0, session_date=session_date)
    assert len(zones_5m) == 2

