"""Unit Tests for Phase C MultiTimeframeAnalyzer and TechnicalCandidate Synthesis."""

from datetime import datetime, timedelta

import pytz

from nifty_engine.analysis.technical.breakouts import (
    BreakoutContext,
    BreakoutDirection,
)
from nifty_engine.analysis.technical.engine import (
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
)
from nifty_engine.analysis.technical.multi_timeframe import (
    MultiTimeframeAnalyzer,
    TechnicalCandidate,
)
from nifty_engine.analysis.technical.support_resistance import (
    LevelType,
    PriceZone,
)
from nifty_engine.core.interfaces import BreakoutState, Candle

IST = pytz.timezone("Asia/Kolkata")


def test_multi_timeframe_macro_trend_alignment_filter():
    """Verify Tier 1: 5m setups are rejected if they contradict the 15m macro regime."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # 1. 15m Macro Trend is BEARISH
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25200.0,
        high=25210.0,
        low=25000.0,
        close=25020.0,
        volume=0,
        trend_bias="BEARISH",
        is_warmed_up=True,
    )

    # 5m Bullish Breakout Setup attempts to qualify
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=1.8,
    )
    setup_bullish = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25070.0,
        target_price=25160.0,
    )

    # Invariant: Bullish setup rejected against 15m Bearish trend
    res_rejected = analyzer.on_5m_setup(setup_bullish, snap_5m)
    assert res_rejected is None
    assert len(analyzer.active_setups) == 0

    # 2. 15m Macro Trend is updated to BULLISH
    state.latest_snapshots["15m"].trend_bias = "BULLISH"
    res_accepted = analyzer.on_5m_setup(setup_bullish, snap_5m)
    assert res_accepted is not None
    assert setup_bullish.setup_id in analyzer.active_setups


def test_multi_timeframe_headroom_to_resistance_filter():
    """Verify Tier 1: 5m setups are rejected if opposing 15m S/R barrier is too close (<1.5x ATR)."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine, min_headroom_atr_multiple=1.5)
    state = engine.states[sym]

    # Major 15m resistance right overhead at 25115 (nominal), lower_bound 25112
    barrier_zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=25115.0,
        upper_bound=25118.0,
        lower_bound=25112.0,
        description="15m Major Resistance",
        weight=3.0,
    )
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25050.0,
        high=25115.0,
        low=25040.0,
        close=25090.0,
        volume=0,
        trend_bias="BULLISH",
        nearest_resistance=barrier_zone,
    )

    # 5m breakout at 25100 with close at 25108. Headroom is only 4 points (25112 - 25108) < 1.5 * 20.0 = 30 pts!
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25110.0,
        low=25075.0,
        close=25108.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    setup = BreakoutContext(
        level_id="ORH_25100.0",
        level_type="ORH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
    )

    # Invariant: Rejected due to inadequate headroom to overhead 15m resistance
    assert analyzer.on_5m_setup(setup, snap_5m) is None

    # Now move resistance barrier far away (e.g. 25250)
    barrier_zone_far = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=25250.0,
        upper_bound=25255.0,
        lower_bound=25245.0,
        description="Far Resistance",
        weight=3.0,
    )
    state.latest_snapshots["15m"].nearest_resistance = barrier_zone_far
    res = analyzer.on_5m_setup(setup, snap_5m)
    assert res is not None
    assert res.headroom_points > 100.0


def test_multi_timeframe_1m_entry_timing_and_candidate_dispatch():
    """Verify Tier 3: Qualified 5m setup waits for 1m entry confirmation and emits TechnicalCandidate."""
    candidates = []

    def on_cand(c: TechnicalCandidate):
        candidates.append(c)

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    analyzer = MultiTimeframeAnalyzer(engine=engine, on_candidate=on_cand)
    state = engine.states[sym]

    # 1. 15m Macro Context
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )

    # 2. 5m Setup Qualifies at 10:00
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.1,
        vwap=25080.0,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25075.0,
        target_price=25175.0,
    )
    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None
    assert len(candidates) == 0

    # 3. 1m Candle Closes at 10:01 with confirmation
    t_1m = base_time + timedelta(minutes=1)
    candle_1m = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25105.0,
        high=25118.0,
        low=25102.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    snap_1m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25118.0,
        low=25102.0,
        close=25115.0,
        volume=0,
        ema_9=25108.0,
        vwap=25095.0,
        rsi=60.0,
    )

    generated = analyzer.on_1m_candle(candle_1m, snap_1m)
    assert len(generated) == 1
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.symbol == sym
    assert cand.direction == BreakoutDirection.BULLISH
    assert cand.entry_price == 25115.0
    assert cand.stop_loss <= 25100.0
    assert cand.target_1 > cand.entry_price
    assert cand.risk_points > 0
    assert cand.risk_reward_ratio >= 1.5
    assert cand.confluence_score >= 60.0
    assert cand.is_paper_trading is True
    assert "15m_trend" in cand.timeframe_breakdown

    # Invariant: Active setup consumed, does not re-fire on subsequent 1m bar
    assert len(analyzer.active_setups) == 0
    t_1m_next = t_1m + timedelta(minutes=1)
    candle_1m_2 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m_next,
        open=25115.0,
        high=25125.0,
        low=25112.0,
        close=25120.0,
        volume=0,
        is_closed=True,
    )
    generated_2 = analyzer.on_1m_candle(candle_1m_2, snap_1m)
    assert len(generated_2) == 0
    assert len(candidates) == 1


def test_multi_timeframe_1m_momentum_invalidation():
    """Verify Tier 3: If 1m bar collapses back below broken level, candidate is not generated."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )

    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25075.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=1.8,
    )
    analyzer.on_5m_setup(setup, snap_5m)

    # 1m candle collapses below nominal breakout price (25100) to 25095
    t_1m = base_time + timedelta(minutes=1)
    candle_fail = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25105.0,
        high=25108.0,
        low=25090.0,
        close=25095.0,
        volume=0,
        is_closed=True,
    )
    snap_fail = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25108.0,
        low=25090.0,
        close=25095.0,
        volume=0,
    )

    cands = analyzer.on_1m_candle(candle_fail, snap_fail)
    assert len(cands) == 0


def test_multi_timeframe_stop_loss_optimization():
    """Verify Tier 3: Stop-loss is tightened using 1m micro-support when structurally safe."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )

    # Base 5m setup stop loss is deep at 25050 (55 pts risk from 25105)
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25050.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=1.8,
    )
    analyzer.on_5m_setup(setup, snap_5m)

    # 1m snapshot provides a tight micro-support zone at 25088 (lower_bound 25085)
    micro_sup = PriceZone(
        level_type=LevelType.SWING_LOW_CLUSTER,
        nominal_price=25088.0,
        upper_bound=25091.0,
        lower_bound=25085.0,
        description="1m Micro Support",
        weight=2.0,
    )
    t_1m = base_time + timedelta(minutes=1)
    candle_1m = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25102.0,
        high=25112.0,
        low=25100.0,
        close=25108.0,
        volume=0,
        is_closed=True,
    )
    snap_1m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25102.0,
        high=25112.0,
        low=25100.0,
        close=25108.0,
        volume=0,
        nearest_support=micro_sup,
    )

    cands = analyzer.on_1m_candle(candle_1m, snap_1m)
    assert len(cands) == 1
    # Optimized stop loss: 25085 instead of 25050
    assert cands[0].stop_loss == 25085.0
    assert cands[0].risk_points == 23.0  # (25108 - 25085)


def test_multi_timeframe_active_setup_expiration():
    """Verify setup expires if 1m entry trigger does not occur within entry window."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    analyzer = MultiTimeframeAnalyzer(engine=engine, entry_window_minutes=15)
    state = engine.states[sym]
    state.latest_snapshots["15m"] = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )

    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=1.8,
    )
    analyzer.on_5m_setup(setup, snap_5m)
    assert len(analyzer.active_setups) == 1

    # Candle arrives 20 minutes later (10:20 > 10:15 expiry)
    t_late = base_time + timedelta(minutes=20)
    c_late = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_late,
        open=25105.0,
        high=25115.0,
        low=25100.0,
        close=25110.0,
        volume=0,
        is_closed=True,
    )
    snap_late = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_late,
        timeframe="1m",
        open=25105.0,
        high=25115.0,
        low=25100.0,
        close=25110.0,
        volume=0,
    )

    cands = analyzer.on_1m_candle(c_late, snap_late)
    # Invariant: Expired setup cleared, zero candidates dispatched
    assert len(cands) == 0
    assert len(analyzer.active_setups) == 0


def test_multi_timeframe_engine_end_to_end_callback_integration():
    """Verify TechnicalAnalysisEngine automatically drives MultiTimeframeAnalyzer end-to-end."""
    candidates = []

    def on_cand(cand: TechnicalCandidate):
        candidates.append(cand)

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_technical_candidate=on_cand,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY50-INDEX"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. 15m candle forms bullish structure
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=base_time,
            open=25020.0,
            high=25080.0,
            low=25010.0,
            close=25075.0,
            volume=0,
            is_closed=True,
        )
    )

    # 2. Warm up 22 5m candles below resistance (25080)
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=2000,
                is_closed=True,
            )
        )
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=0,
                is_closed=True,
            )
        )

    # 3. 5m Breakout bar (close 25120 > 25100) with volume surge (8000 vs 2000)
    ts_bo = base_time + timedelta(minutes=5 * 22)
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="5m",
            timestamp=ts_bo,
            open=25085.0,
            high=25125.0,
            low=25080.0,
            close=25120.0,
            volume=8000,
            is_closed=True,
        )
    )
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=ts_bo,
            open=25085.0,
            high=25125.0,
            low=25080.0,
            close=25120.0,
            volume=0,
            is_closed=True,
        )
    )

    # 4. 5m Retest bar (dips to 25098, closes 25110) -> qualifies 5m setup
    ts_re = base_time + timedelta(minutes=5 * 23)
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="5m",
            timestamp=ts_re,
            open=25105.0,
            high=25115.0,
            low=25098.0,
            close=25110.0,
            volume=3000,
            is_closed=True,
        )
    )
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=ts_re,
            open=25105.0,
            high=25115.0,
            low=25098.0,
            close=25110.0,
            volume=0,
            is_closed=True,
        )
    )

    # 5. 1m candle at ts_re closes green above 25100 -> triggers TechnicalCandidate
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="1m",
            timestamp=ts_re,
            open=25105.0,
            high=25115.0,
            low=25102.0,
            close=25112.0,
            volume=0,
            is_closed=True,
        )
    )

    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.symbol == sym
    assert cand.direction == BreakoutDirection.BULLISH
    assert cand.entry_price == 25112.0
    assert cand.stop_loss <= 25100.0
    assert cand.confluence_score >= 60.0
