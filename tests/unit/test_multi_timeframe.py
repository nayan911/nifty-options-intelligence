"""Unit Tests for Phase C MultiTimeframeAnalyzer and TechnicalCandidate Synthesis.

Includes rigorous validation for:
- Causal timestamps & bar completion (zero look-ahead bias)
- Missing data scoring (zero points awarded for missing indicators)
- Setup lifecycle tracking & permanent invalidation
- Macro 15m context revalidation prior to 1m entry
- Cash-futures basis independence in VWAP evaluation
- Risk-reward & headroom validation before opposing obstacles
"""

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
    SetupLifecycleState,
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

    # 1. 15m candle from 09:45-10:00 (closed at 10:00) with BEARISH trend
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25200.0,
        high=25210.0,
        low=25000.0,
        close=25020.0,
        volume=0,
        trend_bias="BEARISH",
        is_warmed_up=True,
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 5m Bullish Breakout Setup attempts to qualify at 10:05 (bar 10:00-10:05)
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
    snap_15m.trend_bias = "BULLISH"
    res_accepted = analyzer.on_5m_setup(setup_bullish, snap_5m)
    assert res_accepted is not None
    assert setup_bullish.setup_id in analyzer.active_setups
    assert res_accepted.state == SetupLifecycleState.WAITING_ENTRY


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
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25050.0,
        high=25115.0,
        low=25040.0,
        close=25090.0,
        volume=0,
        trend_bias="BULLISH",
        nearest_resistance=barrier_zone,
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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
    snap_15m.nearest_resistance = barrier_zone_far
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

    # 1. 15m Macro Context (09:45-10:00, closed at 10:00)
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 2. 5m Setup Qualifies at 10:05 (bar 10:00-10:05)
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
    state.latest_snapshots["5m"] = snap_5m
    state.snapshot_history["5m"].append(snap_5m)

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
    assert active.available_at == base_time + timedelta(minutes=5)
    assert len(candidates) == 0

    # 3. 1m Candle Closes at 10:06 (opening at 10:05 >= available_at) with confirmation
    t_1m = base_time + timedelta(minutes=5)
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
    state.latest_snapshots["1m"] = snap_1m
    state.snapshot_history["1m"].append(snap_1m)

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
    snap_1m_2 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m_next,
        timeframe="1m",
        open=25115.0,
        high=25125.0,
        low=25112.0,
        close=25120.0,
        volume=0,
        ema_9=25112.0,
        vwap=25100.0,
        rsi=62.0,
    )
    generated_2 = analyzer.on_1m_candle(candle_1m_2, snap_1m_2)
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
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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
    state.latest_snapshots["5m"] = snap_5m
    state.snapshot_history["5m"].append(snap_5m)
    analyzer.on_5m_setup(setup, snap_5m)

    # 1m candle collapses below nominal breakout price (25100) to 25095
    t_1m = base_time + timedelta(minutes=5)
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
    # Permanently invalidated: purged from active_setups!
    assert len(analyzer.active_setups) == 0


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
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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
        vwap=25090.0,
    )
    state.latest_snapshots["5m"] = snap_5m
    state.snapshot_history["5m"].append(snap_5m)
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
    t_1m = base_time + timedelta(minutes=5)
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
        ema_9=25105.0,
        vwap=25090.0,
        rsi=55.0,
    )
    state.latest_snapshots["1m"] = snap_1m
    state.snapshot_history["1m"].append(snap_1m)

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
    t_15m = base_time - timedelta(minutes=15)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_15m,
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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

    # Candle arrives 21 minutes after 5m completion (10:05 + 21m = 10:26 > 10:20 expiry)
    t_late = base_time + timedelta(minutes=26)
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
    if engine.multi_timeframe_analyzer:
        engine.multi_timeframe_analyzer.min_confluence_score = 50.0
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY50-INDEX"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. 15m candle forms bullish structure (09:15-09:30)
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

    # 2. Warm up 22 5m candles below resistance (25080) and keep 15m context fresh
    for i in range(22):
        ts_cur = base_time + timedelta(minutes=5 * i)
        if i % 3 == 0:
            engine.on_candle(
                Candle(
                    symbol=sym,
                    timeframe="15m",
                    timestamp=ts_cur,
                    open=25020.0,
                    high=25080.0,
                    low=25010.0,
                    close=25075.0,
                    volume=0,
                    is_closed=True,
                )
            )
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=ts_cur,
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
                timestamp=ts_cur,
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

    # 4. 5m Retest bar (dips to 25098, closes 25110) -> qualifies 5m setup (bar 11:10-11:15)
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

    # 5. 1m candle arrives AFTER 5m retest candle completed (timestamp = 11:15 >= 11:15 available_at)
    ts_1m = ts_re + timedelta(minutes=5)
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="1m",
            timestamp=ts_1m,
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
    assert cand.confluence_score >= 50.0


# ==============================================================================
# Dedicated Regression Tests for Timing, Lifecycle, and Missing-Data Correctness
# ==============================================================================


def test_multi_timeframe_lookahead_rejection_future_15m():
    """Verify Tier 1: Reject 5m setup qualification if 15m snapshot closed in the future."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # 15m candle from 10:00 to 10:15 (closes at 10:15: in the future relative to 10:05!)
    snap_15m_future = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="15m",
        open=25050.0,
        high=25120.0,
        low=25040.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m_future
    state.snapshot_history["15m"].append(snap_15m_future)

    # 5m candle from 10:00 to 10:05 (closes at 10:05)
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

    # Invariant: 15m bar close (10:15) > 5m bar close (10:05) -> Rejected due to lookahead bias!
    assert analyzer.on_5m_setup(setup, snap_5m) is None
    assert len(analyzer.active_setups) == 0


def test_multi_timeframe_causal_sequencing_1m_candle_before_5m_close():
    """Verify Tier 3: 1m candles that closed before or during the 5m bar cannot confirm entry."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # Valid 15m macro snapshot from 09:45-10:00 (closed at 10:00)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 5m candle from 10:00-10:05 (available_at = 10:05)
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
        rvol=2.0,
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
    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None
    assert active.available_at == base_time + timedelta(minutes=5)  # 10:05

    # 1m candle from 10:02 to 10:03 (timestamp 10:02 < 10:05)
    c_early = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=2),
        open=25105.0,
        high=25118.0,
        low=25102.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    snap_early = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=2),
        timeframe="1m",
        open=25105.0,
        high=25118.0,
        low=25102.0,
        close=25115.0,
        volume=0,
        ema_9=25108.0,
        rsi=60.0,
    )

    # Invariant: Early 1m candle is skipped; zero candidates generated
    cands = analyzer.on_1m_candle(c_early, snap_early)
    assert len(cands) == 0
    assert len(analyzer.active_setups) == 1


def test_multi_timeframe_missing_data_receives_zero_confluence_points():
    """Verify Confluence Scoring: Missing indicators award strictly 0 points."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)

    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        breakout_rvol=None,  # Missing breakout RVOL
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25110.0,
        low=25075.0,
        close=25105.0,
        volume=0,
        atr=20.0,
        rvol=None,  # Missing 5m RVOL
        vwap=None,  # Missing 5m VWAP
    )
    snap_1m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=5),
        timeframe="1m",
        open=25102.0,
        high=25110.0,
        low=25100.0,
        close=25105.0,
        volume=0,
        rsi=None,  # Missing 1m RSI
        vwap=None,  # Missing 1m VWAP
    )

    # 15m snapshot with INSUFFICIENT_DATA and inadequate headroom (< 1.5 ATR = 30 pts)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25110.0,
        low=24980.0,
        close=25105.0,
        volume=0,
        trend_bias="INSUFFICIENT_DATA",
    )

    score = analyzer._compute_confluence_score(
        setup=setup,
        snap_1m=snap_1m,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        direction=BreakoutDirection.BULLISH,
        headroom_points=10.0,  # Below 1.5 * 20.0 ATR
        entry_time=base_time + timedelta(minutes=6),
    )

    # Invariant: Strictly 0.0 points across all 5 dimensions
    assert score == 0.0


def test_multi_timeframe_permanent_invalidation_cannot_rebound():
    """Verify Lifecycle: Once a setup fails level holding, it is permanently INVALIDATED and purged."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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
        rvol=2.0,
    )
    analyzer.on_5m_setup(setup, snap_5m)
    assert len(analyzer.active_setups) == 1

    # Bar 1 (10:05-10:06): Price cracks below breakout level 25100 -> closes at 25095
    t_1m_1 = base_time + timedelta(minutes=5)
    c1 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m_1,
        open=25105.0,
        high=25108.0,
        low=25090.0,
        close=25095.0,
        volume=0,
        is_closed=True,
    )
    s1 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m_1,
        timeframe="1m",
        open=25105.0,
        high=25108.0,
        low=25090.0,
        close=25095.0,
        volume=0,
    )
    analyzer.on_1m_candle(c1, s1)

    # Invariant: Active setup is permanently dropped
    assert len(analyzer.active_setups) == 0

    # Bar 2 (10:06-10:07): Price rebounds back above 25100 to 25115
    t_1m_2 = base_time + timedelta(minutes=6)
    c2 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m_2,
        open=25095.0,
        high=25120.0,
        low=25095.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    s2 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m_2,
        timeframe="1m",
        open=25095.0,
        high=25120.0,
        low=25095.0,
        close=25115.0,
        volume=0,
        ema_9=25108.0,
        rsi=62.0,
    )
    cands = analyzer.on_1m_candle(c2, s2)
    # Invariant: Cannot emit candidate because the setup was permanently invalidated
    assert len(cands) == 0


def test_multi_timeframe_revalidate_15m_macro_trend_reversal_before_entry():
    """Verify Tier 3: Reversal of 15m trend to opposing direction before 1m entry invalidates setup."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # 15m snapshot at 09:45-10:00 is BULLISH
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

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
        rvol=2.0,
    )
    analyzer.on_5m_setup(setup, snap_5m)
    assert len(analyzer.active_setups) == 1

    # Before 1m entry, 15m trend reverses to BEARISH
    snap_15m_rev = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25020.0,
        volume=0,
        trend_bias="BEARISH",
    )
    state.latest_snapshots["15m"] = snap_15m_rev
    state.snapshot_history["15m"].append(snap_15m_rev)

    # 1m candle arrives at 10:05
    t_1m = base_time + timedelta(minutes=5)
    c1 = Candle(
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
    s1 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25118.0,
        low=25102.0,
        close=25115.0,
        volume=0,
        ema_9=25108.0,
        rsi=60.0,
    )

    cands = analyzer.on_1m_candle(c1, s1)
    # Invariant: Rejected and purged due to macro trend reversal!
    assert len(cands) == 0
    assert len(analyzer.active_setups) == 0


def test_multi_timeframe_headroom_blocks_target_1():
    """Verify Tier 3: If opposing 15m barrier blocks Target 1 (1.5R), candidate is rejected."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # Opposing 15m resistance is at 25122 (lower bound 25120)
    barrier_zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=25122.0,
        upper_bound=25125.0,
        lower_bound=25120.0,
        description="15m Major Resistance",
        weight=3.0,
    )
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25080.0,
        volume=0,
        trend_bias="BULLISH",
        nearest_resistance=barrier_zone,
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 5m setup qualifies at 25100
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25095.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25110.0,
        low=25075.0,
        close=25105.0,
        volume=0,
        atr=10.0,
        rvol=2.0,
    )
    analyzer.on_5m_setup(setup, snap_5m)

    # 1m entry candle enters at 25112 with stop loss at 25095 -> risk_points = 17.0
    # Target 1 would be: 25112 + 1.5 * 17 = 25137.5
    # But 15m resistance lower_bound is 25120! Available headroom is only 8.0 points < 25.5 points!
    t_1m = base_time + timedelta(minutes=5)
    c1 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25105.0,
        high=25115.0,
        low=25102.0,
        close=25112.0,
        volume=0,
        is_closed=True,
    )
    s1 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25115.0,
        low=25102.0,
        close=25112.0,
        volume=0,
        ema_9=25108.0,
        rsi=60.0,
    )

    cands = analyzer.on_1m_candle(c1, s1)
    # Invariant: Candidate rejected because 15m resistance is too close to allow Target 1
    assert len(cands) == 0


def test_multi_timeframe_futures_vwap_basis_independence():
    """Verify VWAP Evaluation: Spot index evaluates paired futures price vs futures VWAP without cash-futures basis distortion."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"
    analyzer = MultiTimeframeAnalyzer(engine=engine)

    # Spot trades at 25000 with no spot VWAP
    snap_5m_spot = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_time,
        timeframe="5m",
        open=24980.0,
        high=25020.0,
        low=24970.0,
        close=25000.0,
        volume=0,
        vwap=None,
    )

    # Futures trades at 25080 (premium basis = +80), and Futures VWAP is 25050
    # Spot price (25000) < Futures VWAP (25050), BUT Futures price (25080) >= Futures VWAP (25050)!
    snap_5m_fut = TechnicalSnapshot(
        symbol=fut_sym,
        timestamp=base_time,
        timeframe="5m",
        open=25060.0,
        high=25090.0,
        low=25045.0,
        close=25080.0,
        volume=10000,
        vwap=25050.0,
    )
    engine.states[fut_sym].latest_snapshots["5m"] = snap_5m_fut
    engine.states[fut_sym].snapshot_history["5m"].append(snap_5m_fut)

    # Invariant: Bullish alignment is True because Futures close (25080) >= Futures VWAP (25050)
    is_aligned = analyzer._evaluate_vwap_alignment(
        snap=snap_5m_spot,
        direction=BreakoutDirection.BULLISH,
        timeframe="5m",
        as_of=base_time + timedelta(minutes=5),
    )
    assert is_aligned is True

    # Invariant: Bearish alignment is False
    is_aligned_bearish = analyzer._evaluate_vwap_alignment(
        snap=snap_5m_spot,
        direction=BreakoutDirection.BEARISH,
        timeframe="5m",
        as_of=base_time + timedelta(minutes=5),
    )
    assert is_aligned_bearish is False


def test_multi_timeframe_bearish_candidate_flow():
    """Verify Bearish Setup: 15m downtrend, 5m breakdown qualification, 1m confirmation and candidate dispatch."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # 1. 15m Macro Trend is BEARISH with far support at 24800 (upper bound 24805)
    far_support = PriceZone(
        level_type=LevelType.PDL,
        nominal_price=24800.0,
        upper_bound=24805.0,
        lower_bound=24795.0,
        description="15m Far Support",
        weight=3.0,
    )
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25150.0,
        high=25160.0,
        low=24950.0,
        close=24960.0,
        volume=0,
        trend_bias="BEARISH",
        nearest_support=far_support,
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 2. 5m Breakdown setup qualifies at 25000 (PDL)
    setup = BreakoutContext(
        level_id="PDL_25000.0",
        level_type="PDL",
        nominal_price=25000.0,
        direction=BreakoutDirection.BEARISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25040.0,
        target_price=24920.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25020.0,
        high=25025.0,
        low=24970.0,
        close=24975.0,
        volume=0,
        atr=20.0,
        rvol=2.2,
        vwap=25010.0,
    )
    state.latest_snapshots["5m"] = snap_5m
    state.snapshot_history["5m"].append(snap_5m)

    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None
    assert active.headroom_points > 100.0

    # 3. 1m candle confirms entry at 10:05 (red candle closing below 25000 at 24970)
    t_1m = base_time + timedelta(minutes=5)
    micro_res = PriceZone(
        level_type=LevelType.SWING_HIGH_CLUSTER,
        nominal_price=24985.0,
        upper_bound=24988.0,
        lower_bound=24982.0,
        description="1m Micro Resistance",
        weight=2.0,
    )
    candle_1m = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=24980.0,
        high=24982.0,
        low=24968.0,
        close=24970.0,
        volume=0,
        is_closed=True,
    )
    snap_1m = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=24980.0,
        high=24982.0,
        low=24968.0,
        close=24970.0,
        volume=0,
        ema_9=24978.0,
        vwap=25005.0,
        rsi=40.0,  # Sweet spot for bearish: 32 - 50
        nearest_resistance=micro_res,
    )
    state.latest_snapshots["1m"] = snap_1m
    state.snapshot_history["1m"].append(snap_1m)

    cands = analyzer.on_1m_candle(candle_1m, snap_1m)
    assert len(cands) == 1
    cand = cands[0]
    assert cand.direction == BreakoutDirection.BEARISH
    assert cand.entry_price == 24970.0
    # Stop-loss tightened to 1m micro resistance (24988.0) instead of base SL (25040.0)
    assert cand.stop_loss == 24988.0
    assert cand.risk_points == 18.0
    assert cand.target_1 == round(24970.0 - 1.5 * 18.0, 1)  # 24943.0
    assert cand.confluence_score >= 60.0
    assert "Score:" in cand.summary()


def test_multi_timeframe_actual_available_at_vs_bar_close():
    """Verify Timing: When futures confirmation arrives late, 1m entry must close after actual available_at."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # 5m candle closes at 10:05, but confirmation arrives at 10:06:20
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
        rvol=2.0,
        available_at=base_time + timedelta(minutes=6, seconds=20),
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
        stop_loss=25080.0,
    )
    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None
    assert active.bar_close_time == base_time + timedelta(minutes=5)
    assert active.available_at == base_time + timedelta(minutes=6, seconds=20)

    # 1. 1m candle for 10:05-10:06 (closes at 10:06:00 < 10:06:20)
    c_early = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=5),
        open=25105.0,
        high=25115.0,
        low=25102.0,
        close=25112.0,
        volume=0,
        is_closed=True,
    )
    s_early = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=5),
        timeframe="1m",
        open=25105.0,
        high=25115.0,
        low=25102.0,
        close=25112.0,
        volume=0,
        ema_9=25108.0,
        rsi=60.0,
    )
    # Invariant: 10:05-10:06 candle closed before futures arrived at 10:06:20 -> Skipped!
    cands_early = analyzer.on_1m_candle(c_early, s_early)
    assert len(cands_early) == 0
    assert len(analyzer.active_setups) == 1

    # 2. 1m candle for 10:06-10:07 (closes at 10:07:00 > 10:06:20)
    c_valid = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=6),
        open=25110.0,
        high=25120.0,
        low=25108.0,
        close=25116.0,
        volume=0,
        is_closed=True,
    )
    s_valid = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=6),
        timeframe="1m",
        open=25110.0,
        high=25120.0,
        low=25108.0,
        close=25116.0,
        volume=0,
        ema_9=25110.0,
        vwap=25095.0,
        rsi=62.0,
    )
    cands_valid = analyzer.on_1m_candle(c_valid, s_valid)
    assert len(cands_valid) == 1
    assert cands_valid[0].entry_price == 25116.0


def test_multi_timeframe_rejected_duplicate_candle_not_forwarded():
    """Verify Isolation: Rejected duplicate/out-of-order candles do not invoke MultiTimeframeAnalyzer."""
    called_candidates = []

    def on_cand(c: TechnicalCandidate):
        called_candidates.append(c)

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_technical_candidate=on_cand,
    )
    if engine.multi_timeframe_analyzer:
        engine.multi_timeframe_analyzer.min_confluence_score = 50.0
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    # Feed 1m candle at 10:05
    c1 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=5),
        open=25100.0,
        high=25110.0,
        low=25095.0,
        close=25105.0,
        volume=0,
        is_closed=True,
    )
    snap1 = engine.on_candle(c1)
    assert snap1 is not None
    assert engine.states[sym].last_candle_accepted["1m"] is True

    # Re-feed identical duplicate candle at 10:05
    snap2 = engine.on_candle(c1)
    assert snap2 is not None
    # Invariant: Duplicate was rejected
    assert engine.states[sym].last_candle_accepted["1m"] is False

    # Feed older candle at 10:03 (out-of-order)
    c_old = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=3),
        open=25090.0,
        high=25100.0,
        low=25085.0,
        close=25095.0,
        volume=0,
        is_closed=True,
    )
    snap_old = engine.on_candle(c_old)
    assert snap_old is not None
    # Invariant: Out-of-order candle was rejected
    assert engine.states[sym].last_candle_accepted["1m"] is False


def test_multi_timeframe_stale_15m_macro_rejected():
    """Verify Freshness: 15m macro snapshot older than max_15m_macro_age_seconds is rejected."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    # Set 15m max age to 15 minutes (900 seconds)
    analyzer = MultiTimeframeAnalyzer(engine=engine, max_15m_macro_age_seconds=900.0)
    state = engine.states[sym]

    # 15m candle from 09:15 to 09:30 (closed at 09:30)
    snap_15m_stale = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=45),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m_stale
    state.snapshot_history["15m"].append(snap_15m_stale)

    # 5m candle at 10:00-10:05 (available at 10:05 -> 35 minutes after 09:30 close > 15m limit!)
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
        rvol=2.0,
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

    # Invariant: Rejected due to stale 15m macro data!
    assert analyzer.on_5m_setup(setup, snap_5m) is None


def test_multi_timeframe_missing_resistance_headroom_none_no_inflated_score():
    """Verify Scoring: Unknown headroom is None and awards 0 confluence points."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # 15m snapshot with no nearest resistance
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
        nearest_resistance=None,  # Open air / unknown
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25080.0,
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
        rvol=2.0,
        vwap=25080.0,
    )
    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None
    # Invariant: Headroom is None (NOT 999.0!)
    assert active.headroom_points is None

    # Score with unknown headroom receives 0 headroom points
    snap_1m_no_vwap = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=5),
        timeframe="1m",
        open=25102.0,
        high=25110.0,
        low=25100.0,
        close=25105.0,
        volume=0,
        vwap=None,
    )
    score = analyzer._compute_confluence_score(
        setup=setup,
        snap_1m=snap_1m_no_vwap,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        direction=BreakoutDirection.BULLISH,
        headroom_points=None,
        entry_time=base_time + timedelta(minutes=5),
    )
    # Max without headroom (+15) and without 1m RSI (+10) is 30(trend) + 25(rvol) + 10(5m vwap) = 65
    assert score == 65.0


def test_multi_timeframe_target_2_capped_at_obstacle():
    """Verify Targets: Target 2 is strictly capped at opposing 15m barrier if nominal 2.0R exceeds it."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    # Opposing 15m resistance barrier at 25142 (lower bound 25140)
    barrier = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=25142.0,
        upper_bound=25145.0,
        lower_bound=25140.0,
        description="15m Resistance",
        weight=3.0,
    )
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
        nearest_resistance=barrier,
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25095.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25110.0,
        volume=0,
        atr=10.0,
        rvol=2.0,
        vwap=25090.0,
    )
    active = analyzer.on_5m_setup(setup, snap_5m)
    assert active is not None

    # 1m entry candle enters at 25110 with stop loss at 25095 -> risk_points = 15.0
    # Nominal T1 = 25110 + 1.5 * 15 = 25132.5 (< 25140 resistance -> OK!)
    # Nominal T2 = 25110 + 2.0 * 15 = 25140 + 10 = 25150.0 (> 25140 resistance -> MUST BE CAPPED!)
    t_1m = base_time + timedelta(minutes=5)
    c1 = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25105.0,
        high=25112.0,
        low=25102.0,
        close=25110.0,
        volume=0,
        is_closed=True,
    )
    s1 = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25112.0,
        low=25102.0,
        close=25110.0,
        volume=0,
        ema_9=25108.0,
        vwap=25095.0,
        rsi=60.0,
    )
    cands = analyzer.on_1m_candle(c1, s1)
    assert len(cands) == 1
    cand = cands[0]
    assert cand.target_1 == 25132.5
    # Invariant: Target 2 is strictly capped at barrier lower_bound (25140.0), NOT extending to 25150.0!
    assert cand.target_2 == 25140.0


def test_multi_timeframe_intrabar_stop_loss_breach():
    """Verify Invalidation: Intrabar low crossing stop loss invalidates setup even if close recovers."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    state = engine.states[sym]

    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time - timedelta(minutes=15),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    state.latest_snapshots["15m"] = snap_15m
    state.snapshot_history["15m"].append(snap_15m)

    # Base stop-loss is 25090.0
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time,
        symbol=sym,
        timeframe="5m",
        stop_loss=25090.0,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25110.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    analyzer.on_5m_setup(setup, snap_5m)
    assert len(analyzer.active_setups) == 1

    # 1m candle dips intrabar to low = 25088 (< 25090 SL breach!), but closes at 25105 (above 25100)
    t_1m = base_time + timedelta(minutes=5)
    c_wick = Candle(
        symbol=sym,
        timeframe="1m",
        timestamp=t_1m,
        open=25105.0,
        high=25112.0,
        low=25088.0,  # Intrabar breach!
        close=25105.0,
        volume=0,
        is_closed=True,
    )
    s_wick = TechnicalSnapshot(
        symbol=sym,
        timestamp=t_1m,
        timeframe="1m",
        open=25105.0,
        high=25112.0,
        low=25088.0,
        close=25105.0,
        volume=0,
        ema_9=25102.0,
        rsi=55.0,
    )

    cands = analyzer.on_1m_candle(c_wick, s_wick)
    # Invariant: Intrabar breach invalidates setup; no candidate dispatched
    assert len(cands) == 0
    assert len(analyzer.active_setups) == 0


def test_multi_timeframe_live_pipeline_actual_available_at_propagation():
    """Verify Live Pipeline: When futures arrives late with observed_at, actual_available_at is propagated end-to-end."""
    candidates = []
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_technical_candidate=lambda c: candidates.append(c),
    )
    if engine.multi_timeframe_analyzer:
        engine.multi_timeframe_analyzer.min_confluence_score = 40.0
    start_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"

    # Set daily references
    engine.set_daily_references(spot_sym, pdh=25100.0, pdl=24900.0, pdc=25000.0, cdo=25020.0)

    # Warm up 21 5m candles below PDH (25050-25080) and keep 15m context fresh
    for i in range(21):
        ts_cur = start_time + timedelta(minutes=5 * i)
        if i % 3 == 0:
            engine.on_candle(
                Candle(
                    symbol=spot_sym,
                    timeframe="15m",
                    timestamp=ts_cur,
                    open=25020.0,
                    high=25080.0,
                    low=25010.0,
                    close=25075.0,
                    volume=0,
                    is_closed=True,
                )
            )
        engine.on_candle(
            Candle(
                symbol=fut_sym,
                timeframe="5m",
                timestamp=ts_cur,
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
                symbol=spot_sym,
                timeframe="5m",
                timestamp=ts_cur,
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=0,
                is_closed=True,
            )
        )

    # Breakout candle at 11:00 (closes at 11:05)
    ts_bo = start_time + timedelta(minutes=5 * 21)
    engine.on_candle(
        Candle(
            symbol=fut_sym,
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
            symbol=spot_sym,
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

    # Retest candle at 11:05 (closes at 11:10)
    ts_re = start_time + timedelta(minutes=5 * 22)
    # Spot 5m candle closes at 11:10 (interval 11:05-11:10), but has no volume -> waits for futures
    c_spot_5m = Candle(
        symbol=spot_sym,
        timeframe="5m",
        timestamp=ts_re,
        open=25105.0,
        high=25115.0,
        low=25098.0,
        close=25110.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(c_spot_5m)
    # Spot 5m is waiting for futures volume
    assert len(engine.multi_timeframe_analyzer.active_setups) == 0

    # Futures 5m candle arrives at 11:11:20 (observed_at = 11:11:20)
    actual_arrival = ts_re + timedelta(minutes=6, seconds=20)  # 11:11:20
    c_fut_5m = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=ts_re,
        open=25105.0,
        high=25115.0,
        low=25098.0,
        close=25110.0,
        volume=3000,
        is_closed=True,
        observed_at=actual_arrival,
    )
    engine.on_candle(c_fut_5m)

    # Invariant: Active setup qualified with genuine available_at = 11:11:20!
    assert len(engine.multi_timeframe_analyzer.active_setups) >= 1
    active = list(engine.multi_timeframe_analyzer.active_setups.values())[0]
    assert active.available_at == actual_arrival

    # 1m candle for 11:10-11:11 (closes at 11:11:00 < 11:11:20 actual arrival)
    c_1m_early = Candle(
        symbol=spot_sym,
        timeframe="1m",
        timestamp=ts_re + timedelta(minutes=5),  # 11:10-11:11
        open=25110.0,
        high=25118.0,
        low=25108.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(c_1m_early)
    # Invariant: 11:10-11:11 candle closed before futures arrived; no candidate emitted
    assert len(candidates) == 0

    # 1m candle for 11:11-11:12 (closes at 11:12:00 > 11:11:20)
    c_1m_valid = Candle(
        symbol=spot_sym,
        timeframe="1m",
        timestamp=ts_re + timedelta(minutes=6),  # 11:11-11:12
        open=25112.0,
        high=25122.0,
        low=25110.0,
        close=25118.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(c_1m_valid)
    # Invariant: Candidate emitted on the first 1m bar that closed after actual available_at!
    assert len(candidates) >= 1


def test_multi_timeframe_simultaneous_arrival_order_invariance():
    """Verify Invariance: When 5m and 15m close on the same minute, order of arrival does not change output."""
    base_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    sym = "NSE:NIFTY50-INDEX"

    _c_5m = Candle(
        symbol=sym,
        timeframe="5m",
        timestamp=base_time + timedelta(minutes=10),  # 10:10-10:15
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time + timedelta(minutes=10),
        timeframe="5m",
        open=25080.0,
        high=25125.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_time + timedelta(minutes=10),
        symbol=sym,
        timeframe="5m",
        stop_loss=25080.0,
    )

    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_time,  # 10:00-10:15 (closes at 10:15)
        timeframe="15m",
        open=25020.0,
        high=25120.0,
        low=25010.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )

    # Run A: 15m processed first, then 5m processed
    engine_a = TechnicalAnalysisEngine(primary_index=sym, primary_futures="NSE:NIFTY26NOVFUT")
    analyzer_a = MultiTimeframeAnalyzer(engine=engine_a)
    engine_a.states[sym].latest_snapshots["15m"] = snap_15m
    engine_a.states[sym].snapshot_history["15m"].append(snap_15m)
    res_a = analyzer_a.on_5m_setup(setup, snap_5m)

    # Run B: 5m processed first (buffered in _pending_5m_setups), then 15m processed
    engine_b = TechnicalAnalysisEngine(primary_index=sym, primary_futures="NSE:NIFTY26NOVFUT")
    analyzer_b = MultiTimeframeAnalyzer(engine=engine_b)
    # At first, 15m has not arrived yet: on_5m_setup buffers
    res_b_early = analyzer_b.on_5m_setup(setup, snap_5m)
    assert res_b_early is None
    assert len(analyzer_b._pending_5m_setups) == 1

    # Now 15m arrives
    engine_b.states[sym].latest_snapshots["15m"] = snap_15m
    engine_b.states[sym].snapshot_history["15m"].append(snap_15m)
    analyzer_b.on_15m_snapshot(snap_15m)

    # Invariant: Both runs end up with the identical qualified active setup!
    assert res_a is not None
    assert setup.setup_id in analyzer_a.active_setups
    assert setup.setup_id in analyzer_b.active_setups
    assert analyzer_a.active_setups[setup.setup_id].available_at == analyzer_b.active_setups[setup.setup_id].available_at


def test_multi_timeframe_independent_rr_telemetry():
    """Verify Telemetry: TechnicalCandidate reports independent actual R:R for both Target 1 and Target 2."""
    candidate = TechnicalCandidate(
        candidate_id="NSE_NIFTY50-INDEX_20261123_101500_BULLISH_PDH_25100.0",
        symbol="NSE:NIFTY50-INDEX",
        timestamp=IST.localize(datetime(2026, 11, 23, 10, 15)),
        direction=BreakoutDirection.BULLISH,
        setup_level="PDH_25100.0",
        nominal_price=25100.0,
        entry_price=25000.0,
        stop_loss=24980.0,  # Risk = 20 pts
        target_1=25030.0,  # +30 pts = 1.5R
        target_2=25035.0,  # +35 pts = 1.75R (capped at resistance)
        risk_points=20.0,
        reward_points=30.0,
        risk_reward_ratio=1.5,
        confluence_score=85.0,
        target_1_rr=1.50,
        target_2_rr=1.75,
        headroom_points=35.0,
    )

    summary = candidate.summary()
    # Invariant: Telemetry accurately distinguishes 1.50R from 1.75R
    assert "T1: 25030.00 (1.50R)" in summary
    assert "T2: 25035.00 (1.75R)" in summary


def test_market_data_runner_callback_none_headroom(capsys):
    """Verify Runner Callback: _on_technical_candidate formats safely when headroom_points is None."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.run_market_data import MarketDataEngine

    candidate = TechnicalCandidate(
        candidate_id="NSE_NIFTY50-INDEX_20261123_101500_BULLISH_PDH_25100.0",
        symbol="NSE:NIFTY50-INDEX",
        timestamp=IST.localize(datetime(2026, 11, 23, 10, 15)),
        direction=BreakoutDirection.BULLISH,
        setup_level="PDH_25100.0",
        nominal_price=25100.0,
        entry_price=25110.0,
        stop_loss=25095.0,
        target_1=25132.5,
        target_2=25140.0,
        risk_points=15.0,
        reward_points=22.5,
        risk_reward_ratio=1.5,
        confluence_score=75.0,
        target_1_rr=1.50,
        target_2_rr=2.00,
        headroom_points=None,  # None headroom must not crash formatting!
        timeframe_breakdown={"15m_trend": "BULLISH", "5m_rvol": 2.1},
    )

    runner = object.__new__(MarketDataEngine)
    # Invariant: Executes cleanly without raising TypeError
    runner._on_technical_candidate(candidate)
    captured = capsys.readouterr()
    assert "15m Headroom: N/A" in captured.out
    assert "T1: 25132.50 (1.50R)" in captured.out


def test_candle_builder_observed_at_and_duckdb_persistence(tmp_path):
    """Verify Candle Builder: observed_at captures local receipt time and persists to DuckDB."""
    from nifty_engine.core.interfaces import Tick
    from nifty_engine.data.candle_builder import RealtimeCandleBuilder
    from nifty_engine.data.storage.duckdb_store import DuckDBStore

    builder = RealtimeCandleBuilder()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0, 0))
    t1 = IST.localize(datetime(2026, 11, 23, 10, 0, 30))
    t2 = IST.localize(datetime(2026, 11, 23, 10, 1, 5))
    rec1 = IST.localize(datetime(2026, 11, 23, 10, 1, 5, 250000))

    # Bar 1 ticks (10:00:00 - 10:01:00)
    builder.process_tick(Tick(symbol="NSE:NIFTY50-INDEX", ltp=25000.0, exchange_timestamp=t0, local_timestamp=t0, is_index=True))
    builder.process_tick(Tick(symbol="NSE:NIFTY50-INDEX", ltp=25010.0, exchange_timestamp=t1, local_timestamp=t1, is_index=True))

    # Tick crossing boundary at 10:01:05 with local_timestamp = rec1
    closed = builder.process_tick(Tick(symbol="NSE:NIFTY50-INDEX", ltp=25015.0, exchange_timestamp=t2, local_timestamp=rec1, is_index=True))
    assert len(closed) == 1
    c1m = closed[0]
    assert c1m.timeframe == "1m"
    assert c1m.timestamp == t0
    # Invariant: observed_at captures local receipt timestamp, NOT exchange timestamp!
    assert c1m.observed_at == rec1

    # Clock-driven flush test
    t_flush = IST.localize(datetime(2026, 11, 23, 10, 3, 0))
    flushed = builder.flush_closed_bars(current_time=t_flush)
    assert len(flushed) >= 1
    c_flushed = next(c for c in flushed if c.timeframe == "1m")
    assert c_flushed.observed_at == t_flush

    # Persist and restore from DuckDB
    db_file = tmp_path / "test_pipeline.duckdb"
    store = DuckDBStore(db_path=str(db_file))
    store.insert_candle(c1m)
    restored = store.get_candles("NSE:NIFTY50-INDEX", "1m", t0 - timedelta(minutes=1), t0 + timedelta(minutes=1))
    assert len(restored) == 1
    assert restored[0].observed_at == rec1


def test_engine_regular_snapshots_populate_available_at():
    """Verify TechnicalAnalysisEngine: Regular 1m, 5m, 15m snapshots populate available_at."""
    engine = TechnicalAnalysisEngine()
    sym = "NSE:NIFTY50-INDEX"
    base_t = IST.localize(datetime(2026, 11, 23, 10, 0))
    obs_1m = base_t + timedelta(minutes=1, seconds=2)
    obs_5m = base_t + timedelta(minutes=5, seconds=3)
    obs_15m = base_t + timedelta(minutes=15, seconds=4)

    s1 = engine.on_candle(Candle(symbol=sym, timeframe="1m", timestamp=base_t, open=25000.0, high=25010.0, low=24995.0, close=25005.0, volume=0, is_closed=True, observed_at=obs_1m))
    s5 = engine.on_candle(Candle(symbol=sym, timeframe="5m", timestamp=base_t, open=25000.0, high=25020.0, low=24990.0, close=25015.0, volume=0, is_closed=True, observed_at=obs_5m))
    s15 = engine.on_candle(Candle(symbol=sym, timeframe="15m", timestamp=base_t, open=25000.0, high=25050.0, low=24980.0, close=25040.0, volume=0, is_closed=True, observed_at=obs_15m))

    assert s1 is not None and s1.available_at == obs_1m
    assert s5 is not None and s5.available_at == obs_5m
    assert s15 is not None and s15.available_at == obs_15m


def test_multi_timeframe_15m_coordination_uses_later_availability():
    """Verify 15m Coordination: Unblocked 5m setup uses the later of 5m and 15m availability."""
    engine = TechnicalAnalysisEngine()
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    base_t = IST.localize(datetime(2026, 11, 23, 10, 10))  # 10:10-10:15
    sym = "NSE:NIFTY50-INDEX"

    t_5m_avail = IST.localize(datetime(2026, 11, 23, 10, 15, 2))
    t_15m_avail = IST.localize(datetime(2026, 11, 23, 10, 15, 4))

    snap_5m = TechnicalSnapshot(
        symbol=sym,
        timestamp=base_t,
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
        available_at=t_5m_avail,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_t,
        symbol=sym,
        timeframe="5m",
        stop_loss=25080.0,
    )

    # 5m setup arrives at 10:15:02 on 15m boundary -> buffered because 15m is not ready yet
    res = analyzer.on_5m_setup(setup, snap_5m, available_at=t_5m_avail)
    assert res is None
    assert len(analyzer._pending_5m_setups) == 1

    # 15m snapshot arrives at 10:15:04 (effective_available_at = 10:15:04)
    snap_15m = TechnicalSnapshot(
        symbol=sym,
        timestamp=IST.localize(datetime(2026, 11, 23, 10, 0)),  # 10:00-10:15
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
        available_at=t_15m_avail,
    )
    engine.states[sym].latest_snapshots["15m"] = snap_15m
    engine.states[sym].snapshot_history["15m"].append(snap_15m)

    # Unblock
    analyzer.on_15m_snapshot(snap_15m)
    assert len(analyzer._pending_5m_setups) == 0
    assert len(analyzer.active_setups) == 1
    active = list(analyzer.active_setups.values())[0]
    # Invariant: Setup availability is 10:15:04 (the later timestamp), not 10:15:02!
    assert active.available_at == t_15m_avail


def test_multi_timeframe_pending_setups_isolated_by_symbol():
    """Verify Symbol Isolation: Unrelated instrument candles do not flush pending setups."""
    engine = TechnicalAnalysisEngine()
    analyzer = MultiTimeframeAnalyzer(engine=engine)
    base_t = IST.localize(datetime(2026, 11, 23, 10, 10))  # 10:10-10:15
    sym_nifty = "NSE:NIFTY50-INDEX"
    sym_rel = "NSE:RELIANCE-EQ"

    snap_5m = TechnicalSnapshot(
        symbol=sym_nifty,
        timestamp=base_t,
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        available_at=base_t + timedelta(minutes=5),
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_t,
        symbol=sym_nifty,
        timeframe="5m",
        stop_loss=25080.0,
    )

    analyzer.on_5m_setup(setup, snap_5m)
    assert len(analyzer._pending_5m_setups) == 1

    # 1m candle for RELIANCE arrives at 10:16 (past 10:15)
    c_rel = Candle(symbol=sym_rel, timeframe="1m", timestamp=base_t + timedelta(minutes=5), open=2500.0, high=2510.0, low=2495.0, close=2505.0, volume=1000, is_closed=True)
    s_rel = TechnicalSnapshot(symbol=sym_rel, timestamp=base_t + timedelta(minutes=5), timeframe="1m", open=2500.0, high=2510.0, low=2495.0, close=2505.0, volume=1000)

    analyzer.on_1m_candle(c_rel, s_rel)
    # Invariant: NIFTY setup was NOT flushed by RELIANCE candle!
    assert len(analyzer._pending_5m_setups) == 1
    assert setup.setup_id in analyzer._pending_5m_setups

    # 1m candle for NIFTY arrives past 10:15
    c_nifty_1m = Candle(symbol=sym_nifty, timeframe="1m", timestamp=base_t + timedelta(minutes=5), open=25110.0, high=25118.0, low=25108.0, close=25115.0, volume=0, is_closed=True)
    s_nifty_1m = TechnicalSnapshot(symbol=sym_nifty, timestamp=base_t + timedelta(minutes=5), timeframe="1m", open=25110.0, high=25118.0, low=25108.0, close=25115.0, volume=0)

    analyzer.on_1m_candle(c_nifty_1m, s_nifty_1m)
    # Invariant: NIFTY setup is now flushed/evaluated for NIFTY
    assert len(analyzer._pending_5m_setups) == 0


def test_multi_timeframe_1m_entry_futures_arrival_order_invariance():
    """Verify 1m Synchronization: Order of spot vs futures 1m candle arrival does not alter entry score or candidate."""
    base_t = IST.localize(datetime(2026, 11, 23, 10, 5))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"

    def run_pipeline(spot_first: bool):
        candidates = []
        engine = TechnicalAnalysisEngine(
            primary_index=spot_sym,
            primary_futures=fut_sym,
            on_technical_candidate=lambda c: candidates.append(c),
        )
        if engine.multi_timeframe_analyzer:
            engine.multi_timeframe_analyzer.min_confluence_score = 40.0

        # Setup 15m macro snapshot
        snap_15m = TechnicalSnapshot(symbol=spot_sym, timestamp=base_t - timedelta(minutes=20), timeframe="15m", open=25000.0, high=25120.0, low=24980.0, close=25110.0, volume=0, trend_bias="BULLISH")
        engine.states[spot_sym].latest_snapshots["15m"] = snap_15m
        engine.states[spot_sym].snapshot_history["15m"].append(snap_15m)

        # Active 5m setup
        snap_5m = TechnicalSnapshot(symbol=spot_sym, timestamp=base_t - timedelta(minutes=5), timeframe="5m", open=25080.0, high=25120.0, low=25075.0, close=25115.0, volume=0, atr=20.0, rvol=2.0)
        setup = BreakoutContext(level_id="PDH_25100.0", level_type="PDH", nominal_price=25100.0, direction=BreakoutDirection.BULLISH, current_state=BreakoutState.SETUP_ELIGIBLE, start_time=base_t - timedelta(minutes=5), symbol=spot_sym, timeframe="5m", stop_loss=25080.0)
        engine.multi_timeframe_analyzer.on_5m_setup(setup, snap_5m)

        # Mark futures active with a preceding candle
        engine.on_candle(Candle(symbol=fut_sym, timeframe="1m", timestamp=base_t - timedelta(minutes=1), open=25100.0, high=25110.0, low=25095.0, close=25105.0, volume=1000, is_closed=True))

        c_spot = Candle(symbol=spot_sym, timeframe="1m", timestamp=base_t, open=25110.0, high=25120.0, low=25105.0, close=25115.0, volume=0, is_closed=True)
        # Futures candle has VWAP = 25100; close 25118 > VWAP 25100 -> bullish VWAP alignment (+10 pts)
        c_fut = Candle(symbol=fut_sym, timeframe="1m", timestamp=base_t, open=25110.0, high=25122.0, low=25105.0, close=25118.0, volume=5000, is_closed=True)

        if spot_first:
            engine.on_candle(c_spot)
            # Spot buffered; no candidate yet
            assert len(candidates) == 0
            engine.on_candle(c_fut)
            # Futures arrives; candidate dispatched
            assert len(candidates) == 1
        else:
            engine.on_candle(c_fut)
            assert len(candidates) == 0
            engine.on_candle(c_spot)
            # Spot arrives; candidate dispatched immediately
            assert len(candidates) == 1

        return candidates[0]

    cand_spot_first = run_pipeline(spot_first=True)
    cand_fut_first = run_pipeline(spot_first=False)

    # Invariant: Confluence scores, entry prices, and targets are strictly identical regardless of arrival order!
    assert cand_spot_first.confluence_score == cand_fut_first.confluence_score
    assert cand_spot_first.entry_price == cand_fut_first.entry_price
    assert cand_spot_first.stop_loss == cand_fut_first.stop_loss
    assert cand_spot_first.target_1 == cand_fut_first.target_1
    assert cand_spot_first.target_2 == cand_fut_first.target_2





def test_multi_timeframe_entry_evaluation_respects_decision_at_and_filtering():
    """Verify P0: Multi-timeframe entry evaluation strictly uses decision_at for causal filtering."""
    base_t = IST.localize(datetime(2026, 11, 23, 10, 5))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"

    candidates = []
    engine = TechnicalAnalysisEngine(
        primary_index=spot_sym,
        primary_futures=fut_sym,
        on_technical_candidate=lambda c: candidates.append(c),
    )
    analyzer = engine.multi_timeframe_analyzer
    analyzer.min_confluence_score = 40.0

    # Mark futures feed active with a preceding candle
    engine.on_candle(
        Candle(
            symbol=fut_sym,
            timeframe="1m",
            timestamp=base_t - timedelta(minutes=1),
            open=25100.0,
            high=25110.0,
            low=25095.0,
            close=25105.0,
            volume=1000,
            is_closed=True,
        )
    )

    # 15m snapshot available at 10:00:02 (from 9:45-10:00 bar)
    t_15m_avail = IST.localize(datetime(2026, 11, 23, 10, 0, 2))
    snap_15m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=20),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
        available_at=t_15m_avail,
    )
    engine.states[spot_sym].snapshot_history["15m"].append(snap_15m)

    # 5m setup
    snap_5m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=5),
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_t - timedelta(minutes=5),
        symbol=spot_sym,
        timeframe="5m",
        stop_loss=25080.0,
    )
    analyzer.on_5m_setup(setup, snap_5m)

    # Spot 1m candle closes at 10:06:00, but observed at 10:06:01
    c_spot = Candle(
        symbol=spot_sym,
        timeframe="1m",
        timestamp=base_t,
        open=25110.0,
        high=25120.0,
        low=25105.0,
        close=25115.0,
        volume=0,
        is_closed=True,
        observed_at=IST.localize(datetime(2026, 11, 23, 10, 6, 1)),
    )
    # Futures arrives at 10:06:04
    t_fut_avail = IST.localize(datetime(2026, 11, 23, 10, 6, 4))
    c_fut = Candle(
        symbol=fut_sym,
        timeframe="1m",
        timestamp=base_t,
        open=25110.0,
        high=25122.0,
        low=25105.0,
        close=25118.0,
        volume=5000,
        is_closed=True,
        observed_at=t_fut_avail,
    )

    engine.on_candle(c_spot)
    assert len(candidates) == 0

    engine.on_candle(c_fut)
    assert len(candidates) == 1
    cand = candidates[0]
    # Decision at is exactly the latest evidence availability (10:06:04)
    assert cand.decision_at == t_fut_avail
    assert cand.confirmation_status == "CONFIRMED"
    assert cand.timeframe_breakdown["decision_at"] == t_fut_avail.isoformat()


def test_direct_5m_breakout_callback_preserves_availability_timestamp():
    """Verify P1: Direct 5m breakout callback preserves authentic availability timestamp."""
    from nifty_engine.analysis.technical.support_resistance import LevelType, PriceZone

    base_t = IST.localize(datetime(2026, 11, 23, 10, 10))
    spot_sym = "NSE:NIFTY50-INDEX"
    observed_t = IST.localize(datetime(2026, 11, 23, 10, 15, 3))

    engine = TechnicalAnalysisEngine()
    state = engine.states[spot_sym]
    state.current_trend = "BULLISH"

    pdh_zone = PriceZone(
        level_type=LevelType.PDH,
        nominal_price=25100.0,
        upper_bound=25105.0,
        lower_bound=25095.0,
        description="PDH Resistance",
        weight=1.0,
    )

    emitted_setups = []

    def on_setup(setup, snap):
        emitted_setups.append((setup, snap))

    # Pre-configure an active breakout setup in BREAKOUT_CONFIRMED awaiting retest qualification
    breakout_ctx = BreakoutContext(
        level_id="PDH_25100.0",
        level_type=LevelType.PDH,
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.BREAKOUT_CONFIRMED,
        start_time=base_t - timedelta(minutes=5),
        symbol=spot_sym,
        timeframe="5m",
        stop_loss=25085.0,
        breakout_price=25115.0,
        breakout_candle_index=0,
        zone=pdh_zone,
    )
    state.active_breakouts_5m[breakout_ctx.level_id] = breakout_ctx

    # Retest candle touches zone (low 25098 in [25095, 25105]) and closes green (25112 > open 25106)
    breakout_c = Candle(
        symbol=spot_sym,
        timeframe="5m",
        timestamp=base_t,
        open=25106.0,
        high=25120.0,
        low=25098.0,
        close=25112.0,
        volume=2500,
        is_closed=True,
        observed_at=observed_t,
    )

    # Evaluate breakouts directly
    state._evaluate_breakouts(
        timeframe="5m",
        candle=breakout_c,
        zones=[pdh_zone],
        nearest_sup=None,
        nearest_res=pdh_zone,
        n=5,
        atr_val=20.0,
        rvol_val=2.0,
        is_warmed_up=True,
        is_provisional=False,
        ema_9=25105.0,
        ema_21=25095.0,
        rsi_val=62.0,
        trend_bias="BULLISH",
        on_setup_candidate=on_setup,
        actual_available_at=observed_t,
    )

    assert len(emitted_setups) == 1
    setup, snap = emitted_setups[0]
    assert setup.current_state == BreakoutState.SETUP_ELIGIBLE
    # Invariant: Availability timestamp passed through to snapshot preview
    assert snap.effective_available_at == observed_t


def test_older_futures_candle_does_not_remove_pending_spot_entry():
    """Verify P1: Delayed older futures candle does not discard pending spot 1m entry."""
    base_t = IST.localize(datetime(2026, 11, 23, 10, 5))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"

    candidates = []
    engine = TechnicalAnalysisEngine(
        primary_index=spot_sym,
        primary_futures=fut_sym,
        on_technical_candidate=lambda c: candidates.append(c),
    )
    if engine.multi_timeframe_analyzer:
        engine.multi_timeframe_analyzer.min_confluence_score = 40.0

    # Mark futures active with a preceding candle
    engine.on_candle(
        Candle(
            symbol=fut_sym,
            timeframe="1m",
            timestamp=base_t - timedelta(minutes=1),
            open=25100.0,
            high=25110.0,
            low=25095.0,
            close=25105.0,
            volume=1000,
            is_closed=True,
        )
    )

    snap_15m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=20),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    engine.states[spot_sym].snapshot_history["15m"].append(snap_15m)

    snap_5m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=5),
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_t - timedelta(minutes=5),
        symbol=spot_sym,
        timeframe="5m",
        stop_loss=25080.0,
    )
    engine.multi_timeframe_analyzer.on_5m_setup(setup, snap_5m)

    # Spot 10:05 candle arrives and is buffered
    c_spot_1005 = Candle(
        symbol=spot_sym,
        timeframe="1m",
        timestamp=base_t,
        open=25110.0,
        high=25120.0,
        low=25105.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(c_spot_1005)
    assert spot_sym in engine._pending_1m_entry

    # Delayed older futures candle for 10:04 arrives!
    c_fut_1004 = Candle(
        symbol=fut_sym,
        timeframe="1m",
        timestamp=base_t - timedelta(minutes=1),
        open=25100.0,
        high=25110.0,
        low=25095.0,
        close=25105.0,
        volume=1000,
        is_closed=True,
    )
    engine.on_candle(c_fut_1004)

    # Invariant: Pending spot entry for 10:05 must STILL be preserved!
    assert spot_sym in engine._pending_1m_entry
    assert len(candidates) == 0

    # Now matching futures candle for 10:05 arrives
    c_fut_1005 = Candle(
        symbol=fut_sym,
        timeframe="1m",
        timestamp=base_t,
        open=25110.0,
        high=25122.0,
        low=25105.0,
        close=25118.0,
        volume=5000,
        is_closed=True,
    )
    engine.on_candle(c_fut_1005)

    # Spot entry consumed and candidate emitted!
    assert spot_sym not in engine._pending_1m_entry
    assert len(candidates) == 1
    assert candidates[0].confirmation_status == "CONFIRMED"


def test_timeout_and_missing_confirmation_status_telemetry():
    """Verify P2: Expired 1m entry timeout sets TIMED_OUT status and awards 0 confluence for unconfirmed indicators."""
    base_t = IST.localize(datetime(2026, 11, 23, 10, 5))
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"

    candidates = []
    engine = TechnicalAnalysisEngine(
        primary_index=spot_sym,
        primary_futures=fut_sym,
        on_technical_candidate=lambda c: candidates.append(c),
    )
    if engine.multi_timeframe_analyzer:
        engine.multi_timeframe_analyzer.min_confluence_score = 30.0

    # Mark futures active with a preceding candle
    engine.on_candle(
        Candle(
            symbol=fut_sym,
            timeframe="1m",
            timestamp=base_t - timedelta(minutes=1),
            open=25100.0,
            high=25110.0,
            low=25095.0,
            close=25105.0,
            volume=1000,
            is_closed=True,
        )
    )

    snap_15m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=20),
        timeframe="15m",
        open=25000.0,
        high=25120.0,
        low=24980.0,
        close=25110.0,
        volume=0,
        trend_bias="BULLISH",
    )
    engine.states[spot_sym].snapshot_history["15m"].append(snap_15m)

    snap_5m = TechnicalSnapshot(
        symbol=spot_sym,
        timestamp=base_t - timedelta(minutes=5),
        timeframe="5m",
        open=25080.0,
        high=25120.0,
        low=25075.0,
        close=25115.0,
        volume=0,
        atr=20.0,
        rvol=2.0,
    )
    setup = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        current_state=BreakoutState.SETUP_ELIGIBLE,
        start_time=base_t - timedelta(minutes=5),
        symbol=spot_sym,
        timeframe="5m",
        stop_loss=25080.0,
    )
    engine.multi_timeframe_analyzer.on_5m_setup(setup, snap_5m)

    # Spot 1m candle arrives at 10:05 (bar close 10:06)
    c_spot = Candle(
        symbol=spot_sym,
        timeframe="1m",
        timestamp=base_t,
        open=25110.0,
        high=25120.0,
        low=25105.0,
        close=25115.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(c_spot)
    assert spot_sym in engine._pending_1m_entry

    # 35 seconds after bar close (10:06:35), clock-driven expiry runs
    expiry_time = base_t + timedelta(minutes=1, seconds=35)
    engine.expire_pending_evaluations(as_of_time=expiry_time)

    # Invariant: Pending entry timed out and emitted with TIMED_OUT status
    assert spot_sym not in engine._pending_1m_entry
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.confirmation_status == "TIMED_OUT"
    assert cand.decision_at == expiry_time
