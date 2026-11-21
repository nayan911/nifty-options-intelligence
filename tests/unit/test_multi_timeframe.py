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
    assert cand.confluence_score >= 60.0


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

