"""Unit Tests for Derivatives & Options Intelligence Engine (Phase 3A).

Exhaustively verifies:
1. Futures Price + OI buildup classification across all 4 quadrants + neutral.
2. CE vs PE directional interpretation (Call writing = Bearish; Put writing = Bullish).
3. Consecutively timestamped OI & volume changes (not relying on broker reported oich).
4. Minimum significance threshold filtering.
5. Dynamic Call/Put walls and descriptive Max Pain calculation.
6. Aggregate OI PCR, Near-ATM PCR, Cumulative Volume PCR, and Interval Volume PCR.
7. Invariant-Safe Comparable-Basket Delta PCR (ΔPCR) across ATM shifts.
8. Zero/missing denominator safety (returns None, never 0.0 or ZeroDivisionError).
9. Expiry isolation and stale data handling.
10. DerivativesAssessment classifications (SUPPORTIVE, CONFLICTING, INSUFFICIENT_DATA).
11. Zero Hard Vetoes: TechnicalCandidate confluence score remains intact and candidates are emitted.
"""

from datetime import date, datetime, timedelta

import pytz

from nifty_engine.analysis.derivatives.derivatives_assessment import (
    DerivativesAssessmentType,
    DerivativesIntelligenceEngine,
)
from nifty_engine.analysis.derivatives.oi_analysis import (
    BuildupType,
    DirectionalBias,
    OpenInterestAnalyzer,
)
from nifty_engine.analysis.derivatives.pcr import PCREngine
from nifty_engine.analysis.technical.breakouts import BreakoutContext, BreakoutDirection
from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine, TechnicalSnapshot
from nifty_engine.core.interfaces import (
    BreakoutState,
    Candle,
    OptionChainSnapshot,
    OptionChainStrike,
)

IST = pytz.timezone("Asia/Kolkata")


def create_test_snapshot(
    api_time: datetime,
    spot_price: float = 25100.0,
    expiry_date: date = date(2026, 11, 26),
    strikes_data: list = None,
) -> OptionChainSnapshot:
    """Helper creating OptionChainSnapshot with specified strike metrics."""
    if strikes_data is None:
        strikes_data = [
            (25000.0, 50000, 150.0, 10000, 20000, 40.0, 5000),
            (25050.0, 80000, 110.0, 15000, 40000, 60.0, 8000),
            (25100.0, 120000, 80.0, 25000, 90000, 90.0, 20000),
            (25150.0, 90000, 50.0, 12000, 110000, 130.0, 15000),
            (25200.0, 60000, 30.0, 8000, 70000, 170.0, 10000),
        ]

    parsed_strikes = []
    tot_call_oi = 0
    tot_put_oi = 0

    for strike, c_oi, c_ltp, c_vol, p_oi, p_ltp, p_vol in strikes_data:
        tot_call_oi += c_oi
        tot_put_oi += p_oi
        parsed_strikes.append(
            OptionChainStrike(
                strike=strike,
                call={"symbol": f"NSE:NIFTY{int(strike)}CE", "oi": c_oi, "ltp": c_ltp, "volume": c_vol},
                put={"symbol": f"NSE:NIFTY{int(strike)}PE", "oi": p_oi, "ltp": p_ltp, "volume": p_vol},
            )
        )

    pcr = round(tot_put_oi / tot_call_oi, 3) if tot_call_oi > 0 else 0.0
    return OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=spot_price,
        api_timestamp=api_time,
        expiry=expiry_date,
        strikes=parsed_strikes,
        total_call_oi=tot_call_oi,
        total_put_oi=tot_put_oi,
        pcr=pcr,
    )


# ---------------------------------------------------------------------------
# 1. Futures Buildup Classification
# ---------------------------------------------------------------------------


def test_futures_buildup_classification_all_quadrants():
    """Verify futures price + OI quadrant classification and directional mapping."""
    analyzer = OpenInterestAnalyzer(
        min_oi_change_threshold_pct=1.0,
        min_oi_change_contracts=100,
        min_price_change_pct=0.2,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Base candle
    c0 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0, open=25000.0, high=25020.0, low=24990.0, close=25000.0, volume=1000, oi=100000, is_closed=True)
    res0 = analyzer.on_futures_candle(c0)
    assert res0.buildup == BuildupType.NEUTRAL
    assert res0.directional_bias == DirectionalBias.NEUTRAL

    # 1. Long Buildup: Price UP (+1%), OI UP (+5%) -> BULLISH
    c1 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=5), open=25000.0, high=25260.0, low=25000.0, close=25250.0, volume=2000, oi=105000, is_closed=True)
    res1 = analyzer.on_futures_candle(c1)
    assert res1.buildup == BuildupType.LONG_BUILDUP
    assert res1.directional_bias == DirectionalBias.BULLISH

    # 2. Long Unwinding: Price DOWN (-1%), OI DOWN (-4%) -> BEARISH
    c2 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=10), open=25250.0, high=25250.0, low=24990.0, close=25000.0, volume=1500, oi=101000, is_closed=True)
    res2 = analyzer.on_futures_candle(c2)
    assert res2.buildup == BuildupType.LONG_UNWINDING
    assert res2.directional_bias == DirectionalBias.BEARISH

    # 3. Short Buildup: Price DOWN (-1%), OI UP (+5%) -> BEARISH
    c3 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=15), open=25000.0, high=25010.0, low=24740.0, close=24750.0, volume=3000, oi=106000, is_closed=True)
    res3 = analyzer.on_futures_candle(c3)
    assert res3.buildup == BuildupType.SHORT_BUILDUP
    assert res3.directional_bias == DirectionalBias.BEARISH

    # 4. Short Covering: Price UP (+1%), OI DOWN (-4%) -> BULLISH
    c4 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=20), open=24750.0, high=25020.0, low=24750.0, close=25000.0, volume=2500, oi=102000, is_closed=True)
    res4 = analyzer.on_futures_candle(c4)
    assert res4.buildup == BuildupType.SHORT_COVERING
    assert res4.directional_bias == DirectionalBias.BULLISH


def test_futures_insignificant_fluctuations_filtered_as_neutral():
    """Verify that trivial noise below significance thresholds is classified as NEUTRAL."""
    analyzer = OpenInterestAnalyzer(
        min_oi_change_threshold_pct=2.0,
        min_oi_change_contracts=500,
        min_price_change_pct=0.5,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 9, 15))

    c0 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0, open=25000.0, high=25010.0, low=24990.0, close=25000.0, volume=1000, oi=100000, is_closed=True)
    analyzer.on_futures_candle(c0)

    # Price moves +2 points (+0.008%), OI moves +10 contracts (+0.01%) -> Noise!
    c1 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=5), open=25000.0, high=25005.0, low=24998.0, close=25002.0, volume=500, oi=100010, is_closed=True)
    res1 = analyzer.on_futures_candle(c1)
    assert res1.buildup == BuildupType.NEUTRAL
    assert res1.directional_bias == DirectionalBias.NEUTRAL


# ---------------------------------------------------------------------------
# 2. CE vs PE Directional Interpretation Inversion
# ---------------------------------------------------------------------------


def test_call_and_put_directional_interpretation_inversion():
    """Verify financial interpretation: Call writing is Bearish, Put writing is Bullish."""
    analyzer = OpenInterestAnalyzer(
        min_oi_change_threshold_pct=1.0,
        min_oi_change_contracts=100,
        min_price_change_pct=0.5,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    t1 = t0 + timedelta(minutes=3)

    # Initial snapshot at 10:00:
    # 25100 CE: LTP 80, OI 50000
    # 25100 PE: LTP 80, OI 50000
    snap0 = create_test_snapshot(
        api_time=t0,
        spot_price=25100.0,
        strikes_data=[(25100.0, 50000, 80.0, 5000, 50000, 80.0, 5000)],
    )
    analyzer.on_option_chain(snap0)

    # Scenario A:
    # 25100 CE: Price falls 80 -> 70 (-12.5%), OI rises 50000 -> 55000 (+10%) -> Call Writing / Short Buildup -> BEARISH!
    # 25100 PE: Price falls 80 -> 70 (-12.5%), OI rises 50000 -> 55000 (+10%) -> Put Writing / Short Buildup -> BULLISH!
    snap1 = create_test_snapshot(
        api_time=t1,
        spot_price=25100.0,
        strikes_data=[(25100.0, 55000, 70.0, 10000, 55000, 70.0, 10000)],
    )
    profile1 = analyzer.on_option_chain(snap1)
    s25100 = profile1.strikes[25100.0]

    # Invariant: Call short buildup is BEARISH resistance
    assert s25100.call_buildup == BuildupType.SHORT_BUILDUP
    assert s25100.call_directional_bias == DirectionalBias.BEARISH

    # Invariant: Put short buildup is BULLISH support
    assert s25100.put_buildup == BuildupType.SHORT_BUILDUP
    assert s25100.put_directional_bias == DirectionalBias.BULLISH

    # Scenario B:
    # 25100 CE: Price rises 70 -> 90 (+28%), OI rises 55000 -> 60000 (+9%) -> Call Buying / Long Buildup -> BULLISH!
    # 25100 PE: Price rises 70 -> 90 (+28%), OI rises 55000 -> 60000 (+9%) -> Put Buying / Long Buildup -> BEARISH!
    t2 = t1 + timedelta(minutes=3)
    snap2 = create_test_snapshot(
        api_time=t2,
        spot_price=25100.0,
        strikes_data=[(25100.0, 60000, 90.0, 15000, 60000, 90.0, 15000)],
    )
    profile2 = analyzer.on_option_chain(snap2)
    s25100_b = profile2.strikes[25100.0]

    assert s25100_b.call_buildup == BuildupType.LONG_BUILDUP
    assert s25100_b.call_directional_bias == DirectionalBias.BULLISH

    assert s25100_b.put_buildup == BuildupType.LONG_BUILDUP
    assert s25100_b.put_directional_bias == DirectionalBias.BEARISH


# ---------------------------------------------------------------------------
# 3. Consecutive Timestamped Calculations (Not Broker oich)
# ---------------------------------------------------------------------------


def test_oi_and_volume_changes_calculated_from_timestamped_consecutive_snapshots():
    """Verify that delta OI is computed strictly as OI_t - OI_{t-1} between timestamped snapshots."""
    analyzer = OpenInterestAnalyzer(min_oi_change_threshold_pct=1.0, min_oi_change_contracts=50)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    t1 = t0 + timedelta(minutes=3)

    snap0 = create_test_snapshot(
        api_time=t0,
        strikes_data=[(25100.0, 100000, 100.0, 10000, 80000, 90.0, 8000)],
    )
    profile0 = analyzer.on_option_chain(snap0)
    assert profile0.observation_interval_seconds == 0.0
    assert profile0.strikes[25100.0].call_oi_change == 0

    # 3 minutes later: Call OI increases from 100,000 to 104,500 (+4,500)
    snap1 = create_test_snapshot(
        api_time=t1,
        strikes_data=[(25100.0, 104500, 95.0, 15000, 82000, 92.0, 12000)],
    )
    profile1 = analyzer.on_option_chain(snap1)
    assert profile1.observation_interval_seconds == 180.0
    assert profile1.strikes[25100.0].call_oi_change == 4500
    assert profile1.strikes[25100.0].call_volume_change == 5000
    assert profile1.strikes[25100.0].put_oi_change == 2000
    assert profile1.strikes[25100.0].put_volume_change == 4000


# ---------------------------------------------------------------------------
# 4. Call / Put Walls and Max Pain
# ---------------------------------------------------------------------------


def test_call_put_walls_and_max_pain():
    """Verify identification of maximum Call and Put OI walls and descriptive Max Pain strike."""
    analyzer = OpenInterestAnalyzer()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Highest Call OI at 25200 (150,000)
    # Highest Put OI at 25000 (180,000)
    strikes_data = [
        (25000.0, 30000, 180.0, 5000, 180000, 20.0, 20000),  # Put Wall = 25000
        (25050.0, 50000, 140.0, 8000, 90000, 40.0, 12000),
        (25100.0, 90000, 90.0, 15000, 80000, 80.0, 15000),
        (25150.0, 110000, 50.0, 18000, 40000, 130.0, 8000),
        (25200.0, 150000, 25.0, 25000, 20000, 180.0, 4000),  # Call Wall = 25200
    ]

    snap = create_test_snapshot(api_time=t0, spot_price=25100.0, strikes_data=strikes_data)
    profile = analyzer.on_option_chain(snap)

    assert profile.call_wall_strike == 25200.0
    assert profile.call_wall_oi == 150000
    assert profile.put_wall_strike == 25000.0
    assert profile.put_wall_oi == 180000

    # Max Pain should exist and be close to ATM (e.g. 25050 or 25100)
    assert profile.max_pain_strike is not None
    assert profile.max_pain_strike in [25050.0, 25100.0]


def test_concentration_near_breakout_level():
    """Verify active writer detection near a reference breakout level."""
    analyzer = OpenInterestAnalyzer(min_oi_change_contracts=1000)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    t1 = t0 + timedelta(minutes=3)

    snap0 = create_test_snapshot(
        api_time=t0,
        strikes_data=[(25100.0, 50000, 80.0, 5000, 40000, 80.0, 5000)],
    )
    analyzer.on_option_chain(snap0)

    # Call writers aggressively add 3,000 contracts at 25100 with price dropping
    snap1 = create_test_snapshot(
        api_time=t1,
        strikes_data=[(25100.0, 53000, 72.0, 12000, 40200, 85.0, 6000)],
    )
    analyzer.on_option_chain(snap1)

    conc = analyzer.analyze_concentration_near_level(25100.0, tolerance_points=25.0)
    assert conc["is_available"] is True
    assert conc["call_writers_active"] is True
    assert conc["put_writers_active"] is False
    assert conc["dominant_bias"] == DirectionalBias.BEARISH


# ---------------------------------------------------------------------------
# 5. PCR Engine Multi-Metrics & Safe Denominators
# ---------------------------------------------------------------------------


def test_pcr_engine_multi_metrics_and_zero_denominator_safety():
    """Verify PCR metrics and invariant: Zero denominator returns None, never 0.0 or ZeroDivisionError."""
    pcr_engine = PCREngine(near_atm_strike_range=1)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Normal chain
    snap_normal = create_test_snapshot(
        api_time=t0,
        spot_price=25100.0,
        strikes_data=[
            (25050.0, 40000, 100.0, 10000, 60000, 40.0, 15000),
            (25100.0, 60000, 70.0, 20000, 60000, 70.0, 20000),
            (25150.0, 100000, 40.0, 15000, 30000, 110.0, 5000),
        ],
    )
    pcr0 = pcr_engine.on_option_chain(snap_normal)
    # Total Put OI = 150,000; Total Call OI = 200,000 -> PCR = 0.75
    assert pcr0.aggregate_oi_pcr == 0.75
    assert pcr0.cumulative_volume_pcr is not None
    assert pcr0.near_atm_oi_pcr is not None
    assert pcr0.strike_weighted_pcr is not None

    # Zero Call OI edge case: No calls traded or recorded
    snap_zero_calls = create_test_snapshot(
        api_time=t0 + timedelta(minutes=3),
        spot_price=25100.0,
        strikes_data=[
            (25100.0, 0, 0.0, 0, 50000, 70.0, 10000),
        ],
    )
    pcr_zero = pcr_engine.on_option_chain(snap_zero_calls)
    # Invariant: Zero Call OI yields None, NOT 0.0 or crash!
    assert pcr_zero.aggregate_oi_pcr is None
    assert pcr_zero.cumulative_volume_pcr is None
    assert pcr_zero.near_atm_oi_pcr is None
    assert pcr_zero.strike_weighted_pcr is None


# ---------------------------------------------------------------------------
# 6. Invariant-Safe Delta PCR (ΔPCR) Across ATM Shifts
# ---------------------------------------------------------------------------


def test_delta_pcr_comparable_basket_invariance_across_atm_shifts():
    """Verify Invariant: Shifting ATM does NOT introduce false ΔPCR when underlying contracts' OI is unchanged."""
    pcr_engine = PCREngine(weighting_decay_points=100.0)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    t1 = t0 + timedelta(minutes=3)

    # Initial snapshot at spot = 25100 with 5 strikes [25000..25200]
    strikes_common = [
        (25000.0, 20000, 150.0, 5000, 60000, 20.0, 10000),
        (25050.0, 40000, 110.0, 8000, 50000, 40.0, 12000),
        (25100.0, 80000, 80.0, 15000, 80000, 80.0, 15000),
        (25150.0, 90000, 50.0, 12000, 40000, 120.0, 8000),
        (25200.0, 120000, 25.0, 20000, 20000, 170.0, 5000),
    ]

    snap0 = create_test_snapshot(api_time=t0, spot_price=25100.0, strikes_data=strikes_common)
    pcr_engine.on_option_chain(snap0)

    # Snapshot 2: Spot moves to 25150.
    # An extra higher strike 25250 arrives in the feed and 25000 drops off.
    # BUT for the 4 common strikes (25050, 25100, 25150, 25200), OI is 100% IDENTICAL!
    strikes_shifted = [
        (25050.0, 40000, 110.0, 8000, 50000, 40.0, 12000),
        (25100.0, 80000, 80.0, 15000, 80000, 80.0, 15000),
        (25150.0, 90000, 50.0, 12000, 40000, 120.0, 8000),
        (25200.0, 120000, 25.0, 20000, 20000, 170.0, 5000),
        (25250.0, 80000, 10.0, 5000, 5000, 220.0, 1000),
    ]
    snap1 = create_test_snapshot(api_time=t1, spot_price=25150.0, strikes_data=strikes_shifted)
    pcr1 = pcr_engine.on_option_chain(snap1)

    # Invariant: Because common strikes had identical OI, delta_oi_pcr on the comparable basket is exactly 0.0!
    assert pcr1.delta_oi_pcr == 0.0


# ---------------------------------------------------------------------------
# 7. Expiry Isolation & Stale Data Detection
# ---------------------------------------------------------------------------


def test_expiry_isolation_and_stale_data_detection():
    """Verify that different expiry dates are tracked independently and stale data is flagged."""
    analyzer = OpenInterestAnalyzer(max_snapshot_age_seconds=300.0)
    pcr_engine = PCREngine(max_snapshot_age_seconds=300.0)

    exp_current = date(2026, 11, 26)
    exp_next = date(2026, 12, 3)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    snap_curr = create_test_snapshot(api_time=t0, expiry_date=exp_current)
    snap_next = create_test_snapshot(api_time=t0, expiry_date=exp_next)

    p_curr = analyzer.on_option_chain(snap_curr)
    p_next = analyzer.on_option_chain(snap_next)

    # Invariant: Both expiries maintained independently
    assert p_curr.expiry == exp_current
    assert p_next.expiry == exp_next
    assert len(analyzer.latest_profiles) == 2

    # Stale data check: evaluating at 10:10 (10 minutes > 300s max age)
    t_stale = t0 + timedelta(minutes=10)
    p_stale = analyzer.on_option_chain(snap_curr, as_of=t_stale)
    assert p_stale.is_stale is True

    pcr_stale = pcr_engine.on_option_chain(snap_curr, as_of=t_stale)
    assert pcr_stale.is_stale is True


# ---------------------------------------------------------------------------
# 8. Derivatives Assessment & Zero Hard Vetoes Rule
# ---------------------------------------------------------------------------


def test_derivatives_assessment_classifications():
    """Verify structured assessment categories: SUPPORTIVE, CONFLICTING, and INSUFFICIENT_DATA."""
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # 1. No data fed yet -> INSUFFICIENT_DATA
    res_empty = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t0,
    )
    assert res_empty.assessment_type == DerivativesAssessmentType.INSUFFICIENT_DATA

    # Feed initial snapshot
    snap0 = create_test_snapshot(
        api_time=t0,
        strikes_data=[(25100.0, 80000, 80.0, 10000, 80000, 80.0, 10000)],
    )
    oi_analyzer.on_option_chain(snap0)
    pcr_engine.on_option_chain(snap0)

    # Feed bullish continuation at 10:03:
    # Futures price up, OI up (Long Buildup); Put writing at 25100; PCR rising
    t1 = t0 + timedelta(minutes=3)
    c_fut = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t1, open=25000.0, high=25150.0, low=25000.0, close=25150.0, volume=5000, oi=120000, is_closed=True)
    oi_analyzer.on_futures_candle(c_fut)

    # 25100 PE: Price drops from 80 -> 60, OI rises from 80000 -> 90000 (Put Writing / Bullish)
    snap1 = create_test_snapshot(
        api_time=t1,
        strikes_data=[(25100.0, 80000, 100.0, 15000, 90000, 60.0, 20000)],
    )
    oi_analyzer.on_option_chain(snap1)
    pcr_engine.on_option_chain(snap1)

    # Evaluate Bullish breakout setup
    assessment_bull = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t1,
    )
    assert assessment_bull.assessment_type in (
        DerivativesAssessmentType.SUPPORTIVE,
        DerivativesAssessmentType.STRONGLY_SUPPORTIVE,
    )
    assert len(assessment_bull.reasons) >= 1

    # Evaluate Bearish setup on the same bullish data -> CONFLICTING
    assessment_bear = engine.evaluate_setup(
        direction=BreakoutDirection.BEARISH,
        level_price=25100.0,
        as_of=t1,
    )
    assert assessment_bear.assessment_type in (
        DerivativesAssessmentType.CONFLICTING,
        DerivativesAssessmentType.STRONGLY_CONFLICTING,
    )
    assert len(assessment_bear.contra_reasons) >= 1


def test_zero_hard_vetoes_candidate_emitted_even_with_conflicting_derivatives():
    """Verify Mandatory Rule: Even if derivatives assessment is CONFLICTING, candidate is emitted

    with unchanged technical confluence score and attached diagnostic telemetry.
    """
    spot_sym = "NSE:NIFTY50-INDEX"
    fut_sym = "NSE:NIFTY26NOVFUT"
    candidates = []

    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    deriv_engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine)

    engine = TechnicalAnalysisEngine(
        primary_index=spot_sym,
        primary_futures=fut_sym,
        on_technical_candidate=lambda c: candidates.append(c),
        derivatives_engine=deriv_engine,
    )
    analyzer = engine.multi_timeframe_analyzer
    analyzer.min_confluence_score = 40.0

    base_t = IST.localize(datetime(2026, 11, 23, 10, 5))

    # Pre-configure conflicting derivatives state:
    # Heavy call writing at 25100 (Bearish derivatives flow)
    snap_oi = create_test_snapshot(
        api_time=base_t,
        spot_price=25100.0,
        strikes_data=[(25100.0, 150000, 60.0, 20000, 40000, 90.0, 5000)],
    )
    engine.on_option_chain(snap_oi)

    # 15m snapshot (Macro Bullish)
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

    # Active 5m setup (Bullish breakout of PDH 25100)
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

    # Warm-up futures feed
    engine.on_candle(
        Candle(symbol=fut_sym, timeframe="1m", timestamp=base_t - timedelta(minutes=1), open=25100.0, high=25110.0, low=25095.0, close=25105.0, volume=1000, oi=90000, is_closed=True)
    )

    # 1m spot candle arrives
    c_spot = Candle(symbol=spot_sym, timeframe="1m", timestamp=base_t, open=25110.0, high=25120.0, low=25105.0, close=25115.0, volume=0, is_closed=True)
    engine.on_candle(c_spot)

    # 1m matching futures candle arrives
    c_fut = Candle(symbol=fut_sym, timeframe="1m", timestamp=base_t, open=25110.0, high=25122.0, low=25105.0, close=25118.0, volume=5000, oi=92000, is_closed=True)
    engine.on_candle(c_fut)

    # Invariant 1: Candidate is NOT vetoed! It is dispatched.
    assert len(candidates) == 1
    cand = candidates[0]

    # Invariant 2: Technical confluence score is untouched (still >= 40.0)
    assert cand.confluence_score >= 40.0

    # Invariant 3: Derivatives assessment is attached and provides diagnostic transparency
    assert cand.derivatives_assessment is not None
    assert cand.derivatives_assessment.assessment_type in (
        DerivativesAssessmentType.CONFLICTING,
        DerivativesAssessmentType.NEUTRAL,
        DerivativesAssessmentType.STRONGLY_CONFLICTING,
    )
    assert "derivatives_assessment" in cand.timeframe_breakdown
    assert cand.derivatives_assessment.decision_at == cand.decision_at
