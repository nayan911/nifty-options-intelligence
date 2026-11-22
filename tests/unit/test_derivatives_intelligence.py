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
from nifty_engine.data.fyers.rest_client import FyersRestClient

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
    pcr_engine.on_option_chain(snap_curr)
    pcr_engine.on_option_chain(snap_next)

    # Invariant: Both expiries maintained independently
    assert p_curr.expiry == exp_current
    assert p_next.expiry == exp_next
    assert len(analyzer.latest_profiles) == 2

    # Stale data check: evaluating at 10:10 (10 minutes > 300s max age)
    t_stale = t0 + timedelta(minutes=10)
    p_stale = analyzer.get_profile(as_of=t_stale)
    assert p_stale is not None
    assert p_stale.is_stale is True

    pcr_stale = pcr_engine.get_pcr(as_of=t_stale)
    assert pcr_stale is not None
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
    # Snapshot 0 at 10:02: Call OI = 120,000, Call LTP = 80.0
    snap_oi_0 = create_test_snapshot(
        api_time=base_t - timedelta(minutes=3),
        spot_price=25100.0,
        strikes_data=[(25100.0, 120000, 80.0, 10000, 40000, 90.0, 5000)],
    )
    engine.on_option_chain(snap_oi_0)

    # Snapshot 1 at 10:05: Heavy Call writing (120k -> 150k (+30k), LTP falls 80 -> 60)
    snap_oi_1 = create_test_snapshot(
        api_time=base_t,
        spot_price=25100.0,
        strikes_data=[(25100.0, 150000, 60.0, 20000, 40000, 90.0, 5000)],
    )
    engine.on_option_chain(snap_oi_1)

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



def test_futures_symbol_and_timeframe_isolation():
    """Verify P0: Futures OI tracking strictly isolates symbols and timeframes."""
    analyzer = OpenInterestAnalyzer(
        min_fut_oi_change_threshold_pct=0.1,
        min_fut_oi_change_contracts=100,
        min_fut_price_change_points=2.0,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 9, 15))

    # NIFTY 1m
    c_nifty_1m_0 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="1m", timestamp=t0, open=25000.0, high=25010.0, low=24995.0, close=25005.0, volume=1000, oi=100000, is_closed=True)
    analyzer.on_futures_candle(c_nifty_1m_0)

    # BANKNIFTY 1m arriving at 9:15 should NOT contaminate NIFTY history
    c_bank_1m_0 = Candle(symbol="NSE:BANKNIFTY26NOVFUT", timeframe="1m", timestamp=t0, open=52000.0, high=52050.0, low=51980.0, close=52020.0, volume=2000, oi=200000, is_closed=True)
    analyzer.on_futures_candle(c_bank_1m_0)

    # NIFTY 5m candle arriving should NOT compare against 1m candle
    c_nifty_5m_0 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0, open=25000.0, high=25040.0, low=24990.0, close=25030.0, volume=5000, oi=100000, is_closed=True)
    analyzer.on_futures_candle(c_nifty_5m_0)

    # NIFTY 1m second candle at 9:16: price rises 25005 -> 25020 (+15 pts), OI rises 100000 -> 101000 (+1000)
    c_nifty_1m_1 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="1m", timestamp=t0 + timedelta(minutes=1), open=25005.0, high=25025.0, low=25000.0, close=25020.0, volume=1200, oi=101000, is_closed=True)
    res_nifty_1m = analyzer.on_futures_candle(c_nifty_1m_1)

    # Invariant: NIFTY 1m was compared against NIFTY 1m (25005), NOT BANKNIFTY (52020) or NIFTY 5m!
    assert res_nifty_1m.price_change == 15.0
    assert res_nifty_1m.oi_change == 1000
    assert res_nifty_1m.buildup == BuildupType.LONG_BUILDUP
    assert res_nifty_1m.directional_bias == DirectionalBias.BULLISH

    # Invariant: Distinct histories maintained
    assert ("NSE:NIFTY26NOVFUT", "1m") in analyzer._futures_history
    assert ("NSE:BANKNIFTY26NOVFUT", "1m") in analyzer._futures_history
    assert ("NSE:NIFTY26NOVFUT", "5m") in analyzer._futures_history


def test_buildup_requires_both_price_and_oi_significance():
    """Verify P1: Significant price change with zero/flat OI change classifies as NEUTRAL, not short covering/unwinding."""
    analyzer = OpenInterestAnalyzer(
        min_fut_oi_change_threshold_pct=0.5,
        min_fut_oi_change_contracts=500,
        min_fut_price_change_points=10.0,
        min_fut_price_change_pct=0.05,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 9, 15))

    c0 = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0, open=25000.0, high=25020.0, low=24990.0, close=25000.0, volume=1000, oi=100000, is_closed=True)
    analyzer.on_futures_candle(c0)

    # Case 1: Price rallies +50 pts (huge movement), but OI change is 0
    c_flat_oi = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=5), open=25000.0, high=25060.0, low=25000.0, close=25050.0, volume=2000, oi=100000, is_closed=True)
    res_flat_oi = analyzer.on_futures_candle(c_flat_oi)
    # Invariant: Must NOT classify as SHORT_COVERING or LONG_BUILDUP because OI is completely flat!
    assert res_flat_oi.buildup == BuildupType.NEUTRAL
    assert res_flat_oi.directional_bias == DirectionalBias.NEUTRAL

    # Case 2: Price is completely flat (0 pts), but OI increases by +5,000 contracts
    c_flat_price = Candle(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=t0 + timedelta(minutes=10), open=25050.0, high=25052.0, low=25048.0, close=25050.0, volume=2000, oi=105000, is_closed=True)
    res_flat_price = analyzer.on_futures_candle(c_flat_price)
    # Invariant: Must NOT classify as SHORT_BUILDUP or LONG_BUILDUP because Price is completely flat!
    assert res_flat_price.buildup == BuildupType.NEUTRAL
    assert res_flat_price.directional_bias == DirectionalBias.NEUTRAL


def test_derivatives_assessment_dynamic_freshness_recalculation():
    """Verify P0: Derivatives assessment dynamically recomputes freshness relative to decision_at."""
    oi_analyzer = OpenInterestAnalyzer(max_snapshot_age_seconds=360.0)  # 6 minutes max age
    pcr_engine = PCREngine(max_snapshot_age_seconds=360.0)
    engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine)

    t_chain = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Option chain arrives at 10:00 (is_stale was False at receipt time)
    snap = create_test_snapshot(api_time=t_chain)
    oi_analyzer.on_option_chain(snap)
    pcr_engine.on_option_chain(snap)

    # 1. Candidate evaluated at 10:03 (3 minutes elapsed <= 6m) -> Valid / Sufficient data
    t_decision_fresh = t_chain + timedelta(minutes=3)
    assess_fresh = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t_decision_fresh,
    )
    assert assess_fresh.is_stale is False
    assert assess_fresh.assessment_type != DerivativesAssessmentType.INSUFFICIENT_DATA

    # 2. Candidate evaluated at 10:25 (25 minutes elapsed > 6m) -> Must be dynamically marked INSUFFICIENT_DATA!
    t_decision_stale = t_chain + timedelta(minutes=25)
    assess_stale = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t_decision_stale,
    )
    # Invariant: Dynamically evaluated as stale relative to decision timestamp
    assert assess_stale.is_stale is True
    assert assess_stale.assessment_type == DerivativesAssessmentType.INSUFFICIENT_DATA
    assert any("stale" in r.lower() for r in assess_stale.contra_reasons)


def test_derivatives_assessment_retrieves_latest_futures_at_decision_at():
    """Verify P1: Futures positioning is retrieved dynamically as of decision_at, not stuck at last option chain poll."""
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    fut_sym = "NSE:NIFTY26NOVFUT"
    engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine, primary_futures=fut_sym)

    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Option chain polled at 10:00
    snap = create_test_snapshot(api_time=t0)
    oi_analyzer.on_option_chain(snap)
    pcr_engine.on_option_chain(snap)

    # Initial futures candle covering 09:55-10:00 (closes at 10:00)
    c0 = Candle(symbol=fut_sym, timeframe="5m", timestamp=t0 - timedelta(minutes=5), open=25000.0, high=25020.0, low=24990.0, close=25000.0, volume=1000, oi=100000, is_closed=True)
    oi_analyzer.on_futures_candle(c0)

    # At 10:05 (before next option chain poll at 10:06), a new 10:00-10:05 futures candle arrives showing Long Buildup
    t1 = t0 + timedelta(minutes=5)
    c1 = Candle(symbol=fut_sym, timeframe="5m", timestamp=t0, open=25000.0, high=25060.0, low=25000.0, close=25050.0, volume=3000, oi=105000, is_closed=True, observed_at=t1)
    oi_analyzer.on_futures_candle(c1)

    # Invariant P0: At 10:03 (mid-candle), 10:00-10:05 candle is NOT available (zero lookahead)
    assessment_1003 = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t0 + timedelta(minutes=3),
    )
    assert assessment_1003.futures_analysis is not None
    assert assessment_1003.futures_analysis.timestamp == t0 - timedelta(minutes=5)
    assert assessment_1003.futures_bias == DirectionalBias.NEUTRAL

    # Evaluate setup at 10:05 when 10:00-10:05 candle has closed and become available
    assessment = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t1,
    )
    # Invariant: Retrieved futures positioning from 10:05 completed candle (Long Buildup), NOT stale 10:00!
    assert assessment.futures_bias == DirectionalBias.BULLISH
    assert assessment.futures_analysis is not None
    assert assessment.futures_analysis.timestamp == t0
    assert assessment.futures_analysis.bar_close_time == t1
    assert assessment.futures_analysis.buildup == BuildupType.LONG_BUILDUP


def test_missing_strike_data_handling_not_treated_as_zero():
    """Verify P1: Missing strikes or disappeared contracts do not fabricate artificial delta drops to 0."""
    pcr_engine = PCREngine()
    oi_analyzer = OpenInterestAnalyzer()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    t1 = t0 + timedelta(minutes=3)

    # Snapshot 0 has 25000, 25050, 25100
    snap0 = create_test_snapshot(
        api_time=t0,
        strikes_data=[
            (25000.0, 50000, 150.0, 5000, 20000, 30.0, 2000),
            (25050.0, 60000, 110.0, 6000, 40000, 50.0, 4000),
            (25100.0, 80000, 80.0, 10000, 80000, 80.0, 10000),
        ],
    )
    oi_analyzer.on_option_chain(snap0)
    pcr_engine.on_option_chain(snap0)

    # Snapshot 1: Strike 25000 drops out of broker response (missing), but 25050 and 25100 have identical OI
    snap1 = create_test_snapshot(
        api_time=t1,
        strikes_data=[
            (25050.0, 60000, 110.0, 6000, 40000, 50.0, 4000),
            (25100.0, 80000, 80.0, 10000, 80000, 80.0, 10000),
        ],
    )
    prof1 = oi_analyzer.on_option_chain(snap1)
    pcr1 = pcr_engine.on_option_chain(snap1)

    # Invariant: 25000 is simply absent in prof1, not reported with delta of -50,000!
    assert 25000.0 not in prof1.strikes
    assert prof1.strikes[25050.0].call_oi_change == 0
    assert prof1.strikes[25100.0].call_oi_change == 0

    # Invariant: ΔPCR evaluated only on common strikes (25050, 25100), yielding delta_oi_pcr == 0.0
    assert pcr1.delta_oi_pcr == 0.0


# ---------------------------------------------------------------------------
# Dedicated Regression Tests for Phase 3A Fixes
# ---------------------------------------------------------------------------


def test_p0_futures_oi_causal_timing_and_availability():
    """Verify P0: Retrospective query mid-candle (e.g. 10:03) cannot see 10:00-10:05 candle,

    and late-arriving futures candle respects observed_at availability timestamp.
    """
    analyzer = OpenInterestAnalyzer()
    fut_sym = "NSE:NIFTY26NOVFUT"
    t_open = IST.localize(datetime(2026, 11, 23, 10, 0))
    t_close = t_open + timedelta(minutes=5)
    t_mid = t_open + timedelta(minutes=3)

    # 1. Candle opens at 10:00, scheduled close is 10:05
    c0 = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_open,
        open=25000.0,
        high=25050.0,
        low=24990.0,
        close=25040.0,
        volume=2000,
        oi=100000,
        is_closed=True,
    )
    analyzer.on_futures_candle(c0)

    # Mid-candle query at 10:03:00 MUST NOT see the 10:00-10:05 candle!
    obs_1003 = analyzer.get_futures_analysis(symbol=fut_sym, timeframe="5m", as_of=t_mid)
    assert obs_1003 is None

    # Query at scheduled close 10:05:00: Candle is closed and available
    obs_1005 = analyzer.get_futures_analysis(symbol=fut_sym, timeframe="5m", as_of=t_close)
    assert obs_1005 is not None
    assert obs_1005.timestamp == t_open
    assert obs_1005.bar_close_time == t_close
    assert obs_1005.available_at == t_close

    # 2. Delayed candle arrival: Bar covers 10:05-10:10, but observed_at is 10:11:15
    t_open_2 = t_close
    t_close_2 = t_open_2 + timedelta(minutes=5)
    t_observed_2 = t_close_2 + timedelta(seconds=75)  # 10:11:15

    c1 = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_open_2,
        open=25040.0,
        high=25100.0,
        low=25030.0,
        close=25090.0,
        volume=3000,
        oi=108000,
        is_closed=True,
        observed_at=t_observed_2,
    )
    analyzer.on_futures_candle(c1)

    # At 10:10:30, scheduled bar closed but observation had not arrived yet
    obs_1010_30 = analyzer.get_futures_analysis(symbol=fut_sym, timeframe="5m", as_of=t_close_2 + timedelta(seconds=30))
    # Must only return c0, NOT c1!
    assert obs_1010_30 is not None
    assert obs_1010_30.timestamp == t_open

    # At 10:11:20 (after observed_at), c1 is available
    obs_1011_20 = analyzer.get_futures_analysis(symbol=fut_sym, timeframe="5m", as_of=t_observed_2 + timedelta(seconds=5))
    assert obs_1011_20 is not None
    assert obs_1011_20.timestamp == t_open_2
    assert obs_1011_20.available_at == t_observed_2


def test_p0_underlying_and_futures_symbol_routing():
    """Verify P0: Derivatives assessments route correctly per underlying and synchronize active futures."""
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    deriv_engine = DerivativesIntelligenceEngine(
        oi_analyzer=oi_analyzer,
        pcr_engine=pcr_engine,
        primary_futures="NSE:NIFTY26NOVFUT",
    )

    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Ingest NIFTY option chain
    snap_nifty = create_test_snapshot(api_time=t0)
    oi_analyzer.on_option_chain(snap_nifty)
    pcr_engine.on_option_chain(snap_nifty)

    # 1. BANKNIFTY has NO option chain ingested -> Returns INSUFFICIENT_DATA
    assessment_bank = deriv_engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=52000.0,
        as_of=t0,
        underlying="BANKNIFTY",
        futures_symbol="NSE:BANKNIFTY26NOVFUT",
    )
    assert assessment_bank.assessment_type == DerivativesAssessmentType.INSUFFICIENT_DATA

    # 2. Ingest BANKNIFTY option chain
    snap_bank = OptionChainSnapshot(
        underlying="BANKNIFTY",
        spot_price=52000.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=52000.0,
                call={"symbol": "NSE:BANKNIFTY52000CE", "oi": 50000, "ltp": 250.0, "volume": 10000},
                put={"symbol": "NSE:BANKNIFTY52000PE", "oi": 50000, "ltp": 250.0, "volume": 10000},
            )
        ],
        total_call_oi=50000,
        total_put_oi=50000,
        pcr=1.0,
    )
    oi_analyzer.on_option_chain(snap_bank)
    pcr_engine.on_option_chain(snap_bank)

    assessment_bank_2 = deriv_engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=52000.0,
        as_of=t0,
        underlying="BANKNIFTY",
        futures_symbol="NSE:BANKNIFTY26NOVFUT",
    )
    assert assessment_bank_2.oi_profile is not None
    assert assessment_bank_2.oi_profile.underlying == "BANKNIFTY"

    # 3. Verify set_futures_ticker synchronization
    tech_engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        derivatives_engine=deriv_engine,
    )
    tech_engine.set_futures_ticker("NSE:NIFTY26DECFUT")
    assert deriv_engine.primary_futures == "NSE:NIFTY26DECFUT"


def test_p1_session_rollover_resets_baselines_without_overnight_jumps():
    """Verify P1: First observation of a new day establishes clean baseline without overnight jumps."""
    oi_analyzer = OpenInterestAnalyzer(
        min_oi_change_threshold_pct=1.0,
        min_oi_change_contracts=100,
        min_price_change_pct=0.2,
    )
    pcr_engine = PCREngine()
    fut_sym = "NSE:NIFTY26NOVFUT"

    # Day 1: 15:25 candle (closing at 15:30)
    t_d1 = IST.localize(datetime(2026, 11, 23, 15, 25))
    c_d1 = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_d1,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25000.0,
        volume=2000,
        oi=100000,
        is_closed=True,
    )
    oi_analyzer.on_futures_candle(c_d1)

    snap_d1 = create_test_snapshot(api_time=t_d1)
    oi_analyzer.on_option_chain(snap_d1)
    pcr_engine.on_option_chain(snap_d1)

    # Day 2: 09:15 candle (gap up +300 points, OI +20,000)
    t_d2 = IST.localize(datetime(2026, 11, 24, 9, 15))
    c_d2 = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_d2,
        open=25300.0,
        high=25350.0,
        low=25280.0,
        close=25320.0,
        volume=5000,
        oi=120000,
        is_closed=True,
    )
    res_d2 = oi_analyzer.on_futures_candle(c_d2)

    # Invariant: Day 2 first candle must be a clean baseline (zero delta, NEUTRAL buildup)
    assert res_d2.price_change == 0.0
    assert res_d2.oi_change == 0
    assert res_d2.buildup == BuildupType.NEUTRAL
    assert res_d2.directional_bias == DirectionalBias.NEUTRAL

    # Day 2 option chain snapshot at 09:16
    snap_d2 = create_test_snapshot(
        api_time=t_d2 + timedelta(minutes=1),
        spot_price=25320.0,
        strikes_data=[(25100.0, 150000, 150.0, 30000, 30000, 40.0, 10000)],
    )
    prof_d2 = oi_analyzer.on_option_chain(snap_d2)
    pcr_d2 = pcr_engine.on_option_chain(snap_d2)

    # Invariant: Day 2 first option chain snapshot must have zero delta and no overnight interval metrics
    assert prof_d2.strikes[25100.0].call_oi_change == 0
    assert pcr_d2.interval_volume_pcr is None
    assert pcr_d2.delta_oi_pcr is None


def test_p1_missing_unreported_fields_and_data_quality_score():
    """Verify P1: Missing/unreported fields remain None, do not fabricate delta spikes, and factor into quality score."""
    # 1. Test FyersRestClient._extract_contract_fields with missing keys
    raw_partial = {
        "symbol": "NSE:NIFTY26NOV25100CE",
        "bid": 80.5,
        "ask": 81.0,
    }
    extracted = FyersRestClient._extract_contract_fields(raw_partial)
    assert extracted["ltp"] is None
    assert extracted["oi"] is None
    assert extracted["volume"] is None
    assert extracted["oi_change"] is None
    assert extracted["iv"] is None

    # 2. Ingest option chain where a strike has None fields
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine)

    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    snap0 = create_test_snapshot(api_time=t0)
    oi_analyzer.on_option_chain(snap0)

    # Next snapshot has strike 25100 with None for OI
    snap1 = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0 + timedelta(minutes=3),
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={"symbol": "NSE:NIFTY25100CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25100PE", "oi": None, "ltp": None, "volume": None},
            )
        ],
        total_call_oi=0,
        total_put_oi=0,
        pcr=0.0,
    )
    prof1 = oi_analyzer.on_option_chain(snap1)
    # Invariant: Does not compute false -120,000 delta drop
    assert prof1.strikes[25100.0].call_oi_change == 0
    assert prof1.strikes[25100.0].call_buildup == BuildupType.NEUTRAL

    # 3. Data quality score drops when near-ATM strikes lack valid observations
    pcr_engine.on_option_chain(snap1)
    assessment = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t0 + timedelta(minutes=3),
    )
    # Because valid_strikes_count is 0, strike_coverage is 0.0, lowering data_quality_score
    assert assessment.data_quality_score <= 0.6


def test_p1_expired_option_chain_not_selected_returns_insufficient_data():
    """Verify P1: Expired option chains are never selected as fallback; returns None and INSUFFICIENT_DATA."""
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    engine = DerivativesIntelligenceEngine(oi_analyzer=oi_analyzer, pcr_engine=pcr_engine)

    # Ingest option chain expiring 2026-11-20
    t_past = IST.localize(datetime(2026, 11, 20, 15, 0))
    snap_past = create_test_snapshot(
        api_time=t_past,
        expiry_date=date(2026, 11, 20),
    )
    oi_analyzer.on_option_chain(snap_past)
    pcr_engine.on_option_chain(snap_past)

    # Current query is on 2026-11-23 (after 2026-11-20 expiry)
    t_now = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Invariant: get_profile must return None, NOT fall back to expired 2026-11-20 contracts!
    prof = oi_analyzer.get_profile(underlying="NIFTY", as_of=t_now)
    assert prof is None

    pcr_snap = pcr_engine.get_pcr(underlying="NIFTY", as_of=t_now)
    assert pcr_snap is None

    # Invariant: evaluate_setup must return INSUFFICIENT_DATA
    assessment = engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t_now,
    )
    assert assessment.assessment_type == DerivativesAssessmentType.INSUFFICIENT_DATA
    assert assessment.is_stale is True


def test_p1_delayed_older_futures_candle_rejected_and_cannot_corrupt_history():
    """Verify P1: Delayed older futures candles arriving with later observation timestamps

    are strictly rejected by opening timestamp and cannot corrupt chronological OI history.
    """
    analyzer = OpenInterestAnalyzer()
    fut_sym = "NSE:NIFTY26NOVFUT"

    # Candle 1: interval 10:05-10:10 arrives first at 10:10:01
    t_1005 = IST.localize(datetime(2026, 11, 23, 10, 5))
    c_newer = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_1005,
        open=25000.0,
        high=25060.0,
        low=25000.0,
        close=25050.0,
        volume=3000,
        oi=105000,
        is_closed=True,
        observed_at=t_1005 + timedelta(minutes=5, seconds=1),  # 10:10:01
    )
    res_newer = analyzer.on_futures_candle(c_newer)
    assert res_newer.timestamp == t_1005
    assert res_newer.price == 25050.0
    assert res_newer.oi == 105000

    # Candle 0: delayed interval 10:00-10:05 arrives LATER at 10:11:00
    t_1000 = IST.localize(datetime(2026, 11, 23, 10, 0))
    c_older_delayed = Candle(
        symbol=fut_sym,
        timeframe="5m",
        timestamp=t_1000,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25000.0,
        volume=1000,
        oi=100000,
        is_closed=True,
        observed_at=t_1005 + timedelta(minutes=6),  # 10:11:00 (later than 10:10:01!)
    )
    res_rejected = analyzer.on_futures_candle(c_older_delayed)

    # Invariant: Older candle is rejected; history head remains the 10:05 candle
    assert res_rejected.timestamp == t_1005
    assert res_rejected.price == 25050.0
    assert res_newer.oi == 105000
    history = analyzer._futures_history[(fut_sym, "5m")]
    assert len(history) == 1
    assert history[-1].timestamp == t_1005


def test_p1_partial_option_chain_missing_ce_does_not_distort_pcr_or_returns_unavailable():
    """Verify P1: Missing CE contracts do not fabricate artificial bullish PCR;

    PCR is evaluated strictly over valid comparable CE/PE observations and returns None when coverage < threshold.
    """
    pcr_engine = PCREngine(min_strike_coverage=0.5, min_valid_strikes=2)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # Scenario A: 5 strikes, but 4 have missing CE -> coverage = 1/5 = 20% < 50%
    snap_low_coverage = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25000.0,
                call={"symbol": "NSE:NIFTY25000CE", "oi": 50000, "ltp": 150.0, "volume": 5000},
                put={"symbol": "NSE:NIFTY25000PE", "oi": 50000, "ltp": 40.0, "volume": 5000},
            ),
            OptionChainStrike(
                strike=25050.0,
                call={"symbol": "NSE:NIFTY25050CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25050PE", "oi": 100000, "ltp": 60.0, "volume": 10000},
            ),
            OptionChainStrike(
                strike=25100.0,
                call={"symbol": "NSE:NIFTY25100CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25100PE", "oi": 100000, "ltp": 80.0, "volume": 10000},
            ),
            OptionChainStrike(
                strike=25150.0,
                call={"symbol": "NSE:NIFTY25150CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25150PE", "oi": 100000, "ltp": 110.0, "volume": 10000},
            ),
            OptionChainStrike(
                strike=25200.0,
                call={"symbol": "NSE:NIFTY25200CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25200PE", "oi": 100000, "ltp": 150.0, "volume": 10000},
            ),
        ],
        total_call_oi=50000,
        total_put_oi=450000,
        pcr=9.0,
    )
    pcr_low = pcr_engine.on_option_chain(snap_low_coverage)

    # Invariant: When coverage < 50%, aggregate and near-ATM PCR are unavailable (None), NOT distorted 9.0!
    assert pcr_low.strike_coverage == 0.2
    assert pcr_low.valid_strikes_count == 1
    assert pcr_low.aggregate_oi_pcr is None
    assert pcr_low.near_atm_oi_pcr is None
    assert pcr_low.strike_weighted_pcr is None

    # Scenario B: 4 strikes, 3 have valid CE and PE (50k each), 1 has missing CE and 100k PE
    # Coverage = 3/4 = 75% >= 50%
    t1 = t0 + timedelta(minutes=3)
    snap_good_coverage = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t1,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25000.0,
                call={"symbol": "NSE:NIFTY25000CE", "oi": 50000, "ltp": 150.0, "volume": 5000},
                put={"symbol": "NSE:NIFTY25000PE", "oi": 50000, "ltp": 40.0, "volume": 5000},
            ),
            OptionChainStrike(
                strike=25050.0,
                call={"symbol": "NSE:NIFTY25050CE", "oi": 50000, "ltp": 110.0, "volume": 6000},
                put={"symbol": "NSE:NIFTY25050PE", "oi": 50000, "ltp": 60.0, "volume": 6000},
            ),
            OptionChainStrike(
                strike=25100.0,
                call={"symbol": "NSE:NIFTY25100CE", "oi": 50000, "ltp": 80.0, "volume": 10000},
                put={"symbol": "NSE:NIFTY25100PE", "oi": 50000, "ltp": 80.0, "volume": 10000},
            ),
            OptionChainStrike(
                strike=25150.0,
                call={"symbol": "NSE:NIFTY25150CE", "oi": None, "ltp": None, "volume": None},
                put={"symbol": "NSE:NIFTY25150PE", "oi": 100000, "ltp": 110.0, "volume": 10000},
            ),
        ],
        total_call_oi=150000,
        total_put_oi=250000,
        pcr=1.667,
    )
    pcr_good = pcr_engine.on_option_chain(snap_good_coverage)

    # Invariant: Evaluated strictly over the 3 valid comparable strikes (150k PE / 150k CE = 1.0)
    # The missing CE strike (25150) is excluded, preventing artificial bullish skew!
    assert pcr_good.strike_coverage == 0.75
    assert pcr_good.valid_strikes_count == 3
    assert pcr_good.aggregate_oi_pcr == 1.0
    assert pcr_good.near_atm_oi_pcr == 1.0


def test_p2_futures_rollover_cleans_up_obsolete_reverse_mapping():
    """Verify P2: Technical engine removes obsolete reverse mappings on contract rollover

    while preserving historical records for research.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    # Initial state
    assert engine.primary_futures == "NSE:NIFTY26NOVFUT"
    assert engine.index_futures_map["NSE:NIFTY50-INDEX"] == "NSE:NIFTY26NOVFUT"
    assert engine.futures_index_map["NSE:NIFTY26NOVFUT"] == "NSE:NIFTY50-INDEX"

    # Simulate monthly rollover to December contract
    engine.set_futures_ticker("NSE:NIFTY26DECFUT")

    # Invariant: Active mappings point to December contract
    assert engine.primary_futures == "NSE:NIFTY26DECFUT"
    assert engine.index_futures_map["NSE:NIFTY50-INDEX"] == "NSE:NIFTY26DECFUT"
    assert engine.futures_index_map["NSE:NIFTY26DECFUT"] == "NSE:NIFTY50-INDEX"

    # Invariant: Obsolete reverse mapping for November contract is REMOVED
    assert "NSE:NIFTY26NOVFUT" not in engine.futures_index_map

    # Invariant: Historical state for November contract is PRESERVED for research
    assert "NSE:NIFTY26NOVFUT" in engine.states

    # Test explicit set_index_futures_mapping rollover
    engine.set_index_futures_mapping("NSE:BANKNIFTY-INDEX", "NSE:BANKNIFTY26NOVFUT")
    assert engine.futures_index_map["NSE:BANKNIFTY26NOVFUT"] == "NSE:BANKNIFTY-INDEX"

    engine.set_index_futures_mapping("NSE:BANKNIFTY-INDEX", "NSE:BANKNIFTY26DECFUT")
    assert engine.futures_index_map["NSE:BANKNIFTY26DECFUT"] == "NSE:BANKNIFTY-INDEX"
    assert "NSE:BANKNIFTY26NOVFUT" not in engine.futures_index_map
    assert "NSE:BANKNIFTY26NOVFUT" in engine.states


