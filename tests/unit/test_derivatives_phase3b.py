"""Unit tests for Phase 3B: Greeks Engine, Intelligent Contract Selection, and Cost Modeling.

Exhaustively verifies:
1. Black-76 analytical Greeks computation (Delta, Gamma, Theta, Vega, Rho) for Call and Put.
2. Robust Newton-Raphson & Bisection Implied Volatility (IV) solver convergence and boundary safety.
3. Transparent fallback: broker-reported Greeks preferred when valid, Black-76 when missing/zero.
4. Quantitative OptionContractSelector multi-factor scoring (Delta targeting 0.35-0.55, liquidity, spread, theta budget).
5. Second-order Taylor series premium trajectory estimation (Delta + Gamma - Theta + Vega).
6. Statutory Indian transaction costs (FYERS brokerage, STT 0.15% on sell, GST 18%, Exchange fees, Stamp Duty, Slippage).
7. Net vs Gross Reward-to-Risk ratio edge viability filtering.
8. Seamless integration into DerivativesIntelligenceEngine and TechnicalCandidate payload.
"""

from datetime import date, datetime

import pytz

from nifty_engine.analysis.derivatives.contract_selector import (
    OptionContractSelector,
)
from nifty_engine.analysis.derivatives.cost_model import (
    PremiumCostModel,
)
from nifty_engine.analysis.derivatives.derivatives_assessment import (
    DerivativesIntelligenceEngine,
)
from nifty_engine.analysis.derivatives.greeks import (
    GreeksEngine,
    OptionGreeks,
)
from nifty_engine.analysis.derivatives.oi_analysis import OpenInterestAnalyzer
from nifty_engine.analysis.derivatives.pcr import PCREngine
from nifty_engine.analysis.technical.breakouts import BreakoutDirection
from nifty_engine.core.interfaces import OptionChainSnapshot, OptionChainStrike

IST = pytz.timezone("Asia/Kolkata")


def create_phase3b_chain(
    api_time: datetime,
    spot: float = 25100.0,
    expiry_dt: date = date(2026, 11, 26),
) -> OptionChainSnapshot:
    """Helper creating OptionChainSnapshot with full realistic Greeks and quotes."""
    strikes_data = [
        # strike, c_oi, c_ltp, c_bid, c_ask, c_vol, p_oi, p_ltp, p_bid, p_ask, p_vol
        (24900.0, 40000, 245.0, 244.5, 245.5, 12000, 20000, 45.0, 44.5, 45.5, 6000),
        (25000.0, 70000, 160.0, 159.5, 160.5, 25000, 40000, 60.0, 59.5, 60.5, 15000),
        (25050.0, 90000, 125.0, 124.5, 125.5, 30000, 60000, 75.0, 74.5, 75.5, 20000),
        (25100.0, 140000, 95.0, 94.5, 95.5, 45000, 110000, 95.0, 94.5, 95.5, 35000),
        (25150.0, 80000, 70.0, 69.5, 70.5, 20000, 130000, 120.0, 119.5, 120.5, 25000),
        (25200.0, 60000, 50.0, 49.5, 50.5, 15000, 90000, 150.0, 149.5, 150.5, 18000),
        (25300.0, 30000, 25.0, 24.5, 25.5, 8000, 50000, 225.0, 224.5, 225.5, 10000),
    ]

    parsed_strikes = []
    tot_c_oi = 0
    tot_p_oi = 0

    for strike, c_oi, c_ltp, c_bid, c_ask, c_vol, p_oi, p_ltp, p_bid, p_ask, p_vol in strikes_data:
        tot_c_oi += c_oi
        tot_p_oi += p_oi
        parsed_strikes.append(
            OptionChainStrike(
                strike=strike,
                call={
                    "symbol": f"NSE:NIFTY26NOV{int(strike)}CE",
                    "ltp": c_ltp,
                    "oi": c_oi,
                    "volume": c_vol,
                    "bid": c_bid,
                    "ask": c_ask,
                    "iv": 14.5,
                    "delta": 0.50 if strike == 25100 else (0.75 if strike < 25100 else 0.30),
                    "gamma": 0.0004,
                    "theta": -8.5,
                    "vega": 12.0,
                },
                put={
                    "symbol": f"NSE:NIFTY26NOV{int(strike)}PE",
                    "ltp": p_ltp,
                    "oi": p_oi,
                    "volume": p_vol,
                    "bid": p_bid,
                    "ask": p_ask,
                    "iv": 14.8,
                    "delta": -0.50 if strike == 25100 else (-0.25 if strike < 25100 else -0.70),
                    "gamma": 0.0004,
                    "theta": -8.2,
                    "vega": 12.0,
                },
            )
        )

    return OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=spot,
        api_timestamp=api_time,
        expiry=expiry_dt,
        strikes=parsed_strikes,
        total_call_oi=tot_c_oi,
        total_put_oi=tot_p_oi,
        pcr=round(tot_p_oi / tot_c_oi, 3) if tot_c_oi > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# 1. Greeks Engine Unit Tests
# ---------------------------------------------------------------------------


def test_black76_greeks_analytical_accuracy():
    """Verify Black-76 analytical formulas for Call and Put Greeks under Indian index conventions."""
    engine = GreeksEngine(default_risk_free_rate=0.065)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    expiry = date(2026, 11, 26)  # ~3.23 days remaining

    # 1. ATM Call & Put (Strike 25000, Futures 25000, Volatility 15%)
    g_call = engine.calculate_greeks(
        option_type="CE",
        strike=25000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=140.0,
        broker_greeks=None,  # Force analytical Black-76
    )
    g_put = engine.calculate_greeks(
        option_type="PE",
        strike=25000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=140.0,
        broker_greeks=None,
    )

    # Invariants for ATM contracts:
    assert g_call.model_used == "BLACK_76"
    assert g_call.is_calculated is True
    assert 0.48 <= g_call.delta <= 0.53
    assert -0.53 <= g_put.delta <= -0.48

    # Gamma and Vega are identical for Call and Put
    assert g_call.gamma > 0.0
    assert abs(g_call.gamma - g_put.gamma) < 1e-6
    assert g_call.vega > 0.0
    assert abs(g_call.vega - g_put.vega) < 1e-3

    # Theta decay is negative (loss of premium over time)
    assert g_call.theta < 0.0
    assert g_put.theta < 0.0

    # 2. Deep ITM Call (Strike 24000) vs Deep OTM Call (Strike 26000)
    g_itm_call = engine.calculate_greeks(
        option_type="CE",
        strike=24000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=1010.0,
    )
    assert g_itm_call.delta > 0.90

    g_otm_call = engine.calculate_greeks(
        option_type="CE",
        strike=26000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=15.0,
    )
    assert g_otm_call.delta < 0.15


def test_newton_raphson_iv_solver_convergence():
    """Verify Newton-Raphson IV solver reproduces theoretical volatility accurately."""
    engine = GreeksEngine(default_risk_free_rate=0.065)
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    expiry = date(2026, 11, 26)
    T = engine.compute_time_to_expiry_years(expiry=expiry, current_time=t0)

    true_sigma = 0.165  # 16.5% IV
    # Generate theoretical price
    call_price = engine.calculate_black76_price(
        option_type="CE",
        underlying_futures=25000.0,
        strike=25000.0,
        time_to_expiry_years=T,
        volatility=true_sigma,
    )

    # Invert price using solver
    solved_iv = engine.solve_implied_volatility(
        option_type="CE",
        market_price=call_price,
        underlying_futures=25000.0,
        strike=25000.0,
        time_to_expiry_years=T,
    )

    # Invariant: IV solver converges within 0.1% of true parameter
    assert abs(solved_iv - true_sigma) < 0.001


def test_greeks_broker_fallback_preference():
    """Verify primary preference for valid broker Greeks and fallback when missing."""
    engine = GreeksEngine()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    expiry = date(2026, 11, 26)

    # 1. Valid broker Greeks supplied
    broker_payload = {
        "iv": 15.2,
        "delta": 0.52,
        "gamma": 0.00045,
        "theta": -9.1,
        "vega": 11.8,
        "rho": 0.02,
    }
    g_broker = engine.calculate_greeks(
        option_type="CE",
        strike=25000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=145.0,
        broker_greeks=broker_payload,
    )
    assert g_broker.model_used == "FYERS"
    assert g_broker.is_calculated is False
    assert g_broker.delta == 0.52
    assert g_broker.theta == -9.1

    # 2. Broker Greeks missing or zero -> Transparent Black-76 fallback
    empty_payload = {"iv": None, "delta": None}
    g_fallback = engine.calculate_greeks(
        option_type="CE",
        strike=25000.0,
        underlying_futures=25000.0,
        expiry=expiry,
        current_time=t0,
        market_price=145.0,
        broker_greeks=empty_payload,
    )
    assert g_fallback.model_used == "BLACK_76"
    assert g_fallback.is_calculated is True
    assert g_fallback.delta > 0.40


# ---------------------------------------------------------------------------
# 2. Premium Trajectory & Execution Cost Model Tests
# ---------------------------------------------------------------------------


def test_second_order_taylor_premium_movement():
    """Verify second-order Taylor expansion translates index targets into option premium prices."""
    model = PremiumCostModel()
    greeks_call = OptionGreeks(
        iv=0.15,
        delta=0.50,
        gamma=0.0004,
        theta=-8.0,  # -8 points per day
        vega=12.0,
        rho=0.02,
        is_calculated=True,
    )

    # Bullish scenario: Spot 25100, Target 1 = 25130 (+30 pts), Target 2 = 25160 (+60 pts), SL = 25080 (-20 pts)
    # Entry premium = ₹100.0
    est_call = model.estimate_premium_trajectory(
        greeks=greeks_call,
        entry_premium=100.0,
        spot_entry=25100.0,
        spot_target_t1=25130.0,
        spot_target_t2=25160.0,
        spot_sl=25080.0,
        direction="BULLISH",
        holding_minutes=30.0,  # 30 mins holding
    )

    # Invariants:
    # Target 1: delta move (0.50 * 30 = 15.0) + gamma (0.5 * 0.0004 * 900 = 0.18) - calendar-time theta (8.0 * 30 / 1440 = 0.17)
    calendar_theta_per_min = 8.0 / 1440.0
    expected_theta_loss = round(calendar_theta_per_min * 30.0, 2)
    expected_gain_t1 = (0.50 * 30.0) + (0.5 * 0.0004 * 900.0) - expected_theta_loss
    assert abs(est_call.target_premium_t1 - (100.0 + expected_gain_t1)) < 0.1
    assert est_call.target_premium_t2 > est_call.target_premium_t1
    assert est_call.stop_loss_premium < 100.0
    assert est_call.theta_cost == expected_theta_loss

    # Quadratic spot move solver
    req_move = model.calculate_required_spot_move_for_premium_target(greeks_call, target_premium_gain=15.0)
    assert 29.0 <= req_move <= 31.0


def test_indian_statutory_costs_and_net_rr_viability():
    """Verify Indian statutory fee calculations (STT 0.15%, Brokerage ₹40, GST, Stamp Duty, Slippage)."""
    model = PremiumCostModel(lot_size=65, default_slippage_points=0.5, min_net_rr=1.3)

    # Scenario: Buy at ₹100, Target at ₹130 (+30 pts), Stop Loss at ₹85 (-15 pts). Bid-ask spread = 0.5 pt
    costs = model.calculate_execution_costs(
        entry_premium=100.0,
        target_premium_t1=130.0,
        stop_loss_premium=85.0,
        bid_ask_spread=0.5,
    )

    # 1. Statutory fee accuracy
    assert costs.brokerage == 40.0  # ₹20 buy + ₹20 sell
    # STT = 0.15% on sell turnover (130 * 65 * 0.0015 = 12.675 -> ₹12.68)
    assert abs(costs.stt - 12.68) <= 0.02
    assert costs.gst > 7.0  # 18% on brokerage + exchange fees
    assert costs.total_statutory_rupees > 60.0  # Total government/broker cost per lot

    # 2. Friction in points: statutory pts (~1.0 pt) + spread (0.5 pt) + slippage (2 * 0.5 = 1.0 pt)
    assert 2.0 <= costs.total_friction_points <= 3.5

    # 3. Reward-to-Risk conversion
    # Gross RR: 30 / 15 = 2.0
    assert costs.gross_rr_t1 == 2.0
    # Net reward = 30 - friction (~27.5), Net risk = 15 + friction (~17.5) -> Net RR ~ 1.57 >= 1.3
    assert costs.net_rr_t1 >= 1.4
    assert costs.is_cost_viable is True

    # Scenario 2: Unviable trade with tiny target (+3 pts gain)
    costs_tiny = model.calculate_execution_costs(
        entry_premium=100.0,
        target_premium_t1=103.0,
        stop_loss_premium=95.0,
        bid_ask_spread=1.0,
    )
    # Friction (~3.5 pts) completely erodes the +3 pt gross gain!
    assert costs_tiny.is_cost_viable is False
    assert costs_tiny.net_reward_points_t1 == 0.0


# ---------------------------------------------------------------------------
# 3. Option Contract Selector Multi-Factor Scoring Tests
# ---------------------------------------------------------------------------


def test_option_contract_selector_optimal_strike_and_greeks():
    """Verify OptionContractSelector selects the strike in optimal Delta sweet spot (0.35 - 0.55)."""
    selector = OptionContractSelector()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    chain_snap = create_phase3b_chain(api_time=t0, spot=25100.0)

    # Convert snapshot to profile for analyzer
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(chain_snap)
    assert profile is not None

    # 1. Bullish Breakout: Select Call Option
    selected_call = selector.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        futures_price=25110.0,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
        target_points_t2=80.0,
    )

    assert selected_call is not None
    assert selected_call.option_type == "CE"
    # Strike 25100 has Delta ~0.50, optimal premium ₹95, tight spread -> highest score!
    assert selected_call.strike == 25100.0
    assert selected_call.selection_score >= 80.0
    assert 0.35 <= abs(selected_call.greeks.delta) <= 0.55
    assert selected_call.execution_costs.is_cost_viable is True
    assert selected_call.premium_trajectory.target_premium_t1 > selected_call.ltp
    assert selected_call.premium_trajectory.stop_loss_premium < selected_call.ltp

    # 2. Bearish Breakout: Select Put Option
    selected_put = selector.select_contract(
        direction=BreakoutDirection.BEARISH,
        spot_price=25100.0,
        chain_profile=profile,
        futures_price=25110.0,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
        target_points_t2=80.0,
    )

    assert selected_put is not None
    assert selected_put.option_type == "PE"
    assert selected_put.strike == 25100.0
    assert selected_put.selection_score >= 80.0
    assert -0.55 <= selected_put.greeks.delta <= -0.35


# ---------------------------------------------------------------------------
# 4. Seamless Integration into DerivativesAssessment
# ---------------------------------------------------------------------------


def test_derivatives_assessment_enriches_candidate_with_selected_contract():
    """Verify DerivativesIntelligenceEngine automatically coordinates contract selection and cost modeling."""
    oi_analyzer = OpenInterestAnalyzer()
    pcr_engine = PCREngine()
    selector = OptionContractSelector()
    deriv_engine = DerivativesIntelligenceEngine(
        oi_analyzer=oi_analyzer,
        pcr_engine=pcr_engine,
        contract_selector=selector,
    )

    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    chain_snap = create_phase3b_chain(api_time=t0, spot=25100.0)
    oi_analyzer.on_option_chain(chain_snap)
    pcr_engine.on_option_chain(chain_snap)

    assessment = deriv_engine.evaluate_setup(
        direction=BreakoutDirection.BULLISH,
        level_price=25100.0,
        as_of=t0,
        underlying="NIFTY",
        risk_points=15.0,
        target_points_t1=45.0,
        target_points_t2=80.0,
    )

    # Invariants:
    assert assessment.selected_contract is not None
    assert assessment.selected_contract.symbol == "NSE:NIFTY26NOV25100CE"
    assert assessment.selected_contract.option_type == "CE"
    assert assessment.selected_contract.selection_score >= 70.0
    assert assessment.selected_contract.premium_trajectory.target_premium_t1 > 0.0

    # Serialization test
    as_dict = assessment.to_dict()
    assert "selected_contract" in as_dict
    contract_dict = as_dict["selected_contract"]
    assert contract_dict["symbol"] == "NSE:NIFTY26NOV25100CE"
    assert contract_dict["strike"] == 25100.0
    assert contract_dict["is_cost_viable"] is True
    assert "target_premium_t1" in contract_dict


# ---------------------------------------------------------------------------
# 5. Contract Integrity, Eligibility Gates & Asymmetric Cost Tests
# ---------------------------------------------------------------------------


def test_contract_selector_rejects_missing_or_invalid_broker_quotes():
    """Verify OptionContractSelector rejects contracts with missing symbols, inverted/missing bid-ask, or extreme spreads."""
    selector = OptionContractSelector(
        min_liquidity_volume=1000,
        max_acceptable_spread_pct=0.03,
    )
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))

    # 1. Missing symbol -> must reject (never synthesize!)
    invalid_snap1 = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={
                    "symbol": "",  # Missing symbol!
                    "ltp": 95.0,
                    "oi": 10000,
                    "volume": 5000,
                    "bid": 94.5,
                    "ask": 95.5,
                },
            )
        ],
    )
    oi_analyzer = OpenInterestAnalyzer()
    profile1 = oi_analyzer.on_option_chain(invalid_snap1)
    res1 = selector.select_contract(BreakoutDirection.BULLISH, 25100.0, profile1, as_of=t0)
    assert res1 is None, "Should reject contract with missing exchange symbol"

    # 2. Missing/negative bid/ask quotes -> must reject (never manufacture synthetic quotes!)
    invalid_snap2 = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={
                    "symbol": "NSE:NIFTY26NOV25100CE",
                    "ltp": 95.0,
                    "oi": 10000,
                    "volume": 5000,
                    "bid": None,  # Missing bid!
                    "ask": 95.5,
                },
            )
        ],
    )
    profile2 = oi_analyzer.on_option_chain(invalid_snap2)
    res2 = selector.select_contract(BreakoutDirection.BULLISH, 25100.0, profile2, as_of=t0)
    assert res2 is None, "Should reject contract with missing broker bid"

    # 3. Inverted quote (ask < bid) -> must reject
    invalid_snap3 = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={
                    "symbol": "NSE:NIFTY26NOV25100CE",
                    "ltp": 95.0,
                    "oi": 10000,
                    "volume": 5000,
                    "bid": 98.0,
                    "ask": 94.0,  # Ask < Bid!
                },
            )
        ],
    )
    profile3 = oi_analyzer.on_option_chain(invalid_snap3)
    res3 = selector.select_contract(BreakoutDirection.BULLISH, 25100.0, profile3, as_of=t0)
    assert res3 is None, "Should reject inverted broker quote"

    # 4. Wide spread (spread_pct > max_acceptable_spread_pct = 3%) -> must reject
    invalid_snap4 = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25100.0,
        api_timestamp=t0,
        expiry=date(2026, 11, 26),
        strikes=[
            OptionChainStrike(
                strike=25100.0,
                call={
                    "symbol": "NSE:NIFTY26NOV25100CE",
                    "ltp": 95.0,
                    "oi": 10000,
                    "volume": 5000,
                    "bid": 90.0,
                    "ask": 96.0,  # 6.0 pt spread on 96.0 = 6.25% spread (> 3%)
                },
            )
        ],
    )
    profile4 = oi_analyzer.on_option_chain(invalid_snap4)
    res4 = selector.select_contract(BreakoutDirection.BULLISH, 25100.0, profile4, as_of=t0)
    assert res4 is None, "Should reject illiquid contract with wide spread"


def test_contract_selector_exact_technical_entry_alignment():
    """Verify option premium trajectory evaluates remaining move from technical entry_price, not nominal breakout level."""
    selector = OptionContractSelector()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    chain_snap = create_phase3b_chain(api_time=t0, spot=25100.0)
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(chain_snap)

    # Breakout level is 25100.0, but actual entry confirmed at 25120.0
    # Technical targets: T1 = 25165 (+45 pts remaining), T2 = 25200 (+80 pts), SL = 25105 (-15 pts)
    selected = selector.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,  # Nominal level
        chain_profile=profile,
        futures_price=25125.0,
        as_of=t0,
        entry_price=25120.0,  # Actual technical entry
        stop_loss=25105.0,
        target_1=25165.0,
        target_2=25200.0,
    )

    assert selected is not None
    traj = selected.premium_trajectory
    # Required spot move to T1 must be exactly 45.0 (25165 - 25120), not (25165 - 25100 = 65)!
    assert traj.required_spot_move_t1 == 45.0
    # Entry premium used for trajectory must be the Ask price (95.5), not LTP (95.0)
    assert traj.initial_premium == selected.ask


def test_contract_selector_broker_greeks_and_forward_price():
    """Verify contract selector preserves broker Greeks and interpolates forward price for weekly options."""
    selector = OptionContractSelector()
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    chain_snap = create_phase3b_chain(api_time=t0, spot=25100.0)
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(chain_snap)

    # Weekly option expiring 26-Nov, monthly futures at 25200 (+100 pts basis)
    fwd = selector.compute_forward_price(
        spot_price=25100.0,
        expiry=date(2026, 11, 26),
        current_time=t0,
        futures_price=25200.0,
        futures_expiry=date(2026, 12, 31),  # 38 days away
    )
    # The 3-day weekly forward should NOT have the full 100 pt basis of 38-day futures!
    assert 25100.0 < fwd < 25125.0

    selected = selector.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        futures_price=fwd,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert selected is not None
    # Broker Greeks passed through
    assert selected.greeks.model_used == "FYERS"
    assert selected.greeks.delta == 0.50


def test_asymmetric_target_and_sl_execution_costs():
    """Verify execution cost model calculates separate asymmetric statutory fees for Target and SL exits."""
    model = PremiumCostModel(lot_size=65)

    # Buy at ₹100, Target at ₹140 (+40 pts gain), Stop Loss at ₹80 (-20 pts loss)
    costs = model.calculate_execution_costs(
        entry_premium=100.0,
        target_premium_t1=140.0,
        stop_loss_premium=80.0,
        bid_ask_spread=1.0,
        slippage_points=0.5,
    )

    # STT at target (0.15% on 140 * 65 = 13.65) must be higher than STT at SL (0.15% on 80 * 65 = 7.80)
    assert costs.stt_target > costs.stt_sl
    assert abs(costs.stt_target - 13.65) <= 0.05
    assert abs(costs.stt_sl - 7.80) <= 0.05

    # Total statutory fees at Target > Total statutory fees at SL
    assert costs.total_statutory_target_rupees > costs.total_statutory_sl_rupees

    # Friction points at target > friction points at SL
    assert costs.target_friction_points > costs.sl_friction_points

    # Net reward = gross reward (40.0) - target_friction
    assert costs.net_reward_points_t1 == round(40.0 - costs.target_friction_points, 2)
    # Net risk = gross risk (20.0) + sl_friction
    assert costs.net_risk_points == round(20.0 + costs.sl_friction_points, 2)


def test_contract_selector_rejects_timestamp_less_rest_quotes():
    """Verify OptionContractSelector rejects on-demand REST quotes missing broker timestamps."""
    from datetime import timedelta
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    # Option chain snapshot from 100 seconds ago (stale > 30s)
    stale_chain_time = t0 - timedelta(seconds=100)
    snap = create_phase3b_chain(stale_chain_time)
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(snap)

    # 1. Quote provider returns fresh bid/ask but WITHOUT timestamp (timestamp=None)
    def mock_quote_provider_no_ts(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": 95.0,
                "bid": 94.5,
                "ask": 95.5,
                "timestamp": None,
            }
        return None

    selector_no_ts = OptionContractSelector(
        quote_provider=mock_quote_provider_no_ts,
        max_quote_age_seconds=30.0,
    )
    # Because quote has no timestamp, it cannot substitute current_time;
    # underlying chain quote is stale (100s old), so contract MUST be rejected!
    res_no_ts = selector_no_ts.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res_no_ts is None

    # 2. Quote provider returns fresh bid/ask WITH authentic broker timestamp (5s old)
    def mock_quote_provider_valid_ts(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": 95.0,
                "bid": 94.5,
                "ask": 95.5,
                "timestamp": t0 - timedelta(seconds=5),
            }
        return None

    selector_valid = OptionContractSelector(
        quote_provider=mock_quote_provider_valid_ts,
        max_quote_age_seconds=30.0,
    )
    res_valid = selector_valid.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res_valid is not None
    assert res_valid.bid == 94.5
    assert res_valid.ask == 95.5
    assert res_valid.quote_age_seconds == 5.0


def test_contract_selector_rejects_partially_populated_rest_quotes():
    """Verify OptionContractSelector rejects partially populated REST quotes (e.g. bid without ask)."""
    from datetime import timedelta
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    # Stale chain snapshot (100 seconds old > 30s max age)
    stale_chain_time = t0 - timedelta(seconds=100)
    snap = create_phase3b_chain(stale_chain_time)
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(snap)

    # 1. Fresh timestamp and valid bid, but MISSING ask (ask=None)
    def mock_quote_provider_missing_ask(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": 95.0,
                "bid": 94.5,
                "ask": None,  # Incomplete!
                "timestamp": t0 - timedelta(seconds=5),
            }
        return None

    selector1 = OptionContractSelector(
        quote_provider=mock_quote_provider_missing_ask,
        max_quote_age_seconds=30.0,
    )
    res1 = selector1.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    # Must reject incomplete quote rather than combining fresh bid with stale ask from snapshot
    assert res1 is None

    # 2. Fresh timestamp and valid ask, but MISSING bid (bid=None)
    def mock_quote_provider_missing_bid(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": 95.0,
                "bid": None,  # Incomplete!
                "ask": 95.5,
                "timestamp": t0 - timedelta(seconds=5),
            }
        return None

    selector2 = OptionContractSelector(
        quote_provider=mock_quote_provider_missing_bid,
        max_quote_age_seconds=30.0,
    )
    res2 = selector2.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res2 is None

    # 3. Fresh timestamp, but INVERTED bid/ask (ask < bid)
    def mock_quote_provider_inverted(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": 95.0,
                "bid": 95.5,
                "ask": 94.5,  # Inverted!
                "timestamp": t0 - timedelta(seconds=5),
            }
        return None

    selector3 = OptionContractSelector(
        quote_provider=mock_quote_provider_inverted,
        max_quote_age_seconds=30.0,
    )
    res3 = selector3.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res3 is None


def test_contract_selector_derives_midpoint_ltp_when_rest_quote_omits_ltp():
    """Verify OptionContractSelector derives model LTP from fresh bid/ask midpoint when quote omits LTP."""
    from datetime import timedelta
    t0 = IST.localize(datetime(2026, 11, 23, 10, 0))
    # Stale chain snapshot (100 seconds old > 30s max age)
    stale_chain_time = t0 - timedelta(seconds=100)
    snap = create_phase3b_chain(stale_chain_time)
    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(snap)

    # Mock quote provider returning valid contemporaneous bid and ask, but ltp is None
    def mock_quote_provider_no_ltp(symbol: str):
        if "25100CE" in symbol:
            return {
                "ltp": None,
                "bid": 94.0,
                "ask": 96.0,
                "timestamp": t0 - timedelta(seconds=5),
            }
        return None

    selector = OptionContractSelector(
        quote_provider=mock_quote_provider_no_ltp,
        max_quote_age_seconds=30.0,
    )
    res = selector.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res is not None
    assert res.bid == 94.0
    assert res.ask == 96.0
    assert res.ltp == 95.0
    assert res.is_estimated_ltp is True
    assert "Model price derived from fresh bid/ask midpoint (estimated LTP)" in res.selection_reasons
    assert res.to_dict()["is_estimated_ltp"] is True



