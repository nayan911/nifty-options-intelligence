"""Unit tests for Phase 4 Paper-Trading Execution Engine.

Validates simulated order fills, conservative slippage, multi-stage exits (Target 1, Target 2,
Stop Loss, Trailing Breakeven, Spot Invalidation, EOD 15:15 IST Square-Off),
Indian statutory costs (STT 0.15%, Brokerage, GST, Stamp Duty), and DuckDB persistence.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict

import pytz

from nifty_engine.analysis.derivatives.contract_selector import SelectedContract
from nifty_engine.analysis.derivatives.cost_model import ExecutionCosts, PremiumEstimate
from nifty_engine.analysis.derivatives.derivatives_assessment import DerivativesAssessment, DerivativesAssessmentType
from nifty_engine.analysis.derivatives.greeks import OptionGreeks
from nifty_engine.analysis.derivatives.oi_analysis import DirectionalBias
from nifty_engine.analysis.technical.breakouts import BreakoutDirection
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.interfaces import Candle, OptionContract, Tick
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.paper_trading.engine import PaperTradingEngine
from nifty_engine.paper_trading.models import ExitReason, OrderSide, PositionStatus

IST = pytz.timezone("Asia/Kolkata")


def create_mock_candidate(
    candidate_id: str = "TEST_CAND_001",
    timestamp: datetime = None,
    trigger_price: float = 25100.0,
    target_1: float = 25140.0,
    target_2: float = 25180.0,
    stop_loss: float = 25070.0,
    opt_symbol: str = "NSE:NIFTY26NOV25100CE",
    entry_ltp: float = 100.0,
    bid: float = 99.8,
    ask: float = 100.2,
    spread: float = 0.4,
    quote_age: float = 2.0,
    net_rr: float = 1.6,
    is_cost_viable: bool = True,
    t1_premium: float = 120.0,
    t2_premium: float = 140.0,
    sl_premium: float = 85.0,
) -> TechnicalCandidate:
    """Helper to generate a rich mock TechnicalCandidate with complete SelectedContract."""
    ts = timestamp or IST.localize(datetime(2026, 11, 26, 10, 30, 0))

    trajectory = PremiumEstimate(
        initial_premium=entry_ltp,
        target_premium_t1=t1_premium,
        target_premium_t2=t2_premium,
        stop_loss_premium=sl_premium,
        delta_contribution_t1=18.0,
        gamma_contribution_t1=1.5,
        theta_cost=0.5,
        vega_contribution=0.0,
        required_spot_move_t1=40.0,
    )

    costs = ExecutionCosts(
        brokerage=40.0,
        stt_target=12.0,
        stt_sl=8.0,
        exchange_fee_target=8.0,
        exchange_fee_sl=7.0,
        sebi_charges=0.2,
        stamp_duty=0.3,
        gst_target=8.5,
        gst_sl=8.0,
        slippage_points=0.5,
        spread_cost_points=spread,
        total_statutory_target_rupees=69.0,
        total_statutory_sl_rupees=63.3,
        target_friction_points=2.5,
        sl_friction_points=2.4,
        gross_reward_points_t1=20.0,
        gross_risk_points=15.0,
        gross_rr_t1=1.33,
        net_reward_points_t1=17.5,
        net_risk_points=17.4,
        net_rr_t1=net_rr,
        is_cost_viable=is_cost_viable,
        lot_size=65,
    )

    greeks = OptionGreeks(
        iv=0.14,
        delta=0.52,
        gamma=0.0004,
        theta=-7.5,
        vega=11.0,
        rho=0.02,
        is_calculated=True,
    )

    selected = SelectedContract(
        symbol=opt_symbol,
        underlying="NIFTY",
        strike=25100.0,
        option_type="CE",
        expiry=date(2026, 11, 26),
        ltp=entry_ltp,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread / entry_ltp,
        volume=50000,
        oi=1500000,
        greeks=greeks,
        selection_score=88.5,
        selection_reasons=["High liquidity", "Optimal delta", "Tight spread"],
        premium_trajectory=trajectory,
        execution_costs=costs,
        quote_timestamp=ts,
        quote_age_seconds=quote_age,
        holding_minutes=30.0,
    )

    da = DerivativesAssessment(
        assessment_type=DerivativesAssessmentType.STRONGLY_SUPPORTIVE,
        direction=BreakoutDirection.BULLISH,
        alignment_score=0.85,
        data_quality_score=0.95,
        futures_bias=DirectionalBias.BULLISH,
        options_bias=DirectionalBias.BULLISH,
        pcr_snapshot=None,
        oi_profile=None,
        selected_contract=selected,
        reasons=["Strong Call buying and Put writing"],
        decision_at=ts,
    )

    risk_pts = max(0.5, trigger_price - stop_loss)
    reward_pts = max(1.0, target_1 - trigger_price)
    return TechnicalCandidate(
        candidate_id=candidate_id,
        symbol="NSE:NIFTY50-INDEX",
        timestamp=ts,
        direction=BreakoutDirection.BULLISH,
        setup_level="RESISTANCE",
        nominal_price=trigger_price,
        entry_price=trigger_price,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        risk_points=risk_pts,
        reward_points=reward_pts,
        risk_reward_ratio=round(reward_pts / risk_pts, 2),
        confluence_score=85.0,
        target_1_rr=1.33,
        target_2_rr=2.67,
        decision_at=ts,
        derivatives_assessment=da,
        is_paper_trading=True,
    )


def test_paper_entry_gating_and_realistic_fill():
    """Verify paper entry respects market hours, risk gates, quote age, and conservative ask fills."""
    engine = PaperTradingEngine(slippage_points=0.5, min_net_rr=1.3)

    # 1. Reject before market open (09:10 IST)
    early_ts = IST.localize(datetime(2026, 11, 26, 9, 10, 0))
    cand_early = create_mock_candidate(timestamp=early_ts)
    pos = engine.process_technical_candidate(cand_early, current_time=early_ts)
    assert pos is None

    # 2. Reject after 15:00 IST cutoff
    late_ts = IST.localize(datetime(2026, 11, 26, 15, 5, 0))
    cand_late = create_mock_candidate(timestamp=late_ts)
    pos = engine.process_technical_candidate(cand_late, current_time=late_ts)
    assert pos is None

    # 3. Reject stale quote (> 30s)
    valid_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand_stale = create_mock_candidate(timestamp=valid_ts, quote_age=45.0)
    pos = engine.process_technical_candidate(cand_stale, current_time=valid_ts)
    assert pos is None

    # 4. Reject unviable net R:R
    cand_unviable = create_mock_candidate(timestamp=valid_ts, is_cost_viable=False, net_rr=0.9)
    pos = engine.process_technical_candidate(cand_unviable, current_time=valid_ts)
    assert pos is None

    # 5. Successful Entry Fill at Ask (100.2) + Slippage (0.5) = 100.70
    cand_valid = create_mock_candidate(timestamp=valid_ts, ask=100.2, quote_age=5.0, net_rr=1.6)
    pos = engine.process_technical_candidate(cand_valid, current_time=valid_ts)
    assert pos is not None
    assert pos.status == PositionStatus.OPEN
    assert pos.entry_price == 100.70
    assert pos.remaining_quantity == 65
    assert pos.total_costs > 20.0
    assert len(engine.get_open_positions()) == 1


def test_multi_lot_target1_partial_booking_and_target2_full_exit():
    """Verify 2-lot position books 50% on Target 1, trails SL to breakeven, and exits 50% on Target 2."""
    engine = PaperTradingEngine(default_lots=2, slippage_points=0.5)  # 2 lots = 130 qty
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        t2_premium=140.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.initial_quantity == 130
    assert pos.remaining_quantity == 130
    assert pos.entry_price == 100.50

    # Tick 1: Price rallies to 110.0 (Unrealized gain)
    tick1 = Tick(
        symbol=pos.symbol,
        ltp=110.0,
        bid=109.8,
        ask=110.2,
        volume=1000,
        local_timestamp=entry_ts,
    )
    events = engine.on_tick(tick1)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN
    assert pos.gross_pnl == (109.8 - 100.50) * 130

    # Tick 2: Target 1 Hit (Bid touches 120.5 >= 120.0)
    t1_ts = IST.localize(datetime(2026, 11, 26, 10, 15, 0))
    tick_t1 = Tick(
        symbol=pos.symbol,
        ltp=120.5,
        bid=120.5,
        ask=121.0,
        volume=5000,
        local_timestamp=t1_ts,
    )
    events_t1 = engine.on_tick(tick_t1)
    assert len(events_t1) == 1
    p, reason, exit_price = events_t1[0]
    assert reason == ExitReason.TARGET_1
    assert exit_price == 120.0  # 120.5 bid - 0.5 slippage
    assert p.status == PositionStatus.PARTIALLY_CLOSED
    assert p.remaining_quantity == 65
    assert p.is_t1_hit is True
    assert p.is_breakeven_sl_active is True
    assert p.stop_loss_premium == p.entry_price + 1.0

    # Tick 3: Target 2 Hit (Bid reaches 140.0 >= 140.0)
    t2_ts = IST.localize(datetime(2026, 11, 26, 10, 25, 0))
    tick_t2 = Tick(
        symbol=pos.symbol,
        ltp=140.5,
        bid=140.0,
        ask=140.5,
        volume=8000,
        local_timestamp=t2_ts,
    )
    events_t2 = engine.on_tick(tick_t2)
    assert len(events_t2) == 1
    p2, reason2, exit_price2 = events_t2[0]
    assert reason2 == ExitReason.TARGET_2
    assert exit_price2 == 139.50  # 140.0 bid - 0.5 slippage
    assert p2.status == PositionStatus.CLOSED
    assert p2.remaining_quantity == 0
    assert len(engine.get_open_positions()) == 0
    assert len(engine.get_closed_positions()) == 1

    # Gross & Net PnL check:
    # 65 sold at 120.0, 65 sold at 139.50, 130 bought at 100.50
    # Gross = (65 * 19.50) + (65 * 39.00) = 1267.5 + 2535.0 = 3802.50
    assert abs(p2.gross_pnl - 3802.50) < 1.0
    assert p2.net_pnl > 3500.0


def test_one_lot_target1_breakeven_trailing_stop():
    """Verify 1-lot position cannot trade fractional lots on NSE: moves SL to breakeven on T1."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 11, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        t2_premium=140.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos.remaining_quantity == 65
    assert pos.entry_price == 100.50

    # T1 Hit
    tick_t1 = Tick(
        symbol=pos.symbol,
        ltp=120.5,
        bid=120.5,
        ask=121.0,
        volume=2000,
        local_timestamp=entry_ts,
    )
    events_t1 = engine.on_tick(tick_t1)
    assert len(events_t1) == 1
    assert events_t1[0][1] == ExitReason.TARGET_1
    assert pos.remaining_quantity == 65
    assert pos.is_breakeven_sl_active is True
    assert pos.stop_loss_premium == 101.50

    # Market retraces sharply below entry: drops to 98.0
    retrace_tick = Tick(
        symbol=pos.symbol,
        ltp=98.0,
        bid=98.0,
        ask=98.5,
        volume=3000,
        local_timestamp=entry_ts,
    )
    events_retrace = engine.on_tick(retrace_tick)
    assert len(events_retrace) == 1
    p, reason, exit_price = events_retrace[0]
    assert reason == ExitReason.TRAILING_STOP
    assert p.status == PositionStatus.CLOSED


def test_stop_loss_exit_on_option_drop():
    """Verify stop loss triggers when option Bid drops to or below stop_loss_premium."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 11, 30, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)

    sl_tick = Tick(
        symbol=pos.symbol,
        ltp=84.5,
        bid=84.5,
        ask=85.0,
        volume=5000,
        local_timestamp=entry_ts,
    )
    events = engine.on_tick(sl_tick)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 84.0  # 84.5 bid - 0.5 slippage
    assert p.status == PositionStatus.CLOSED
    assert p.net_pnl < 0.0


def test_emergency_exit_on_underlying_spot_sl_breach_with_fresh_quote():
    """Verify underlying spot price breach forces immediate option position liquidation when quote is fresh."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 12, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Underlying index cracks down to 25068.0 (< 25070.0) within fresh window
    index_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25068.0,
        volume=100000,
        local_timestamp=entry_ts + timedelta(seconds=10),
    )
    events = engine.on_tick(index_tick)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert p.status == PositionStatus.CLOSED
    assert "Spot SL breached" in p.notes


def test_stale_option_quote_delays_spot_trigger_until_fresh_tick():
    """Verify stale option quote prevents erroneous execution; awaits fresh option tick."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5, max_quote_age_seconds=30.0)
    entry_ts = IST.localize(datetime(2026, 11, 26, 12, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Spot breach arrives 5 minutes later without intermediate option quotes!
    stale_ts = entry_ts + timedelta(minutes=5)
    index_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25068.0,
        volume=100000,
        local_timestamp=stale_ts,
    )
    events = engine.on_tick(index_tick)
    # MUST NOT execute against the 5-minute-old entry price!
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN
    assert pos.is_exit_triggered is True
    assert pos.pending_exit_reason == ExitReason.STOP_LOSS

    # Now fresh option tick arrives at actual current market (e.g. ₹75.0)
    fresh_opt_tick = Tick(
        symbol=pos.symbol,
        ltp=75.0,
        bid=74.8,
        ask=75.2,
        volume=1000,
        local_timestamp=stale_ts + timedelta(seconds=2),
    )
    fresh_events = engine.on_tick(fresh_opt_tick)
    assert len(fresh_events) == 1
    p, reason, exit_price = fresh_events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 74.30  # 74.8 bid - 0.5 slippage (NOT the stale 100.0 quote!)
    assert p.status == PositionStatus.CLOSED


def test_on_candle_intrabar_adverse_first_evaluation():
    """Verify on_candle evaluates intrabar high and low with conservative adverse-first logic and fresh bid validation."""
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        quote_provider=lambda sym: {"ltp": 83.0, "bid": 83.0, "ask": 83.5, "timestamp": entry_ts + timedelta(minutes=1)},
    )

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Candle that spiked to 83.0 (SL breached) but closed at 95.0
    bar = Candle(
        symbol=pos.symbol,
        timeframe="1m",
        timestamp=entry_ts + timedelta(minutes=1),
        open=98.0,
        high=102.0,
        low=83.0,  # Breached 85.0 SL!
        close=95.0,
        volume=10000,
        is_closed=True,
    )
    events = engine.on_candle(bar)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 82.50  # 83.0 low - 0.5 slippage
    assert p.status == PositionStatus.CLOSED


def test_no_duplicate_target1_events():
    """Verify spot target 1 branch returns exactly one event without duplicates."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        target_1=25140.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Spot tick reaching Target 1
    spot_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25145.0,
        volume=50000,
        local_timestamp=entry_ts + timedelta(seconds=15),
    )
    events = engine.on_tick(spot_tick)
    assert len(events) == 1  # Exactly 1 event, not duplicated!
    assert events[0][1] == ExitReason.TARGET_1


def test_realized_risk_to_reward_preserves_initial_stop_and_averages_all_legs():
    """Verify realized R:R preserves the initial stop loss and computes quantity-weighted average exit price."""
    engine = PaperTradingEngine(default_lots=2, slippage_points=0.5)  # 2 lots = 130 qty
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        t2_premium=140.0,
        sl_premium=85.0,  # Initial risk = 100.50 - 85.0 = 15.50 pts
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.initial_stop_loss_premium == 85.0

    # 1. Hit Target 1: books 65 at 120.0
    tick_t1 = Tick(
        symbol=pos.symbol,
        ltp=120.5,
        bid=120.5,
        ask=121.0,
        volume=2000,
        local_timestamp=entry_ts + timedelta(minutes=5),
    )
    engine.on_tick(tick_t1)
    assert pos.is_t1_hit is True
    # Stop loss is moved to breakeven (101.50), but initial_stop_loss_premium MUST remain 85.0!
    assert pos.stop_loss_premium == 101.50
    assert pos.initial_stop_loss_premium == 85.0

    # 2. Hit Target 2: books remaining 65 at 139.50
    tick_t2 = Tick(
        symbol=pos.symbol,
        ltp=140.5,
        bid=140.0,
        ask=140.5,
        volume=5000,
        local_timestamp=entry_ts + timedelta(minutes=15),
    )
    engine.on_tick(tick_t2)
    assert pos.status == PositionStatus.CLOSED

    # Average exit price = (65 * 120.0 + 65 * 139.50) / 130 = 129.75
    assert pos.exit_price == 129.75
    # Initial risk = 100.50 - 85.0 = 15.50 pts
    # Average gain = 129.75 - 100.50 = 29.25 pts
    # Realized R:R = 29.25 / 15.50 = 1.89R
    expected_rr = round(29.25 / 15.50, 2)
    assert pos.realized_rr == expected_rr


def test_duckdb_recovery_on_engine_restart():
    """Verify a newly initialized engine recovers active positions, closed positions, and cumulative P&L."""
    store = DuckDBStore(db_path=":memory:")
    engine1 = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)

    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand = create_mock_candidate(
        candidate_id="RECOVERY_TEST_1",
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        t2_premium=140.0,
    )
    pos1 = engine1.process_technical_candidate(cand, current_time=entry_ts)
    assert pos1 is not None

    # Execute full exit on engine1
    tick_t2 = Tick(
        symbol=pos1.symbol,
        ltp=140.5,
        bid=140.0,
        ask=140.5,
        volume=5000,
        local_timestamp=entry_ts + timedelta(minutes=10),
    )
    engine1.on_tick(tick_t2)
    assert pos1.status == PositionStatus.CLOSED
    pnl1 = engine1.total_realized_net_pnl
    assert pnl1 > 0.0

    # Simulate restart: create brand-new engine2 connected to the same DuckDB store!
    engine2 = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)

    # Verify state was restored
    assert len(engine2.get_closed_positions()) == 1
    recovered_pos = engine2.get_closed_positions()[0]
    assert recovered_pos.signal_id == "RECOVERY_TEST_1"
    assert recovered_pos.exit_reason == ExitReason.TARGET_2
    assert engine2.total_realized_net_pnl == pnl1
    assert engine2.get_account_summary()["total_trades"] == 1
    assert engine2.get_account_summary()["winning_trades"] == 1

    store.close()


def test_eod_square_off_at_1515_ist():
    """Verify open positions automatically square off at 15:15 IST."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))

    cand = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.status == PositionStatus.OPEN

    # Time event at 15:15:02 IST
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 2))
    events = engine.check_time_events(eod_ts)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.EOD_SQUARE_OFF
    assert p.status == PositionStatus.CLOSED
    assert len(engine.get_open_positions()) == 0


def test_put_option_spot_sl_and_intrabar_boundaries():
    """Verify Put option triggers spot SL when underlying index surges upward."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    # Bearish Put: Spot Entry = 25100, Spot SL = 25130 (higher than entry), Spot Targets = 25060, 25020
    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25130.0,  # Index surge stops Put
        target_1=25060.0,
        target_2=25020.0,
        opt_symbol="NSE:NIFTY26NOV25100PE",
        entry_ltp=100.0,
        ask=100.0,
    )
    # Modify candidate direction & contract option_type to Put
    cand.direction = BreakoutDirection.BEARISH
    cand.derivatives_assessment.selected_contract.option_type = "PE"

    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.option_type == "PE"

    # Index surges to 25135 (> 25130 spot SL for Put) within a fresh 1m bar
    candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=15),
        open=25110.0,
        high=25135.0,  # Surged above spot SL!
        low=25095.0,
        close=25120.0,
        volume=50000,
        is_closed=True,
    )
    events = engine.on_candle(candle)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert p.status == PositionStatus.CLOSED


def test_quote_provider_on_demand_refresh():
    """Verify on-demand quote_provider refreshes stale option quote during spot trigger."""
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    def mock_quote_provider(symbol: str):
        return {"ltp": 82.0, "bid": 81.8, "ask": 82.2, "timestamp": entry_ts + timedelta(minutes=3)}

    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        max_quote_age_seconds=30.0,
        quote_provider=mock_quote_provider,
    )

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Spot breach 3 minutes later (option quote in memory is stale)
    spot_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25065.0,
        volume=10000,
        local_timestamp=entry_ts + timedelta(minutes=3),
    )
    events = engine.on_tick(spot_tick)
    # quote_provider was queried on-demand and returned fresh quote (bid=81.8)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 81.30  # 81.8 bid - 0.5 slippage
    assert p.status == PositionStatus.CLOSED


def test_holding_timeout_and_eod_stale_conservative_penalty():
    """Verify holding timeout triggers when max duration exceeded, and stale EOD receives penalty."""
    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        max_holding_minutes=45.0,
        max_quote_age_seconds=30.0,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Stale EOD event at 15:15:00 IST (quote is 1h15m old)
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 0))
    events = engine.check_time_events(eod_ts)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.EOD_SQUARE_OFF
    # Received conservative penalty (bid 99.8 - 2 * slippage 0.5 = 98.80)
    assert exit_price == 98.80
    assert "conservative penalty" in p.notes


def test_on_candle_spot_targets_call_and_put():
    """Verify on_candle handles spot Target 1 and Target 2 for both Call and Put positions."""
    engine = PaperTradingEngine(default_lots=2, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    # 1. Bullish Call setup
    cand_call = create_mock_candidate(
        candidate_id="CAND_CALL_BAR",
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        target_1=25140.0,
        target_2=25180.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos_call = engine.process_technical_candidate(cand_call, current_time=entry_ts)
    assert pos_call is not None

    # Spot candle hits Target 1 (high = 25145 >= 25140)
    c1 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=10),
        open=25105.0,
        high=25145.0,
        low=25100.0,
        close=25130.0,
        volume=20000,
        is_closed=True,
    )
    ev_c1 = engine.on_candle(c1)
    assert len(ev_c1) == 1
    assert ev_c1[0][1] == ExitReason.TARGET_1
    assert pos_call.remaining_quantity == 65

    # Spot candle hits Target 2 (high = 25185 >= 25180)
    c2 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=20),
        open=25135.0,
        high=25185.0,
        low=25130.0,
        close=25175.0,
        volume=30000,
        is_closed=True,
    )
    ev_c2 = engine.on_candle(c2)
    assert len(ev_c2) == 1
    assert ev_c2[0][1] == ExitReason.TARGET_2
    assert pos_call.status == PositionStatus.CLOSED

    # 2. Bearish Put setup
    cand_put = create_mock_candidate(
        candidate_id="CAND_PUT_BAR",
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25130.0,
        target_1=25060.0,
        target_2=25020.0,
        opt_symbol="NSE:NIFTY26NOV25100PE",
        entry_ltp=100.0,
        ask=100.0,
    )
    cand_put.direction = BreakoutDirection.BEARISH
    cand_put.derivatives_assessment.selected_contract.option_type = "PE"

    pos_put = engine.process_technical_candidate(cand_put, current_time=entry_ts)
    assert pos_put is not None

    # Put spot hits Target 2 directly (low = 25015 <= 25020)
    c_put = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=15),
        open=25090.0,
        high=25100.0,
        low=25015.0,
        close=25030.0,
        volume=40000,
        is_closed=True,
    )
    ev_put = engine.on_candle(c_put)
    assert len(ev_put) == 1
    assert ev_put[0][1] == ExitReason.TARGET_2
    assert pos_put.status == PositionStatus.CLOSED


def test_tick_freshness_uses_receipt_time_vs_exchange_time():
    """Verify tick freshness validation evaluates receipt time against exchange time."""
    engine = PaperTradingEngine(max_quote_age_seconds=30.0)
    receipt_ts = IST.localize(datetime(2026, 11, 26, 10, 5, 0))
    stale_exchange_ts = receipt_ts - timedelta(seconds=45)  # 45s latency > 30s max

    tick_stale = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25100.0,
        volume=1000,
        exchange_timestamp=stale_exchange_ts,
        local_timestamp=receipt_ts,
    )
    # Stale tick should be rejected
    events = engine.on_tick(tick_stale)
    assert events == []

    # Fresh tick (latency = 2s <= 30s)
    fresh_exchange_ts = receipt_ts - timedelta(seconds=2)
    tick_fresh = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25105.0,
        volume=1000,
        exchange_timestamp=fresh_exchange_ts,
        local_timestamp=receipt_ts,
    )
    # Process fresh tick (no positions open, returns empty list without dropping)
    events = engine.on_tick(tick_fresh)
    assert events == []


def test_duckdb_recovery_restores_pending_triggers_quotes_and_trades(tmp_path):
    """Verify DuckDB state recovery restores pending exit triggers, quotes, and execution trades."""
    db_file = str(tmp_path / "paper_recovery_hardening.duckdb")
    store = DuckDBStore(db_path=db_file)

    engine1 = PaperTradingEngine(store=store, default_lots=2, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 9, 30, 0))

    cand = create_mock_candidate(
        candidate_id="RECOVERY_HARDENED",
        timestamp=entry_ts,
        trigger_price=25100.0,
        target_1=25140.0,
        target_2=25180.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos1 = engine1.process_technical_candidate(cand, current_time=entry_ts)
    assert pos1 is not None

    # T1 triggered within fresh quote window -> partial exit
    spot_tick_t1 = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25145.0,
        volume=50000,
        local_timestamp=entry_ts + timedelta(seconds=10),
    )
    engine1.on_tick(spot_tick_t1)
    assert pos1.status == PositionStatus.PARTIALLY_CLOSED
    assert len(pos1.trades) == 2  # Buy + T1 Sell

    # Simulate pending exit trigger with stale quote
    pos1.is_exit_triggered = True
    pos1.pending_exit_reason = ExitReason.STOP_LOSS
    pos1.current_ltp = 105.0
    pos1.current_bid = 104.5
    pos1.current_ask = 105.5
    store.save_paper_position(pos1.to_dict())

    # Now create engine2 from the same DuckDB store
    engine2 = PaperTradingEngine(store=store, default_lots=2, slippage_points=0.5)
    recovered = engine2.get_open_positions()
    assert len(recovered) == 1
    pos2 = recovered[0]

    assert pos2.position_id == pos1.position_id
    assert pos2.status == PositionStatus.PARTIALLY_CLOSED
    assert pos2.is_exit_triggered is True
    assert pos2.pending_exit_reason == ExitReason.STOP_LOSS
    assert pos2.current_ltp == 105.0
    assert pos2.current_bid == 104.5
    assert pos2.current_ask == 105.5
    assert len(pos2.trades) == 2  # Reconstructed from paper orders!
    assert pos2.trades[0].side == OrderSide.BUY
    assert pos2.trades[1].side == OrderSide.SELL

    store.close()


def test_quote_provider_fallback_in_on_candle():
    """Verify on_candle queries quote_provider when quote is stale during spot breach."""
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    fetched = False

    def mock_quote_provider(symbol: str) -> Dict[str, Any]:
        nonlocal fetched
        fetched = True
        return {"ltp": 82.0, "bid": 81.5, "ask": 82.5, "timestamp": entry_ts + timedelta(seconds=60)}

    engine = PaperTradingEngine(
        quote_provider=mock_quote_provider,
        default_lots=1,
        slippage_points=0.5,
        max_quote_age_seconds=10.0,
    )

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Candle 60s later breaches spot SL
    stale_bar = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=60),
        open=25080.0,
        high=25085.0,
        low=25065.0,  # Breached 25070 SL
        close=25075.0,
        volume=20000,
        is_closed=True,
    )

    events = engine.on_candle(stale_bar)
    assert fetched is True
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 81.0  # 81.5 bid - 0.5 slippage
    assert p.status == PositionStatus.CLOSED


def test_market_data_engine_wires_quote_provider_and_candle_pipeline():
    """Verify MarketDataEngine constructor wires quote provider and on_candle properly."""
    from run_market_data import MarketDataEngine

    # Use in-memory DuckDB for test isolation
    store = DuckDBStore(db_path=":memory:")
    runner = MarketDataEngine(store=store)

    assert runner.paper_engine is not None
    assert runner.paper_engine.quote_provider is not None
    assert callable(runner.paper_engine.quote_provider)

    # Verify candle close forwards to paper_engine
    candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=IST.localize(datetime(2026, 11, 26, 10, 0, 0)),
        open=25100.0,
        high=25110.0,
        low=25095.0,
        close=25105.0,
        volume=1000,
        is_closed=True,
    )
    runner._on_candle_closed(candle)
    assert runner.candle_count == 1
    store.close()


def test_candle_timing_prevents_retroactive_trades():
    """Verify on_candle ignores higher timeframes and bars that started before position entry."""
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 10, 0))
    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        quote_provider=lambda sym: {"ltp": 80.0, "bid": 80.0, "ask": 80.5, "timestamp": entry_ts + timedelta(minutes=1)},
    )

    cand = create_mock_candidate(
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.entry_time == entry_ts

    # 1. 15m candle spanning 10:00 to 10:15 with breached SL -> MUST BE DROPPED
    bar_15m = Candle(
        symbol=pos.symbol,
        timeframe="15m",
        timestamp=entry_ts - timedelta(minutes=10),  # 10:00
        open=98.0,
        high=102.0,
        low=80.0,  # Below 85.0 SL
        close=95.0,
        volume=50000,
        is_closed=True,
    )
    events_15m = engine.on_candle(bar_15m)
    assert events_15m == []  # Not executed
    assert pos.status == PositionStatus.OPEN

    # 2. 1m candle starting at 10:09 (prior to 10:10 entry) closing at 10:10 -> MUST BE DROPPED
    bar_prior = Candle(
        symbol=pos.symbol,
        timeframe="1m",
        timestamp=entry_ts - timedelta(minutes=1),  # 10:09
        open=98.0,
        high=100.0,
        low=80.0,  # Below 85.0 SL
        close=95.0,
        volume=1000,
        is_closed=True,
    )
    events_prior = engine.on_candle(bar_prior)
    assert events_prior == []
    assert pos.status == PositionStatus.OPEN

    # 3. 1m candle strictly after entry (10:10 to 10:11) breaching SL -> EXECUTED at availability time (10:11)
    bar_valid = Candle(
        symbol=pos.symbol,
        timeframe="1m",
        timestamp=entry_ts,  # 10:10
        observed_at=entry_ts + timedelta(minutes=1),  # 10:11
        open=98.0,
        high=100.0,
        low=80.0,  # Below 85.0 SL
        close=82.0,
        volume=1000,
        is_closed=True,
    )
    events_valid = engine.on_candle(bar_valid)
    assert len(events_valid) == 1
    p, reason, exit_price = events_valid[0]
    assert reason == ExitReason.STOP_LOSS
    assert p.exit_time == entry_ts + timedelta(minutes=1)  # Availability time!
    assert p.status == PositionStatus.CLOSED


def test_atm_rotation_derives_protected_symbols_from_paper_engine():
    """Verify check_and_rotate_options preserves open paper position contracts from being unsubscribed."""
    from run_market_data import MarketDataEngine

    store = DuckDBStore(db_path=":memory:")
    runner = MarketDataEngine(store=store)

    entry_ts = IST.localize(datetime(2026, 11, 26, 9, 30, 0))
    trade_symbol = "NSE:NIFTY26NOV25000CE"
    cand = create_mock_candidate(
        opt_symbol=trade_symbol,
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = runner.paper_engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.symbol == trade_symbol

    # Mock initial runner state
    runner.current_spot_price = 25000.0
    runner.active_atm_strike = 25000.0
    runner.subscribed_option_symbols = {trade_symbol, "NSE:NIFTY26NOV25000PE"}

    # Mock unsubscriptions tracker
    unsubscribed_symbols = []
    runner.ws_client.unsubscribe = lambda syms: unsubscribed_symbols.extend(syms)

    # Mock instrument master resolution
    runner.master.get_active_expiries = lambda und: [entry_ts.date()]
    runner.master.resolve_option = lambda und, exp, strk, opt_type: OptionContract(
        symbol=f"NSE:NIFTY26NOV{int(strk)}{opt_type.value}",
        fytoken="dummy",
        underlying="NIFTY",
        expiry=exp,
        strike=strk,
        option_type=opt_type,
    )

    # Shift spot significantly: 25000 -> 25300 so 25000 is no longer in ATM +/- 50 strikes
    runner.current_spot_price = 25300.0
    runner.check_and_rotate_options()

    # trade_symbol MUST NOT be in unsubscribed_symbols because it is protected by open paper positions
    assert trade_symbol not in unsubscribed_symbols
    assert trade_symbol in runner.subscribed_option_symbols
    assert trade_symbol in runner.active_paper_trade_symbols
    store.close()


def test_one_lot_target1_and_pending_triggers_survive_restart(tmp_path):
    """Verify 1-lot Target 1 breakeven SL and pending exit triggers survive engine restart."""
    db_file = str(tmp_path / "paper_1lot_pending.duckdb")
    store = DuckDBStore(db_path=db_file)

    engine1 = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 9, 30, 0))

    cand = create_mock_candidate(
        candidate_id="ONE_LOT_RESTART",
        timestamp=entry_ts,
        trigger_price=25100.0,
        target_1=25140.0,
        target_2=25180.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos1 = engine1.process_technical_candidate(cand, current_time=entry_ts)
    assert pos1 is not None
    assert pos1.initial_quantity == 65  # 1 lot
    initial_sl = pos1.stop_loss_premium

    # 1. Reach Target 1 within fresh quote window
    spot_tick_t1 = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25145.0,
        volume=50000,
        local_timestamp=entry_ts + timedelta(seconds=10),
    )
    events_t1 = engine1.on_tick(spot_tick_t1)
    assert len(events_t1) == 1
    assert events_t1[0][1] == ExitReason.TARGET_1
    assert pos1.is_t1_hit is True
    assert pos1.is_breakeven_sl_active is True
    assert pos1.stop_loss_premium >= pos1.entry_price + 1.0
    assert pos1.stop_loss_premium > initial_sl

    # 2. Simulate restart: verify 1-lot position restores is_t1_hit, is_breakeven_sl_active, and modified SL
    engine2 = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)
    open_pos = engine2.get_open_positions()
    assert len(open_pos) == 1
    pos2 = open_pos[0]
    assert pos2.position_id == pos1.position_id
    assert pos2.is_t1_hit is True
    assert pos2.is_breakeven_sl_active is True
    assert pos2.stop_loss_premium == pos1.stop_loss_premium

    # 3. Simulate pending trigger when quote is stale
    stale_time = entry_ts + timedelta(minutes=10)
    spot_tick_sl = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25050.0,  # Breached spot SL
        volume=50000,
        local_timestamp=stale_time,
    )
    events_sl = engine2.on_tick(spot_tick_sl)
    assert len(events_sl) == 0  # Awaiting fresh quote
    assert pos2.is_exit_triggered is True
    assert pos2.pending_exit_reason == ExitReason.TRAILING_STOP

    # 4. Simulate second restart: verify pending exit trigger is preserved in DuckDB
    engine3 = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)
    open_pos3 = engine3.get_open_positions()
    assert len(open_pos3) == 1
    pos3 = open_pos3[0]
    assert pos3.is_exit_triggered is True
    assert pos3.pending_exit_reason == ExitReason.TRAILING_STOP

    store.close()


def test_recovered_open_positions_included_in_runner_initial_watchlist():
    """Verify runner _build_watchlist includes symbols for recovered open positions outside current ATM."""
    from run_market_data import MarketDataEngine

    store = DuckDBStore(db_path=":memory:")
    # Seed an open position at an older strike (24800) into DuckDB
    engine_seed = PaperTradingEngine(store=store, default_lots=1)
    entry_ts = IST.localize(datetime(2026, 11, 26, 9, 30, 0))
    cand = create_mock_candidate(
        candidate_id="RECOVERED_STRIKE_TEST",
        opt_symbol="NSE:NIFTY26NOV24800CE",
        timestamp=entry_ts,
        entry_ltp=150.0,
        ask=150.0,
    )
    pos_seed = engine_seed.process_technical_candidate(cand, current_time=entry_ts)
    assert pos_seed is not None

    # Simulate fresh runner restart
    runner = MarketDataEngine(store=store)
    open_pos = runner.paper_engine.get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0].symbol == "NSE:NIFTY26NOV24800CE"

    # Mock spot at 25100 (so 24800 is NOT in ATM strikes [25050, 25100, 25150])
    runner.current_spot_price = 25100.0
    runner.master.get_active_expiries = lambda und: [entry_ts.date()]
    runner.master.resolve_option = lambda und, exp, strk, opt_type: OptionContract(
        symbol=f"NSE:NIFTY26NOV{int(strk)}{opt_type.value}",
        fytoken="dummy",
        underlying="NIFTY",
        expiry=exp,
        strike=strk,
        option_type=opt_type,
    )

    watchlist = runner._build_watchlist()

    # The recovered position's symbol MUST be in watchlist, subscribed_option_symbols, and active_paper_trade_symbols
    assert "NSE:NIFTY26NOV24800CE" in watchlist
    assert "NSE:NIFTY26NOV24800CE" in runner.subscribed_option_symbols
    assert "NSE:NIFTY26NOV24800CE" in runner.active_paper_trade_symbols
    store.close()


def test_candle_closing_does_not_refresh_quote_timestamp():
    """Verify on_candle tracks last_candle_time and does not falsely refresh last_quote_time."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5, max_quote_age_seconds=30.0)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Authentic broker option tick arrives at 10:00:05
    opt_tick = Tick(
        symbol=pos.symbol,
        ltp=100.5,
        bid=100.0,
        ask=101.0,
        volume=5000,
        local_timestamp=entry_ts + timedelta(seconds=5),
    )
    engine.on_tick(opt_tick)
    assert pos.last_quote_time == entry_ts + timedelta(seconds=5)

    # 1m option candle closes at 10:01:05 (60 seconds after option tick)
    candle_time = entry_ts + timedelta(seconds=65)
    opt_candle = Candle(
        symbol=pos.symbol,
        timeframe="1m",
        timestamp=entry_ts + timedelta(seconds=5),
        observed_at=candle_time,
        open=100.5,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=10000,
        is_closed=True,
    )
    engine.on_candle(opt_candle)

    # pos.last_candle_time should be updated, but last_quote_time MUST NOT be refreshed to candle_time
    assert pos.last_candle_time == candle_time
    assert pos.last_quote_time == entry_ts + timedelta(seconds=5)  # Preserved!
    # Quote age at candle_time is 60 seconds (stale, > 30s max)
    assert pos.quote_age_seconds(candle_time) == 60.0

    # Spot SL breach arrives at candle_time + 1s. Because quote is 61s old, it must NOT execute against stale bid!
    spot_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25060.0,  # Below 25070 SL
        volume=50000,
        local_timestamp=candle_time + timedelta(seconds=1),
    )
    events = engine.on_tick(spot_tick)
    assert len(events) == 0  # Does not execute stale fill!
    assert pos.is_exit_triggered is True
    assert pos.pending_exit_reason == ExitReason.STOP_LOSS


def test_ltp_only_ticks_do_not_refresh_bid_freshness():
    """Verify that LTP-only option ticks do not refresh last_bid_time or make stale bids executable."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5, max_quote_age_seconds=30.0)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    # Initially, bid from candidate is 99.8
    assert pos.current_bid == 99.8
    assert pos.last_bid_time == entry_ts
    assert pos.has_fresh_bid(entry_ts, 30.0) is True

    # At 10:01:30 (90 seconds later), pos.last_bid_time is now 90s old
    t1 = entry_ts + timedelta(seconds=90)
    assert pos.has_fresh_bid(t1, 30.0) is False

    # An LTP-only tick arrives with LTP 80.0 (below stop-loss premium)
    # bid is None!
    ltp_tick = Tick(
        symbol=pos.symbol,
        ltp=80.0,
        bid=None,
        ask=None,
        volume=12000,
        local_timestamp=t1,
    )
    events = engine.on_tick(ltp_tick)

    # 1. LTP should be updated, and last_quote_time refreshed to t1
    assert pos.current_ltp == 80.0
    assert pos.last_quote_time == t1

    # 2. But last_bid_time MUST NOT be refreshed; bid_fresh must still be False!
    assert pos.last_bid_time == entry_ts
    assert pos.has_fresh_bid(t1, 30.0) is False

    # 3. Simulated sell fill must NOT execute against outdated 99.8 bid!
    # Instead, exit must be marked as pending awaiting fresh bid
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN
    assert pos.is_exit_triggered is True
    assert pos.pending_exit_reason == ExitReason.STOP_LOSS

    # 4. Now a fresh tick with authentic bid arrives at 10:01:32
    t2 = entry_ts + timedelta(seconds=92)
    fresh_tick = Tick(
        symbol=pos.symbol,
        ltp=79.5,
        bid=79.0,
        ask=80.0,
        volume=15000,
        local_timestamp=t2,
    )
    events2 = engine.on_tick(fresh_tick)

    # 5. Pending exit should execute against the fresh bid of 79.0 (minus 0.5 slippage = 78.50)
    assert len(events2) == 1
    p, reason, exit_price = events2[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 78.50
    assert p.status == PositionStatus.CLOSED
    assert p.is_exit_triggered is False
    assert p.pending_exit_reason is None
    assert p.last_bid_time == t2


def test_duckdb_stores_and_recovers_bid_ask_timestamps(tmp_path):
    """Verify last_bid_time and last_ask_time persist and recover accurately in DuckDB."""
    db_file = str(tmp_path / "test_bid_ask_recovery.duckdb")
    store = DuckDBStore(db_path=db_file)

    engine = PaperTradingEngine(
        store=store,
        default_lots=1,
        slippage_points=0.5,
        max_quote_age_seconds=30.0,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Option tick updates bid and ask and automatically checkpoints to DuckDB
    bid_time = entry_ts + timedelta(seconds=15)
    tick = Tick(
        symbol=pos.symbol,
        ltp=101.0,
        bid=100.5,
        ask=101.5,
        volume=1000,
        local_timestamp=bid_time,
    )
    engine.on_tick(tick)
    assert pos.last_bid_time == bid_time
    assert pos.last_ask_time == bid_time
    # Automatic controlled-interval checkpointing eliminates need for manual store.save_paper_position!

    # Recover in new engine instance
    engine2 = PaperTradingEngine(
        store=store,
        default_lots=1,
        slippage_points=0.5,
        max_quote_age_seconds=30.0,
    )
    recovered_positions = engine2.get_open_positions()
    assert len(recovered_positions) == 1
    rec_pos = recovered_positions[0]
    assert rec_pos.last_bid_time == bid_time
    assert rec_pos.last_ask_time == bid_time
    assert rec_pos.current_bid == 100.5
    assert rec_pos.has_fresh_bid(bid_time + timedelta(seconds=10), 30.0) is True

    store.close()


def test_on_candle_stale_bid_defers_to_pending_exit():
    """Verify on_candle detects intrabar trigger on low, but defers simulated sell if bid is stale."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5, max_quote_age_seconds=30.0)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Candle closes at 10:01:00 (60s after entry). Bid from candidate is 60s old (> 30s max)
    candle_ts = entry_ts + timedelta(minutes=1)
    bar = Candle(
        symbol=pos.symbol,
        timeframe="1m",
        timestamp=candle_ts,
        observed_at=candle_ts,
        open=95.0,
        high=98.0,
        low=83.0,  # Below 85.0 SL!
        close=86.0,
        volume=5000,
        is_closed=True,
    )
    # on_candle must NOT fill against stale bid or arbitrary candle.low
    events = engine.on_candle(bar)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN
    assert pos.is_exit_triggered is True
    assert pos.pending_exit_reason == ExitReason.STOP_LOSS

    # When fresh option tick with authentic bid arrives, pending exit executes against fresh bid
    fresh_tick = Tick(
        symbol=pos.symbol,
        ltp=84.0,
        bid=83.5,
        ask=84.5,
        volume=6000,
        local_timestamp=candle_ts + timedelta(seconds=2),
    )
    events2 = engine.on_tick(fresh_tick)
    assert len(events2) == 1
    p, reason, exit_price = events2[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 83.0  # 83.5 bid - 0.5 slippage
    assert p.status == PositionStatus.CLOSED
    assert p.is_exit_triggered is False
    assert p.pending_exit_reason is None


def test_rest_quote_fallback_rejects_missing_or_stale_bid():
    """Verify on-demand quote provider rejects missing/zero bids and stale broker timestamps."""
    current_mock_quote = {"ltp": 80.0, "bid": 0.0, "ask": 81.0, "timestamp": None}

    def mock_quote_provider(symbol: str):
        return current_mock_quote

    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        max_quote_age_seconds=30.0,
        quote_provider=mock_quote_provider,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
        sl_premium=85.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Trigger time: 10:02:00 (quote from entry is stale, 120s old)
    trigger_time = entry_ts + timedelta(minutes=2)

    # 1. Quote provider returns zero bid (missing bid from broker): REJECTED
    current_mock_quote = {"ltp": 80.0, "bid": 0.0, "ask": 81.0, "timestamp": trigger_time}
    spot_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25060.0,  # Below spot SL
        volume=1000,
        local_timestamp=trigger_time,
    )
    events = engine.on_tick(spot_tick)
    assert len(events) == 0
    assert pos.is_exit_triggered is True

    # 2. Quote provider returns positive bid but broker timestamp is 60s stale (> 30s): REJECTED
    current_mock_quote = {
        "ltp": 80.0,
        "bid": 79.5,
        "ask": 80.5,
        "timestamp": trigger_time - timedelta(seconds=60),
    }
    events = engine.on_tick(spot_tick)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 3. Quote provider returns positive bid but broker timestamp is missing (None): REJECTED
    current_mock_quote = {
        "ltp": 80.0,
        "bid": 79.5,
        "ask": 80.5,
        "timestamp": None,
    }
    events = engine.on_tick(spot_tick)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 4. Quote provider returns positive bid but broker timestamp is materially future-dated (10s > 2s skew tolerance): REJECTED
    current_mock_quote = {
        "ltp": 80.0,
        "bid": 79.5,
        "ask": 80.5,
        "timestamp": trigger_time + timedelta(seconds=10),
    }
    events = engine.on_tick(spot_tick)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 5. Quote provider returns authentic positive bid with fresh timestamp (5s old): ACCEPTED
    current_mock_quote = {
        "ltp": 80.0,
        "bid": 79.5,
        "ask": 80.5,
        "timestamp": trigger_time - timedelta(seconds=5),
    }
    events = engine.on_tick(spot_tick)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert exit_price == 79.0  # 79.5 bid - 0.5 slippage
    assert p.status == PositionStatus.CLOSED


def test_checkpoint_does_not_redundantly_write_unchanged_positions():
    """Verify checkpointing avoids redundant DuckDB writes when position quote state is unchanged."""
    save_count = 0

    class MockStore(DuckDBStore):
        def __init__(self):
            super().__init__(db_path=":memory:")

        def save_paper_position(self, pos_dict):
            nonlocal save_count
            save_count += 1
            return super().save_paper_position(pos_dict)

    store = MockStore()
    engine = PaperTradingEngine(store=store, default_lots=1, checkpoint_interval_seconds=1.0)
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))

    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    # 1 save from initial candidate creation
    assert save_count == 1

    # Call checkpoint multiple times at 10:00:05, 10:00:10, 10:00:15 without new quotes
    for secs in (5, 10, 15):
        engine.checkpoint_active_positions(
            force=False,
            current_time=entry_ts + timedelta(seconds=secs),
        )
    # save_count MUST remain 1 because no new quote/candle telemetry arrived!
    assert save_count == 1

    # Now a new option tick arrives at 10:00:20
    tick_time = entry_ts + timedelta(seconds=20)
    tick = Tick(
        symbol=pos.symbol,
        ltp=100.5,
        bid=100.0,
        ask=101.0,
        volume=1000,
        local_timestamp=tick_time,
    )
    engine.on_tick(tick)
    # Because quote state mutated and > 1.0s elapsed, checkpoint triggers exactly once
    assert save_count == 2

    # Subsequent checkpoints with no new quotes do NOT re-write
    for secs in (25, 30):
        engine.checkpoint_active_positions(
            force=False,
            current_time=entry_ts + timedelta(seconds=secs),
        )
    assert save_count == 2
    store.close()


def test_weekend_and_holiday_guards_prevent_eod_square_off():
    """Verify recovered/open positions do not square off on weekends, holidays, or after-hours."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))  # Thursday

    cand = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.status == PositionStatus.OPEN

    # 1. After-hours on Thursday (22:00 IST) - Market closed
    after_hours_ts = IST.localize(datetime(2026, 11, 26, 22, 0, 0))
    events = engine.check_time_events(after_hours_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 2. Gazetted exchange holiday (Tuesday, 2026-11-24 Guru Nanak Jayanti at 15:15:05)
    holiday_ts = IST.localize(datetime(2026, 11, 24, 15, 15, 5))
    events = engine.check_time_events(holiday_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 3. Saturday (2026-11-28 at 15:15:05)
    saturday_ts = IST.localize(datetime(2026, 11, 28, 15, 15, 5))
    events = engine.check_time_events(saturday_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 4. Sunday (2026-11-29 at 15:15:05)
    sunday_ts = IST.localize(datetime(2026, 11, 29, 15, 15, 5))
    events = engine.check_time_events(sunday_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # 5. Active trading session at 15:15:05 IST on Thursday
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 5))
    events = engine.check_time_events(eod_ts)
    assert len(events) == 1
    assert events[0][1] == ExitReason.EOD_SQUARE_OFF
    assert pos.status == PositionStatus.CLOSED


def test_special_session_uses_custom_square_off_time():
    """Verify special session (e.g. Muhurat Trading 18:15-19:15) uses its own closing time."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    # Muhurat trading date: 2026-11-08, session 18:15 - 19:15
    entry_ts = IST.localize(datetime(2026, 11, 8, 18, 20, 0))

    cand = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # During afternoon (15:15 IST) market is closed for Muhurat day
    afternoon_ts = IST.localize(datetime(2026, 11, 8, 15, 15, 0))
    events = engine.check_time_events(afternoon_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # Mid-session (18:30 IST) - 15m before 19:15 close is 19:00, so 18:30 should not square off
    mid_ts = IST.localize(datetime(2026, 11, 8, 18, 30, 0))
    events = engine.check_time_events(mid_ts)
    assert len(events) == 0
    assert pos.status == PositionStatus.OPEN

    # Cutoff time (19:01 IST) - should square off
    cutoff_ts = IST.localize(datetime(2026, 11, 8, 19, 1, 0))
    events = engine.check_time_events(cutoff_ts)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.EOD_SQUARE_OFF
    assert p.status == PositionStatus.CLOSED


def test_distinguish_estimated_eod_fill_from_authenticated_fill():
    """Verify paper engine differentiates authentic broker bid fills from penalty estimated fills."""
    # 1. Authenticated broker bid fill
    engine_auth = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        quote_provider=lambda sym: {
            "ltp": 95.0,
            "bid": 94.8,
            "ask": 95.2,
            "timestamp": IST.localize(datetime(2026, 11, 26, 15, 15, 0)),
        },
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))
    cand1 = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos1 = engine_auth.process_technical_candidate(cand1, current_time=entry_ts)
    assert pos1 is not None

    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 2))
    events1 = engine_auth.check_time_events(eod_ts)
    assert len(events1) == 1
    p1, reason1, exit_price1 = events1[0]
    assert reason1 == ExitReason.EOD_SQUARE_OFF
    assert p1.is_estimated_exit is False
    assert p1.trades[-1].is_estimated_fill is False
    assert exit_price1 == 94.30  # 94.8 bid - 0.5 slippage
    assert "authenticated broker bid" in p1.notes

    # 2. Estimated penalty fallback fill (quote stale, quote provider None)
    engine_est = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    cand2 = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos2 = engine_est.process_technical_candidate(cand2, current_time=entry_ts)
    assert pos2 is not None

    events2 = engine_est.check_time_events(eod_ts)
    assert len(events2) == 1
    p2, reason2, exit_price2 = events2[0]
    assert reason2 == ExitReason.EOD_SQUARE_OFF
    assert p2.is_estimated_exit is True
    assert p2.trades[-1].is_estimated_fill is True
    assert "conservative penalty" in p2.notes


def test_duckdb_recovery_restores_estimated_exit_state(tmp_path):
    """Verify DuckDB correctly persists and recovers is_estimated_exit state."""
    db_path = tmp_path / "test_estimated.duckdb"
    store = DuckDBStore(db_path=db_path)

    engine = PaperTradingEngine(
        store=store,
        default_lots=1,
        slippage_points=0.5,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))
    cand = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Trigger estimated EOD exit
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 2))
    events = engine.check_time_events(eod_ts)
    assert len(events) == 1
    assert pos.is_estimated_exit is True

    # Restart engine and recover from DuckDB
    engine_recovered = PaperTradingEngine(store=store)
    assert len(engine_recovered.closed_positions) == 1
    rec_pos = engine_recovered.closed_positions[0]
    assert rec_pos.position_id == pos.position_id
    assert rec_pos.is_estimated_exit is True
    assert rec_pos.trades[-1].is_estimated_fill is True
    store.close()


def test_future_dated_rest_quote_rejection_and_skew_tolerance():
    """Verify quote provider rejects materially future-dated quotes while permitting small clock-skew."""
    engine = PaperTradingEngine(
        default_lots=1,
        slippage_points=0.5,
        clock_skew_tolerance_seconds=2.0,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand = create_mock_candidate(timestamp=entry_ts, entry_ltp=100.0, ask=100.0)
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    current_eval_time = entry_ts + timedelta(minutes=5)

    # 1. Quote is 10 seconds in the future (skew 10s > 2.0s allowed) -> REJECTED
    engine.quote_provider = lambda sym: {
        "ltp": 99.0,
        "bid": 98.5,
        "ask": 99.5,
        "timestamp": current_eval_time + timedelta(seconds=10),
    }
    applied = engine._apply_fresh_quote(pos, current_eval_time)
    assert applied is False

    # 2. Quote is 1 second in the future (skew 1.0s <= 2.0s allowed tolerance) -> ACCEPTED
    engine.quote_provider = lambda sym: {
        "ltp": 99.0,
        "bid": 98.5,
        "ask": 99.5,
        "timestamp": current_eval_time + timedelta(seconds=1),
    }
    applied = engine._apply_fresh_quote(pos, current_eval_time)
    assert applied is True
    assert pos.current_bid == 98.5


def test_account_summary_separates_verified_and_all_trades():
    """Verify get_account_summary calculates verified-only metrics cleanly separated from estimated fills."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)

    # Position 1: Verified winning trade (closed via authentic broker bid at Target 2)
    t1_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand1 = create_mock_candidate(
        candidate_id="POS_1_VERIFIED",
        timestamp=t1_ts,
        entry_ltp=100.0,
        ask=100.0,
        t2_premium=130.0,
    )
    pos1 = engine.process_technical_candidate(cand1, current_time=t1_ts)
    assert pos1 is not None

    tick1 = Tick(
        symbol=pos1.symbol,
        ltp=131.0,
        bid=130.5,
        ask=131.5,
        volume=5000,
        local_timestamp=t1_ts + timedelta(minutes=5),
    )
    engine.on_tick(tick1)
    assert pos1.status == PositionStatus.CLOSED
    assert pos1.is_estimated_exit is False
    assert pos1.net_pnl > 0.0

    # Position 2: Stale EOD exit resulting in estimated fallback penalty fill
    t2_ts = IST.localize(datetime(2026, 11, 26, 14, 0, 0))
    cand2 = create_mock_candidate(
        candidate_id="POS_2_ESTIMATED",
        timestamp=t2_ts,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos2 = engine.process_technical_candidate(cand2, current_time=t2_ts)
    assert pos2 is not None

    # Stale EOD trigger (quote is 1h15m old, no quote provider)
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 2))
    events = engine.check_time_events(eod_ts)
    assert len(events) == 1
    assert pos2.status == PositionStatus.CLOSED
    assert pos2.is_estimated_exit is True

    # Account summary telemetry check
    summary = engine.get_account_summary()

    # Primary metrics strictly reflect verified-only trades (1 trade, 100% win rate)
    assert summary["total_trades"] == 1
    assert summary["winning_trades"] == 1
    assert summary["losing_trades"] == 0
    assert summary["win_rate_pct"] == 100.0
    assert summary["estimated_trades_count"] == 1

    # Verified-only sub-dictionary
    assert summary["verified_only"]["total_trades"] == 1
    assert summary["verified_only"]["winning_trades"] == 1
    assert summary["verified_only"]["win_rate_pct"] == 100.0

    # All-trades sub-dictionary contains both verified and estimated
    assert summary["all_trades"]["total_trades"] == 2
    assert summary["all_trades"]["winning_trades"] == 1
    assert summary["all_trades"]["losing_trades"] == 1
    assert summary["all_trades"]["win_rate_pct"] == 50.0


def test_order_level_estimated_fill_persistence_and_recovery(tmp_path):
    """Verify partial exit preserves is_estimated_fill=False even when final exit is estimated."""
    db_path = tmp_path / "test_order_level_recovery.duckdb"
    store = DuckDBStore(db_path=db_path)

    engine = PaperTradingEngine(
        store=store,
        default_lots=2,
        slippage_points=0.5,
    )
    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand = create_mock_candidate(
        candidate_id="MULTI_LOT_TEST",
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos.initial_quantity == 130  # 2 lots of 65

    # 1. Target 1 achieved with authentic broker bid -> partial exit order (is_estimated_fill=False)
    t1_tick = Tick(
        symbol=pos.symbol,
        ltp=121.0,
        bid=120.5,
        ask=121.5,
        volume=2000,
        local_timestamp=entry_ts + timedelta(minutes=5),
    )
    events_t1 = engine.on_tick(t1_tick)
    assert len(events_t1) == 1
    assert pos.status == PositionStatus.PARTIALLY_CLOSED
    assert pos.remaining_quantity == 65
    assert len(pos.trades) == 2
    assert pos.trades[1].is_estimated_fill is False

    # 2. Final exit at EOD with stale quote -> estimated penalty exit (is_estimated_fill=True)
    eod_ts = IST.localize(datetime(2026, 11, 26, 15, 15, 2))
    events_eod = engine.check_time_events(eod_ts)
    assert len(events_eod) == 1
    assert pos.status == PositionStatus.CLOSED
    assert pos.is_estimated_exit is True
    assert len(pos.trades) == 3
    assert pos.trades[2].is_estimated_fill is True

    # Check orders table in DuckDB
    orders = store.get_paper_orders(pos.position_id)
    assert len(orders) == 3
    # Order 0: BUY, Order 1: SELL T1 (is_estimated_fill=False), Order 2: SELL EOD (is_estimated_fill=True)
    assert orders[0]["is_estimated_fill"] is False
    assert orders[1]["is_estimated_fill"] is False
    assert orders[2]["is_estimated_fill"] is True

    # 3. Simulate process restart and recover from DuckDB
    engine_recovered = PaperTradingEngine(store=store)
    assert len(engine_recovered.closed_positions) == 1
    rec_pos = engine_recovered.closed_positions[0]
    assert rec_pos.position_id == pos.position_id
    assert rec_pos.is_estimated_exit is True
    assert len(rec_pos.trades) == 3

    # CRITICAL CHECK: Trade 1 (T1 partial exit) MUST remain is_estimated_fill=False upon recovery!
    assert rec_pos.trades[0].is_estimated_fill is False  # BUY
    assert rec_pos.trades[1].is_estimated_fill is False  # T1 partial exit (verified)
    assert rec_pos.trades[2].is_estimated_fill is True   # EOD exit (estimated)

    store.close()








