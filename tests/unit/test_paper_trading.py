"""Unit tests for Phase 4 Paper-Trading Execution Engine.

Validates simulated order fills, conservative slippage, multi-stage exits (Target 1, Target 2,
Stop Loss, Trailing Breakeven, Spot Invalidation, EOD 15:15 IST Square-Off),
Indian statutory costs (STT 0.15%, Brokerage, GST, Stamp Duty), and DuckDB persistence.
"""

from datetime import date, datetime

import pytz

from nifty_engine.analysis.derivatives.contract_selector import SelectedContract
from nifty_engine.analysis.derivatives.cost_model import ExecutionCosts, PremiumEstimate
from nifty_engine.analysis.derivatives.derivatives_assessment import DerivativesAssessment, DerivativesAssessmentType
from nifty_engine.analysis.derivatives.greeks import OptionGreeks
from nifty_engine.analysis.derivatives.oi_analysis import DirectionalBias
from nifty_engine.analysis.technical.breakouts import BreakoutDirection
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.interfaces import Tick
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.paper_trading.engine import PaperTradingEngine
from nifty_engine.paper_trading.models import ExitReason, PositionStatus

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
    assert pos.total_costs > 20.0  # Entry statutory charges deducted
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
    assert pos is not None
    assert pos.initial_quantity == 130
    assert pos.remaining_quantity == 130
    assert pos.entry_price == 100.50  # 100.0 ask + 0.5 slippage

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
    assert p.remaining_quantity == 65  # 50% booked!
    assert p.is_t1_hit is True
    assert p.is_breakeven_sl_active is True
    assert p.stop_loss_premium == p.entry_price + 1.0  # Breakeven SL locked in

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
    assert p2.net_pnl > 3500.0  # Subtracted STT 0.15% on sells + brokerage ₹60 + GST


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
    assert pos is not None
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
    assert pos.remaining_quantity == 65  # Intact
    assert pos.is_breakeven_sl_active is True
    assert pos.stop_loss_premium == 101.50  # 100.50 + 1.0 buffer

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
    # Protected from original 85.0 SL loss!


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
    assert pos is not None

    # Option collapses to 84.5
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


def test_emergency_exit_on_underlying_spot_sl_breach():
    """Verify underlying spot price breach forces immediate option position liquidation."""
    engine = PaperTradingEngine(default_lots=1, slippage_points=0.5)
    entry_ts = IST.localize(datetime(2026, 11, 26, 12, 0, 0))

    # Spot entry = 25100.0, Spot SL = 25070.0 (Bullish Call)
    cand = create_mock_candidate(
        timestamp=entry_ts,
        trigger_price=25100.0,
        stop_loss=25070.0,
        entry_ltp=100.0,
        ask=100.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None

    # Underlying index cracks down to 25068.0 (< 25070.0)
    index_tick = Tick(
        symbol="NSE:NIFTY50-INDEX",
        ltp=25068.0,
        volume=100000,
        local_timestamp=entry_ts,
    )
    events = engine.on_tick(index_tick)
    assert len(events) == 1
    p, reason, exit_price = events[0]
    assert reason == ExitReason.STOP_LOSS
    assert p.status == PositionStatus.CLOSED
    assert "Spot SL breached" in p.notes


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


def test_duckdb_store_journal_integration():
    """Verify DuckDBStore saves and retrieves paper orders, positions, and summary metrics."""
    store = DuckDBStore(db_path=":memory:")
    engine = PaperTradingEngine(store=store, default_lots=1, slippage_points=0.5)

    entry_ts = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    cand = create_mock_candidate(
        candidate_id="JOURNAL_TEST_1",
        timestamp=entry_ts,
        entry_ltp=100.0,
        ask=100.0,
        t1_premium=120.0,
        t2_premium=140.0,
    )
    pos = engine.process_technical_candidate(cand, current_time=entry_ts)
    assert pos is not None
    assert pos is not None

    # Check order persisted
    orders = store.get_paper_orders(position_id=pos.position_id)
    assert len(orders) == 1
    assert orders[0]["side"] == "BUY"
    assert orders[0]["filled_price"] == 100.50

    # Execute T2 exit
    tick_t2 = Tick(
        symbol=pos.symbol,
        ltp=140.5,
        bid=140.0,
        ask=140.5,
        volume=5000,
        local_timestamp=entry_ts,
    )
    engine.on_tick(tick_t2)

    # Check position in DuckDB
    positions_db = store.get_paper_positions(status="CLOSED")
    assert len(positions_db) == 1
    assert positions_db[0]["signal_id"] == "JOURNAL_TEST_1"
    assert positions_db[0]["exit_reason"] == "TARGET_2"
    assert positions_db[0]["gross_pnl"] > 2000.0

    summary = engine.get_account_summary()
    assert summary["total_trades"] == 1
    assert summary["winning_trades"] == 1
    assert summary["win_rate_pct"] == 100.0
    assert summary["total_realized_net_pnl"] > 0.0
    assert summary["total_costs_paid"] > 40.0

    store.close()
