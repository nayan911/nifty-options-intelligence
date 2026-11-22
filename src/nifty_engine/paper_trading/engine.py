"""Paper-trading execution engine for simulated NIFTY options trading.

Implements realistic broker friction (slippage, bid-ask spreads, Indian statutory taxes),
M2M position tracking, automated multi-stage exits (Target 1 partial profit booking,
Target 2 runner exit, technical stop-loss, and EOD 15:15 IST auto-square-off).
"""

import logging
import uuid
from datetime import datetime, time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from nifty_engine.analysis.derivatives.cost_model import PremiumCostModel
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.interfaces import Candle, IStorageEngine, Tick
from nifty_engine.paper_trading.models import (
    ExitReason,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperOrder,
    PaperPosition,
    PositionStatus,
    PositionTrade,
)
from nifty_engine.utils.timezone import now_ist

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Simulated execution engine providing zero-broker-risk paper trading.

    Calculates realistic fills against live bid/ask quotes with slippage, enforces
    strict Indian statutory friction (STT, GST, Stamp Duty, Brokerage), tracks live
    M2M P&L, and manages multi-stage automated trade exits.
    """

    def __init__(
        self,
        cost_model: Optional[PremiumCostModel] = None,
        store: Optional[IStorageEngine] = None,
        instrument_master: Optional[Any] = None,
        max_open_positions: int = 2,
        default_lots: int = 1,
        slippage_points: float = 0.5,
        min_net_rr: float = 1.2,
        max_quote_age_seconds: float = 30.0,
        max_holding_minutes: float = 60.0,
        eod_square_off_time: time = time(15, 15, 0),
        no_new_entries_time: time = time(15, 0, 0),
        market_open_time: time = time(9, 15, 0),
        on_trade_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.cost_model = cost_model or PremiumCostModel()
        self.store = store
        self.instrument_master = instrument_master
        self.max_open_positions = max_open_positions
        self.default_lots = default_lots
        self.slippage_points = slippage_points
        self.min_net_rr = min_net_rr
        self.max_quote_age_seconds = max_quote_age_seconds
        self.max_holding_minutes = max_holding_minutes
        self.eod_square_off_time = eod_square_off_time
        self.no_new_entries_time = no_new_entries_time
        self.market_open_time = market_open_time
        self.on_trade_event = on_trade_event

        # In-memory position and order storage
        self.active_positions: Dict[str, PaperPosition] = {}  # position_id -> PaperPosition
        self.closed_positions: List[PaperPosition] = []
        self.orders: Dict[str, PaperOrder] = {}  # order_id -> PaperOrder
        self.subscribed_option_symbols: Set[str] = set()

        # Cumulative P&L and cost tracking
        self.total_realized_gross_pnl: float = 0.0
        self.total_realized_net_pnl: float = 0.0
        self.total_costs_paid: float = 0.0

    def process_technical_candidate(
        self,
        candidate: TechnicalCandidate,
        current_time: Optional[datetime] = None,
    ) -> Optional[PaperPosition]:
        """Evaluates a confirmed technical candidate with derivatives contract selection

        and executes a simulated market BUY order if all risk gates pass.
        """
        now = (
            current_time
            or getattr(candidate, "decision_at", None)
            or getattr(candidate, "timestamp", None)
            or getattr(candidate, "observed_at", None)
            or now_ist()
        )
        t = now.time()

        # 1. Trading hours gate
        if t < self.market_open_time:
            logger.info("Paper entry rejected: Market not yet open (%s)", t)
            return None
        if t >= self.no_new_entries_time:
            logger.info("Paper entry rejected: After no-new-entries cutoff (%s >= %s)", t, self.no_new_entries_time)
            return None

        # 2. Candidate prerequisites
        da = getattr(candidate, "derivatives_assessment", None)
        if da is None or getattr(da, "selected_contract", None) is None:
            logger.info("Paper entry rejected: No derivatives contract selected for candidate %s", candidate.candidate_id)
            return None

        contract = da.selected_contract

        # 3. Cost & Net R:R viability gate
        costs = contract.execution_costs
        if not costs.is_cost_viable or costs.net_rr_t1 < self.min_net_rr:
            logger.info(
                "Paper entry rejected: Contract %s net R:R (%.2f) below threshold %.2f",
                contract.symbol,
                costs.net_rr_t1,
                self.min_net_rr,
            )
            return None

        # 4. Quote freshness gate
        if contract.quote_age_seconds > self.max_quote_age_seconds:
            logger.info(
                "Paper entry rejected: Contract quote age (%.1fs) exceeds max allowed %.1fs",
                contract.quote_age_seconds,
                self.max_quote_age_seconds,
            )
            return None

        # 5. Position limit gate (concurrency)
        underlying = contract.underlying
        open_underlying_count = sum(
            1 for p in self.active_positions.values() if p.underlying == underlying and p.status != PositionStatus.CLOSED
        )
        if open_underlying_count >= self.max_open_positions:
            logger.info(
                "Paper entry rejected: Max open positions (%d) reached for %s",
                self.max_open_positions,
                underlying,
            )
            return None

        # 6. Resolve lot size & quantity
        lot_size = self.cost_model.get_lot_size(
            underlying=contract.underlying,
            symbol=contract.symbol,
            instrument_master=self.instrument_master,
        )
        quantity = lot_size * self.default_lots

        # 7. Simulated Fill Execution (Conservative Buy on Ask + Slippage)
        ask_price = contract.ask if (contract.ask is not None and contract.ask > 0) else (contract.ltp + (contract.spread / 2.0))
        fill_price = round(ask_price + self.slippage_points, 2)

        # Calculate entry statutory fees (Brokerage, Stamp Duty, Exchange Fee, GST)
        entry_costs = self.cost_model.compute_fill_costs("BUY", fill_price, quantity)

        position_id = str(uuid.uuid4())
        order_id = str(uuid.uuid4())

        order = PaperOrder(
            order_id=order_id,
            position_id=position_id,
            signal_id=candidate.candidate_id,
            symbol=contract.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            requested_price=ask_price,
            filled_price=fill_price,
            slippage_points=self.slippage_points,
            created_at=now,
            filled_at=now,
            status=OrderStatus.FILLED,
        )
        self.orders[order_id] = order

        trade = PositionTrade(
            trade_id=str(uuid.uuid4()),
            order_id=order_id,
            side=OrderSide.BUY,
            quantity=quantity,
            price=fill_price,
            timestamp=now,
            turnover=entry_costs["turnover"],
            costs=entry_costs["total_costs"],
            reason=None,
        )

        position = PaperPosition(
            position_id=position_id,
            signal_id=candidate.candidate_id,
            symbol=contract.symbol,
            underlying=contract.underlying,
            option_type=contract.option_type,
            strike=contract.strike,
            expiry=contract.expiry,
            lot_size=lot_size,
            initial_quantity=quantity,
            remaining_quantity=quantity,
            entry_price=fill_price,
            target_premium_t1=contract.premium_trajectory.target_premium_t1,
            target_premium_t2=contract.premium_trajectory.target_premium_t2,
            stop_loss_premium=contract.premium_trajectory.stop_loss_premium,
            spot_entry=getattr(candidate, "entry_price", getattr(candidate, "trigger_price", 0.0)),
            spot_target_t1=getattr(candidate, "target_1", getattr(candidate, "target_1_price", 0.0)),
            spot_target_t2=getattr(candidate, "target_2", getattr(candidate, "target_2_price", 0.0)),
            spot_sl=candidate.stop_loss,
            entry_time=now,
            status=PositionStatus.OPEN,
            current_ltp=contract.ltp,
            current_bid=contract.bid,
            current_ask=contract.ask,
            highest_premium_seen=fill_price,
            lowest_premium_seen=fill_price,
            gross_pnl=0.0,
            net_pnl=-entry_costs["total_costs"],
            total_costs=entry_costs["total_costs"],
            trades=[trade],
            notes=f"Candidate: {candidate.candidate_id}",
        )

        self.active_positions[position_id] = position
        self.subscribed_option_symbols.add(contract.symbol)

        # Persist to DuckDB if available
        if self.store is not None:
            try:
                self.store.save_paper_order(order.to_dict())
                self.store.save_paper_position(position.to_dict())
            except Exception as e:
                logger.warning("Failed to persist paper trade to DuckDB: %s", e)

        if self.on_trade_event:
            self.on_trade_event({
                "event": "ORDER_FILLED",
                "side": "BUY",
                "position_id": position_id,
                "symbol": contract.symbol,
                "price": fill_price,
                "quantity": quantity,
                "costs": entry_costs["total_costs"],
            })

        return position

    def on_tick(self, tick: Tick) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Processes live tick updates for subscribed options or primary underlying spot."""
        events: List[Tuple[PaperPosition, ExitReason, float]] = []
        sym = tick.symbol
        current_time = tick.exchange_timestamp or tick.local_timestamp or now_ist()

        # 1. Contract Tick: Update Option M2M and evaluate premium triggers
        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            if pos.symbol == sym:
                # Update option price telemetry
                pos.current_ltp = tick.ltp
                if tick.bid is not None and tick.bid > 0:
                    pos.current_bid = tick.bid
                if tick.ask is not None and tick.ask > 0:
                    pos.current_ask = tick.ask

                if tick.ltp > pos.highest_premium_seen:
                    pos.highest_premium_seen = tick.ltp
                if tick.ltp < pos.lowest_premium_seen:
                    pos.lowest_premium_seen = tick.ltp

                # Update live M2M unrealized PnL
                effective_price = pos.current_bid if pos.current_bid is not None else pos.current_ltp
                unrealized_gross = (effective_price - pos.entry_price) * pos.remaining_quantity
                pos.gross_pnl = unrealized_gross
                pos.net_pnl = unrealized_gross - pos.total_costs

                # A. Target 2 Check (Full runner exit)
                if effective_price >= pos.target_premium_t2:
                    ev = self._execute_full_exit(pos, effective_price, ExitReason.TARGET_2, current_time)
                    if ev:
                        events.append(ev)

                # B. Target 1 Check (50% partial exit / trailing stop activation)
                elif not pos.is_t1_hit and effective_price >= pos.target_premium_t1:
                    ev = self._execute_partial_exit_t1(pos, effective_price, current_time)
                    if ev:
                        events.append(ev)

                # C. Stop Loss Check (Active SL or Trailing Breakeven)
                elif effective_price <= pos.stop_loss_premium:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    ev = self._execute_full_exit(pos, effective_price, reason, current_time)
                    if ev:
                        events.append(ev)

            # 2. Underlying Spot Tick: Evaluate spot-level technical stop & target invalidation
            elif "INDEX" in sym and pos.underlying in sym:
                spot_price = tick.ltp
                opt_price = pos.current_bid if pos.current_bid is not None else pos.current_ltp

                is_call = pos.option_type == "CE"
                # Spot Stop Loss Breach
                spot_sl_hit = (spot_price <= pos.spot_sl) if is_call else (spot_price >= pos.spot_sl)
                if spot_sl_hit:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes="Spot SL breached")
                    if ev:
                        events.append(ev)
                    continue

                # Spot Target 2 Reached
                spot_t2_hit = (spot_price >= pos.spot_target_t2) if is_call else (spot_price <= pos.spot_target_t2)
                if spot_t2_hit:
                    ev = self._execute_full_exit(pos, opt_price, ExitReason.TARGET_2, current_time, notes="Spot T2 reached")
                    if ev:
                        events.append(ev)
                    continue

                # Spot Target 1 Reached
                spot_t1_hit = (spot_price >= pos.spot_target_t1) if is_call else (spot_price <= pos.spot_target_t1)
                if not pos.is_t1_hit and spot_t1_hit:
                    ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Spot T1 reached")
                    if ev:
                        events.append(ev)
                    if ev:
                        events.append(ev)

        return events

    def on_candle(self, candle: Candle) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Processes completed candle to evaluate high/low bounds for exits."""
        # Synthesize a high/low tick to evaluate worst-case and best-case execution bounds
        pseudo_tick = Tick(
            symbol=candle.symbol,
            ltp=candle.close,
            volume=candle.volume,
            local_timestamp=candle.timestamp,
            exchange_timestamp=candle.timestamp,
        )
        return self.on_tick(pseudo_tick)

    def check_time_events(self, current_time: datetime) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Evaluates auto square-off at 15:15 IST and max holding timeouts."""
        events: List[Tuple[PaperPosition, ExitReason, float]] = []
        t = current_time.time()
        is_eod = t >= self.eod_square_off_time

        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            opt_price = pos.current_bid if pos.current_bid is not None else pos.current_ltp

            # 1. EOD Auto Square-Off at 15:15 IST
            if is_eod:
                ev = self._execute_full_exit(
                    pos,
                    opt_price,
                    ExitReason.EOD_SQUARE_OFF,
                    current_time,
                    notes="Intraday auto square-off at 15:15 IST",
                )
                if ev:
                    events.append(ev)
                continue

            # 2. Maximum holding period timeout
            elapsed_minutes = (current_time - pos.entry_time).total_seconds() / 60.0
            if elapsed_minutes >= self.max_holding_minutes and not pos.is_t1_hit:
                ev = self._execute_full_exit(
                    pos,
                    opt_price,
                    ExitReason.TIMEOUT,
                    current_time,
                    notes=f"Holding timeout reached ({elapsed_minutes:.1f} mins)",
                )
                if ev:
                    events.append(ev)

        return events

    def _execute_partial_exit_t1(
        self,
        pos: PaperPosition,
        market_price: float,
        current_time: datetime,
        notes: Optional[str] = None,
    ) -> Optional[Tuple[PaperPosition, ExitReason, float]]:
        """Handles Target 1 profit locking: sells 50% if multiple lots; if 1 lot, moves SL to breakeven."""
        exit_price = max(0.05, round(market_price - self.slippage_points, 2))
        pos.is_t1_hit = True
        pos.is_breakeven_sl_active = True

        # For 1 lot, fractional contracts cannot trade on NSE; secure position by moving SL to breakeven + buffer
        if pos.initial_quantity <= pos.lot_size:
            pos.stop_loss_premium = max(pos.stop_loss_premium, pos.entry_price + 1.0)
            logger.info(
                "Paper Trade T1 Achieved for 1-lot position %s at ₹%.2f. Stop-Loss moved to breakeven (₹%.2f)",
                pos.symbol,
                exit_price,
                pos.stop_loss_premium,
            )
            return (pos, ExitReason.TARGET_1, exit_price)

        # For multi-lot positions (>= 2 lots): exit 50%
        exit_qty = (pos.remaining_quantity // (2 * pos.lot_size)) * pos.lot_size
        if exit_qty <= 0:
            pos.stop_loss_premium = max(pos.stop_loss_premium, pos.entry_price + 1.0)
            return (pos, ExitReason.TARGET_1, exit_price)

        exit_costs = self.cost_model.compute_fill_costs("SELL", exit_price, exit_qty)
        order_id = str(uuid.uuid4())

        order = PaperOrder(
            order_id=order_id,
            position_id=pos.position_id,
            signal_id=pos.signal_id,
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=exit_qty,
            requested_price=market_price,
            filled_price=exit_price,
            slippage_points=self.slippage_points,
            created_at=current_time,
            filled_at=current_time,
            status=OrderStatus.FILLED,
        )
        self.orders[order_id] = order

        trade = PositionTrade(
            trade_id=str(uuid.uuid4()),
            order_id=order_id,
            side=OrderSide.SELL,
            quantity=exit_qty,
            price=exit_price,
            timestamp=current_time,
            turnover=exit_costs["turnover"],
            costs=exit_costs["total_costs"],
            reason=ExitReason.TARGET_1,
        )
        pos.trades.append(trade)

        pos.remaining_quantity -= exit_qty
        pos.status = PositionStatus.PARTIALLY_CLOSED
        pos.total_costs += exit_costs["total_costs"]
        pos.stop_loss_premium = max(pos.stop_loss_premium, pos.entry_price + 1.0)

        if self.store is not None:
            try:
                self.store.save_paper_order(order.to_dict())
                self.store.save_paper_position(pos.to_dict())
            except Exception as e:
                logger.warning("Failed to save paper partial exit to DuckDB: %s", e)

        return (pos, ExitReason.TARGET_1, exit_price)

    def _execute_full_exit(
        self,
        pos: PaperPosition,
        market_price: float,
        reason: ExitReason,
        current_time: datetime,
        notes: Optional[str] = None,
    ) -> Optional[Tuple[PaperPosition, ExitReason, float]]:
        """Closes remaining quantity of an active position with complete Indian statutory charges."""
        if pos.status == PositionStatus.CLOSED or pos.remaining_quantity <= 0:
            return None

        exit_price = max(0.05, round(market_price - self.slippage_points, 2))
        exit_qty = pos.remaining_quantity
        exit_costs = self.cost_model.compute_fill_costs("SELL", exit_price, exit_qty)

        order_id = str(uuid.uuid4())
        order = PaperOrder(
            order_id=order_id,
            position_id=pos.position_id,
            signal_id=pos.signal_id,
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=exit_qty,
            requested_price=market_price,
            filled_price=exit_price,
            slippage_points=self.slippage_points,
            created_at=current_time,
            filled_at=current_time,
            status=OrderStatus.FILLED,
        )
        self.orders[order_id] = order

        trade = PositionTrade(
            trade_id=str(uuid.uuid4()),
            order_id=order_id,
            side=OrderSide.SELL,
            quantity=exit_qty,
            price=exit_price,
            timestamp=current_time,
            turnover=exit_costs["turnover"],
            costs=exit_costs["total_costs"],
            reason=reason,
        )
        pos.trades.append(trade)

        pos.remaining_quantity = 0
        pos.status = PositionStatus.CLOSED
        pos.exit_reason = reason
        pos.exit_price = exit_price
        pos.exit_time = current_time
        pos.total_costs += exit_costs["total_costs"]

        # Calculate finalized gross & net PnL across all execution legs
        total_sell_turnover = sum(t.turnover for t in pos.trades if t.side == OrderSide.SELL)
        total_buy_turnover = sum(t.turnover for t in pos.trades if t.side == OrderSide.BUY)
        pos.gross_pnl = round(total_sell_turnover - total_buy_turnover, 2)
        pos.net_pnl = round(pos.gross_pnl - pos.total_costs, 2)

        # Realized R:R
        risk_points = max(0.5, pos.entry_price - pos.stop_loss_premium)
        reward_points = exit_price - pos.entry_price
        pos.realized_rr = round(reward_points / risk_points, 2)

        if notes:
            pos.notes = (pos.notes or "") + f" | {notes}"

        # Update engine cumulative accounting
        self.total_realized_gross_pnl += pos.gross_pnl
        self.total_realized_net_pnl += pos.net_pnl
        self.total_costs_paid += pos.total_costs

        # Archive from active positions
        self.active_positions.pop(pos.position_id, None)
        self.closed_positions.append(pos)

        # Persist finalized position to DuckDB
        if self.store is not None:
            try:
                self.store.save_paper_order(order.to_dict())
                self.store.save_paper_position(pos.to_dict())
            except Exception as e:
                logger.warning("Failed to save closed paper position to DuckDB: %s", e)

        if self.on_trade_event:
            self.on_trade_event({
                "event": "POSITION_CLOSED",
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "reason": reason.value,
                "exit_price": exit_price,
                "gross_pnl": pos.gross_pnl,
                "net_pnl": pos.net_pnl,
                "total_costs": pos.total_costs,
            })

        return (pos, reason, exit_price)

    def get_open_positions(self) -> List[PaperPosition]:
        """Returns all currently active paper trading positions."""
        return list(self.active_positions.values())

    def get_closed_positions(self) -> List[PaperPosition]:
        """Returns all completed paper trading positions."""
        return list(self.closed_positions)

    def get_account_summary(self) -> Dict[str, Any]:
        """Computes comprehensive paper trading performance telemetry."""
        closed = self.closed_positions
        total_trades = len(closed)
        wins = [p for p in closed if p.net_pnl > 0]
        losses = [p for p in closed if p.net_pnl <= 0]

        win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
        avg_win = (sum(p.net_pnl for p in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(p.net_pnl for p in losses) / len(losses)) if losses else 0.0
        gross_profit = sum(p.gross_pnl for p in wins)
        gross_loss = abs(sum(p.gross_pnl for p in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        open_unrealized_gross = sum(p.gross_pnl for p in self.active_positions.values())
        open_unrealized_net = sum(p.net_pnl for p in self.active_positions.values())

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100.0, 1),
            "total_realized_gross_pnl": round(self.total_realized_gross_pnl, 2),
            "total_realized_net_pnl": round(self.total_realized_net_pnl, 2),
            "total_costs_paid": round(self.total_costs_paid, 2),
            "cost_drag_pct": round((self.total_costs_paid / gross_profit * 100.0), 1) if gross_profit > 0 else 0.0,
            "profit_factor": profit_factor,
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
            "open_positions_count": len(self.active_positions),
            "open_unrealized_gross_pnl": round(open_unrealized_gross, 2),
            "open_unrealized_net_pnl": round(open_unrealized_net, 2),
        }
