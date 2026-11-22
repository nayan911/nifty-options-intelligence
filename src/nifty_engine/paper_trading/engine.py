"""Paper-trading execution engine for simulated NIFTY options trading.

Implements realistic broker friction (slippage, bid-ask spreads, Indian statutory taxes),
M2M position tracking, automated multi-stage exits (Target 1 partial profit booking,
Target 2 runner exit, technical stop-loss, and EOD 15:15 IST auto-square-off).
"""

import logging
import uuid
from datetime import date, datetime, time
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
        quote_provider: Optional[Any] = None,
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
        self.quote_provider = quote_provider
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

        # Automatic recovery from persistent DuckDB store upon process startup
        self._recover_from_store()

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
            initial_stop_loss_premium=contract.premium_trajectory.stop_loss_premium,
            spot_entry=getattr(candidate, "entry_price", getattr(candidate, "trigger_price", 0.0)),
            spot_target_t1=getattr(candidate, "target_1", getattr(candidate, "target_1_price", 0.0)),
            spot_target_t2=getattr(candidate, "target_2", getattr(candidate, "target_2_price", 0.0)),
            spot_sl=candidate.stop_loss,
            entry_time=now,
            last_quote_time=now,
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
        """Processes verified live tick updates for subscribed options or primary underlying spot."""
        events: List[Tuple[PaperPosition, ExitReason, float]] = []
        sym = tick.symbol
        receipt_time = tick.local_timestamp or now_ist()
        exchange_time = tick.exchange_timestamp or receipt_time

        # Reject stale ticks if exchange timestamp is significantly behind receipt clock
        if tick.exchange_timestamp:
            tick_latency = (receipt_time - tick.exchange_timestamp).total_seconds()
            if tick_latency > self.max_quote_age_seconds:
                logger.debug(
                    "Dropping stale tick for %s (latency %.1fs > %.1fs)",
                    sym, tick_latency, self.max_quote_age_seconds,
                )
                return events

        current_time = exchange_time

        # 1. Contract Tick: Update Option M2M and evaluate premium triggers
        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            if pos.symbol == sym:
                # Update option price telemetry and stamp fresh quote time
                pos.current_ltp = tick.ltp
                if tick.bid is not None and tick.bid > 0:
                    pos.current_bid = tick.bid
                if tick.ask is not None and tick.ask > 0:
                    pos.current_ask = tick.ask
                pos.last_quote_time = current_time

                if tick.ltp > pos.highest_premium_seen:
                    pos.highest_premium_seen = tick.ltp
                if tick.ltp < pos.lowest_premium_seen:
                    pos.lowest_premium_seen = tick.ltp

                # Update live M2M unrealized PnL
                effective_price = pos.current_bid if pos.current_bid is not None else pos.current_ltp
                unrealized_gross = (effective_price - pos.entry_price) * pos.remaining_quantity
                pos.gross_pnl = unrealized_gross
                pos.net_pnl = unrealized_gross - pos.total_costs

                # Handle pending triggered exits awaiting fresh quote
                if pos.is_exit_triggered and pos.pending_exit_reason:
                    reason = pos.pending_exit_reason
                    pos.is_exit_triggered = False
                    pos.pending_exit_reason = None
                    if reason == ExitReason.TARGET_1:
                        ev = self._execute_partial_exit_t1(pos, effective_price, current_time, notes="Triggered T1 filled on fresh quote")
                    else:
                        ev = self._execute_full_exit(pos, effective_price, reason, current_time, notes=f"Triggered {reason.value} filled on fresh quote")
                    if ev:
                        events.append(ev)
                    continue

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
                is_call = pos.option_type == "CE"

                # Check spot-level triggers
                spot_sl_hit = (spot_price <= pos.spot_sl) if is_call else (spot_price >= pos.spot_sl)
                spot_t2_hit = (spot_price >= pos.spot_target_t2) if is_call else (spot_price <= pos.spot_target_t2)
                spot_t1_hit = (spot_price >= pos.spot_target_t1) if is_call else (spot_price <= pos.spot_target_t1)

                if not (spot_sl_hit or spot_t2_hit or (not pos.is_t1_hit and spot_t1_hit)):
                    continue

                # Validate option quote freshness before executing fill
                quote_age = pos.quote_age_seconds(current_time)
                is_fresh = quote_age <= self.max_quote_age_seconds
                opt_price = pos.current_bid if (pos.current_bid is not None and pos.current_bid > 0) else pos.current_ltp

                # Try on-demand quote provider if quote is stale
                if not is_fresh and self.quote_provider is not None:
                    fresh_quote = self._fetch_fresh_quote(pos.symbol)
                    if fresh_quote and fresh_quote.get("bid", 0) > 0:
                        opt_price = fresh_quote["bid"]
                        pos.current_bid = opt_price
                        pos.last_quote_time = current_time
                        is_fresh = True

                # Spot Stop Loss Breach
                if spot_sl_hit:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    if is_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes="Spot SL breached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = reason
                        logger.info("Spot SL breached for %s. Option quote stale (age %.1fs). Awaiting fresh quote.", pos.symbol, quote_age)
                    continue

                # Spot Target 2 Reached
                if spot_t2_hit:
                    if is_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, ExitReason.TARGET_2, current_time, notes="Spot T2 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_2
                        logger.info("Spot T2 reached for %s. Option quote stale (age %.1fs). Awaiting fresh quote.", pos.symbol, quote_age)
                    continue

                # Spot Target 1 Reached
                if not pos.is_t1_hit and spot_t1_hit:
                    if is_fresh and opt_price > 0:
                        ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Spot T1 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_1
                        logger.info("Spot T1 reached for %s. Option quote stale (age %.1fs). Awaiting fresh quote.", pos.symbol, quote_age)

        return events

    def on_candle(self, candle: Candle) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Processes completed candle evaluating intrabar boundaries (high, low, close)

        with conservative adverse-first ordering.
        """
        events: List[Tuple[PaperPosition, ExitReason, float]] = []
        current_time = candle.timestamp or now_ist()
        sym = candle.symbol

        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            # Case 1: Option contract candle
            if pos.symbol == sym:
                pos.last_quote_time = current_time
                pos.current_ltp = candle.close

                if candle.high > pos.highest_premium_seen:
                    pos.highest_premium_seen = candle.high
                if candle.low < pos.lowest_premium_seen:
                    pos.lowest_premium_seen = candle.low

                # Conservative adverse-first evaluation: check Stop-Loss on candle.low first!
                effective_sl = pos.stop_loss_premium
                if candle.low <= effective_sl:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    # Execute fill at low bound minus slippage
                    ev = self._execute_full_exit(
                        pos,
                        candle.low,
                        reason,
                        current_time,
                        notes=f"Intrabar SL low: {candle.low:.2f}",
                    )
                    if ev:
                        events.append(ev)
                    continue

                # Target 2 (runner completion)
                if candle.high >= pos.target_premium_t2:
                    ev = self._execute_full_exit(
                        pos,
                        pos.target_premium_t2,
                        ExitReason.TARGET_2,
                        current_time,
                        notes=f"Intrabar T2 high: {candle.high:.2f}",
                    )
                    if ev:
                        events.append(ev)
                    continue

                # Target 1 (partial profit booking)
                if not pos.is_t1_hit and candle.high >= pos.target_premium_t1:
                    ev = self._execute_partial_exit_t1(
                        pos,
                        pos.target_premium_t1,
                        current_time,
                        notes=f"Intrabar T1 high: {candle.high:.2f}",
                    )
                    if ev:
                        events.append(ev)

            # Case 2: Underlying index candle (e.g. NIFTY 1m bar)
            elif "INDEX" in sym and pos.underlying in sym:
                is_call = pos.option_type == "CE"
                quote_age = pos.quote_age_seconds(current_time)
                is_fresh = quote_age <= self.max_quote_age_seconds
                opt_price = pos.current_bid if (pos.current_bid is not None and pos.current_bid > 0) else pos.current_ltp

                # Adverse-first check: spot SL breach
                spot_sl_hit = (candle.low <= pos.spot_sl) if is_call else (candle.high >= pos.spot_sl)
                if spot_sl_hit:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    if not is_fresh and self.quote_provider is not None:
                        fresh_quote = self._fetch_fresh_quote(pos.symbol)
                        if fresh_quote and fresh_quote.get("bid", 0) > 0:
                            opt_price = fresh_quote["bid"]
                            pos.current_bid = opt_price
                            pos.last_quote_time = current_time
                            is_fresh = True
                    if is_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes="Intrabar spot SL breach")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = reason
                    continue

                # Favorable check: Target 2
                spot_t2_hit = (candle.high >= pos.spot_target_t2) if is_call else (candle.low <= pos.spot_target_t2)
                if spot_t2_hit:
                    if not is_fresh and self.quote_provider is not None:
                        fresh_quote = self._fetch_fresh_quote(pos.symbol)
                        if fresh_quote and fresh_quote.get("bid", 0) > 0:
                            opt_price = fresh_quote["bid"]
                            pos.current_bid = opt_price
                            pos.last_quote_time = current_time
                            is_fresh = True
                    if is_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, ExitReason.TARGET_2, current_time, notes="Intrabar spot T2 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_2
                    continue

                # Favorable check: Target 1
                spot_t1_hit = (candle.high >= pos.spot_target_t1) if is_call else (candle.low <= pos.spot_target_t1)
                if not pos.is_t1_hit and spot_t1_hit:
                    if not is_fresh and self.quote_provider is not None:
                        fresh_quote = self._fetch_fresh_quote(pos.symbol)
                        if fresh_quote and fresh_quote.get("bid", 0) > 0:
                            opt_price = fresh_quote["bid"]
                            pos.current_bid = opt_price
                            pos.last_quote_time = current_time
                            is_fresh = True
                    if is_fresh and opt_price > 0:
                        ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Intrabar spot T1 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_1

        return events

    def check_time_events(self, current_time: datetime) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Evaluates auto square-off at 15:15 IST and max holding timeouts."""
        events: List[Tuple[PaperPosition, ExitReason, float]] = []
        t = current_time.time()
        is_eod = t >= self.eod_square_off_time

        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            quote_age = pos.quote_age_seconds(current_time)
            is_fresh = quote_age <= self.max_quote_age_seconds
            opt_price = pos.current_bid if (pos.current_bid is not None and pos.current_bid > 0) else pos.current_ltp

            # Try quote provider if stale
            if not is_fresh and self.quote_provider is not None:
                fresh_quote = self._fetch_fresh_quote(pos.symbol)
                if fresh_quote and fresh_quote.get("bid", 0) > 0:
                    opt_price = fresh_quote["bid"]
                    pos.current_bid = opt_price
                    pos.last_quote_time = current_time
                    is_fresh = True

            # 1. EOD Auto Square-Off at 15:15 IST
            if is_eod:
                if is_fresh:
                    exit_notes = "Intraday auto square-off at 15:15 IST"
                    fill_ref = opt_price
                else:
                    # Explicit conservative fallback fill: apply double slippage penalty
                    fill_ref = max(0.05, opt_price - self.slippage_points)
                    exit_notes = f"EOD square-off filled with conservative penalty (quote age: {quote_age:.1f}s)"

                ev = self._execute_full_exit(
                    pos,
                    fill_ref,
                    ExitReason.EOD_SQUARE_OFF,
                    current_time,
                    notes=exit_notes,
                )
                if ev:
                    events.append(ev)
                continue

            # 2. Maximum holding period timeout
            elapsed_minutes = (current_time - pos.entry_time).total_seconds() / 60.0
            if elapsed_minutes >= self.max_holding_minutes and not pos.is_t1_hit:
                if is_fresh and opt_price > 0:
                    ev = self._execute_full_exit(
                        pos,
                        opt_price,
                        ExitReason.TIMEOUT,
                        current_time,
                        notes=f"Holding timeout reached ({elapsed_minutes:.1f} mins)",
                    )
                    if ev:
                        events.append(ev)
                else:
                    pos.is_exit_triggered = True
                    pos.pending_exit_reason = ExitReason.TIMEOUT
                    logger.info("Holding timeout reached for %s. Option quote stale (age %.1fs). Awaiting fresh quote.", pos.symbol, quote_age)

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
            if notes:
                pos.notes = (pos.notes or "") + f" | {notes}"
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
        if notes:
            pos.notes = (pos.notes or "") + f" | {notes}"

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
        pos.exit_time = current_time
        pos.total_costs += exit_costs["total_costs"]

        # Calculate finalized gross & net PnL across all execution legs
        total_sell_turnover = sum(t.turnover for t in pos.trades if t.side == OrderSide.SELL)
        total_sell_quantity = sum(t.quantity for t in pos.trades if t.side == OrderSide.SELL)
        total_buy_turnover = sum(t.turnover for t in pos.trades if t.side == OrderSide.BUY)
        pos.gross_pnl = round(total_sell_turnover - total_buy_turnover, 2)
        pos.net_pnl = round(pos.gross_pnl - pos.total_costs, 2)

        # Quantity-weighted average exit price across all partial fills
        avg_exit_price = (total_sell_turnover / total_sell_quantity) if total_sell_quantity > 0 else exit_price
        pos.exit_price = round(avg_exit_price, 2)

        # Realized R:R preserves the INITIAL stop loss (original risk before breakeven trailing)
        initial_sl = pos.initial_stop_loss_premium if pos.initial_stop_loss_premium > 0.0 else pos.stop_loss_premium
        risk_points = max(0.5, pos.entry_price - initial_sl)
        reward_points = avg_exit_price - pos.entry_price
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
                "exit_price": pos.exit_price,
                "gross_pnl": pos.gross_pnl,
                "net_pnl": pos.net_pnl,
                "total_costs": pos.total_costs,
            })

        return (pos, reason, exit_price)

    def _fetch_fresh_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Queries quote_provider for an on-demand quote when external triggers fire."""
        if self.quote_provider is None:
            return None
        try:
            if callable(self.quote_provider):
                return self.quote_provider(symbol)
            if hasattr(self.quote_provider, "quotes"):
                res = self.quote_provider.quotes(data={"symbols": symbol})
                if res and res.get("s") == "ok" and res.get("d"):
                    val = res["d"][0].get("v", {})
                    ltp = float(val.get("lp", 0.0))
                    bid = float(val.get("bid", ltp))
                    ask = float(val.get("ask", ltp))
                    return {"ltp": ltp, "bid": bid, "ask": ask}
        except Exception as e:
            logger.debug("Failed to fetch on-demand quote for %s: %s", symbol, e)
        return None

    def _recover_from_store(self) -> None:
        """Reconstructs active positions, closed positions, and cumulative accounting from DuckDB on startup."""
        if self.store is None:
            return

        try:
            # 1. Recover orders first so trade histories can be mapped to positions
            orders_rows = self.store.get_paper_orders()
            orders_by_pos: Dict[str, List[PaperOrder]] = {}
            for o_row in orders_rows:
                order = self._row_to_order(o_row)
                self.orders[order.order_id] = order
                if order.position_id:
                    orders_by_pos.setdefault(order.position_id, []).append(order)

            # 2. Recover closed positions and accumulate historical performance
            closed_rows = self.store.get_paper_positions(status=PositionStatus.CLOSED.value)
            for row in closed_rows:
                pos = self._row_to_position(row, orders=orders_by_pos.get(str(row.get("position_id")), []))
                self.closed_positions.append(pos)
                self.total_realized_gross_pnl += pos.gross_pnl
                self.total_realized_net_pnl += pos.net_pnl
                self.total_costs_paid += pos.total_costs

            # 3. Recover active open and partially closed positions
            for st in (PositionStatus.OPEN.value, PositionStatus.PARTIALLY_CLOSED.value):
                active_rows = self.store.get_paper_positions(status=st)
                for row in active_rows:
                    pos = self._row_to_position(row, orders=orders_by_pos.get(str(row.get("position_id")), []))
                    self.active_positions[pos.position_id] = pos
                    self.subscribed_option_symbols.add(pos.symbol)

            logger.info(
                "Recovered %d active and %d closed paper positions from DuckDB. Historical Net P&L: ₹%.2f",
                len(self.active_positions),
                len(self.closed_positions),
                self.total_realized_net_pnl,
            )
        except Exception as e:
            logger.warning("Failed to recover paper trading state from DuckDB: %s", e)

    def _row_to_position(self, row: Dict[str, Any], orders: Optional[List[PaperOrder]] = None) -> PaperPosition:
        """Constructs PaperPosition instance from DuckDB dictionary row and order history."""
        entry_time = self._parse_dt(row.get("entry_time")) or now_ist()
        exit_time = self._parse_dt(row.get("exit_time"))
        last_quote_time = self._parse_dt(row.get("last_quote_time")) or entry_time
        expiry = self._parse_date(row.get("expiry")) or entry_time.date()

        status_str = row.get("status", "OPEN")
        status = PositionStatus(status_str) if status_str in PositionStatus._value2member_map_ else PositionStatus.OPEN

        reason_str = row.get("exit_reason")
        exit_reason = ExitReason(reason_str) if reason_str and reason_str in ExitReason._value2member_map_ else None

        is_exit_triggered = bool(row.get("is_exit_triggered", False))
        pending_exit_str = row.get("pending_exit_reason")
        pending_exit_reason = (
            ExitReason(pending_exit_str)
            if pending_exit_str and pending_exit_str in ExitReason._value2member_map_
            else None
        )

        pos = PaperPosition(
            position_id=str(row.get("position_id")),
            signal_id=str(row.get("signal_id")),
            symbol=str(row.get("symbol")),
            underlying=str(row.get("underlying", "NIFTY")),
            option_type=str(row.get("option_type", "CE")),
            strike=float(row.get("strike", 0.0)),
            expiry=expiry,
            lot_size=int(row.get("lot_size", 65)),
            initial_quantity=int(row.get("initial_quantity", 65)),
            remaining_quantity=int(row.get("remaining_quantity", 65)),
            entry_price=float(row.get("entry_price", 0.0)),
            target_premium_t1=float(row.get("target_premium_t1", 0.0)),
            target_premium_t2=float(row.get("target_premium_t2", 0.0)),
            stop_loss_premium=float(row.get("stop_loss_premium", 0.0)),
            initial_stop_loss_premium=float(row.get("initial_stop_loss_premium") or row.get("stop_loss_premium", 0.0)),
            spot_entry=float(row.get("spot_entry", 0.0)),
            spot_target_t1=float(row.get("spot_target_t1", 0.0)),
            spot_target_t2=float(row.get("spot_target_t2", 0.0)),
            spot_sl=float(row.get("spot_sl", 0.0)),
            entry_time=entry_time,
            exit_time=exit_time,
            exit_price=float(row["exit_price"]) if row.get("exit_price") is not None else None,
            last_quote_time=last_quote_time,
            status=status,
            exit_reason=exit_reason,
            is_t1_hit=(status == PositionStatus.PARTIALLY_CLOSED or exit_reason == ExitReason.TARGET_2),
            is_breakeven_sl_active=(status == PositionStatus.PARTIALLY_CLOSED),
            current_ltp=float(row.get("current_ltp", 0.0)),
            current_bid=float(row["current_bid"]) if row.get("current_bid") is not None else None,
            current_ask=float(row["current_ask"]) if row.get("current_ask") is not None else None,
            is_exit_triggered=is_exit_triggered,
            pending_exit_reason=pending_exit_reason,
            gross_pnl=float(row.get("gross_pnl", 0.0)),
            net_pnl=float(row.get("net_pnl", 0.0)),
            total_costs=float(row.get("total_costs", 0.0)),
            realized_rr=float(row["realized_rr"]) if row.get("realized_rr") is not None else None,
            notes=row.get("notes"),
        )

        # Restore execution history (trades)
        trades: List[PositionTrade] = []
        if orders:
            for order in sorted(orders, key=lambda o: o.created_at):
                if order.status != OrderStatus.FILLED:
                    continue
                price = order.filled_price or pos.entry_price
                qty = order.quantity
                is_buy = (order.side == OrderSide.BUY)
                costs = self.cost_model.compute_fill_costs(
                    side="BUY" if is_buy else "SELL",
                    price=price,
                    quantity=qty,
                )["total_costs"]
                trade_reason = None
                if not is_buy:
                    trade_reason = pos.exit_reason if (pos.status == PositionStatus.CLOSED and qty == pos.remaining_quantity) else ExitReason.TARGET_1
                trade = PositionTrade(
                    trade_id=f"tr_{order.order_id}",
                    order_id=order.order_id,
                    side=order.side,
                    quantity=qty,
                    price=price,
                    timestamp=order.filled_at or order.created_at,
                    turnover=round(price * qty, 2),
                    costs=costs,
                    reason=trade_reason,
                )
                trades.append(trade)

        if not trades and pos.entry_price > 0:
            entry_costs = self.cost_model.compute_fill_costs(
                side="BUY",
                price=pos.entry_price,
                quantity=pos.initial_quantity,
            )["total_costs"]
            trades.append(
                PositionTrade(
                    trade_id=f"tr_init_{pos.position_id}",
                    order_id=f"ord_init_{pos.position_id}",
                    side=OrderSide.BUY,
                    quantity=pos.initial_quantity,
                    price=pos.entry_price,
                    timestamp=pos.entry_time,
                    turnover=round(pos.entry_price * pos.initial_quantity, 2),
                    costs=entry_costs,
                    reason=None,
                )
            )
            if pos.status == PositionStatus.PARTIALLY_CLOSED:
                closed_qty = pos.initial_quantity - pos.remaining_quantity
                if closed_qty > 0:
                    t1_price = pos.target_premium_t1 or pos.entry_price
                    t1_costs = self.cost_model.compute_fill_costs(
                        side="SELL",
                        price=t1_price,
                        quantity=closed_qty,
                    )["total_costs"]
                    trades.append(
                        PositionTrade(
                            trade_id=f"tr_t1_{pos.position_id}",
                            order_id=f"ord_t1_{pos.position_id}",
                            side=OrderSide.SELL,
                            quantity=closed_qty,
                            price=t1_price,
                            timestamp=pos.last_quote_time or pos.entry_time,
                            turnover=round(t1_price * closed_qty, 2),
                            costs=t1_costs,
                            reason=ExitReason.TARGET_1,
                        )
                    )

        pos.trades = trades
        return pos

    def _row_to_order(self, row: Dict[str, Any]) -> PaperOrder:
        """Constructs PaperOrder instance from DuckDB dictionary row."""
        side_str = row.get("side", "BUY")
        side = OrderSide(side_str) if side_str in OrderSide._value2member_map_ else OrderSide.BUY

        type_str = row.get("order_type", "MARKET")
        order_type = OrderType(type_str) if type_str in OrderType._value2member_map_ else OrderType.MARKET

        st_str = row.get("status", "FILLED")
        st = OrderStatus(st_str) if st_str in OrderStatus._value2member_map_ else OrderStatus.FILLED

        return PaperOrder(
            order_id=str(row.get("order_id")),
            position_id=row.get("position_id"),
            signal_id=row.get("signal_id"),
            symbol=str(row.get("symbol")),
            side=side,
            order_type=order_type,
            quantity=int(row.get("quantity", 0)),
            created_at=self._parse_dt(row.get("created_at")) or now_ist(),
            filled_at=self._parse_dt(row.get("filled_at")),
            status=st,
            requested_price=float(row["requested_price"]) if row.get("requested_price") is not None else None,
            filled_price=float(row["filled_price"]) if row.get("filled_price") is not None else None,
            slippage_points=float(row.get("slippage", 0.0)),
            rejection_reason=row.get("rejection_reason"),
        )

    @staticmethod
    def _parse_dt(val: Any) -> Optional[datetime]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_date(val: Any) -> Optional[date]:
        if val is None:
            return None
        if isinstance(val, date) and not isinstance(val, datetime):
            return val
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, str):
            try:
                return date.fromisoformat(val)
            except Exception:
                return None
        return None

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
