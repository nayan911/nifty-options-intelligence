"""Paper-trading execution engine for simulated NIFTY options trading.

Implements realistic broker friction (slippage, bid-ask spreads, Indian statutory taxes),
M2M position tracking, automated multi-stage exits (Target 1 partial profit booking,
Target 2 runner exit, technical stop-loss, and EOD 15:15 IST auto-square-off).
"""

import logging
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pytz

from nifty_engine.analysis.derivatives.cost_model import PremiumCostModel
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle, IStorageEngine, SystemReadiness, Tick
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
from nifty_engine.utils.timezone import now_ist, to_ist

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
        checkpoint_interval_seconds: float = 1.0,
        clock_skew_tolerance_seconds: float = 2.0,
        calendar: Optional[ExchangeCalendar] = None,
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
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.clock_skew_tolerance_seconds = clock_skew_tolerance_seconds
        self.calendar = calendar or ExchangeCalendar()
        self.on_trade_event = on_trade_event

        # In-memory position and order storage
        self.active_positions: Dict[str, PaperPosition] = {}  # position_id -> PaperPosition
        self.closed_positions: List[PaperPosition] = []
        self.orders: Dict[str, PaperOrder] = {}  # order_id -> PaperOrder
        self.subscribed_option_symbols: Set[str] = set()
        self._last_persisted_time: Dict[str, datetime] = {}
        self._last_persisted_quote_time: Dict[str, Optional[datetime]] = {}
        self._last_persisted_candle_time: Dict[str, Optional[datetime]] = {}

        # Cumulative P&L and cost tracking
        self.total_realized_gross_pnl: float = 0.0
        self.total_realized_net_pnl: float = 0.0
        self.total_costs_paid: float = 0.0
        self.system_readiness: SystemReadiness = SystemReadiness.READY

        # Automatic recovery from persistent DuckDB store upon process startup
        self._recover_from_store()

    def set_system_readiness(self, readiness: Union[SystemReadiness, str]) -> None:
        """Sets the system operational readiness level, gating new trade entries when NOT_READY."""
        if isinstance(readiness, str):
            try:
                self.system_readiness = SystemReadiness(readiness.upper())
            except ValueError:
                self.system_readiness = SystemReadiness.NOT_READY
        else:
            self.system_readiness = readiness
        logger.info("Paper trading system readiness updated to: %s", self.system_readiness.value)

    def process_technical_candidate(
        self,
        candidate: TechnicalCandidate,
        current_time: Optional[datetime] = None,
    ) -> Optional[PaperPosition]:
        """Evaluates a confirmed technical candidate with derivatives contract selection

        and executes a simulated market BUY order if all risk gates pass.
        """
        # 0. System Readiness Gate
        if self.system_readiness == SystemReadiness.NOT_READY:
            logger.warning(
                "Paper entry rejected: System readiness is NOT_READY (historical data / indicators incomplete)",
            )
            return None

        now = (
            current_time
            or getattr(candidate, "decision_at", None)
            or getattr(candidate, "timestamp", None)
            or getattr(candidate, "observed_at", None)
            or now_ist()
        )
        t = now.time()

        # 1. Trading hours gate
        if self.calendar:
            if not self.calendar.is_market_open(now, is_derivative=True):
                logger.info("Paper entry rejected: Market is closed (%s)", now)
                return None

            now_ist_dt = to_ist(now) if now else now_ist()
            current_date = now_ist_dt.date()
            if current_date in self.calendar.special_sessions:
                cfg = self.calendar.special_sessions[current_date]
                end_t = datetime.strptime(cfg["session_end"], "%H:%M:%S").time()
                end_dt = datetime.combine(current_date, end_t)
                cutoff_t = (end_dt - timedelta(minutes=15)).time()
                if t >= cutoff_t:
                    logger.info("Paper entry rejected: After special session cutoff (%s >= %s)", t, cutoff_t)
                    return None
            else:
                if t < self.market_open_time:
                    logger.info("Paper entry rejected: Market not yet open (%s)", t)
                    return None
                if t >= self.no_new_entries_time:
                    logger.info("Paper entry rejected: After no-new-entries cutoff (%s >= %s)", t, self.no_new_entries_time)
                    return None
        else:
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
            last_bid_time=now if (contract.bid is not None and contract.bid > 0) else None,
            last_ask_time=now if (contract.ask is not None and contract.ask > 0) else None,
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
        self._last_persisted_time[position_id] = now
        self._last_persisted_quote_time[position_id] = now
        self._last_persisted_candle_time[position_id] = None

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
                    pos.last_bid_time = current_time
                if tick.ask is not None and tick.ask > 0:
                    pos.current_ask = tick.ask
                    pos.last_ask_time = current_time
                pos.last_quote_time = current_time

                if tick.ltp > pos.highest_premium_seen:
                    pos.highest_premium_seen = tick.ltp
                if tick.ltp < pos.lowest_premium_seen:
                    pos.lowest_premium_seen = tick.ltp

                # Check bid freshness
                bid_fresh = pos.has_fresh_bid(current_time, self.max_quote_age_seconds)

                # Update live M2M unrealized PnL
                effective_price = pos.current_bid if (bid_fresh and pos.current_bid is not None and pos.current_bid > 0) else pos.current_ltp
                unrealized_gross = (effective_price - pos.entry_price) * pos.remaining_quantity
                pos.gross_pnl = unrealized_gross
                pos.net_pnl = unrealized_gross - pos.total_costs

                # Controlled-interval checkpoint: automatically persist updated quote state to DuckDB
                last_persisted = self._last_persisted_time.get(pos.position_id, pos.entry_time)
                if (current_time - last_persisted).total_seconds() >= self.checkpoint_interval_seconds:
                    self._save_position(pos, persist_time=current_time)

                # Handle pending triggered exits awaiting fresh quote
                if pos.is_exit_triggered and pos.pending_exit_reason:
                    # Require fresh bid before executing pending sell exits
                    if not bid_fresh:
                        bid_fresh = self._apply_fresh_quote(pos, current_time)

                    if bid_fresh and pos.current_bid and pos.current_bid > 0:
                        reason = pos.pending_exit_reason
                        pos.is_exit_triggered = False
                        pos.pending_exit_reason = None
                        fill_price = pos.current_bid
                        if reason == ExitReason.TARGET_1:
                            ev = self._execute_partial_exit_t1(pos, fill_price, current_time, notes="Triggered T1 filled on fresh quote")
                        else:
                            ev = self._execute_full_exit(pos, fill_price, reason, current_time, notes=f"Triggered {reason.value} filled on fresh quote")
                        if ev:
                            events.append(ev)
                    continue

                # Evaluate option-level triggers (Target 2, Target 1, Stop Loss)
                # Triggers evaluate against current trade LTP or verified bid
                t2_hit = effective_price >= pos.target_premium_t2 or tick.ltp >= pos.target_premium_t2
                t1_hit = not pos.is_t1_hit and (effective_price >= pos.target_premium_t1 or tick.ltp >= pos.target_premium_t1)
                sl_hit = effective_price <= pos.stop_loss_premium or tick.ltp <= pos.stop_loss_premium

                if t2_hit or t1_hit or sl_hit:
                    # Require fresh bid before simulating sell fill
                    if not bid_fresh:
                        bid_fresh = self._apply_fresh_quote(pos, current_time)

                    if t2_hit:
                        if bid_fresh and pos.current_bid and pos.current_bid > 0:
                            ev = self._execute_full_exit(pos, pos.current_bid, ExitReason.TARGET_2, current_time)
                            if ev:
                                events.append(ev)
                        else:
                            pos.is_exit_triggered = True
                            pos.pending_exit_reason = ExitReason.TARGET_2
                            self._save_position(pos, persist_time=current_time)
                            logger.info("Option T2 reached for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))

                    elif t1_hit:
                        if bid_fresh and pos.current_bid and pos.current_bid > 0:
                            ev = self._execute_partial_exit_t1(pos, pos.current_bid, current_time)
                            if ev:
                                events.append(ev)
                        else:
                            pos.is_exit_triggered = True
                            pos.pending_exit_reason = ExitReason.TARGET_1
                            self._save_position(pos, persist_time=current_time)
                            logger.info("Option T1 reached for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))

                    elif sl_hit:
                        reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                        if bid_fresh and pos.current_bid and pos.current_bid > 0:
                            ev = self._execute_full_exit(pos, pos.current_bid, reason, current_time)
                            if ev:
                                events.append(ev)
                        else:
                            pos.is_exit_triggered = True
                            pos.pending_exit_reason = reason
                            self._save_position(pos, persist_time=current_time)
                            logger.info("Option SL hit for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))

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

                # Validate option bid freshness before executing sell fill
                bid_fresh = pos.has_fresh_bid(current_time, self.max_quote_age_seconds)
                if not bid_fresh:
                    bid_fresh = self._apply_fresh_quote(pos, current_time)
                opt_price = pos.current_bid if (bid_fresh and pos.current_bid is not None and pos.current_bid > 0) else 0.0

                # Spot Stop Loss Breach
                if spot_sl_hit:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes="Spot SL breached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = reason
                        self._save_position(pos, persist_time=current_time)
                        logger.info("Spot SL breached for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))
                    continue

                # Spot Target 2 Reached
                if spot_t2_hit:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, ExitReason.TARGET_2, current_time, notes="Spot T2 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_2
                        self._save_position(pos, persist_time=current_time)
                        logger.info("Spot T2 reached for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))
                    continue

                # Spot Target 1 Reached
                if not pos.is_t1_hit and spot_t1_hit:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Spot T1 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_1
                        self._save_position(pos, persist_time=current_time)
                        logger.info("Spot T1 reached for %s. Option bid stale (age %.1fs). Awaiting fresh bid.", pos.symbol, pos.bid_age_seconds(current_time))

        return events

    def on_candle(self, candle: Candle) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Processes completed candle evaluating intrabar boundaries (high, low, close)

        with conservative adverse-first ordering.
        """
        events: List[Tuple[PaperPosition, ExitReason, float]] = []

        # 1. Restrict intrabar boundary evaluations to base 1m bars to prevent replaying
        # overlapping multi-timeframe boundaries (e.g. 5m and 15m).
        if candle.timeframe and candle.timeframe not in ("1m", "1M"):
            logger.debug(
                "Skipping non-base candle timeframe %s for %s in paper engine",
                candle.timeframe,
                candle.symbol,
            )
            return events

        # 2. Derive candle availability / completion timestamp
        availability_time = candle.observed_at or candle.timestamp or now_ist()
        current_time = availability_time
        sym = candle.symbol

        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            # Causal guard: Candle availability must be strictly after position was entered
            if availability_time < pos.entry_time:
                continue

            # Causal guard: Prevent retroactive lookahead fills.
            # If candle start time is strictly before pos.entry_time, part of the bar elapsed before
            # the position was open, so its high/low could reflect pre-trade movements.
            if candle.timestamp and candle.timestamp < pos.entry_time:
                continue

            # Case 1: Option contract candle
            if pos.symbol == sym:
                pos.last_candle_time = current_time
                pos.current_ltp = candle.close

                if candle.high > pos.highest_premium_seen:
                    pos.highest_premium_seen = candle.high
                if candle.low < pos.lowest_premium_seen:
                    pos.lowest_premium_seen = candle.low

                # Validate executable bid freshness
                bid_fresh = pos.has_fresh_bid(current_time, self.max_quote_age_seconds)
                if not bid_fresh:
                    bid_fresh = self._apply_fresh_quote(pos, current_time)
                opt_price = pos.current_bid if (bid_fresh and pos.current_bid is not None and pos.current_bid > 0) else 0.0

                # Process pending exit if bid has become fresh
                if pos.is_exit_triggered and pos.pending_exit_reason:
                    if bid_fresh and opt_price > 0:
                        reason = pos.pending_exit_reason
                        pos.is_exit_triggered = False
                        pos.pending_exit_reason = None
                        if reason == ExitReason.TARGET_1:
                            ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Triggered T1 filled on fresh candle quote")
                        else:
                            ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes=f"Triggered {reason.value} filled on fresh candle quote")
                        if ev:
                            events.append(ev)
                    continue

                # Conservative adverse-first evaluation: check Stop-Loss on candle.low first!
                effective_sl = pos.stop_loss_premium
                if candle.low <= effective_sl:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(
                            pos,
                            opt_price,
                            reason,
                            current_time,
                            notes=f"Intrabar SL low: {candle.low:.2f}",
                        )
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = reason
                        self._save_position(pos, persist_time=current_time)
                        logger.info(
                            "Intrabar SL low breached for %s (low=%.2f <= sl=%.2f). Option bid stale. Retaining pending exit.",
                            pos.symbol,
                            candle.low,
                            effective_sl,
                        )
                    continue

                # Target 2 (runner completion)
                if candle.high >= pos.target_premium_t2:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(
                            pos,
                            opt_price,
                            ExitReason.TARGET_2,
                            current_time,
                            notes=f"Intrabar T2 high: {candle.high:.2f}",
                        )
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_2
                        self._save_position(pos, persist_time=current_time)
                        logger.info(
                            "Intrabar T2 reached for %s (high=%.2f >= t2=%.2f). Option bid stale. Retaining pending exit.",
                            pos.symbol,
                            candle.high,
                            pos.target_premium_t2,
                        )
                    continue

                # Target 1 (partial profit booking)
                if not pos.is_t1_hit and candle.high >= pos.target_premium_t1:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_partial_exit_t1(
                            pos,
                            opt_price,
                            current_time,
                            notes=f"Intrabar T1 high: {candle.high:.2f}",
                        )
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_1
                        self._save_position(pos, persist_time=current_time)
                        logger.info(
                            "Intrabar T1 reached for %s (high=%.2f >= t1=%.2f). Option bid stale. Retaining pending exit.",
                            pos.symbol,
                            candle.high,
                            pos.target_premium_t1,
                        )

            # Case 2: Underlying index candle (e.g. NIFTY 1m bar)
            elif "INDEX" in sym and pos.underlying in sym:
                is_call = pos.option_type == "CE"
                bid_fresh = pos.has_fresh_bid(current_time, self.max_quote_age_seconds)
                if not bid_fresh:
                    bid_fresh = self._apply_fresh_quote(pos, current_time)
                opt_price = pos.current_bid if (bid_fresh and pos.current_bid is not None and pos.current_bid > 0) else 0.0

                # Adverse-first check: spot SL breach
                spot_sl_hit = (candle.low <= pos.spot_sl) if is_call else (candle.high >= pos.spot_sl)
                if spot_sl_hit:
                    reason = ExitReason.TRAILING_STOP if pos.is_breakeven_sl_active else ExitReason.STOP_LOSS
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, reason, current_time, notes="Intrabar spot SL breach")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = reason
                        self._save_position(pos, persist_time=current_time)
                    continue

                # Favorable check: Target 2
                spot_t2_hit = (candle.high >= pos.spot_target_t2) if is_call else (candle.low <= pos.spot_target_t2)
                if spot_t2_hit:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_full_exit(pos, opt_price, ExitReason.TARGET_2, current_time, notes="Intrabar spot T2 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_2
                        self._save_position(pos, persist_time=current_time)
                    continue

                # Favorable check: Target 1
                spot_t1_hit = (candle.high >= pos.spot_target_t1) if is_call else (candle.low <= pos.spot_target_t1)
                if not pos.is_t1_hit and spot_t1_hit:
                    if bid_fresh and opt_price > 0:
                        ev = self._execute_partial_exit_t1(pos, opt_price, current_time, notes="Intrabar spot T1 reached")
                        if ev:
                            events.append(ev)
                    else:
                        pos.is_exit_triggered = True
                        pos.pending_exit_reason = ExitReason.TARGET_1
                        self._save_position(pos, persist_time=current_time)

        # Controlled-interval checkpoint active positions
        self.checkpoint_active_positions(force=False, current_time=current_time)
        return events

    def check_time_events(self, current_time: datetime) -> List[Tuple[PaperPosition, ExitReason, float]]:
        """Evaluates auto square-off at 15:15 IST (or session cutoff) and max holding timeouts."""
        events: List[Tuple[PaperPosition, ExitReason, float]] = []

        # Calendar gate: Do not run any auto square-offs or timeouts on non-trading days/sessions
        if self.calendar and not self.calendar.is_market_open(current_time, is_derivative=True):
            return events

        sq_off_time = self.eod_square_off_time
        if self.calendar:
            cal_sq_off = self.calendar.get_auto_square_off_time(current_time, self.eod_square_off_time)
            if cal_sq_off is not None:
                sq_off_time = cal_sq_off

        t = current_time.time()
        is_eod = t >= sq_off_time

        for pos_id, pos in list(self.active_positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue

            bid_fresh = pos.has_fresh_bid(current_time, self.max_quote_age_seconds)
            if not bid_fresh:
                bid_fresh = self._apply_fresh_quote(pos, current_time)
            opt_price = pos.current_bid if (bid_fresh and pos.current_bid is not None and pos.current_bid > 0) else 0.0

            # 1. EOD Auto Square-Off at session cutoff
            if is_eod:
                is_estimated = not (bid_fresh and opt_price > 0)
                if not is_estimated:
                    exit_notes = "Intraday auto square-off at session cutoff (authenticated broker bid fill)"
                    fill_ref = opt_price
                else:
                    # Explicit conservative fallback fill: apply double slippage penalty
                    fallback_base = pos.current_bid if (pos.current_bid is not None and pos.current_bid > 0) else pos.current_ltp
                    fill_ref = max(0.05, fallback_base - self.slippage_points)
                    exit_notes = f"EOD square-off filled with conservative penalty (bid age: {pos.bid_age_seconds(current_time):.1f}s)"

                ev = self._execute_full_exit(
                    pos,
                    fill_ref,
                    ExitReason.EOD_SQUARE_OFF,
                    current_time,
                    notes=exit_notes,
                    is_estimated_fill=is_estimated,
                )
                if ev:
                    events.append(ev)
                continue

            # 2. Maximum holding period timeout
            elapsed_minutes = (current_time - pos.entry_time).total_seconds() / 60.0
            if elapsed_minutes >= self.max_holding_minutes and not pos.is_t1_hit:
                if bid_fresh and opt_price > 0:
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
                    self._save_position(pos, persist_time=current_time)
                    logger.info("Holding timeout reached for %s. Option bid stale (age %.1fs). Awaiting fresh quote.", pos.symbol, pos.bid_age_seconds(current_time))

        # Controlled-interval checkpoint active positions
        self.checkpoint_active_positions(force=False, current_time=current_time)
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
            self._save_position(pos)
            return (pos, ExitReason.TARGET_1, exit_price)

        # For multi-lot positions (>= 2 lots): exit 50%
        exit_qty = (pos.remaining_quantity // (2 * pos.lot_size)) * pos.lot_size
        if exit_qty <= 0:
            pos.stop_loss_premium = max(pos.stop_loss_premium, pos.entry_price + 1.0)
            self._save_position(pos)
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
            is_estimated_fill=False,
            reason=ExitReason.TARGET_1,
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
        is_estimated_fill: bool = False,
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
            is_estimated_fill=is_estimated_fill,
            reason=reason,
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
            is_estimated_fill=is_estimated_fill,
        )
        pos.trades.append(trade)

        pos.remaining_quantity = 0
        pos.status = PositionStatus.CLOSED
        pos.exit_reason = reason
        pos.exit_time = current_time
        pos.is_estimated_exit = is_estimated_fill
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

    def _save_position(self, pos: PaperPosition, persist_time: Optional[datetime] = None) -> None:
        """Persists position state transition to DuckDB store if available."""
        if self.store is not None:
            try:
                self.store.save_paper_position(pos.to_dict())
                actual_time = persist_time or now_ist()
                self._last_persisted_time[pos.position_id] = actual_time
                self._last_persisted_quote_time[pos.position_id] = pos.last_quote_time
                self._last_persisted_candle_time[pos.position_id] = pos.last_candle_time
            except Exception as e:
                logger.warning("Failed to persist paper position %s to DuckDB: %s", pos.position_id, e)

    def checkpoint_active_positions(
        self,
        force: bool = False,
        current_time: Optional[datetime] = None,
    ) -> None:
        """Periodically or forcefully checkpoints live quote/bid telemetry for all active positions to DuckDB."""
        if self.store is None:
            return
        now = current_time or now_ist()
        for pos in list(self.active_positions.values()):
            if pos.status == PositionStatus.CLOSED:
                continue
            last_persisted = self._last_persisted_time.get(pos.position_id, pos.entry_time)
            if force or (now - last_persisted).total_seconds() >= self.checkpoint_interval_seconds:
                has_updates = (
                    pos.last_quote_time != self._last_persisted_quote_time.get(pos.position_id)
                    or pos.last_candle_time != self._last_persisted_candle_time.get(pos.position_id)
                )
                if force or has_updates:
                    self._save_position(pos, persist_time=now)

    def _fetch_fresh_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
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
                    raw_bid = val.get("bid")
                    raw_ask = val.get("ask")
                    bid = float(raw_bid) if (raw_bid is not None and float(raw_bid) > 0) else 0.0
                    ask = float(raw_ask) if (raw_ask is not None and float(raw_ask) > 0) else 0.0
                    quote_ts = None
                    tt = val.get("tt")
                    if tt is not None and int(tt) > 0:
                        try:
                            quote_ts = datetime.fromtimestamp(int(tt), tz=pytz.timezone("Asia/Kolkata"))
                        except Exception:
                            quote_ts = None
                    return {"ltp": ltp, "bid": bid, "ask": ask, "timestamp": quote_ts}
        except Exception as e:
            logger.debug("Failed to fetch on-demand quote for %s: %s", symbol, e)
        return None

    def _apply_fresh_quote(
        self,
        pos: PaperPosition,
        current_time: datetime,
    ) -> bool:
        """Attempts to fetch an on-demand quote and apply it if the bid is authentic, positive, and fresh.

        Returns True if a valid, fresh bid was successfully applied.
        """
        if self.quote_provider is None:
            return False
        fresh_quote = self._fetch_fresh_quote(pos.symbol)
        if not fresh_quote:
            return False

        raw_bid = fresh_quote.get("bid")
        bid = float(raw_bid) if (raw_bid is not None and float(raw_bid) > 0) else 0.0
        if bid <= 0.0:
            logger.debug("Rejected on-demand quote for %s: missing or non-positive bid (%.2f)", pos.symbol, bid)
            return False

        # Validate broker quote timestamp freshness
        quote_ts = fresh_quote.get("timestamp")
        if quote_ts is None:
            logger.debug("Rejected on-demand quote for %s: missing broker quote timestamp", pos.symbol)
            return False

        delta_seconds = (current_time - quote_ts).total_seconds()
        if delta_seconds < -self.clock_skew_tolerance_seconds:
            logger.debug(
                "Rejected on-demand quote for %s: quote timestamp is in the future (skew %.1fs > allowed %.1fs)",
                pos.symbol,
                -delta_seconds,
                self.clock_skew_tolerance_seconds,
            )
            return False

        quote_age = max(0.0, delta_seconds)
        if quote_age > self.max_quote_age_seconds:
            logger.debug(
                "Rejected on-demand quote for %s: quote timestamp is stale (age %.1fs > %.1fs)",
                pos.symbol,
                quote_age,
                self.max_quote_age_seconds,
            )
            return False

        pos.current_bid = bid
        pos.last_bid_time = quote_ts
        pos.last_quote_time = quote_ts
        if fresh_quote.get("ltp") is not None and float(fresh_quote["ltp"]) > 0:
            pos.current_ltp = float(fresh_quote["ltp"])
        if fresh_quote.get("ask") is not None and float(fresh_quote["ask"]) > 0:
            pos.current_ask = float(fresh_quote["ask"])
            pos.last_ask_time = quote_ts
        return True

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
                    self._last_persisted_time[pos.position_id] = now_ist()
                    self._last_persisted_quote_time[pos.position_id] = pos.last_quote_time
                    self._last_persisted_candle_time[pos.position_id] = pos.last_candle_time

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
        last_candle_time = self._parse_dt(row.get("last_candle_time"))
        last_bid_time = self._parse_dt(row.get("last_bid_time"))
        last_ask_time = self._parse_dt(row.get("last_ask_time"))
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
        is_t1_hit = bool(row.get("is_t1_hit", False)) or (
            status == PositionStatus.PARTIALLY_CLOSED or exit_reason == ExitReason.TARGET_2
        )
        is_breakeven_sl_active = bool(row.get("is_breakeven_sl_active", False)) or (
            status == PositionStatus.PARTIALLY_CLOSED
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
            last_candle_time=last_candle_time,
            last_bid_time=last_bid_time,
            last_ask_time=last_ask_time,
            status=status,
            exit_reason=exit_reason,
            is_t1_hit=is_t1_hit,
            is_breakeven_sl_active=is_breakeven_sl_active,
            current_ltp=float(row.get("current_ltp", 0.0)),
            current_bid=float(row["current_bid"]) if row.get("current_bid") is not None else None,
            current_ask=float(row["current_ask"]) if row.get("current_ask") is not None else None,
            is_exit_triggered=is_exit_triggered,
            pending_exit_reason=pending_exit_reason,
            gross_pnl=float(row.get("gross_pnl", 0.0)),
            net_pnl=float(row.get("net_pnl", 0.0)),
            total_costs=float(row.get("total_costs", 0.0)),
            realized_rr=float(row["realized_rr"]) if row.get("realized_rr") is not None else None,
            is_estimated_exit=bool(row.get("is_estimated_exit", False)),
            notes=row.get("notes"),
        )

        # Restore execution history (trades)
        trades: List[PositionTrade] = []
        if orders:
            filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
            sell_orders = [o for o in filled_orders if o.side == OrderSide.SELL]
            for order in sorted(filled_orders, key=lambda o: o.created_at):
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
                    trade_reason = getattr(order, "reason", None)
                    if trade_reason is None:
                        is_final_sell = (
                            pos.status == PositionStatus.CLOSED
                            and (not sell_orders or order == sell_orders[-1])
                        )
                        trade_reason = pos.exit_reason if is_final_sell else ExitReason.TARGET_1
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
                    is_estimated_fill=getattr(order, "is_estimated_fill", False) if not is_buy else False,
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

        reason_str = row.get("reason")
        order_reason = (
            ExitReason(reason_str)
            if reason_str and reason_str in ExitReason._value2member_map_
            else None
        )

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
            is_estimated_fill=bool(row.get("is_estimated_fill", False)),
            reason=order_reason,
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

    @staticmethod
    def _calc_performance_stats(positions: List[PaperPosition]) -> Dict[str, Any]:
        """Calculates performance statistics for a collection of closed positions."""
        total_trades = len(positions)
        wins = [p for p in positions if p.net_pnl > 0]
        losses = [p for p in positions if p.net_pnl <= 0]

        win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
        avg_win = (sum(p.net_pnl for p in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(p.net_pnl for p in losses) / len(losses)) if losses else 0.0
        gross_profit = sum(p.gross_pnl for p in wins)
        gross_loss = abs(sum(p.gross_pnl for p in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        realized_gross = sum(p.gross_pnl for p in positions)
        realized_net = sum(p.net_pnl for p in positions)
        costs_paid = sum(p.total_costs for p in positions)

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100.0, 1),
            "total_realized_gross_pnl": round(realized_gross, 2),
            "total_realized_net_pnl": round(realized_net, 2),
            "total_costs_paid": round(costs_paid, 2),
            "cost_drag_pct": round((costs_paid / gross_profit * 100.0), 1) if gross_profit > 0 else 0.0,
            "profit_factor": profit_factor,
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
        }

    def get_account_summary(self) -> Dict[str, Any]:
        """Computes comprehensive paper trading performance telemetry.

        Maintains separate verified-only (authentic broker fills) and all-trades performance summaries,
        ensuring estimated penalty fills do not distort verified strategy profitability.
        """
        closed = self.closed_positions
        verified_closed = [p for p in closed if not p.is_estimated_exit]
        estimated_closed = [p for p in closed if p.is_estimated_exit]

        verified_stats = self._calc_performance_stats(verified_closed)
        all_stats = self._calc_performance_stats(closed)

        open_unrealized_gross = sum(p.gross_pnl for p in self.active_positions.values())
        open_unrealized_net = sum(p.net_pnl for p in self.active_positions.values())

        return {
            # Primary performance metrics reflect verified-only trades
            **verified_stats,
            "estimated_trades_count": len(estimated_closed),
            "open_positions_count": len(self.active_positions),
            "open_unrealized_gross_pnl": round(open_unrealized_gross, 2),
            "open_unrealized_net_pnl": round(open_unrealized_net, 2),
            # Explicit separate summaries
            "verified_only": verified_stats,
            "all_trades": all_stats,
        }
