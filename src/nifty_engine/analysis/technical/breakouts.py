"""Breakout and Retest State Machine Engine.

Implements rigorous deterministic state transitions for technical breakouts:
APPROACHING_LEVEL -> BREAKOUT_DETECTED -> BREAKOUT_CONFIRMED -> RETEST_PENDING -> RETEST_CONFIRMED -> SETUP_ELIGIBLE
With terminal fail-safes:
FALSE_BREAKOUT, RETEST_FAILED, SETUP_EXPIRED, INVALIDATED.

All volume confirmation metrics strictly mandate NIFTY Futures RVOL (never 0-volume spot index).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from nifty_engine.analysis.technical.support_resistance import PriceZone
from nifty_engine.core.interfaces import BreakoutState, Candle


class BreakoutDirection(str, Enum):
    BULLISH = "BULLISH"  # Breaking above resistance
    BEARISH = "BEARISH"  # Breaking below support


@dataclass
class BreakoutContext:
    """Tracks state and telemetry for an active level breakout setup."""
    level_id: str
    level_type: str
    nominal_price: float
    direction: BreakoutDirection
    current_state: BreakoutState = BreakoutState.APPROACHING_LEVEL
    start_time: Optional[datetime] = None
    breakout_time: Optional[datetime] = None
    breakout_candle_index: Optional[int] = None
    breakout_price: Optional[float] = None
    breakout_rvol: Optional[float] = None
    retest_time: Optional[datetime] = None
    retest_candle_index: Optional[int] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    candles_since_breakout: int = 0
    invalidation_reason: Optional[str] = None
    history: List[Tuple[BreakoutState, datetime, str]] = field(default_factory=list)
    setup_id: str = ""
    is_notified: bool = False
    zone: Optional[PriceZone] = None
    symbol: str = ""
    timeframe: str = ""

    def __post_init__(self):
        if not self.setup_id:
            ts_str = str(int(self.start_time.timestamp())) if self.start_time else "0"
            parts = [p for p in (self.symbol, self.timeframe, self.level_id, self.direction.value, ts_str) if p]
            self.setup_id = "_".join(parts)

    def transition_to(self, new_state: BreakoutState, timestamp: datetime, reason: str):
        self.current_state = new_state
        self.history.append((new_state, timestamp, reason))


class BreakoutStateMachine:
    """Evaluates candle streams against key support/resistance levels and trendlines."""

    def __init__(
        self,
        min_breakout_rvol: float = 1.3,
        min_body_atr_ratio: float = 0.35,
        approach_buffer_atr: float = 0.25,
        invalidation_buffer_atr: float = 0.20,
        max_retest_candles: int = 6,
    ):
        self.min_breakout_rvol = min_breakout_rvol
        self.min_body_atr_ratio = min_body_atr_ratio
        self.approach_buffer_atr = approach_buffer_atr
        self.invalidation_buffer_atr = invalidation_buffer_atr
        self.max_retest_candles = max_retest_candles

    def update_candle(
        self,
        candle: Candle,
        candle_index: int,
        zone: Optional[PriceZone],
        atr_value: float,
        futures_rvol: float,
        context: Optional[BreakoutContext] = None,
    ) -> BreakoutContext:
        """Processes a newly closed candle against a PriceZone and transitions state."""
        # 1. Initialize context if starting from scratch
        if context is None:
            if zone is None:
                raise ValueError("Initial BreakoutContext creation requires a valid PriceZone")
            # If zone is above candle open, it was resistance -> candidate BULLISH breakout
            # If zone is below candle open, it was support -> candidate BEARISH breakdown
            direction = (
                BreakoutDirection.BULLISH
                if zone.nominal_price >= candle.open
                else BreakoutDirection.BEARISH
            )
            context = BreakoutContext(
                level_id=f"{zone.level_type.value}_{zone.nominal_price:.1f}",
                level_type=zone.level_type.value,
                nominal_price=zone.nominal_price,
                direction=direction,
                current_state=BreakoutState.APPROACHING_LEVEL,
                start_time=candle.timestamp,
                zone=zone,
                symbol=candle.symbol,
                timeframe=candle.timeframe,
            )
        else:
            if zone is not None:
                context.zone = zone
            else:
                zone = context.zone

        if zone is None:
            context.transition_to(
                BreakoutState.INVALIDATED,
                candle.timestamp,
                "Reference zone unavailable and no stored zone in context",
            )
            return context

        effective_atr = atr_value if atr_value is not None and atr_value > 0.0 else max(zone.upper_bound - zone.nominal_price, 5.0)
        approach_dist = max(self.approach_buffer_atr * effective_atr, 4.0)
        effective_rvol = futures_rvol if futures_rvol is not None else 1.0

        state = context.current_state

        # If already in a terminal state, return without modification
        if state in (
            BreakoutState.SETUP_ELIGIBLE,
            BreakoutState.FALSE_BREAKOUT,
            BreakoutState.RETEST_FAILED,
            BreakoutState.SETUP_EXPIRED,
            BreakoutState.INVALIDATED,
        ):
            return context

        # 2. State: APPROACHING_LEVEL
        if state == BreakoutState.APPROACHING_LEVEL:
            if context.direction == BreakoutDirection.BULLISH:
                # Bullish: candle high touches or pierces resistance
                if candle.high >= zone.lower_bound:
                    # Check if it already closed above zone
                    if candle.close > zone.upper_bound:
                        self._evaluate_breakout_confirmation(
                            context, candle, candle_index, zone, effective_atr, effective_rvol
                        )
                    else:
                        context.transition_to(
                            BreakoutState.BREAKOUT_DETECTED,
                            candle.timestamp,
                            f"Price reached resistance zone [{zone.lower_bound:.1f}, {zone.upper_bound:.1f}]",
                        )
            else:
                # Bearish: candle low touches or pierces support
                if candle.low <= zone.upper_bound:
                    if candle.close < zone.lower_bound:
                        self._evaluate_breakout_confirmation(
                            context, candle, candle_index, zone, effective_atr, effective_rvol
                        )
                    else:
                        context.transition_to(
                            BreakoutState.BREAKOUT_DETECTED,
                            candle.timestamp,
                            f"Price reached support zone [{zone.lower_bound:.1f}, {zone.upper_bound:.1f}]",
                        )

        # 3. State: BREAKOUT_DETECTED
        elif state == BreakoutState.BREAKOUT_DETECTED:
            self._evaluate_breakout_confirmation(
                context, candle, candle_index, zone, effective_atr, effective_rvol
            )

        # 4. State: BREAKOUT_CONFIRMED / RETEST_PENDING
        elif state in (BreakoutState.BREAKOUT_CONFIRMED, BreakoutState.RETEST_PENDING):
            context.candles_since_breakout += 1

            # Check expiration
            if context.candles_since_breakout > self.max_retest_candles:
                context.transition_to(
                    BreakoutState.SETUP_EXPIRED,
                    candle.timestamp,
                    f"Retest did not materialize within {self.max_retest_candles} bars",
                )
                return context

            if context.direction == BreakoutDirection.BULLISH:
                # If candle closes back below support zone
                if candle.close < zone.lower_bound:
                    if state == BreakoutState.RETEST_PENDING:
                        context.transition_to(
                            BreakoutState.RETEST_FAILED,
                            candle.timestamp,
                            f"Retest failed: candle closed below support at {candle.close:.1f}",
                        )
                    else:
                        context.transition_to(
                            BreakoutState.FALSE_BREAKOUT,
                            candle.timestamp,
                            f"Candle closed back below broken resistance ({candle.close:.1f} < {zone.lower_bound:.1f})",
                        )
                    return context

                # Check for Retest in progress
                if candle.low <= zone.upper_bound + approach_dist:
                    # Retest Confirmation: candle low tested zone, close remained above zone and formed green bar
                    if candle.close >= zone.nominal_price and candle.close > candle.open:
                        context.retest_time = candle.timestamp
                        context.retest_candle_index = candle_index
                        context.stop_loss = round(candle.low - (0.1 * effective_atr), 2)
                        risk = max(candle.close - context.stop_loss, 10.0)
                        context.target_price = round(candle.close + (1.5 * risk), 2)

                        context.transition_to(
                            BreakoutState.RETEST_CONFIRMED,
                            candle.timestamp,
                            f"Bullish retest rejection wick formed at {candle.low:.1f}, close {candle.close:.1f}",
                        )
                        context.transition_to(
                            BreakoutState.SETUP_ELIGIBLE,
                            candle.timestamp,
                            f"Ready for CE options selection. SL={context.stop_loss}, Target={context.target_price}",
                        )
                    else:
                        if state != BreakoutState.RETEST_PENDING:
                            context.transition_to(
                                BreakoutState.RETEST_PENDING,
                                candle.timestamp,
                                f"Retest in progress: price dipping towards {zone.nominal_price:.1f}",
                            )

            elif context.direction == BreakoutDirection.BEARISH:
                # If candle closes back above resistance zone
                if candle.close > zone.upper_bound:
                    if state == BreakoutState.RETEST_PENDING:
                        context.transition_to(
                            BreakoutState.RETEST_FAILED,
                            candle.timestamp,
                            f"Retest failed: candle closed above resistance at {candle.close:.1f}",
                        )
                    else:
                        context.transition_to(
                            BreakoutState.FALSE_BREAKOUT,
                            candle.timestamp,
                            f"Candle closed back above broken support ({candle.close:.1f} > {zone.upper_bound:.1f})",
                        )
                    return context

                # Check for Retest in progress
                if candle.high >= zone.lower_bound - approach_dist:
                    if candle.close <= zone.nominal_price and candle.close < candle.open:
                        context.retest_time = candle.timestamp
                        context.retest_candle_index = candle_index
                        context.stop_loss = round(candle.high + (0.1 * effective_atr), 2)
                        risk = max(context.stop_loss - candle.close, 10.0)
                        context.target_price = round(candle.close - (1.5 * risk), 2)

                        context.transition_to(
                            BreakoutState.RETEST_CONFIRMED,
                            candle.timestamp,
                            f"Bearish retest rejection wick formed at {candle.high:.1f}, close {candle.close:.1f}",
                        )
                        context.transition_to(
                            BreakoutState.SETUP_ELIGIBLE,
                            candle.timestamp,
                            f"Ready for PE options selection. SL={context.stop_loss}, Target={context.target_price}",
                        )
                    else:
                        if state != BreakoutState.RETEST_PENDING:
                            context.transition_to(
                                BreakoutState.RETEST_PENDING,
                                candle.timestamp,
                                f"Retest in progress: price rising towards {zone.nominal_price:.1f}",
                            )

        return context

    def _evaluate_breakout_confirmation(
        self,
        context: BreakoutContext,
        candle: Candle,
        candle_index: int,
        zone: PriceZone,
        atr_value: float,
        futures_rvol: float,
    ):
        body = abs(candle.close - candle.open)
        body_atr_ratio = body / atr_value if atr_value > 0 else 0.0

        if context.direction == BreakoutDirection.BULLISH:
            if candle.close > zone.upper_bound:
                # Check volume & momentum filters
                if futures_rvol >= self.min_breakout_rvol and body_atr_ratio >= self.min_body_atr_ratio:
                    context.breakout_time = candle.timestamp
                    context.breakout_candle_index = candle_index
                    context.breakout_price = candle.close
                    context.breakout_rvol = futures_rvol
                    context.transition_to(
                        BreakoutState.BREAKOUT_CONFIRMED,
                        candle.timestamp,
                        f"Bullish breakout confirmed: Close={candle.close:.1f} > {zone.upper_bound:.1f}, RVOL={futures_rvol:.2f}x >= {self.min_breakout_rvol}x, Body/ATR={body_atr_ratio:.2f}",
                    )
                else:
                    context.transition_to(
                        BreakoutState.FALSE_BREAKOUT,
                        candle.timestamp,
                        f"Breakout lacked volume/momentum confirmation: RVOL={futures_rvol:.2f}x (req {self.min_breakout_rvol}x), Body/ATR={body_atr_ratio:.2f}",
                    )
            elif candle.close < zone.lower_bound:
                context.transition_to(
                    BreakoutState.FALSE_BREAKOUT,
                    candle.timestamp,
                    f"Failed to close above resistance: closed at {candle.close:.1f}",
                )

        elif context.direction == BreakoutDirection.BEARISH:
            if candle.close < zone.lower_bound:
                if futures_rvol >= self.min_breakout_rvol and body_atr_ratio >= self.min_body_atr_ratio:
                    context.breakout_time = candle.timestamp
                    context.breakout_candle_index = candle_index
                    context.breakout_price = candle.close
                    context.breakout_rvol = futures_rvol
                    context.transition_to(
                        BreakoutState.BREAKOUT_CONFIRMED,
                        candle.timestamp,
                        f"Bearish breakdown confirmed: Close={candle.close:.1f} < {zone.lower_bound:.1f}, RVOL={futures_rvol:.2f}x >= {self.min_breakout_rvol}x, Body/ATR={body_atr_ratio:.2f}",
                    )
                else:
                    context.transition_to(
                        BreakoutState.FALSE_BREAKOUT,
                        candle.timestamp,
                        f"Breakdown lacked volume/momentum confirmation: RVOL={futures_rvol:.2f}x (req {self.min_breakout_rvol}x), Body/ATR={body_atr_ratio:.2f}",
                    )
            elif candle.close > zone.upper_bound:
                context.transition_to(
                    BreakoutState.FALSE_BREAKOUT,
                    candle.timestamp,
                    f"Failed to close below support: closed at {candle.close:.1f}",
                )
