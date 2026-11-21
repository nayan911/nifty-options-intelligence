"""Multi-Timeframe Technical Synthesis Engine.

Coordinates the 3-tier hierarchical technical analysis funnel:
1. Tier 1 (15m Macro Regime & Filter):
   - Trend direction validation (BULLISH, BEARISH, SIDEWAYS).
   - Headroom verification: requires >= 1.5x ATR distance to opposing 15m S/R barriers.
   - Causal snapshot retrieval with zero lookahead bias and strict freshness thresholds.
2. Tier 2 (5m Setup Qualification):
   - Listens for 5m Breakout & Retest transitions to SETUP_ELIGIBLE.
   - Validates futures volume surge (RVOL >= 1.3x).
   - Distinguishes scheduled bar_close_time from genuine data availability timestamp (available_at).
   - Pre-qualifies trade setups and buffers them for 1m entry timing.
3. Tier 3 (1m Precision Entry Timing):
   - Causal sequencing: 1m entry candle must close after genuine available_at timestamp.
   - Revalidates 15m macro context and opposing levels immediately prior to entry.
   - Intrabar stop-loss breach check using high/low extremes.
   - Validates short-term momentum alignment (EMA 9, basis-synchronized VWAP, RSI sweet-spot).
   - Permanent invalidation for failed breakout levels or stop-loss breaches.
   - Optimizes stop-loss anchoring to recent 1m micro-swings with correct directional boundary.
   - Enforces Target 1 obstacle clearance and strictly caps Target 2 at opposing 15m barriers.
   - Strict zero-tolerance confluence scoring: missing/unknown data awards 0 points.
   - Emits a unified, deterministic TechnicalCandidate.
"""

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from nifty_engine.analysis.technical.breakouts import (
    BreakoutContext,
    BreakoutDirection,
)
from nifty_engine.analysis.technical.engine import (
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
    get_bar_close_time,
)
from nifty_engine.analysis.technical.support_resistance import PriceZone
from nifty_engine.core.interfaces import Candle


class SetupLifecycleState(str, Enum):
    """Lifecycle states of an active 5m breakout setup awaiting entry confirmation."""

    WAITING_ENTRY = "WAITING_ENTRY"
    CANDIDATE_EMITTED = "CANDIDATE_EMITTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass
class TechnicalCandidate:
    """Actionable quantitative multi-timeframe trade candidate.

    Ready for Phase 3 Options Flow, OI & Greek validation.
    Strictly paper-trading mode (zero broker order placement).
    """

    candidate_id: str
    symbol: str
    timestamp: datetime
    direction: BreakoutDirection
    setup_level: str
    nominal_price: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_points: float
    reward_points: float
    risk_reward_ratio: float
    confluence_score: float
    headroom_points: Optional[float] = None
    timeframe_breakdown: Dict[str, Any] = field(default_factory=dict)
    is_paper_trading: bool = True

    def summary(self) -> str:
        """Formatted human-readable summary of the technical candidate."""
        hd_str = f"{self.headroom_points:.1f}pts" if self.headroom_points is not None else "N/A"
        return (
            f"[{self.candidate_id}] {self.symbol} {self.direction.value} @ {self.entry_price:.2f} | "
            f"SL: {self.stop_loss:.2f} | T1: {self.target_1:.2f} (1.5R) | T2: {self.target_2:.2f} (2.0R) | "
            f"Risk: {self.risk_points:.1f}pts | Headroom: {hd_str} | Score: {self.confluence_score:.0f}/100"
        )


@dataclass
class Active5mSetup:
    """Tracks an eligible 5m setup awaiting 1m confirmation with causal timestamps and lifecycle."""

    setup: BreakoutContext
    snapshot_5m: TechnicalSnapshot
    qualified_time: datetime  # Opening timestamp of the 5m candle
    bar_close_time: datetime  # Scheduled completion timestamp of 5m candle (snap_5m.bar_close_time)
    available_at: datetime  # Actual timestamp when confirmation (e.g. futures volume) completed
    expiry_time: datetime  # available_at + entry_window_minutes
    headroom_points: Optional[float] = None
    opposing_zone: Optional[PriceZone] = None
    state: SetupLifecycleState = SetupLifecycleState.WAITING_ENTRY
    invalidation_reason: Optional[str] = None


class MultiTimeframeAnalyzer:
    """Synthesizes 15m structure, 5m setups, and 1m execution into unified candidates."""

    def __init__(
        self,
        engine: TechnicalAnalysisEngine,
        on_candidate: Optional[Callable[[TechnicalCandidate], None]] = None,
        min_confluence_score: float = 60.0,
        min_headroom_atr_multiple: float = 1.5,
        min_risk_points: float = 8.0,
        entry_window_minutes: int = 15,
        max_1m_futures_age_seconds: float = 60.0,
        max_5m_futures_age_seconds: float = 300.0,
        max_15m_macro_age_seconds: float = 1200.0,
    ):
        self._lock = threading.RLock()
        self.engine = engine
        self.on_candidate = on_candidate
        self.min_confluence_score = min_confluence_score
        self.min_headroom_atr_multiple = min_headroom_atr_multiple
        self.min_risk_points = min_risk_points
        self.entry_window_minutes = entry_window_minutes
        self.max_1m_futures_age_seconds = max_1m_futures_age_seconds
        self.max_5m_futures_age_seconds = max_5m_futures_age_seconds
        self.max_15m_macro_age_seconds = max_15m_macro_age_seconds

        # In-flight setups waiting for 1m entry timing: {setup_id: Active5mSetup}
        self.active_setups: Dict[str, Active5mSetup] = {}
        # Avoid duplicate candidate dispatches: {candidate_id}
        self.notified_candidates: Set[str] = set()
        self.current_session_date: Optional[date] = None

    def _check_session_rollover(self, timestamp: datetime) -> None:
        """Resets active setups and candidate tracking on new trading session date."""
        candle_date = timestamp.date()
        if self.current_session_date is None:
            self.current_session_date = candle_date
            return
        if candle_date > self.current_session_date:
            self.current_session_date = candle_date
            self.active_setups.clear()
            self.notified_candidates.clear()

    def on_5m_setup(
        self,
        setup: BreakoutContext,
        snap_5m: TechnicalSnapshot,
        available_at: Optional[datetime] = None,
    ) -> Optional[Active5mSetup]:
        """Tier 1 & Tier 2: Evaluates 15m macro regime & headroom for a newly eligible 5m setup."""
        with self._lock:
            self._check_session_rollover(snap_5m.timestamp)

            bar_close = snap_5m.bar_close_time
            # Genuine availability timestamp when all volume confirmations completed
            actual_available_at = available_at or getattr(snap_5m, "available_at", None) or bar_close
            actual_available_at = max(actual_available_at, bar_close)

            # 1. Retrieve 15m snapshot available as of actual_available_at (strictly zero lookahead)
            snap_15m = self.engine.get_snapshot(setup.symbol, timeframe="15m", as_of=actual_available_at)
            if not snap_15m:
                # 15m structure not yet completed/observed by available_at; cannot qualify without macro context
                return None

            # Verify no future leakage: 15m bar close must be on or before actual_available_at
            if snap_15m.bar_close_time > actual_available_at:
                return None

            # Verify freshness of 15m market context
            age_15m = (actual_available_at - snap_15m.bar_close_time).total_seconds()
            if age_15m > self.max_15m_macro_age_seconds or snap_15m.timestamp.date() != actual_available_at.date():
                return None

            # 2. Tier 1 Macro Trend Bias Filter
            macro_trend = snap_15m.trend_bias
            if macro_trend == "INSUFFICIENT_DATA":
                return None

            if setup.direction == BreakoutDirection.BULLISH:
                if macro_trend == "BEARISH":
                    # Strictly reject counter-trend bullish breakout against 15m downtrend
                    return None
            elif setup.direction == BreakoutDirection.BEARISH:
                if macro_trend == "BULLISH":
                    # Strictly reject counter-trend bearish breakout against 15m uptrend
                    return None

            # 3. Tier 1 Headroom to Opposing 15m S/R Obstacles
            atr_5m = snap_5m.atr
            min_headroom = max(self.min_headroom_atr_multiple * atr_5m, 25.0) if atr_5m is not None else 25.0

            opposing_zone: Optional[PriceZone] = None
            headroom_points: Optional[float] = None

            if setup.direction == BreakoutDirection.BULLISH:
                if snap_15m.nearest_resistance and snap_15m.nearest_resistance.nominal_price > setup.nominal_price:
                    opposing_zone = snap_15m.nearest_resistance
                    headroom_points = max(0.0, opposing_zone.lower_bound - snap_5m.close)
                    if headroom_points < min_headroom:
                        # Major 15m resistance is too close; unfavorable risk-reward
                        return None
            elif setup.direction == BreakoutDirection.BEARISH:
                if snap_15m.nearest_support and snap_15m.nearest_support.nominal_price < setup.nominal_price:
                    opposing_zone = snap_15m.nearest_support
                    headroom_points = max(0.0, snap_5m.close - opposing_zone.upper_bound)
                    if headroom_points < min_headroom:
                        # Major 15m support is too close; unfavorable risk-reward
                        return None

            # 4. Qualify 5m setup and wait for 1m entry trigger
            expiry_dt = actual_available_at + timedelta(minutes=self.entry_window_minutes)
            active = Active5mSetup(
                setup=setup,
                snapshot_5m=snap_5m,
                qualified_time=snap_5m.timestamp,
                bar_close_time=bar_close,
                available_at=actual_available_at,
                expiry_time=expiry_dt,
                headroom_points=headroom_points,
                opposing_zone=opposing_zone,
                state=SetupLifecycleState.WAITING_ENTRY,
            )
            self.active_setups[setup.setup_id] = active
            return active

    def on_1m_candle(
        self,
        candle: Candle,
        snap_1m: TechnicalSnapshot,
    ) -> List[TechnicalCandidate]:
        """Tier 3: Precision entry timing and candidate generation on 1m candle close."""
        with self._lock:
            self._check_session_rollover(candle.timestamp)

            candle_close_time = get_bar_close_time(candle.timestamp, "1m")

            # 1. Purge expired active setups
            for sid, active in list(self.active_setups.items()):
                if candle_close_time > active.expiry_time or candle.timestamp.date() > active.qualified_time.date():
                    active.state = SetupLifecycleState.EXPIRED
                    del self.active_setups[sid]

            generated_candidates: List[TechnicalCandidate] = []

            # 2. Evaluate remaining active setups
            for sid, active in list(self.active_setups.items()):
                if active.setup.symbol != candle.symbol:
                    continue

                # Causal Check: 1m candle must complete strictly AFTER the 5m setup became available
                # and its opening must not precede the 5m bar close
                if candle_close_time <= active.available_at or candle.timestamp < active.bar_close_time:
                    continue

                candidate = self._evaluate_1m_entry(
                    candle=candle,
                    snap_1m=snap_1m,
                    active=active,
                    candle_close_time=candle_close_time,
                )

                if candidate:
                    active.state = SetupLifecycleState.CANDIDATE_EMITTED
                    generated_candidates.append(candidate)
                    del self.active_setups[sid]
                    if self.on_candidate:
                        self.on_candidate(candidate)
                elif active.state == SetupLifecycleState.INVALIDATED:
                    # Structurally broken setup: permanently drop from active setups
                    del self.active_setups[sid]

            return generated_candidates

    def _evaluate_vwap_holding(
        self,
        snap: TechnicalSnapshot,
        price: float,
        direction: BreakoutDirection,
        as_of: datetime,
    ) -> bool:
        """Verifies price is holding the correct side of VWAP without spot/futures basis distortion."""
        if snap.vwap is not None:
            if direction == BreakoutDirection.BULLISH:
                return price >= snap.vwap
            else:
                return price <= snap.vwap

        # Spot index has no traded volume: check paired futures price against paired futures VWAP
        paired_futures = self.engine.index_futures_map.get(snap.symbol)
        if paired_futures:
            fut_snap = self.engine.get_snapshot(paired_futures, timeframe=snap.timeframe, as_of=as_of)
            if fut_snap and fut_snap.vwap is not None:
                max_age = self.max_1m_futures_age_seconds if snap.timeframe == "1m" else self.max_5m_futures_age_seconds
                age = (as_of - fut_snap.bar_close_time).total_seconds()
                if age <= max_age:
                    if direction == BreakoutDirection.BULLISH:
                        return fut_snap.close >= fut_snap.vwap
                    else:
                        return fut_snap.close <= fut_snap.vwap

        # No fresh VWAP available on either: does not fail holding check
        return True

    def _evaluate_vwap_alignment(
        self,
        snap: TechnicalSnapshot,
        direction: BreakoutDirection,
        timeframe: str,
        as_of: Optional[datetime] = None,
    ) -> bool:
        """Checks if symbol or paired futures confirms VWAP positioning (used for positive scoring)."""
        cutoff = as_of or snap.bar_close_time

        if snap.vwap is not None:
            if direction == BreakoutDirection.BULLISH:
                return snap.close >= snap.vwap
            else:
                return snap.close <= snap.vwap

        paired_futures = self.engine.index_futures_map.get(snap.symbol)
        if paired_futures:
            fut_snap = self.engine.get_snapshot(paired_futures, timeframe=timeframe, as_of=cutoff)
            if fut_snap and fut_snap.vwap is not None:
                max_age = self.max_1m_futures_age_seconds if timeframe == "1m" else self.max_5m_futures_age_seconds
                age = (cutoff - fut_snap.bar_close_time).total_seconds()
                if age <= max_age:
                    if direction == BreakoutDirection.BULLISH:
                        return fut_snap.close >= fut_snap.vwap
                    else:
                        return fut_snap.close <= fut_snap.vwap

        return False

    def _evaluate_1m_entry(
        self,
        candle: Candle,
        snap_1m: TechnicalSnapshot,
        active: Active5mSetup,
        candle_close_time: datetime,
    ) -> Optional[TechnicalCandidate]:
        """Validates 1m execution criteria, optimizes stop loss, and calculates confluence score."""
        setup = active.setup
        nominal_lvl = setup.nominal_price

        # 1. Intrabar & Close Stop-Loss Breach Check: hard invalidation boundary
        base_sl = setup.stop_loss if setup.stop_loss is not None else (
            nominal_lvl - 25.0 if setup.direction == BreakoutDirection.BULLISH else nominal_lvl + 25.0
        )
        if setup.direction == BreakoutDirection.BULLISH:
            if candle.low <= base_sl or candle.close <= base_sl:
                active.state = SetupLifecycleState.INVALIDATED
                active.invalidation_reason = f"1m low {candle.low:.2f} breached base stop loss {base_sl:.2f}"
                return None
        else:
            if candle.high >= base_sl or candle.close >= base_sl:
                active.state = SetupLifecycleState.INVALIDATED
                active.invalidation_reason = f"1m high {candle.high:.2f} breached base stop loss {base_sl:.2f}"
                return None

        # 2. Level Holding Check: Breakout level must hold on 1m close
        if setup.direction == BreakoutDirection.BULLISH:
            if candle.close < nominal_lvl:
                active.state = SetupLifecycleState.INVALIDATED
                active.invalidation_reason = f"1m close {candle.close:.2f} failed to hold breakout level {nominal_lvl:.2f}"
                return None
        else:
            if candle.close > nominal_lvl:
                active.state = SetupLifecycleState.INVALIDATED
                active.invalidation_reason = f"1m close {candle.close:.2f} failed to hold breakdown level {nominal_lvl:.2f}"
                return None

        # 3. Revalidate 15m Macro Context as of entry candle close
        snap_15m = self.engine.get_snapshot(setup.symbol, timeframe="15m", as_of=candle_close_time)
        if not snap_15m:
            return None

        # Freshness: snapshot must be from current session and within max_15m_macro_age_seconds
        age_15m = (candle_close_time - snap_15m.bar_close_time).total_seconds()
        if age_15m > self.max_15m_macro_age_seconds or snap_15m.timestamp.date() != candle.timestamp.date():
            return None

        # Macro trend bias must still align (no counter-trend entry)
        if setup.direction == BreakoutDirection.BULLISH and snap_15m.trend_bias == "BEARISH":
            active.state = SetupLifecycleState.INVALIDATED
            active.invalidation_reason = "15m macro trend reversed to BEARISH before entry"
            return None
        elif setup.direction == BreakoutDirection.BEARISH and snap_15m.trend_bias == "BULLISH":
            active.state = SetupLifecycleState.INVALIDATED
            active.invalidation_reason = "15m macro trend reversed to BULLISH before entry"
            return None

        # 4. 1m Momentum Confirmation
        if setup.direction == BreakoutDirection.BULLISH:
            # Bullish candle momentum: green candle OR confirmed above EMA 9
            if snap_1m.ema_9 is not None:
                candle_momentum_ok = (candle.close >= snap_1m.ema_9) or (candle.close > candle.open)
            else:
                candle_momentum_ok = (candle.close > candle.open)

            if not candle_momentum_ok:
                return None

            # VWAP filter: price or paired futures must not be below VWAP
            if not self._evaluate_vwap_holding(snap_1m, candle.close, BreakoutDirection.BULLISH, candle_close_time):
                return None

            # Overbought RSI exhaustion filter
            if snap_1m.rsi is not None and snap_1m.rsi > 75.0:
                return None
        else:
            # Bearish candle momentum: red candle OR confirmed below EMA 9
            if snap_1m.ema_9 is not None:
                candle_momentum_ok = (candle.close <= snap_1m.ema_9) or (candle.close < candle.open)
            else:
                candle_momentum_ok = (candle.close < candle.open)

            if not candle_momentum_ok:
                return None

            # VWAP filter: price or paired futures must not be above VWAP
            if not self._evaluate_vwap_holding(snap_1m, candle.close, BreakoutDirection.BEARISH, candle_close_time):
                return None

            # Oversold RSI exhaustion filter
            if snap_1m.rsi is not None and snap_1m.rsi < 25.0:
                return None

        # 5. Stop-Loss Optimization using 1m Micro-Structure
        entry_price = candle.close
        effective_sl = base_sl

        if setup.direction == BreakoutDirection.BULLISH:
            if snap_1m.nearest_support:
                sup_lvl = snap_1m.nearest_support.lower_bound
                if base_sl < sup_lvl < entry_price - self.min_risk_points:
                    effective_sl = round(sup_lvl, 1)
            # Guarantee stop-loss is strictly on the correct side of entry
            if effective_sl >= entry_price:
                effective_sl = round(entry_price - self.min_risk_points, 1)
            risk_points = entry_price - effective_sl
            if risk_points < self.min_risk_points:
                effective_sl = round(entry_price - self.min_risk_points, 1)
                risk_points = self.min_risk_points
        else:
            if snap_1m.nearest_resistance:
                res_lvl = snap_1m.nearest_resistance.upper_bound
                if base_sl > res_lvl > entry_price + self.min_risk_points:
                    effective_sl = round(res_lvl, 1)
            # Guarantee stop-loss is strictly on the correct side of entry
            if effective_sl <= entry_price:
                effective_sl = round(entry_price + self.min_risk_points, 1)
            risk_points = effective_sl - entry_price
            if risk_points < self.min_risk_points:
                effective_sl = round(entry_price + self.min_risk_points, 1)
                risk_points = self.min_risk_points

        # 6. Targets & Realized Headroom Validation (Target 1 check + Target 2 capping)
        nominal_target_1 = (
            round(entry_price + 1.5 * risk_points, 1)
            if setup.direction == BreakoutDirection.BULLISH
            else round(entry_price - 1.5 * risk_points, 1)
        )
        nominal_target_2 = (
            round(entry_price + 2.0 * risk_points, 1)
            if setup.direction == BreakoutDirection.BULLISH
            else round(entry_price - 2.0 * risk_points, 1)
        )

        target_1 = nominal_target_1
        target_2 = nominal_target_2
        live_headroom: Optional[float] = None

        if setup.direction == BreakoutDirection.BULLISH:
            if snap_15m.nearest_resistance and snap_15m.nearest_resistance.nominal_price > entry_price:
                barrier_price = snap_15m.nearest_resistance.lower_bound
                live_headroom = max(0.0, barrier_price - entry_price)
                # Opposing 15m barrier must not block Target 1 (1.5R)
                if live_headroom < 1.5 * risk_points:
                    return None
                # Cap Target 2 at the barrier price if nominal 2.0R extends past it
                if nominal_target_2 > barrier_price:
                    target_2 = round(barrier_price, 1)
            reward_points = target_1 - entry_price
        else:
            if snap_15m.nearest_support and snap_15m.nearest_support.nominal_price < entry_price:
                barrier_price = snap_15m.nearest_support.upper_bound
                live_headroom = max(0.0, entry_price - barrier_price)
                # Opposing 15m barrier must not block Target 1 (1.5R)
                if live_headroom < 1.5 * risk_points:
                    return None
                # Cap Target 2 at the barrier price if nominal 2.0R extends past it
                if nominal_target_2 < barrier_price:
                    target_2 = round(barrier_price, 1)
            reward_points = entry_price - target_1

        rr_ratio = round(reward_points / risk_points, 2) if risk_points > 0 else 0.0

        # 7. Confluence Scoring (0-100) — STRICT: 0 pts for missing/unknown data
        score = self._compute_confluence_score(
            setup=setup,
            snap_1m=snap_1m,
            snap_5m=active.snapshot_5m,
            snap_15m=snap_15m,
            direction=setup.direction,
            headroom_points=live_headroom,
            entry_time=candle_close_time,
        )

        if score < self.min_confluence_score:
            return None

        # 8. Build Candidate
        ts_str = candle.timestamp.strftime("%Y%m%d_%H%M%S")
        candidate_id = f"{setup.symbol.replace(':', '_')}_{ts_str}_{setup.direction.value}_{setup.level_id}"

        if candidate_id in self.notified_candidates:
            return None
        self.notified_candidates.add(candidate_id)

        candidate = TechnicalCandidate(
            candidate_id=candidate_id,
            symbol=setup.symbol,
            timestamp=candle.timestamp,
            direction=setup.direction,
            setup_level=setup.level_id,
            nominal_price=nominal_lvl,
            entry_price=entry_price,
            stop_loss=effective_sl,
            target_1=target_1,
            target_2=target_2,
            risk_points=round(risk_points, 1),
            reward_points=round(reward_points, 1),
            risk_reward_ratio=rr_ratio,
            confluence_score=round(score, 1),
            headroom_points=round(live_headroom, 1) if live_headroom is not None else None,
            timeframe_breakdown={
                "15m_trend": snap_15m.trend_bias,
                "5m_rvol": active.snapshot_5m.rvol,
                "5m_vwap": active.snapshot_5m.vwap,
                "1m_rsi": snap_1m.rsi,
                "1m_vwap": snap_1m.vwap,
            },
            is_paper_trading=True,
        )

        return candidate

    def _compute_confluence_score(
        self,
        setup: BreakoutContext,
        snap_1m: TechnicalSnapshot,
        snap_5m: TechnicalSnapshot,
        snap_15m: Optional[TechnicalSnapshot],
        direction: BreakoutDirection,
        headroom_points: Optional[float],
        entry_time: datetime,
    ) -> float:
        """Calculates 0-100 composite score across all three timeframes.

        Strict zero-tolerance scoring: Missing or unknown data receives 0 points.
        """
        score = 0.0

        # 1. 15m Macro Trend Alignment (max +30 pts, missing/insufficient = +0 pts)
        if snap_15m and snap_15m.trend_bias != "INSUFFICIENT_DATA":
            if direction == BreakoutDirection.BULLISH:
                if snap_15m.trend_bias == "BULLISH":
                    score += 30.0 if not snap_15m.is_provisional else 20.0
                elif snap_15m.trend_bias == "SIDEWAYS":
                    score += 15.0
            elif direction == BreakoutDirection.BEARISH:
                if snap_15m.trend_bias == "BEARISH":
                    score += 30.0 if not snap_15m.is_provisional else 20.0
                elif snap_15m.trend_bias == "SIDEWAYS":
                    score += 15.0

        # 2. 5m Volume Surge (max +25 pts, missing/sub-threshold = +0 pts)
        rvol = max(
            setup.breakout_rvol if setup.breakout_rvol is not None else 0.0,
            snap_5m.rvol if snap_5m.rvol is not None else 0.0,
        )
        if rvol >= 2.0:
            score += 25.0
        elif rvol >= 1.5:
            score += 20.0
        elif rvol >= 1.3:
            score += 15.0

        # 3. VWAP Positioning (max +20 pts, missing = +0 pts)
        if self._evaluate_vwap_alignment(snap_5m, direction, "5m", as_of=snap_5m.bar_close_time):
            score += 10.0
        if self._evaluate_vwap_alignment(snap_1m, direction, "1m", as_of=entry_time):
            score += 10.0

        # 4. 15m Headroom (max +15 pts, unknown/missing ATR/inadequate = +0 pts)
        if headroom_points is not None and snap_5m.atr is not None:
            if headroom_points >= 2.0 * snap_5m.atr:
                score += 15.0
            elif headroom_points >= 1.5 * snap_5m.atr:
                score += 10.0

        # 5. 1m Momentum Sweet-Spot (max +10 pts, missing/outside = +0 pts)
        rsi = snap_1m.rsi
        if rsi is not None:
            if direction == BreakoutDirection.BULLISH and 50.0 <= rsi <= 68.0:
                score += 10.0
            elif direction == BreakoutDirection.BEARISH and 32.0 <= rsi <= 50.0:
                score += 10.0

        return min(100.0, score)
