"""Multi-Timeframe Technical Synthesis Engine.

Coordinates the 3-tier hierarchical technical analysis funnel:
1. Tier 1 (15m Macro Regime & Filter):
   - Trend direction validation (BULLISH, BEARISH, SIDEWAYS).
   - Headroom verification: requires >= 1.5x ATR distance to opposing 15m S/R barriers.
2. Tier 2 (5m Setup Qualification):
   - Listens for 5m Breakout & Retest transitions to SETUP_ELIGIBLE.
   - Validates futures volume surge (RVOL >= 1.3x).
   - Pre-qualifies trade setups and buffers them for 1m entry timing.
3. Tier 3 (1m Precision Entry Timing):
   - Validates short-term momentum alignment (EMA 9, VWAP, RSI sweet-spot).
   - Optimizes stop-loss anchoring to recent 1m micro-swings.
   - Emits a unified, deterministic TechnicalCandidate.
"""

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from nifty_engine.analysis.technical.breakouts import (
    BreakoutContext,
    BreakoutDirection,
)
from nifty_engine.analysis.technical.engine import (
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
)
from nifty_engine.analysis.technical.support_resistance import PriceZone
from nifty_engine.core.interfaces import Candle


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
    headroom_points: float
    timeframe_breakdown: Dict[str, Any] = field(default_factory=dict)
    is_paper_trading: bool = True

    def summary(self) -> str:
        """Formatted human-readable summary of the technical candidate."""
        return (
            f"[{self.candidate_id}] {self.symbol} {self.direction.value} @ {self.entry_price:.2f} | "
            f"SL: {self.stop_loss:.2f} | T1: {self.target_1:.2f} (1.5R) | T2: {self.target_2:.2f} (2.0R) | "
            f"Risk: {self.risk_points:.1f}pts | Score: {self.confluence_score:.0f}/100"
        )


@dataclass
class Active5mSetup:
    """Tracks an eligible 5m setup awaiting 1m confirmation."""

    setup: BreakoutContext
    snapshot_5m: TechnicalSnapshot
    qualified_time: datetime
    expiry_time: datetime
    headroom_points: float
    opposing_zone: Optional[PriceZone] = None


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
    ):
        self._lock = threading.RLock()
        self.engine = engine
        self.on_candidate = on_candidate
        self.min_confluence_score = min_confluence_score
        self.min_headroom_atr_multiple = min_headroom_atr_multiple
        self.min_risk_points = min_risk_points
        self.entry_window_minutes = entry_window_minutes

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
    ) -> Optional[Active5mSetup]:
        """Tier 1 & Tier 2: Evaluates 15m macro regime & headroom for a newly eligible 5m setup."""
        with self._lock:
            self._check_session_rollover(snap_5m.timestamp)

            snap_15m = self.engine.get_snapshot(setup.symbol, timeframe="15m")
            if not snap_15m:
                # 15m structure not yet observed; cannot qualify without macro context
                return None

            # 1. Tier 1 Macro Trend Bias Filter
            macro_trend = snap_15m.trend_bias
            if setup.direction == BreakoutDirection.BULLISH:
                if macro_trend == "BEARISH":
                    # Strictly reject counter-trend bullish breakout against 15m downtrend
                    return None
            elif setup.direction == BreakoutDirection.BEARISH:
                if macro_trend == "BULLISH":
                    # Strictly reject counter-trend bearish breakout against 15m uptrend
                    return None

            # 2. Tier 1 Headroom to Opposing 15m S/R Obstacles
            atr_5m = snap_5m.atr if snap_5m.atr is not None else 20.0
            min_headroom = max(self.min_headroom_atr_multiple * atr_5m, 25.0)

            opposing_zone: Optional[PriceZone] = None
            headroom_points = 999.0

            if setup.direction == BreakoutDirection.BULLISH:
                if snap_15m.nearest_resistance and snap_15m.nearest_resistance.nominal_price > setup.nominal_price:
                    opposing_zone = snap_15m.nearest_resistance
                    headroom_points = opposing_zone.lower_bound - snap_5m.close
                    if headroom_points < min_headroom:
                        # Major 15m resistance is too close; unfavorable risk-reward
                        return None
            elif setup.direction == BreakoutDirection.BEARISH:
                if snap_15m.nearest_support and snap_15m.nearest_support.nominal_price < setup.nominal_price:
                    opposing_zone = snap_15m.nearest_support
                    headroom_points = snap_5m.close - opposing_zone.upper_bound
                    if headroom_points < min_headroom:
                        # Major 15m support is too close; unfavorable risk-reward
                        return None

            # 3. Qualify 5m setup and wait for 1m entry trigger
            expiry_dt = snap_5m.timestamp + timedelta(minutes=self.entry_window_minutes)
            active = Active5mSetup(
                setup=setup,
                snapshot_5m=snap_5m,
                qualified_time=snap_5m.timestamp,
                expiry_time=expiry_dt,
                headroom_points=max(0.0, headroom_points),
                opposing_zone=opposing_zone,
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

            # Purge expired active setups
            for sid, active in list(self.active_setups.items()):
                if candle.timestamp > active.expiry_time or candle.timestamp.date() > active.qualified_time.date():
                    del self.active_setups[sid]

            generated_candidates: List[TechnicalCandidate] = []

            for sid, active in list(self.active_setups.items()):
                if active.setup.symbol != candle.symbol:
                    continue

                candidate = self._evaluate_1m_entry(
                    candle=candle,
                    snap_1m=snap_1m,
                    active=active,
                )
                if candidate:
                    generated_candidates.append(candidate)
                    # Remove setup so it doesn't trigger repeatedly on consecutive 1m bars
                    del self.active_setups[sid]
                    if self.on_candidate:
                        self.on_candidate(candidate)

            return generated_candidates

    def _evaluate_1m_entry(
        self,
        candle: Candle,
        snap_1m: TechnicalSnapshot,
        active: Active5mSetup,
    ) -> Optional[TechnicalCandidate]:
        """Validates 1m execution criteria, optimizes stop loss, and calculates confluence score."""
        setup = active.setup
        nominal_lvl = setup.nominal_price

        # 1. Level Holding Check
        if setup.direction == BreakoutDirection.BULLISH:
            if candle.close < nominal_lvl:
                # Price fell back below broken resistance
                return None
        else:
            if candle.close > nominal_lvl:
                # Price pushed back above broken support
                return None

        # 2. 1m Momentum Confirmation
        if setup.direction == BreakoutDirection.BULLISH:
            # Bullish: green candle OR close above 1m EMA 9
            ema9_ok = (snap_1m.ema_9 is None) or (candle.close >= snap_1m.ema_9)
            candle_ok = candle.close >= candle.open
            if not (ema9_ok or candle_ok):
                return None
            # VWAP filter: price should not be below 1m VWAP if computed
            if snap_1m.vwap is not None and candle.close < snap_1m.vwap:
                return None
            # Overbought RSI exhaustion filter
            if snap_1m.rsi is not None and snap_1m.rsi > 75.0:
                return None
        else:
            # Bearish: red candle OR close below 1m EMA 9
            ema9_ok = (snap_1m.ema_9 is None) or (candle.close <= snap_1m.ema_9)
            candle_ok = candle.close <= candle.open
            if not (ema9_ok or candle_ok):
                return None
            # VWAP filter: price should not be above 1m VWAP if computed
            if snap_1m.vwap is not None and candle.close > snap_1m.vwap:
                return None
            # Oversold RSI exhaustion filter
            if snap_1m.rsi is not None and snap_1m.rsi < 25.0:
                return None

        # 3. Stop-Loss Optimization using 1m Micro-Structure
        entry_price = candle.close
        base_sl = setup.stop_loss if setup.stop_loss is not None else (
            entry_price - 25.0 if setup.direction == BreakoutDirection.BULLISH else entry_price + 25.0
        )

        effective_sl = base_sl
        if setup.direction == BreakoutDirection.BULLISH:
            # Look for 1m nearest support above base_sl but below entry_price
            if snap_1m.nearest_support:
                sup_lvl = snap_1m.nearest_support.lower_bound
                if base_sl < sup_lvl < entry_price - self.min_risk_points:
                    effective_sl = round(sup_lvl, 1)
        else:
            # Look for 1m nearest resistance below base_sl but above entry_price
            if snap_1m.nearest_resistance:
                res_lvl = snap_1m.nearest_resistance.upper_bound
                if base_sl > res_lvl > entry_price + self.min_risk_points:
                    effective_sl = round(res_lvl, 1)

        risk_points = abs(entry_price - effective_sl)
        if risk_points < self.min_risk_points:
            # Enforce minimum risk floor to prevent noise invalidation
            if setup.direction == BreakoutDirection.BULLISH:
                effective_sl = round(entry_price - self.min_risk_points, 1)
            else:
                effective_sl = round(entry_price + self.min_risk_points, 1)
            risk_points = self.min_risk_points

        # 4. Multi-Target & Risk-to-Reward Calculation
        if setup.direction == BreakoutDirection.BULLISH:
            target_1 = round(entry_price + 1.5 * risk_points, 1)
            target_2 = round(entry_price + 2.0 * risk_points, 1)
            reward_points = target_1 - entry_price
        else:
            target_1 = round(entry_price - 1.5 * risk_points, 1)
            target_2 = round(entry_price - 2.0 * risk_points, 1)
            reward_points = entry_price - target_1

        rr_ratio = round(reward_points / risk_points, 2) if risk_points > 0 else 0.0

        # 5. Confluence Scoring (0-100)
        score = self._compute_confluence_score(
            setup=setup,
            snap_1m=snap_1m,
            snap_5m=active.snapshot_5m,
            direction=setup.direction,
            headroom_points=active.headroom_points,
        )

        if score < self.min_confluence_score:
            return None

        # 6. Build Candidate
        ts_str = candle.timestamp.strftime("%Y%m%d_%H%M%S")
        candidate_id = f"{setup.symbol.replace(':', '_')}_{ts_str}_{setup.direction.value}_{setup.level_id}"

        if candidate_id in self.notified_candidates:
            return None
        self.notified_candidates.add(candidate_id)

        snap_15m = self.engine.get_snapshot(setup.symbol, timeframe="15m")

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
            headroom_points=round(active.headroom_points, 1),
            timeframe_breakdown={
                "15m_trend": snap_15m.trend_bias if snap_15m else "N/A",
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
        direction: BreakoutDirection,
        headroom_points: float,
    ) -> float:
        """Calculates 0-100 composite score across all three timeframes."""
        score = 0.0
        snap_15m = self.engine.get_snapshot(snap_5m.symbol, timeframe="15m")

        # 1. 15m Macro Trend Alignment (+30 pts)
        if snap_15m:
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
        else:
            score += 10.0

        # 2. 5m Volume Surge (+25 pts)
        rvol = max(
            setup.breakout_rvol if setup.breakout_rvol is not None else 0.0,
            snap_5m.rvol if snap_5m.rvol is not None else 0.0,
        )
        if rvol > 0:
            if rvol >= 2.0:
                score += 25.0
            elif rvol >= 1.5:
                score += 20.0
            elif rvol >= 1.3:
                score += 15.0
            else:
                score += 5.0
        else:
            score += 10.0

        # 3. VWAP Positioning (+20 pts)
        vwap_5m = snap_5m.vwap
        if vwap_5m is None:
            paired_futures = self.engine.index_futures_map.get(snap_5m.symbol)
            if paired_futures and paired_futures in self.engine.states:
                fut_snap = self.engine.states[paired_futures].latest_snapshots.get("5m")
                if fut_snap:
                    vwap_5m = fut_snap.vwap

        if vwap_5m is not None:
            if direction == BreakoutDirection.BULLISH and snap_5m.close >= vwap_5m:
                score += 10.0
            elif direction == BreakoutDirection.BEARISH and snap_5m.close <= vwap_5m:
                score += 10.0
        else:
            score += 5.0

        vwap_1m = snap_1m.vwap
        if vwap_1m is None:
            paired_futures = self.engine.index_futures_map.get(snap_1m.symbol)
            if paired_futures and paired_futures in self.engine.states:
                fut_snap = self.engine.states[paired_futures].latest_snapshots.get("1m")
                if fut_snap:
                    vwap_1m = fut_snap.vwap

        if vwap_1m is not None:
            if direction == BreakoutDirection.BULLISH and snap_1m.close >= vwap_1m:
                score += 10.0
            elif direction == BreakoutDirection.BEARISH and snap_1m.close <= vwap_1m:
                score += 10.0
        else:
            score += 5.0

        # 4. 15m Headroom (+15 pts)
        atr_5m = snap_5m.atr if snap_5m.atr is not None else 20.0
        if headroom_points >= 2.0 * atr_5m:
            score += 15.0
        elif headroom_points >= 1.5 * atr_5m:
            score += 10.0
        else:
            score += 5.0

        # 5. 1m Momentum Sweet-Spot (+10 pts)
        rsi = snap_1m.rsi
        if rsi is not None:
            if direction == BreakoutDirection.BULLISH and 50.0 <= rsi <= 68.0:
                score += 10.0
            elif direction == BreakoutDirection.BEARISH and 32.0 <= rsi <= 50.0:
                score += 10.0
            else:
                score += 5.0
        else:
            score += 5.0

        return min(100.0, score)
