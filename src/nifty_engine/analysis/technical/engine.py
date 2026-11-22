"""Technical Analysis Engine.

Orchestrates streaming technical and structural quantitative analysis across subscribed assets:
1. Multi-timeframe analysis:
   - 15-minute: Market structure, higher-timeframe trend, horizontal S/R, regime context.
   - 5-minute: Primary breakout and retest setup detector, 5m swings, 5m RVOL/VWAP.
   - 1-minute: Precision entry confirmation, rapid momentum and micro-structure.
2. Deterministic warm-up requirements with distinct provisional vs confirmed status.
3. Synchronized cross-asset volume pairing (NIFTY/BANKNIFTY spot inherit paired 1m/5m futures RVOL with freshness checks).
4. Structural swing point detection across all three timeframes without lookahead bias.
5. Dynamic support and resistance clustering (Daily levels, Opening Range, Round strikes, Swing clusters).
6. Timeframe-isolated Breakout & Retest candidate tracking.
7. Candidate notification callbacks with explicit risk & disclaimer tags.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from nifty_engine.analysis.technical.breakouts import (
    BreakoutContext,
    BreakoutStateMachine,
)
from nifty_engine.analysis.technical.indicators import (
    compute_atr,
    compute_ema,
    compute_relative_volume,
    compute_rsi,
    compute_vwap,
)
from nifty_engine.analysis.technical.support_resistance import (
    LevelType,
    PriceZone,
    SupportResistanceEngine,
)
from nifty_engine.analysis.technical.swings import SwingDetector, SwingPoint
from nifty_engine.core.calendar import get_bar_close_time
from nifty_engine.core.interfaces import BreakoutState, Candle


@dataclass
class TechnicalSnapshot:
    """Full quantitative technical telemetry snapshot for a symbol at a specific candle."""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    is_warmed_up: bool = False
    is_provisional: bool = False
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    vwap_upper: Optional[float] = None
    vwap_lower: Optional[float] = None
    rvol: Optional[float] = None
    trend_bias: str = "INSUFFICIENT_DATA"  # BULLISH, BEARISH, SIDEWAYS, INSUFFICIENT_DATA
    nearest_support: Optional[PriceZone] = None
    nearest_resistance: Optional[PriceZone] = None
    active_setups: List[BreakoutContext] = field(default_factory=list)
    available_at: Optional[datetime] = None

    @property
    def bar_close_time(self) -> datetime:
        """Calculates exact candle close/completion timestamp based on timeframe duration."""
        return get_bar_close_time(self.timestamp, self.timeframe)

    @property
    def effective_available_at(self) -> datetime:
        """Returns genuine confirmation timestamp when all indicators and volume were available."""
        return self.available_at if self.available_at is not None else self.bar_close_time


class SymbolTechnicalState:
    """Maintains independent timeframe histories, indicators, and breakout machines per symbol."""

    def __init__(
        self,
        symbol: str,
        is_index: bool = False,
        max_history_1m: int = 400,
        max_history_5m: int = 200,
        max_history_15m: int = 100,
    ):
        self.symbol = symbol
        self.is_index = is_index
        self.max_history_1m = max_history_1m
        self.max_history_5m = max_history_5m
        self.max_history_15m = max_history_15m

        # Strictly isolated timeframe histories
        self.candles_1m: Deque[Candle] = deque(maxlen=max_history_1m)
        self.candles_5m: Deque[Candle] = deque(maxlen=max_history_5m)
        self.candles_15m: Deque[Candle] = deque(maxlen=max_history_15m)

        # Structural swing detectors
        self.swing_detector_1m = SwingDetector(timeframe="1m", left_bars=3, right_bars=3)
        self.swing_detector_5m = SwingDetector(timeframe="5m", left_bars=3, right_bars=3)
        self.swing_detector_15m = SwingDetector(timeframe="15m", left_bars=2, right_bars=2)

        # Support & Resistance Engine
        self.sr_engine = SupportResistanceEngine(zone_atr_multiple=0.15, min_zone_width_points=3.0)

        # Timeframe-isolated Breakout State Machines
        # 1-minute: fast momentum / micro breakouts (max 6 candles = 6m)
        self.breakout_machine_1m = BreakoutStateMachine(
            min_breakout_rvol=1.3,
            min_body_atr_ratio=0.35,
            approach_buffer_atr=0.25,
            invalidation_buffer_atr=0.20,
            max_retest_candles=6,
        )
        # 5-minute: primary setup detector (max 4 candles = 20m)
        self.breakout_machine_5m = BreakoutStateMachine(
            min_breakout_rvol=1.3,
            min_body_atr_ratio=0.35,
            approach_buffer_atr=0.25,
            invalidation_buffer_atr=0.20,
            max_retest_candles=4,
        )

        # Active breakout setups tracked per level_id (strictly separated by timeframe)
        self.active_breakouts_1m: Dict[str, BreakoutContext] = {}
        self.active_breakouts_5m: Dict[str, BreakoutContext] = {}

        # Daily reference levels (PDH, PDL, PDC, CDO)
        self.daily_reference_levels: Optional[Dict[str, float]] = None

        # Multi-timeframe snapshots retained independently (1m, 5m, 15m)
        self.latest_snapshots: Dict[str, Optional[TechnicalSnapshot]] = {
            "1m": None,
            "5m": None,
            "15m": None,
        }
        self.snapshot_history: Dict[str, Deque[TechnicalSnapshot]] = {
            "1m": deque(maxlen=60),
            "5m": deque(maxlen=30),
            "15m": deque(maxlen=20),
        }

        # Telemetry: 1m and 5m Futures RVOL & Timestamps
        self.latest_1m_rvol: Optional[float] = None
        self.latest_1m_rvol_timestamp: Optional[datetime] = None
        self.latest_5m_rvol: Optional[float] = None
        self.latest_5m_rvol_timestamp: Optional[datetime] = None

        # Trading session state & isolation
        self.current_session_date: Optional[date] = None
        self.session_open_price: Optional[float] = None
        self.notified_setup_ids: Set[str] = set()
        self.latest_candle_timestamps: Dict[str, datetime] = {}
        self.last_candle_accepted: Dict[str, bool] = {"1m": False, "5m": False, "15m": False}
        self._pending_breakout_eval: Dict[str, Tuple] = {}

    @property
    def latest_snapshot(self) -> Optional[TechnicalSnapshot]:
        """Backward-compatible property returning the latest 1m snapshot."""
        return self.latest_snapshots.get("1m")

    def get_snapshot(
        self,
        timeframe: str = "1m",
        as_of: Optional[datetime] = None,
    ) -> Optional[TechnicalSnapshot]:
        """Returns the technical snapshot for a timeframe, optionally as of a specific cutoff timestamp."""
        if as_of is None:
            return self.latest_snapshots.get(timeframe)

        # Look in reverse chronological order for the most recent snapshot completed on or before as_of
        history = self.snapshot_history.get(timeframe)
        if history:
            for snap in reversed(history):
                if snap.effective_available_at <= as_of:
                    return snap

        latest = self.latest_snapshots.get(timeframe)
        if latest is not None and latest.effective_available_at <= as_of:
            return latest

        return None

    @property
    def latest_rvol(self) -> Optional[float]:
        """Backward-compatible property returning the latest 1m RVOL."""
        return self.latest_1m_rvol

    @property
    def latest_rvol_timestamp(self) -> Optional[datetime]:
        """Backward-compatible property returning the latest 1m RVOL timestamp."""
        return self.latest_1m_rvol_timestamp

    def set_daily_references(self, pdh: float, pdl: float, pdc: float, cdo: float) -> None:
        """Sets daily reference prices for benchmark S/R calculation."""
        self.daily_reference_levels = {
            "PDH": pdh,
            "PDL": pdl,
            "PDC": pdc,
            "CDO": cdo,
        }

    def _check_session_rollover(self, candle_timestamp: datetime) -> bool:
        """Enforces clean trading-session boundaries across multiple trading days.

        Returns True if candle belongs to current or newer session; False if candle is from an older session.
        """
        candle_date = candle_timestamp.date()
        if self.current_session_date is None:
            self.current_session_date = candle_date
            return True

        if candle_date < self.current_session_date:
            # Reject older trading dates before performing session rollover!
            # Out-of-order or late candles from prior sessions MUST NOT clear or contaminate current session.
            return False

        if candle_date > self.current_session_date:
            self.current_session_date = candle_date
            self.session_open_price = None
            self.daily_reference_levels = None
            self.active_breakouts_1m.clear()
            self.active_breakouts_5m.clear()
            self.notified_setup_ids.clear()
            self.latest_1m_rvol = None
            self.latest_1m_rvol_timestamp = None
            self.latest_5m_rvol = None
            self.latest_5m_rvol_timestamp = None
            self.latest_snapshots = {
                "1m": None,
                "5m": None,
                "15m": None,
            }
            for deq in self.snapshot_history.values():
                deq.clear()
            self.latest_candle_timestamps.clear()
            self._pending_breakout_eval.clear()

        return True

    def process_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
        awaiting_futures_volume: bool = False,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
        is_bootstrap: bool = False,
    ) -> TechnicalSnapshot:
        """Routes candle strictly to its corresponding timeframe handler."""
        if not is_bootstrap and not self._check_session_rollover(candle.timestamp):
            # Candle belongs to a past trading date: strictly reject!
            self.last_candle_accepted[candle.timeframe] = False
            return self.latest_snapshots.get(candle.timeframe) or self._create_generic_snapshot(candle)

        if is_bootstrap:
            candle_date = candle.timestamp.date()
            if self.current_session_date is None or candle_date > self.current_session_date:
                self.current_session_date = candle_date

        # Monotonic timestamp validation: reject duplicate or out-of-order candles
        last_ts = self.latest_candle_timestamps.get(candle.timeframe)
        if last_ts is not None and candle.timestamp <= last_ts:
            self.last_candle_accepted[candle.timeframe] = False
            if candle.timestamp == last_ts and futures_rvol is not None:
                snap = self.latest_snapshots.get(candle.timeframe)
                if snap is not None and snap.rvol is None:
                    snap.rvol = round(futures_rvol, 2)
                    self.latest_snapshots[candle.timeframe] = snap
                    return snap
            return self.latest_snapshots.get(candle.timeframe) or self._create_generic_snapshot(candle)
        self.last_candle_accepted[candle.timeframe] = True
        self.latest_candle_timestamps[candle.timeframe] = candle.timestamp

        # Flush any uncompleted pending breakout evaluation on older candle
        if candle.timeframe in self._pending_breakout_eval:
            self.complete_breakout_evaluation(candle.timeframe, futures_rvol=None, on_setup_candidate=on_setup_candidate)

        if self.session_open_price is None:
            self.session_open_price = candle.open

        if candle.timeframe == "1m":
            return self._process_1m_candle(
                candle,
                futures_rvol=futures_rvol,
                awaiting_futures_volume=awaiting_futures_volume,
                on_setup_candidate=on_setup_candidate,
            )
        elif candle.timeframe == "5m":
            return self._process_5m_candle(
                candle,
                futures_rvol=futures_rvol,
                awaiting_futures_volume=awaiting_futures_volume,
                on_setup_candidate=on_setup_candidate,
            )
        elif candle.timeframe == "15m":
            return self._process_15m_candle(candle)
        else:
            return self._create_generic_snapshot(candle)

    def _evaluate_breakouts(
        self,
        timeframe: str,
        candle: Candle,
        zones: List[PriceZone],
        nearest_sup: Optional[PriceZone],
        nearest_res: Optional[PriceZone],
        n: int,
        atr_val: Optional[float],
        rvol_val: Optional[float],
        is_warmed_up: bool,
        is_provisional: bool,
        ema_9: Optional[float],
        ema_21: Optional[float],
        rsi_val: Optional[float],
        trend_bias: str,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
        actual_available_at: Optional[datetime] = None,
    ) -> List[BreakoutContext]:
        """Evaluates breakout state machine updates and approach transitions exactly once per candle."""
        active_map = self.active_breakouts_1m if timeframe == "1m" else self.active_breakouts_5m
        machine = self.breakout_machine_1m if timeframe == "1m" else self.breakout_machine_5m

        active_setups: List[BreakoutContext] = []
        terminal_states = {
            BreakoutState.SETUP_ELIGIBLE,
            BreakoutState.FALSE_BREAKOUT,
            BreakoutState.RETEST_FAILED,
            BreakoutState.SETUP_EXPIRED,
            BreakoutState.INVALIDATED,
        }

        effective_atr = atr_val if atr_val is not None else 0.0
        effective_rvol = rvol_val if rvol_val is not None else 1.0

        for lvl_id, ctx in list(active_map.items()):
            matching_zone = next((z for z in zones if f"{z.level_type.value}_{z.nominal_price:.1f}" == lvl_id), None)
            if not matching_zone:
                matching_zone = ctx.zone

            if not matching_zone:
                ctx.transition_to(BreakoutState.INVALIDATED, candle.timestamp, "Original reference zone no longer exists")
                del active_map[lvl_id]
                continue

            prev_state = ctx.current_state
            updated_ctx = machine.update_candle(
                candle=candle,
                candle_index=n - 1,
                zone=matching_zone,
                atr_value=effective_atr,
                futures_rvol=effective_rvol,
                context=ctx,
            )
            active_map[lvl_id] = updated_ctx

            if (
                prev_state != BreakoutState.SETUP_ELIGIBLE
                and updated_ctx.current_state == BreakoutState.SETUP_ELIGIBLE
                and updated_ctx.setup_id not in self.notified_setup_ids
            ):
                self.notified_setup_ids.add(updated_ctx.setup_id)
                updated_ctx.is_notified = True
                active_setups.append(updated_ctx)
                if on_setup_candidate:
                    snapshot_preview = TechnicalSnapshot(
                        symbol=self.symbol,
                        timestamp=candle.timestamp,
                        timeframe=timeframe,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        is_warmed_up=is_warmed_up,
                        is_provisional=is_provisional,
                        ema_9=round(ema_9, 2) if ema_9 else None,
                        ema_21=round(ema_21, 2) if ema_21 else None,
                        rsi=round(rsi_val, 1) if rsi_val else None,
                        atr=round(atr_val, 2) if atr_val else None,
                        rvol=round(rvol_val, 2) if rvol_val else None,
                        trend_bias=trend_bias,
                        available_at=actual_available_at,
                    )
                    on_setup_candidate(updated_ctx, snapshot_preview)
                del active_map[lvl_id]
            elif updated_ctx.current_state in terminal_states:
                del active_map[lvl_id]
            else:
                active_setups.append(updated_ctx)

        # Evaluate new breakout approaches
        if is_warmed_up and atr_val is not None and rvol_val is not None:
            eval_zones: List[PriceZone] = []
            if nearest_sup:
                eval_zones.append(nearest_sup)
            if nearest_res:
                eval_zones.append(nearest_res)

            for z in zones:
                if z not in eval_zones and (z.lower_bound <= candle.high and z.upper_bound >= candle.low):
                    eval_zones.append(z)

            for z in eval_zones:
                lvl_id = f"{z.level_type.value}_{z.nominal_price:.1f}"
                if lvl_id not in active_map:
                    ctx = machine.update_candle(
                        candle=candle,
                        candle_index=n - 1,
                        zone=z,
                        atr_value=atr_val,
                        futures_rvol=rvol_val,
                        context=None,
                    )
                    if ctx.current_state != BreakoutState.APPROACHING_LEVEL:
                        active_map[lvl_id] = ctx
                        active_setups.append(ctx)

        return active_setups

    def complete_breakout_evaluation(
        self,
        timeframe: str,
        futures_rvol: Optional[float],
        futures_timestamp: Optional[datetime] = None,
        futures_observed_at: Optional[datetime] = None,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> None:
        """Executes deferred breakout evaluation for a spot candle once synchronized futures volume arrives."""
        if timeframe not in self._pending_breakout_eval:
            return
        params = self._pending_breakout_eval.pop(timeframe)
        (candle, zones, nearest_sup, nearest_res, n, atr_val, is_warmed_up, is_provisional, ema_9, ema_21, rsi_val, trend_bias) = params

        # Interval alignment validation: verify futures observation matches spot candle interval
        effective_rvol = futures_rvol
        if futures_timestamp is not None and abs((candle.timestamp - futures_timestamp).total_seconds()) > 5.0:
            effective_rvol = None

        actual_available_at = futures_observed_at
        if actual_available_at is None and futures_timestamp is not None:
            fut_close = get_bar_close_time(futures_timestamp, timeframe)
            spot_close = get_bar_close_time(candle.timestamp, timeframe)
            actual_available_at = max(spot_close, fut_close)

        active_setups = self._evaluate_breakouts(
            timeframe=timeframe,
            candle=candle,
            zones=zones,
            nearest_sup=nearest_sup,
            nearest_res=nearest_res,
            n=n,
            atr_val=atr_val,
            rvol_val=effective_rvol,
            is_warmed_up=is_warmed_up,
            is_provisional=is_provisional,
            ema_9=ema_9,
            ema_21=ema_21,
            rsi_val=rsi_val,
            trend_bias=trend_bias,
            on_setup_candidate=on_setup_candidate,
            actual_available_at=actual_available_at,
        )

        snap = self.latest_snapshots.get(timeframe)
        if snap:
            if effective_rvol is not None:
                snap.rvol = round(effective_rvol, 2)
            snap.active_setups = active_setups
            if actual_available_at is not None:
                snap.available_at = actual_available_at
            self.latest_snapshots[timeframe] = snap

    def expire_pending_evaluations(
        self,
        as_of_time: datetime,
        timeout_seconds: Optional[float] = None,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> None:
        """Flushes any pending breakout evaluations that have exceeded timeout without futures volume confirmation."""
        for tf, params in list(self._pending_breakout_eval.items()):
            pending_candle: Candle = params[0]
            interval_secs = 60.0 if tf == "1m" else (300.0 if tf == "5m" else 900.0)
            threshold = timeout_seconds if timeout_seconds is not None else (interval_secs * 2.0)

            # Elapsed time past interval close
            elapsed_past_close = (as_of_time - pending_candle.timestamp).total_seconds() - interval_secs
            if elapsed_past_close >= threshold or as_of_time.date() > pending_candle.timestamp.date():
                self.complete_breakout_evaluation(
                    timeframe=tf,
                    futures_rvol=None,
                    on_setup_candidate=on_setup_candidate,
                )

    def _determine_trend_bias(
        self,
        close: float,
        ema_9: Optional[float],
        ema_21: Optional[float],
        ema_50: Optional[float],
        confirmed_swings: List[SwingPoint],
        orh: Optional[float] = None,
        orl: Optional[float] = None,
        cdo: Optional[float] = None,
    ) -> Tuple[str, bool]:
        """Classifies trend bias as (trend_bias, is_provisional).

        If EMAs are warmed up, returns confirmed trend (is_provisional=False).
        Otherwise falls back to Opening Range / structural swings / daily reference levels (is_provisional=True).
        """
        # 1. Confirmed Indicator-Based Trend
        if ema_9 is not None and ema_21 is not None and ema_50 is not None:
            if ema_9 > ema_21 > ema_50 and close > ema_9:
                return "BULLISH", False
            elif ema_9 < ema_21 < ema_50 and close < ema_9:
                return "BEARISH", False
            else:
                return "SIDEWAYS", False
        elif ema_9 is not None and ema_21 is not None:
            if ema_9 > ema_21 and close > ema_9:
                return "BULLISH", False
            elif ema_9 < ema_21 and close < ema_9:
                return "BEARISH", False
            else:
                return "SIDEWAYS", False

        # 2. Provisional Structural Trend (Indicator Warmup Fallback)
        # A. Breakout Beyond Opening Range (Highest structural priority)
        if orh is not None and orl is not None:
            if close > orh:
                return "BULLISH", True
            elif close < orl:
                return "BEARISH", True

        # B. Daily Reference Levels (PDH, PDL, PDC, CDO)
        effective_cdo = cdo
        if self.daily_reference_levels:
            pdh = self.daily_reference_levels.get("PDH")
            pdl = self.daily_reference_levels.get("PDL")
            cdo_lvl = self.daily_reference_levels.get("CDO") or effective_cdo
            pdc = self.daily_reference_levels.get("PDC")
            if pdh and close > pdh:
                return "BULLISH", True
            if pdl and close < pdl:
                return "BEARISH", True
            if cdo_lvl and pdc:
                if close > cdo_lvl and close > pdc:
                    return "BULLISH", True
                elif close < cdo_lvl and close < pdc:
                    return "BEARISH", True
                else:
                    return "SIDEWAYS", True
            elif cdo_lvl:
                if close > cdo_lvl:
                    return "BULLISH", True
                elif close < cdo_lvl:
                    return "BEARISH", True
        elif effective_cdo is not None:
            if close > effective_cdo:
                return "BULLISH", True
            elif close < effective_cdo:
                return "BEARISH", True

        # C. Inside Opening Range Balance
        if orh is not None and orl is not None:
            return "SIDEWAYS", True

        # C. Confirmed Swings
        high_swings = [s for s in confirmed_swings if s.is_high]
        low_swings = [s for s in confirmed_swings if s.is_low]
        if len(high_swings) >= 2 and len(low_swings) >= 2:
            if high_swings[-1].price > high_swings[-2].price and low_swings[-1].price > low_swings[-2].price:
                return "BULLISH", True
            elif high_swings[-1].price < high_swings[-2].price and low_swings[-1].price < low_swings[-2].price:
                return "BEARISH", True
            else:
                return "SIDEWAYS", True

        return "INSUFFICIENT_DATA", True

    def _process_1m_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
        awaiting_futures_volume: bool = False,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> TechnicalSnapshot:
        """Processes 1m candle: precision entry timing, fast momentum, and micro S/R."""
        self.candles_1m.append(candle)
        c_list = list(self.candles_1m)
        n = len(c_list)

        closes = pd.Series([c.close for c in c_list], dtype=float)
        highs = pd.Series([c.high for c in c_list], dtype=float)
        lows = pd.Series([c.low for c in c_list], dtype=float)
        volumes = pd.Series([c.volume for c in c_list], dtype=float)

        # 1. Warm-up Enforced Indicators
        ema_9 = float(compute_ema(closes, 9).iloc[-1]) if n >= 9 else None
        ema_21 = float(compute_ema(closes, 21).iloc[-1]) if n >= 21 else None
        ema_50 = float(compute_ema(closes, 50).iloc[-1]) if n >= 50 else None
        ema_200 = float(compute_ema(closes, 200).iloc[-1]) if n >= 200 else None

        rsi_val: Optional[float] = None
        if n >= 15:
            rsi_series = compute_rsi(closes, period=14)
            if not np.isnan(rsi_series.iloc[-1]):
                rsi_val = float(rsi_series.iloc[-1])

        atr_val: Optional[float] = None
        if n >= 15:
            atr_series = compute_atr(highs, lows, closes, period=14)
            if not np.isnan(atr_series.iloc[-1]):
                atr_val = float(atr_series.iloc[-1])

        is_warmed_up = (n >= 21 and atr_val is not None)

        # Relative Volume (RVOL 20)
        rvol_val: Optional[float] = None
        if not self.is_index and volumes.sum() > 0:
            if n >= 20:
                rvol_series = compute_relative_volume(volumes, period=20)
                rvol_val = float(rvol_series.iloc[-1])
            else:
                rvol_val = 1.0
            self.latest_1m_rvol = rvol_val
            self.latest_1m_rvol_timestamp = candle.timestamp
        elif futures_rvol is not None:
            rvol_val = futures_rvol
            self.latest_1m_rvol = futures_rvol
            self.latest_1m_rvol_timestamp = candle.timestamp

        # VWAP & Bands (Tradeable Volume Only)
        vwap_val: Optional[float] = None
        vwap_up: Optional[float] = None
        vwap_dn: Optional[float] = None
        if not self.is_index and volumes.sum() > 0:
            try:
                df = pd.DataFrame({
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                    "date": [c.timestamp.date() for c in c_list],
                })
                vwap_s, up_s, dn_s = compute_vwap(df, anchor_col="date")
                vwap_val = round(float(vwap_s.iloc[-1]), 2)
                vwap_up = round(float(up_s.iloc[-1]), 2)
                vwap_dn = round(float(dn_s.iloc[-1]), 2)
            except Exception:
                pass

        # 2. Structural Swings
        self.swing_detector_1m.on_candle(candle)
        confirmed_swings = self.swing_detector_1m.get_visible_swings(as_of_time=candle.timestamp)

        # 3. Dynamic Support & Resistance Zones
        effective_atr = atr_val if atr_val is not None else 0.0
        zones: List[PriceZone] = []

        if self.daily_reference_levels:
            zones.extend(
                self.sr_engine.compute_daily_levels(
                    pdh=self.daily_reference_levels["PDH"],
                    pdl=self.daily_reference_levels["PDL"],
                    pdc=self.daily_reference_levels["PDC"],
                    cdo=self.daily_reference_levels["CDO"],
                    atr_value=effective_atr,
                )
            )

        or_zones = self.sr_engine.compute_opening_range(
            c_list, atr_value=effective_atr, session_date=candle.timestamp.date()
        )
        zones.extend(or_zones)
        orh_1m = next((z.nominal_price for z in or_zones if z.level_type == LevelType.ORH), None)
        orl_1m = next((z.nominal_price for z in or_zones if z.level_type == LevelType.ORL), None)

        effective_cdo = None
        if self.daily_reference_levels:
            effective_cdo = self.daily_reference_levels.get("CDO")
        if effective_cdo is None:
            effective_cdo = self.session_open_price

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings,
            orh=orh_1m,
            orl=orl_1m,
            cdo=effective_cdo,
        )

        zones.extend(
            self.sr_engine.compute_psychological_levels(
                current_price=candle.close,
                atr_value=effective_atr,
                range_points=150.0,
            )
        )

        if confirmed_swings:
            zones.extend(
                self.sr_engine.compute_swing_clusters(
                    swings=confirmed_swings,
                    atr_value=effective_atr,
                    as_of_time=candle.timestamp,
                )
            )

        if vwap_val is not None:
            zones.append(
                self.sr_engine.create_zone(
                    level_type=LevelType.VWAP,
                    nominal_price=vwap_val,
                    atr_value=effective_atr,
                    description="Session Anchored VWAP (1m)",
                    weight=2.0,
                )
            )

        nearest_sup, nearest_res = self.sr_engine.find_nearest_zones(candle.close, zones)

        close_time_1m = get_bar_close_time(candle.timestamp, "1m")
        avail_at_1m = max(candle.observed_at, close_time_1m) if candle.observed_at is not None else close_time_1m

        # 4. Breakout State Machine (1m)
        if awaiting_futures_volume:
            self._pending_breakout_eval["1m"] = (
                candle,
                zones,
                nearest_sup,
                nearest_res,
                n,
                atr_val,
                is_warmed_up,
                is_provisional,
                ema_9,
                ema_21,
                rsi_val,
                trend_bias,
            )
            active_setups: List[BreakoutContext] = []
        else:
            active_setups = self._evaluate_breakouts(
                timeframe="1m",
                candle=candle,
                zones=zones,
                nearest_sup=nearest_sup,
                nearest_res=nearest_res,
                n=n,
                atr_val=atr_val,
                rvol_val=rvol_val,
                is_warmed_up=is_warmed_up,
                is_provisional=is_provisional,
                ema_9=ema_9,
                ema_21=ema_21,
                rsi_val=rsi_val,
                trend_bias=trend_bias,
                on_setup_candidate=on_setup_candidate,
                actual_available_at=avail_at_1m,
            )

        snapshot = TechnicalSnapshot(
            symbol=self.symbol,
            timestamp=candle.timestamp,
            timeframe="1m",
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            is_warmed_up=is_warmed_up,
            is_provisional=is_provisional,
            ema_9=round(ema_9, 2) if ema_9 is not None else None,
            ema_21=round(ema_21, 2) if ema_21 is not None else None,
            ema_50=round(ema_50, 2) if ema_50 is not None else None,
            ema_200=round(ema_200, 2) if ema_200 is not None else None,
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            atr=round(atr_val, 2) if atr_val is not None else None,
            vwap=vwap_val,
            vwap_upper=vwap_up,
            vwap_lower=vwap_dn,
            rvol=round(rvol_val, 2) if rvol_val is not None else None,
            trend_bias=trend_bias,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            active_setups=active_setups,
            available_at=avail_at_1m,
        )

        self.latest_snapshots["1m"] = snapshot
        self.snapshot_history["1m"].append(snapshot)
        return snapshot

    def _process_5m_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
        awaiting_futures_volume: bool = False,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> TechnicalSnapshot:
        """Processes 5m candle: primary setup detection, 5m structural swings, RVOL & VWAP."""
        self.candles_5m.append(candle)
        c_list = list(self.candles_5m)
        n = len(c_list)

        closes = pd.Series([c.close for c in c_list], dtype=float)
        highs = pd.Series([c.high for c in c_list], dtype=float)
        lows = pd.Series([c.low for c in c_list], dtype=float)
        volumes = pd.Series([c.volume for c in c_list], dtype=float)

        # 1. Warm-up Enforced 5m Indicators
        ema_9 = float(compute_ema(closes, 9).iloc[-1]) if n >= 9 else None
        ema_21 = float(compute_ema(closes, 21).iloc[-1]) if n >= 21 else None
        ema_50 = float(compute_ema(closes, 50).iloc[-1]) if n >= 50 else None
        ema_200 = float(compute_ema(closes, 200).iloc[-1]) if n >= 200 else None

        rsi_val: Optional[float] = None
        if n >= 15:
            rsi_s = compute_rsi(closes, period=14)
            if not np.isnan(rsi_s.iloc[-1]):
                rsi_val = float(rsi_s.iloc[-1])

        atr_val: Optional[float] = None
        if n >= 15:
            atr_s = compute_atr(highs, lows, closes, period=14)
            if not np.isnan(atr_s.iloc[-1]):
                atr_val = float(atr_s.iloc[-1])

        is_warmed_up = (n >= 21 and atr_val is not None)

        # 5m Relative Volume (RVOL from 5-minute traded volume)
        rvol_val: Optional[float] = None
        if not self.is_index and volumes.sum() > 0:
            if n >= 20:
                rvol_series = compute_relative_volume(volumes, period=20)
                rvol_val = float(rvol_series.iloc[-1])
            else:
                rvol_val = 1.0
            self.latest_5m_rvol = rvol_val
            self.latest_5m_rvol_timestamp = candle.timestamp
        elif futures_rvol is not None:
            rvol_val = futures_rvol
            self.latest_5m_rvol = futures_rvol
            self.latest_5m_rvol_timestamp = candle.timestamp

        # 5m VWAP & Bands (Tradeable Volume Only)
        vwap_val: Optional[float] = None
        vwap_up: Optional[float] = None
        vwap_dn: Optional[float] = None
        if not self.is_index and volumes.sum() > 0:
            try:
                df = pd.DataFrame({
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                    "date": [c.timestamp.date() for c in c_list],
                })
                vwap_s, up_s, dn_s = compute_vwap(df, anchor_col="date")
                vwap_val = round(float(vwap_s.iloc[-1]), 2)
                vwap_up = round(float(up_s.iloc[-1]), 2)
                vwap_dn = round(float(dn_s.iloc[-1]), 2)
            except Exception:
                pass

        # 2. 5m Structural Swings
        self.swing_detector_5m.on_candle(candle)
        confirmed_swings_5m = self.swing_detector_5m.get_visible_swings(as_of_time=candle.timestamp)

        # 3. Dynamic Support & Resistance Zones (5m)
        effective_atr = atr_val if atr_val is not None else 0.0
        zones_5m: List[PriceZone] = []

        if self.daily_reference_levels:
            zones_5m.extend(
                self.sr_engine.compute_daily_levels(
                    pdh=self.daily_reference_levels["PDH"],
                    pdl=self.daily_reference_levels["PDL"],
                    pdc=self.daily_reference_levels["PDC"],
                    cdo=self.daily_reference_levels["CDO"],
                    atr_value=effective_atr,
                )
            )

        # 15m Opening range (from first three 5m bars: 09:15, 09:20, 09:25 of current session)
        or_zones_5m = self.sr_engine.compute_opening_range(
            c_list, atr_value=effective_atr, session_date=candle.timestamp.date()
        )
        zones_5m.extend(or_zones_5m)
        orh_5m = next((z.nominal_price for z in or_zones_5m if z.level_type == LevelType.ORH), None)
        orl_5m = next((z.nominal_price for z in or_zones_5m if z.level_type == LevelType.ORL), None)

        effective_cdo = None
        if self.daily_reference_levels:
            effective_cdo = self.daily_reference_levels.get("CDO")
        if effective_cdo is None:
            effective_cdo = self.session_open_price

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings_5m,
            orh=orh_5m,
            orl=orl_5m,
            cdo=effective_cdo,
        )

        zones_5m.extend(
            self.sr_engine.compute_psychological_levels(
                current_price=candle.close,
                atr_value=effective_atr,
                range_points=200.0,
            )
        )

        if confirmed_swings_5m:
            zones_5m.extend(
                self.sr_engine.compute_swing_clusters(
                    swings=confirmed_swings_5m,
                    atr_value=effective_atr,
                    as_of_time=candle.timestamp,
                )
            )

        if vwap_val is not None:
            zones_5m.append(
                self.sr_engine.create_zone(
                    level_type=LevelType.VWAP,
                    nominal_price=vwap_val,
                    atr_value=effective_atr,
                    description="Session Anchored VWAP (5m)",
                    weight=2.0,
                )
            )

        nearest_sup_5m, nearest_res_5m = self.sr_engine.find_nearest_zones(candle.close, zones_5m)

        close_time_5m = get_bar_close_time(candle.timestamp, "5m")
        avail_at_5m = max(candle.observed_at, close_time_5m) if candle.observed_at is not None else close_time_5m

        # 4. 5m Breakout State Machine (Primary Setup Detector)
        if awaiting_futures_volume:
            self._pending_breakout_eval["5m"] = (
                candle,
                zones_5m,
                nearest_sup_5m,
                nearest_res_5m,
                n,
                atr_val,
                is_warmed_up,
                is_provisional,
                ema_9,
                ema_21,
                rsi_val,
                trend_bias,
            )
            active_setups_5m: List[BreakoutContext] = []
        else:
            active_setups_5m = self._evaluate_breakouts(
                timeframe="5m",
                candle=candle,
                zones=zones_5m,
                nearest_sup=nearest_sup_5m,
                nearest_res=nearest_res_5m,
                n=n,
                atr_val=atr_val,
                rvol_val=rvol_val,
                is_warmed_up=is_warmed_up,
                is_provisional=is_provisional,
                ema_9=ema_9,
                ema_21=ema_21,
                rsi_val=rsi_val,
                trend_bias=trend_bias,
                on_setup_candidate=on_setup_candidate,
                actual_available_at=avail_at_5m,
            )

        snap = TechnicalSnapshot(
            symbol=self.symbol,
            timestamp=candle.timestamp,
            timeframe="5m",
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            is_warmed_up=is_warmed_up,
            is_provisional=is_provisional,
            ema_9=round(ema_9, 2) if ema_9 is not None else None,
            ema_21=round(ema_21, 2) if ema_21 is not None else None,
            ema_50=round(ema_50, 2) if ema_50 is not None else None,
            ema_200=round(ema_200, 2) if ema_200 is not None else None,
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            atr=round(atr_val, 2) if atr_val is not None else None,
            vwap=vwap_val,
            vwap_upper=vwap_up,
            vwap_lower=vwap_dn,
            rvol=round(rvol_val, 2) if rvol_val is not None else None,
            trend_bias=trend_bias,
            nearest_support=nearest_sup_5m,
            nearest_resistance=nearest_res_5m,
            active_setups=active_setups_5m,
            available_at=avail_at_5m,
        )
        self.latest_snapshots["5m"] = snap
        self.snapshot_history["5m"].append(snap)
        return snap

    def _process_15m_candle(self, candle: Candle) -> TechnicalSnapshot:
        """Processes 15m candle: higher-timeframe market structure, directional filter, and key S/R."""
        self.candles_15m.append(candle)
        c_list = list(self.candles_15m)
        n = len(c_list)

        closes = pd.Series([c.close for c in c_list], dtype=float)
        highs = pd.Series([c.high for c in c_list], dtype=float)
        lows = pd.Series([c.low for c in c_list], dtype=float)

        # 1. Warm-up Enforced 15m Indicators
        ema_9 = float(compute_ema(closes, 9).iloc[-1]) if n >= 9 else None
        ema_21 = float(compute_ema(closes, 21).iloc[-1]) if n >= 21 else None
        ema_50 = float(compute_ema(closes, 50).iloc[-1]) if n >= 50 else None

        rsi_val: Optional[float] = None
        if n >= 15:
            rsi_s = compute_rsi(closes, period=14)
            if not np.isnan(rsi_s.iloc[-1]):
                rsi_val = float(rsi_s.iloc[-1])

        atr_val: Optional[float] = None
        if n >= 15:
            atr_s = compute_atr(highs, lows, closes, period=14)
            if not np.isnan(atr_s.iloc[-1]):
                atr_val = float(atr_s.iloc[-1])

        is_warmed_up = (n >= 21 and atr_val is not None)

        # 2. 15m Structural Swings & Support/Resistance Zones (15m Market Structure)
        self.swing_detector_15m.on_candle(candle)
        confirmed_swings_15m = self.swing_detector_15m.get_visible_swings(as_of_time=candle.timestamp)

        effective_atr = atr_val if atr_val is not None else 0.0
        zones_15m: List[PriceZone] = []

        if self.daily_reference_levels:
            zones_15m.extend(
                self.sr_engine.compute_daily_levels(
                    pdh=self.daily_reference_levels["PDH"],
                    pdl=self.daily_reference_levels["PDL"],
                    pdc=self.daily_reference_levels["PDC"],
                    cdo=self.daily_reference_levels["CDO"],
                    atr_value=effective_atr,
                )
            )

        # 15m Opening Range (the 09:15-09:30 candle is the first 15m bar of current session)
        current_date = candle.timestamp.date()
        or_candles_15m = [
            c for c in c_list
            if c.timestamp.date() == current_date and c.timestamp.time() == time(9, 15) and c.is_closed
        ]
        orh_15m: Optional[float] = None
        orl_15m: Optional[float] = None
        if or_candles_15m:
            c0 = or_candles_15m[0]
            orh_15m = c0.high
            orl_15m = c0.low
            zones_15m.append(
                self.sr_engine.create_zone(LevelType.ORH, orh_15m, effective_atr, "15-Min Opening Range High", weight=2.0)
            )
            zones_15m.append(
                self.sr_engine.create_zone(LevelType.ORL, orl_15m, effective_atr, "15-Min Opening Range Low", weight=2.0)
            )

        effective_cdo = None
        if self.daily_reference_levels:
            effective_cdo = self.daily_reference_levels.get("CDO")
        if effective_cdo is None:
            effective_cdo = self.session_open_price

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings_15m,
            orh=orh_15m,
            orl=orl_15m,
            cdo=effective_cdo,
        )

        zones_15m.extend(
            self.sr_engine.compute_psychological_levels(
                current_price=candle.close,
                atr_value=effective_atr,
                range_points=250.0,
            )
        )

        if confirmed_swings_15m:
            zones_15m.extend(
                self.sr_engine.compute_swing_clusters(
                    swings=confirmed_swings_15m,
                    atr_value=effective_atr,
                    as_of_time=candle.timestamp,
                )
            )

        nearest_sup_15m, nearest_res_15m = self.sr_engine.find_nearest_zones(candle.close, zones_15m)

        close_time_15m = get_bar_close_time(candle.timestamp, "15m")
        avail_at_15m = max(candle.observed_at, close_time_15m) if candle.observed_at is not None else close_time_15m

        snap = TechnicalSnapshot(
            symbol=self.symbol,
            timestamp=candle.timestamp,
            timeframe="15m",
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            is_warmed_up=is_warmed_up,
            is_provisional=is_provisional,
            ema_9=round(ema_9, 2) if ema_9 is not None else None,
            ema_21=round(ema_21, 2) if ema_21 is not None else None,
            ema_50=round(ema_50, 2) if ema_50 is not None else None,
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            atr=round(atr_val, 2) if atr_val is not None else None,
            trend_bias=trend_bias,
            nearest_support=nearest_sup_15m,
            nearest_resistance=nearest_res_15m,
            available_at=avail_at_15m,
        )
        self.latest_snapshots["15m"] = snap
        self.snapshot_history["15m"].append(snap)
        return snap

    def _create_generic_snapshot(self, candle: Candle) -> TechnicalSnapshot:
        """Fallback snapshot for unmanaged timeframes."""
        close_time = get_bar_close_time(candle.timestamp, candle.timeframe)
        avail_at = max(candle.observed_at, close_time) if candle.observed_at is not None else close_time
        return TechnicalSnapshot(
            symbol=self.symbol,
            timestamp=candle.timestamp,
            timeframe=candle.timeframe,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            available_at=avail_at,
        )


class TechnicalAnalysisEngine:
    """Multi-asset orchestrator coordinating real-time technical analysis for the live pipeline."""

    def __init__(
        self,
        primary_index: str = "NSE:NIFTY50-INDEX",
        primary_futures: str = "NSE:NIFTY26NOVFUT",
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
        on_technical_candidate: Optional[Callable[[Any], None]] = None,
        derivatives_engine: Optional[Any] = None,
    ):
        self._lock = threading.RLock()
        self.primary_index = primary_index
        self.primary_futures = primary_futures
        self.on_setup_candidate = on_setup_candidate
        self.on_technical_candidate = on_technical_candidate
        self.derivatives_engine = derivatives_engine
        self.multi_timeframe_analyzer: Optional[Any] = None

        if on_technical_candidate:
            from nifty_engine.analysis.technical.multi_timeframe import MultiTimeframeAnalyzer
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(
                engine=self,
                on_candidate=on_technical_candidate,
                derivatives_engine=derivatives_engine,
            )

        # Explicit Index-to-Futures Mapping (prevents NIFTY futures confirming BANKNIFTY)
        self.index_futures_map: Dict[str, str] = {
            primary_index: primary_futures,
        }
        self.futures_index_map: Dict[str, str] = {
            primary_futures: primary_index,
        }

        self.states: Dict[str, SymbolTechnicalState] = {}
        # Pre-initialize primary index and futures states
        self._get_or_create_state(primary_index, is_index=True)
        self._get_or_create_state(primary_futures, is_index=False)

        # Buffer for synchronized 1m entry evaluation: {spot_symbol: (candle, snapshot, arrival_time)}
        self._pending_1m_entry: Dict[str, Tuple[Candle, TechnicalSnapshot, datetime]] = {}

    def _dispatch_setup_candidate(self, setup: BreakoutContext, snap: TechnicalSnapshot) -> None:
        """Internal callback dispatcher notifying both raw setup listeners and MultiTimeframeAnalyzer."""
        if self.on_setup_candidate:
            self.on_setup_candidate(setup, snap)
        if self.multi_timeframe_analyzer and setup.timeframe == "5m":
            self.multi_timeframe_analyzer.on_5m_setup(setup, snap)

    def set_multi_timeframe_candidate_callback(self, callback: Callable[[Any], None]) -> None:
        """Configures Phase C MultiTimeframeAnalyzer candidate callback."""
        with self._lock:
            self.on_technical_candidate = callback
            from nifty_engine.analysis.technical.multi_timeframe import MultiTimeframeAnalyzer
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(
                engine=self,
                on_candidate=callback,
                derivatives_engine=self.derivatives_engine,
            )

    def on_option_chain(self, snapshot: Any, as_of: Optional[datetime] = None) -> None:
        """Ingests option chain snapshots into the derivatives intelligence engine."""
        with self._lock:
            if self.derivatives_engine is not None:
                self.derivatives_engine.oi_analyzer.on_option_chain(snapshot, as_of=as_of)
                self.derivatives_engine.pcr_engine.on_option_chain(snapshot, as_of=as_of)

    def _get_or_create_state(self, symbol: str, is_index: bool = False) -> SymbolTechnicalState:
        if symbol not in self.states:
            self.states[symbol] = SymbolTechnicalState(symbol=symbol, is_index=is_index)
        return self.states[symbol]

    def set_futures_ticker(self, futures_ticker: str) -> None:
        """Dynamically updates active primary futures ticker (e.g. at contract rollover)."""
        with self._lock:
            old_futures = self.index_futures_map.get(self.primary_index)
            if old_futures and old_futures != futures_ticker:
                self.futures_index_map.pop(old_futures, None)

            self.primary_futures = futures_ticker
            self.index_futures_map[self.primary_index] = futures_ticker
            self.futures_index_map[futures_ticker] = self.primary_index
            self._get_or_create_state(futures_ticker, is_index=False)
            if self.derivatives_engine is not None:
                if hasattr(self.derivatives_engine, "set_futures_ticker"):
                    self.derivatives_engine.set_futures_ticker(futures_ticker)
                else:
                    self.derivatives_engine.primary_futures = futures_ticker

    def set_index_futures_mapping(self, index_symbol: str, futures_symbol: str) -> None:
        """Explicitly pairs a spot index with its corresponding futures instrument."""
        with self._lock:
            old_futures = self.index_futures_map.get(index_symbol)
            if old_futures and old_futures != futures_symbol:
                self.futures_index_map.pop(old_futures, None)

            self.index_futures_map[index_symbol] = futures_symbol
            self.futures_index_map[futures_symbol] = index_symbol
            self._get_or_create_state(futures_symbol, is_index=False)

    def set_daily_references(
        self,
        symbol: str,
        pdh: float,
        pdl: float,
        pdc: float,
        cdo: float,
    ) -> None:
        """Sets daily reference levels (PDH/PDL/PDC/CDO) for a symbol."""
        with self._lock:
            state = self._get_or_create_state(symbol, is_index="INDEX" in symbol)
            state.set_daily_references(pdh=pdh, pdl=pdl, pdc=pdc, cdo=cdo)

    def prepare_session(self, session_date: date) -> None:
        """Explicitly initializes a new trading session date across all symbol states."""
        with self._lock:
            for state in self.states.values():
                if state.current_session_date is None or session_date > state.current_session_date:
                    state.current_session_date = session_date
                    state.session_open_price = None
                    state.daily_reference_levels = None
                    state.active_breakouts_1m.clear()
                    state.active_breakouts_5m.clear()
                    state.notified_setup_ids.clear()
                    state.latest_1m_rvol = None
                    state.latest_1m_rvol_timestamp = None
                    state.latest_5m_rvol = None
                    state.latest_5m_rvol_timestamp = None
                    state.latest_candle_timestamps.clear()
                    state._pending_breakout_eval.clear()

    def bootstrap_candles(self, candles: List[Candle]) -> int:
        """Bootstraps technical indicators, moving averages, and swings with historical candles.

        STRICT TRADING SAFETY GUARDS:
        1. Suppresses setup candidate and trade notifications (is_bootstrap=True).
        2. Bypasses live session bounds check during historical replay.
        3. Sets indicators to warmed up when sufficient history is processed.
        """
        if not candles:
            return 0

        # Sort ascending chronologically
        sorted_candles = sorted(candles, key=lambda c: (c.timestamp, c.timeframe))
        bootstrapped_count = 0

        for c in sorted_candles:
            if not c.is_closed:
                continue
            snap = self.on_candle(c, is_bootstrap=True)
            if snap is not None:
                bootstrapped_count += 1

        return bootstrapped_count

    def on_candle(self, candle: Candle, is_bootstrap: bool = False) -> Optional[TechnicalSnapshot]:
        """Dispatches closed candle with strict timeframe routing and synchronized volume freshness checks."""
        # Strictly reject unclosed candles
        if not candle.is_closed:
            return None

        with self._lock:
            is_idx = ("INDEX" in candle.symbol) or (candle.symbol in self.index_futures_map)
            state = self._get_or_create_state(candle.symbol, is_index=is_idx)

            # Reject older trading dates before modifying state or forwarding to derivatives (unless bootstrapping)
            if not is_bootstrap and state.current_session_date is not None and candle.timestamp.date() < state.current_session_date:
                return state.latest_snapshots.get(candle.timeframe) or state._create_generic_snapshot(candle)

            # Phase 3A: Update derivatives futures positioning if candle has OI
            if self.derivatives_engine is not None and (
                candle.symbol in self.futures_index_map or candle.symbol == self.primary_futures
            ):
                self.derivatives_engine.oi_analyzer.on_futures_candle(candle)

            # 1. Spot Index Candle: Check if mapped futures volume is available or pending
            paired_futures = self.index_futures_map.get(candle.symbol)
            awaiting_futures_volume = False
            futures_rvol = None

            if paired_futures and paired_futures in self.states and candle.timeframe in ("1m", "5m"):
                fut_state = self.states[paired_futures]
                fut_candle_count = len(fut_state.candles_1m if candle.timeframe == "1m" else fut_state.candles_5m)

                fut_rvol_ts = fut_state.latest_1m_rvol_timestamp if candle.timeframe == "1m" else fut_state.latest_5m_rvol_timestamp
                fut_rvol = fut_state.latest_1m_rvol if candle.timeframe == "1m" else fut_state.latest_5m_rvol

                has_matching_futures = (
                    fut_rvol_ts is not None
                    and fut_rvol is not None
                    and abs((candle.timestamp - fut_rvol_ts).total_seconds()) <= 5.0
                )

                if has_matching_futures:
                    futures_rvol = fut_rvol
                elif fut_candle_count > 0:
                    # If futures stream is active in this session, check if candle is waiting for current interval's volume
                    interval_secs = 60.0 if candle.timeframe == "1m" else 300.0
                    if fut_rvol_ts is None or (0 <= (candle.timestamp - fut_rvol_ts).total_seconds() <= interval_secs):
                        awaiting_futures_volume = True

            dispatch_cb = None if is_bootstrap else self._dispatch_setup_candidate
            snapshot = state.process_candle(
                candle=candle,
                futures_rvol=futures_rvol,
                awaiting_futures_volume=awaiting_futures_volume,
                on_setup_candidate=dispatch_cb,
                is_bootstrap=is_bootstrap,
            )

            # 2. Futures Candle: Check if paired spot index arrived earlier and was awaiting volume
            paired_index = self.futures_index_map.get(candle.symbol)
            if paired_index and paired_index in self.states and candle.timeframe in ("1m", "5m"):
                spot_state = self.states[paired_index]
                fut_rvol = state.latest_1m_rvol if candle.timeframe == "1m" else state.latest_5m_rvol
                fut_rvol_ts = state.latest_1m_rvol_timestamp if candle.timeframe == "1m" else state.latest_5m_rvol_timestamp

                # Check if spot snapshot was awaiting volume for this exact interval
                if fut_rvol is not None and fut_rvol_ts is not None:
                    spot_snap = spot_state.latest_snapshots.get(candle.timeframe)
                    if (
                        spot_snap is not None
                        and spot_snap.rvol is None
                        and abs((spot_snap.timestamp - candle.timestamp).total_seconds()) <= 5.0
                        and abs((spot_snap.timestamp - fut_rvol_ts).total_seconds()) <= 5.0
                    ):
                        spot_snap.rvol = round(fut_rvol, 2)
                        spot_state.latest_snapshots[candle.timeframe] = spot_snap

                # Check deferred breakout evaluation
                if candle.timeframe in spot_state._pending_breakout_eval:
                    pending_params = spot_state._pending_breakout_eval[candle.timeframe]
                    pending_spot_candle: Candle = pending_params[0]

                    is_same_interval = (
                        abs((pending_spot_candle.timestamp - candle.timestamp).total_seconds()) <= 5.0
                        and fut_rvol_ts is not None
                        and abs((pending_spot_candle.timestamp - fut_rvol_ts).total_seconds()) <= 5.0
                    )

                    if is_same_interval:
                        if fut_rvol is not None:
                            spot_state.complete_breakout_evaluation(
                                timeframe=candle.timeframe,
                                futures_rvol=fut_rvol,
                                futures_timestamp=candle.timestamp,
                                futures_observed_at=candle.observed_at,
                                on_setup_candidate=dispatch_cb,
                            )
                    elif candle.timestamp > pending_spot_candle.timestamp:
                        # Futures candle has progressed to a future interval!
                        # The pending spot candle missed its corresponding futures interval.
                        # Do NOT borrow later volume! Expire pending evaluation with futures_rvol=None.
                        spot_state.complete_breakout_evaluation(
                            timeframe=candle.timeframe,
                            futures_rvol=None,
                            futures_observed_at=candle.observed_at,
                            on_setup_candidate=dispatch_cb,
                        )

            # 3. Phase C: Synchronized precision entry evaluation for 1m candle close (skip during bootstrap)
            if (
                not is_bootstrap
                and snapshot
                and candle.timeframe == "1m"
                and self.multi_timeframe_analyzer
                and state.last_candle_accepted.get("1m", False)
                and snapshot.timestamp == candle.timestamp
            ):
                paired_futures = self.index_futures_map.get(candle.symbol)
                paired_index = self.futures_index_map.get(candle.symbol)

                if paired_futures:
                    # Candle is a spot index: check if paired futures stream is active in this session
                    fut_state = self.states.get(paired_futures)
                    fut_active = fut_state is not None and len(fut_state.candles_1m) > 0
                    if fut_active:
                        fut_snap_1m = fut_state.latest_snapshots.get("1m")
                        futures_ready = (
                            fut_snap_1m is not None
                            and abs((fut_snap_1m.timestamp - candle.timestamp).total_seconds()) <= 5.0
                        )
                        if futures_ready:
                            # Both spot and futures 1m are ready!
                            if candle.symbol in self._pending_1m_entry:
                                old_c, old_s, _ = self._pending_1m_entry.pop(candle.symbol)
                                old_decision = max(old_s.effective_available_at, snapshot.effective_available_at)
                                self.multi_timeframe_analyzer.on_1m_candle(
                                    old_c, old_s, decision_at=old_decision, confirmation_status="MISSING_CONFIRMATION"
                                )
                            decision_at = max(
                                snapshot.effective_available_at,
                                fut_snap_1m.effective_available_at,
                            )
                            self.multi_timeframe_analyzer.on_1m_candle(
                                candle, snapshot, decision_at=decision_at, confirmation_status="CONFIRMED"
                            )
                        else:
                            # Flush older pending 1m candle if exists
                            if candle.symbol in self._pending_1m_entry:
                                old_c, old_s, _ = self._pending_1m_entry.pop(candle.symbol)
                                old_decision = max(old_s.effective_available_at, snapshot.effective_available_at)
                                self.multi_timeframe_analyzer.on_1m_candle(
                                    old_c, old_s, decision_at=old_decision, confirmation_status="MISSING_CONFIRMATION"
                                )
                            # Buffer spot 1m awaiting matching futures 1m candle
                            self._pending_1m_entry[candle.symbol] = (
                                candle,
                                snapshot,
                                candle.observed_at or candle.timestamp,
                            )
                    else:
                        # No active futures stream; evaluate immediately
                        self.multi_timeframe_analyzer.on_1m_candle(
                            candle, snapshot, decision_at=snapshot.effective_available_at, confirmation_status="CONFIRMED"
                        )
                elif paired_index:
                    # Candle is a futures contract: check if paired spot index was waiting
                    if paired_index in self._pending_1m_entry:
                        pending_c, pending_s, _ = self._pending_1m_entry[paired_index]
                        if abs((pending_c.timestamp - candle.timestamp).total_seconds()) <= 5.0:
                            # Matching futures candle arrived! Evaluate spot 1m entry
                            del self._pending_1m_entry[paired_index]
                            decision_at = max(
                                pending_s.effective_available_at,
                                snapshot.effective_available_at,
                            )
                            self.multi_timeframe_analyzer.on_1m_candle(
                                pending_c, pending_s, decision_at=decision_at, confirmation_status="CONFIRMED"
                            )
                        elif candle.timestamp > pending_c.timestamp:
                            # Futures progressed past pending spot interval; confirmation was missed!
                            del self._pending_1m_entry[paired_index]
                            decision_at = max(
                                pending_s.effective_available_at,
                                snapshot.effective_available_at,
                            )
                            self.multi_timeframe_analyzer.on_1m_candle(
                                pending_c, pending_s, decision_at=decision_at, confirmation_status="MISSING_CONFIRMATION"
                            )
                        else:
                            # candle.timestamp < pending_c.timestamp:
                            # Delayed older futures candle; do not remove pending spot entry!
                            pass
                else:
                    # Unpaired symbol: evaluate immediately
                    self.multi_timeframe_analyzer.on_1m_candle(
                        candle, snapshot, decision_at=snapshot.effective_available_at, confirmation_status="CONFIRMED"
                    )

            # Phase C: 15m snapshot coordination unblocking simultaneous 5m setups
            if (
                snapshot
                and candle.timeframe == "15m"
                and self.multi_timeframe_analyzer
                and state.last_candle_accepted.get("15m", False)
            ):
                self.multi_timeframe_analyzer.on_15m_snapshot(snapshot)

            return snapshot

    def expire_pending_evaluations(
        self,
        as_of_time: datetime,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Clock-driven expiry of pending breakout evaluations and 1m entry evaluations across all monitored symbols."""
        with self._lock:
            for state in self.states.values():
                state.expire_pending_evaluations(
                    as_of_time=as_of_time,
                    timeout_seconds=timeout_seconds,
                    on_setup_candidate=self._dispatch_setup_candidate,
                )

            # Expire pending 1m entry evaluations awaiting futures confirmation
            if self.multi_timeframe_analyzer:
                thresh = timeout_seconds if timeout_seconds is not None else 5.0
                for sym, (p_c, p_s, _) in list(self._pending_1m_entry.items()):
                    elapsed = (as_of_time - p_c.timestamp).total_seconds() - 60.0
                    if elapsed >= thresh or as_of_time.date() > p_c.timestamp.date():
                        del self._pending_1m_entry[sym]
                        self.multi_timeframe_analyzer.on_1m_candle(
                            p_c, p_s, decision_at=as_of_time, confirmation_status="TIMED_OUT"
                        )

                self.multi_timeframe_analyzer.expire_pending_5m_setups(
                    as_of_time=as_of_time,
                    timeout_seconds=timeout_seconds or 30.0,
                )

    def get_snapshot(
        self,
        symbol: str,
        timeframe: str = "1m",
        as_of: Optional[datetime] = None,
    ) -> Optional[TechnicalSnapshot]:
        """Returns the technical snapshot for a symbol and timeframe, optionally as of a specific cutoff timestamp."""
        with self._lock:
            if symbol in self.states:
                return self.states[symbol].get_snapshot(timeframe=timeframe, as_of=as_of)
            return None
