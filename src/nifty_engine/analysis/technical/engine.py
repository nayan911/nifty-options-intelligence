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

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, Deque, Dict, List, Optional, Tuple

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

        # Telemetry: 1m and 5m Futures RVOL & Timestamps
        self.latest_1m_rvol: Optional[float] = None
        self.latest_1m_rvol_timestamp: Optional[datetime] = None
        self.latest_5m_rvol: Optional[float] = None
        self.latest_5m_rvol_timestamp: Optional[datetime] = None

    @property
    def latest_snapshot(self) -> Optional[TechnicalSnapshot]:
        """Backward-compatible property returning the latest 1m snapshot."""
        return self.latest_snapshots.get("1m")

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

    def process_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> TechnicalSnapshot:
        """Routes candle strictly to its corresponding timeframe handler."""
        if candle.timeframe == "1m":
            return self._process_1m_candle(
                candle, futures_rvol=futures_rvol, on_setup_candidate=on_setup_candidate
            )
        elif candle.timeframe == "5m":
            return self._process_5m_candle(
                candle, futures_rvol=futures_rvol, on_setup_candidate=on_setup_candidate
            )
        elif candle.timeframe == "15m":
            return self._process_15m_candle(candle)
        else:
            return self._create_generic_snapshot(candle)

    def _determine_trend_bias(
        self,
        close: float,
        ema_9: Optional[float],
        ema_21: Optional[float],
        ema_50: Optional[float],
        confirmed_swings: List[SwingPoint],
    ) -> Tuple[str, bool]:
        """Classifies trend bias as (trend_bias, is_provisional).

        If EMAs are warmed up, returns confirmed trend (is_provisional=False).
        Otherwise falls back to structural swings / daily reference levels (is_provisional=True).
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
        if self.daily_reference_levels:
            pdh = self.daily_reference_levels.get("PDH")
            pdl = self.daily_reference_levels.get("PDL")
            cdo = self.daily_reference_levels.get("CDO")
            pdc = self.daily_reference_levels.get("PDC")
            if pdh and close > pdh:
                return "BULLISH", True
            if pdl and close < pdl:
                return "BEARISH", True
            if cdo and pdc:
                if close > cdo and close > pdc:
                    return "BULLISH", True
                elif close < cdo and close < pdc:
                    return "BEARISH", True
                else:
                    return "SIDEWAYS", True
            elif cdo:
                if close > cdo:
                    return "BULLISH", True
                elif close < cdo:
                    return "BEARISH", True

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

        # 2. Structural Swings & Trend Bias
        self.swing_detector_1m.on_candle(candle)
        confirmed_swings = self.swing_detector_1m.get_visible_swings(as_of_time=candle.timestamp)

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings,
        )

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

        or_zones = self.sr_engine.compute_opening_range(c_list, atr_value=effective_atr)
        zones.extend(or_zones)

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

        # 4. Breakout State Machine (1m)
        active_setups: List[BreakoutContext] = []
        if is_warmed_up and atr_val is not None and rvol_val is not None:
            eval_zones: List[PriceZone] = []
            if nearest_sup:
                eval_zones.append(nearest_sup)
            if nearest_res:
                eval_zones.append(nearest_res)

            for z in zones:
                if z not in eval_zones and (z.lower_bound <= candle.high and z.upper_bound >= candle.low):
                    eval_zones.append(z)

            terminal_states = {
                BreakoutState.SETUP_ELIGIBLE,
                BreakoutState.FALSE_BREAKOUT,
                BreakoutState.RETEST_FAILED,
                BreakoutState.SETUP_EXPIRED,
                BreakoutState.INVALIDATED,
            }

            for lvl_id, ctx in list(self.active_breakouts_1m.items()):
                matching_zone = next((z for z in zones if f"{z.level_type.value}_{z.nominal_price:.1f}" == lvl_id), None)
                if not matching_zone:
                    continue

                updated_ctx = self.breakout_machine_1m.update_candle(
                    candle=candle,
                    candle_index=n - 1,
                    zone=matching_zone,
                    atr_value=atr_val,
                    futures_rvol=rvol_val,
                    context=ctx,
                )
                self.active_breakouts_1m[lvl_id] = updated_ctx

                if updated_ctx.current_state == BreakoutState.SETUP_ELIGIBLE:
                    active_setups.append(updated_ctx)
                    if on_setup_candidate:
                        snapshot_preview = TechnicalSnapshot(
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
                            ema_9=round(ema_9, 2) if ema_9 else None,
                            ema_21=round(ema_21, 2) if ema_21 else None,
                            rsi=round(rsi_val, 1) if rsi_val else None,
                            atr=round(atr_val, 2),
                            rvol=round(rvol_val, 2),
                            trend_bias=trend_bias,
                        )
                        on_setup_candidate(updated_ctx, snapshot_preview)

                elif updated_ctx.current_state not in terminal_states:
                    active_setups.append(updated_ctx)
                else:
                    del self.active_breakouts_1m[lvl_id]

            for z in eval_zones:
                lvl_id = f"{z.level_type.value}_{z.nominal_price:.1f}"
                if lvl_id not in self.active_breakouts_1m:
                    ctx = self.breakout_machine_1m.update_candle(
                        candle=candle,
                        candle_index=n - 1,
                        zone=z,
                        atr_value=atr_val,
                        futures_rvol=rvol_val,
                        context=None,
                    )
                    if ctx.current_state != BreakoutState.APPROACHING_LEVEL:
                        self.active_breakouts_1m[lvl_id] = ctx
                        active_setups.append(ctx)

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
        )

        self.latest_snapshots["1m"] = snapshot
        return snapshot

    def _process_5m_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
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

        # 2. 5m Structural Swings & Trend Bias
        self.swing_detector_5m.on_candle(candle)
        confirmed_swings_5m = self.swing_detector_5m.get_visible_swings(as_of_time=candle.timestamp)

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings_5m,
        )

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

        # 15m Opening range (from first three 5m bars: 09:15, 09:20, 09:25)
        or_zones_5m = self.sr_engine.compute_opening_range(c_list, atr_value=effective_atr)
        zones_5m.extend(or_zones_5m)

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

        # 4. 5m Breakout State Machine (Primary Setup Detector)
        active_setups_5m: List[BreakoutContext] = []
        if is_warmed_up and atr_val is not None and rvol_val is not None:
            eval_zones: List[PriceZone] = []
            if nearest_sup_5m:
                eval_zones.append(nearest_sup_5m)
            if nearest_res_5m:
                eval_zones.append(nearest_res_5m)

            for z in zones_5m:
                if z not in eval_zones and (z.lower_bound <= candle.high and z.upper_bound >= candle.low):
                    eval_zones.append(z)

            terminal_states = {
                BreakoutState.SETUP_ELIGIBLE,
                BreakoutState.FALSE_BREAKOUT,
                BreakoutState.RETEST_FAILED,
                BreakoutState.SETUP_EXPIRED,
                BreakoutState.INVALIDATED,
            }

            for lvl_id, ctx in list(self.active_breakouts_5m.items()):
                matching_zone = next((z for z in zones_5m if f"{z.level_type.value}_{z.nominal_price:.1f}" == lvl_id), None)
                if not matching_zone:
                    continue

                updated_ctx = self.breakout_machine_5m.update_candle(
                    candle=candle,
                    candle_index=n - 1,
                    zone=matching_zone,
                    atr_value=atr_val,
                    futures_rvol=rvol_val,
                    context=ctx,
                )
                self.active_breakouts_5m[lvl_id] = updated_ctx

                if updated_ctx.current_state == BreakoutState.SETUP_ELIGIBLE:
                    active_setups_5m.append(updated_ctx)
                    if on_setup_candidate:
                        snapshot_preview = TechnicalSnapshot(
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
                            ema_9=round(ema_9, 2) if ema_9 else None,
                            ema_21=round(ema_21, 2) if ema_21 else None,
                            rsi=round(rsi_val, 1) if rsi_val else None,
                            atr=round(atr_val, 2),
                            rvol=round(rvol_val, 2),
                            trend_bias=trend_bias,
                        )
                        on_setup_candidate(updated_ctx, snapshot_preview)

                elif updated_ctx.current_state not in terminal_states:
                    active_setups_5m.append(updated_ctx)
                else:
                    del self.active_breakouts_5m[lvl_id]

            for z in eval_zones:
                lvl_id = f"{z.level_type.value}_{z.nominal_price:.1f}"
                if lvl_id not in self.active_breakouts_5m:
                    ctx = self.breakout_machine_5m.update_candle(
                        candle=candle,
                        candle_index=n - 1,
                        zone=z,
                        atr_value=atr_val,
                        futures_rvol=rvol_val,
                        context=None,
                    )
                    if ctx.current_state != BreakoutState.APPROACHING_LEVEL:
                        self.active_breakouts_5m[lvl_id] = ctx
                        active_setups_5m.append(ctx)

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
        )
        self.latest_snapshots["5m"] = snap
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

        # 2. 15m Structural Swings & Higher-Timeframe Trend
        self.swing_detector_15m.on_candle(candle)
        confirmed_swings_15m = self.swing_detector_15m.get_visible_swings(as_of_time=candle.timestamp)

        trend_bias, is_provisional = self._determine_trend_bias(
            close=candle.close,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            confirmed_swings=confirmed_swings_15m,
        )

        # 3. Dynamic Support & Resistance Zones (15m Market Structure)
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

        # 15m Opening Range (the 09:15-09:30 candle is the first 15m bar)
        or_candles_15m = [c for c in c_list if c.timestamp.time() == time(9, 15)]
        if or_candles_15m:
            c0 = or_candles_15m[0]
            zones_15m.append(
                self.sr_engine.create_zone(LevelType.ORH, c0.high, effective_atr, "15-Min Opening Range High", weight=2.0)
            )
            zones_15m.append(
                self.sr_engine.create_zone(LevelType.ORL, c0.low, effective_atr, "15-Min Opening Range Low", weight=2.0)
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
        )
        self.latest_snapshots["15m"] = snap
        return snap

    def _create_generic_snapshot(self, candle: Candle) -> TechnicalSnapshot:
        """Fallback snapshot for unmanaged timeframes."""
        return TechnicalSnapshot(
            symbol=self.symbol,
            timestamp=candle.timestamp,
            timeframe=candle.timeframe,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )


class TechnicalAnalysisEngine:
    """Multi-asset orchestrator coordinating real-time technical analysis for the live pipeline."""

    def __init__(
        self,
        primary_index: str = "NSE:NIFTY50-INDEX",
        primary_futures: str = "NSE:NIFTY26NOVFUT",
        on_setup_candidate: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ):
        self.primary_index = primary_index
        self.primary_futures = primary_futures
        self.on_setup_candidate = on_setup_candidate

        # Explicit Index-to-Futures Mapping (prevents NIFTY futures confirming BANKNIFTY)
        self.index_futures_map: Dict[str, str] = {
            primary_index: primary_futures,
        }

        self.states: Dict[str, SymbolTechnicalState] = {}
        # Pre-initialize primary index and futures states
        self._get_or_create_state(primary_index, is_index=True)
        self._get_or_create_state(primary_futures, is_index=False)

    def _get_or_create_state(self, symbol: str, is_index: bool = False) -> SymbolTechnicalState:
        if symbol not in self.states:
            self.states[symbol] = SymbolTechnicalState(symbol=symbol, is_index=is_index)
        return self.states[symbol]

    def set_futures_ticker(self, futures_ticker: str) -> None:
        """Dynamically updates active primary futures ticker (e.g. at contract rollover)."""
        self.primary_futures = futures_ticker
        self.index_futures_map[self.primary_index] = futures_ticker
        self._get_or_create_state(futures_ticker, is_index=False)

    def set_index_futures_mapping(self, index_symbol: str, futures_symbol: str) -> None:
        """Explicitly pairs a spot index with its corresponding futures instrument."""
        self.index_futures_map[index_symbol] = futures_symbol
        self._get_or_create_state(futures_symbol, is_index=False)

    def set_daily_references(
        self,
        symbol: str,
        pdh: float,
        pdl: float,
        pdc: float,
        cdo: float
    ) -> None:
        """Sets daily reference levels (PDH/PDL/PDC/CDO) for a symbol."""
        state = self._get_or_create_state(symbol, is_index="INDEX" in symbol)
        state.set_daily_references(pdh=pdh, pdl=pdl, pdc=pdc, cdo=cdo)

    def on_candle(self, candle: Candle) -> Optional[TechnicalSnapshot]:
        """Dispatches closed candle with strict timeframe routing and synchronized volume freshness checks."""
        is_idx = ("INDEX" in candle.symbol) or (candle.symbol == self.primary_index)
        state = self._get_or_create_state(candle.symbol, is_index=is_idx)

        # Cross-asset volume pairing: Spot index references its EXPLICIT paired Futures RVOL
        futures_rvol: Optional[float] = None
        paired_futures = self.index_futures_map.get(candle.symbol)
        if paired_futures and paired_futures in self.states:
            fut_state = self.states[paired_futures]

            if candle.timeframe == "1m":
                if fut_state.latest_1m_rvol is not None and fut_state.latest_1m_rvol_timestamp is not None:
                    delta_seconds = (candle.timestamp - fut_state.latest_1m_rvol_timestamp).total_seconds()
                    # 1m tolerance: 0 <= delta <= 120s
                    if 0.0 <= delta_seconds <= 120.0:
                        futures_rvol = fut_state.latest_1m_rvol

            elif candle.timeframe == "5m":
                if fut_state.latest_5m_rvol is not None and fut_state.latest_5m_rvol_timestamp is not None:
                    delta_seconds = (candle.timestamp - fut_state.latest_5m_rvol_timestamp).total_seconds()
                    # 5m tolerance: 0 <= delta <= 300s
                    if 0.0 <= delta_seconds <= 300.0:
                        futures_rvol = fut_state.latest_5m_rvol

        snapshot = state.process_candle(
            candle=candle,
            futures_rvol=futures_rvol,
            on_setup_candidate=self.on_setup_candidate,
        )
        return snapshot

    def get_snapshot(self, symbol: str, timeframe: str = "1m") -> Optional[TechnicalSnapshot]:
        """Returns the latest technical snapshot for a symbol and timeframe."""
        if symbol in self.states:
            return self.states[symbol].latest_snapshots.get(timeframe)
        return None
