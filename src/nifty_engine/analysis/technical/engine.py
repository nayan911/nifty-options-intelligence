"""Technical Analysis Engine.

Orchestrates streaming technical and structural quantitative analysis across subscribed assets:
1. Rolling multi-timeframe candle management (1m streaming and synthesized 5m candles).
2. Real-time indicators: EMA (9, 21, 50, 200), RSI (14), ATR (14), Anchored VWAP + Bands, and RVOL (20).
3. Cross-asset volume pairing: spot indices leverage paired index futures RVOL for volume confirmation.
4. Real-time swing point detection without lookahead bias.
5. Dynamic support and resistance clustering (Daily levels, 15m Opening Range, Round strikes, Swing clusters).
6. State-machine driven Breakout & Retest setup tracking.
7. Callback notifications when actionable setups reach SETUP_ELIGIBLE state.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

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
from nifty_engine.analysis.technical.swings import SwingDetector
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
    trend_bias: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    nearest_support: Optional[PriceZone] = None
    nearest_resistance: Optional[PriceZone] = None
    active_setups: List[BreakoutContext] = field(default_factory=list)


class SymbolTechnicalState:
    """Maintains state, indicators, and breakout machines for an individual instrument."""

    def __init__(
        self,
        symbol: str,
        is_index: bool = False,
        max_history_1m: int = 400,
        max_history_5m: int = 150,
    ):
        self.symbol = symbol
        self.is_index = is_index
        self.max_history_1m = max_history_1m
        self.max_history_5m = max_history_5m

        self.candles_1m: Deque[Candle] = deque(maxlen=max_history_1m)
        self.candles_5m: Deque[Candle] = deque(maxlen=max_history_5m)

        # 5-minute bar synthesis tracking
        self._current_5m_bar: Optional[Candle] = None

        # Structural detectors
        self.swing_detector_1m = SwingDetector(timeframe="1m", left_bars=3, right_bars=3)
        self.swing_detector_5m = SwingDetector(timeframe="5m", left_bars=3, right_bars=3)
        self.sr_engine = SupportResistanceEngine(zone_atr_multiple=0.15, min_zone_width_points=3.0)
        self.breakout_machine = BreakoutStateMachine(
            min_breakout_rvol=1.3,
            min_body_atr_ratio=0.35,
            approach_buffer_atr=0.25,
            invalidation_buffer_atr=0.20,
            max_retest_candles=6,
        )

        # Active breakout setups tracked per level_id
        self.active_breakouts: Dict[str, BreakoutContext] = {}

        # Daily reference levels (PDH, PDL, PDC, CDO)
        self.daily_reference_levels: Optional[Dict[str, float]] = None

        # Latest computed indicators and snapshot
        self.latest_snapshot: Optional[TechnicalSnapshot] = None
        self.latest_rvol: float = 1.0

    def set_daily_references(self, pdh: float, pdl: float, pdc: float, cdo: float) -> None:
        """Sets daily reference prices for benchmark S/R calculation."""
        self.daily_reference_levels = {
            "PDH": pdh,
            "PDL": pdl,
            "PDC": pdc,
            "CDO": cdo,
        }

    def process_1m_candle(
        self,
        candle: Candle,
        futures_rvol: Optional[float] = None,
        on_setup_eligible: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ) -> TechnicalSnapshot:
        """Processes a newly closed 1-minute candle, updating all technical layers."""
        self.candles_1m.append(candle)
        c_list = list(self.candles_1m)
        n = len(c_list)

        # 1. Update 5m candle synthesizer
        self._update_5m_synthesizer(candle)

        # 2. Extract series for indicator computation
        closes = pd.Series([c.close for c in c_list], dtype=float)
        highs = pd.Series([c.high for c in c_list], dtype=float)
        lows = pd.Series([c.low for c in c_list], dtype=float)
        volumes = pd.Series([c.volume for c in c_list], dtype=float)

        # Compute EMAs
        ema_9 = float(compute_ema(closes, 9).iloc[-1]) if n >= 2 else candle.close
        ema_21 = float(compute_ema(closes, 21).iloc[-1]) if n >= 5 else None
        ema_50 = float(compute_ema(closes, 50).iloc[-1]) if n >= 15 else None
        ema_200 = float(compute_ema(closes, 200).iloc[-1]) if n >= 50 else None

        # Compute RSI(14)
        rsi_series = compute_rsi(closes, period=14)
        rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None

        # Compute ATR(14)
        atr_series = compute_atr(highs, lows, closes, period=14)
        atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 15.0

        # Compute RVOL(20)
        rvol_val = 1.0
        if not self.is_index and volumes.sum() > 0:
            rvol_series = compute_relative_volume(volumes, period=20)
            rvol_val = float(rvol_series.iloc[-1])
            self.latest_rvol = rvol_val
        elif futures_rvol is not None:
            rvol_val = futures_rvol
            self.latest_rvol = futures_rvol

        # Compute VWAP and bands (strictly tradeable volume)
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

        # 3. Determine Trend Bias
        trend_bias = "NEUTRAL"
        if ema_21 is not None and ema_50 is not None:
            if ema_9 > ema_21 > ema_50 and candle.close > ema_9:
                trend_bias = "BULLISH"
            elif ema_9 < ema_21 < ema_50 and candle.close < ema_9:
                trend_bias = "BEARISH"
        elif ema_21 is not None:
            if ema_9 > ema_21 and candle.close > ema_9:
                trend_bias = "BULLISH"
            elif ema_9 < ema_21 and candle.close < ema_9:
                trend_bias = "BEARISH"

        # 4. Update Swings
        self.swing_detector_1m.on_candle(candle)
        confirmed_swings = self.swing_detector_1m.get_visible_swings(as_of_time=candle.timestamp)

        # 5. Build Dynamic Support and Resistance Zones
        zones: List[PriceZone] = []

        # Daily benchmark zones
        if self.daily_reference_levels:
            zones.extend(
                self.sr_engine.compute_daily_levels(
                    pdh=self.daily_reference_levels["PDH"],
                    pdl=self.daily_reference_levels["PDL"],
                    pdc=self.daily_reference_levels["PDC"],
                    cdo=self.daily_reference_levels["CDO"],
                    atr_value=atr_val,
                )
            )

        # 15-minute Opening Range (09:15 - 09:30)
        or_zones = self.sr_engine.compute_opening_range(c_list, atr_value=atr_val)
        zones.extend(or_zones)

        # Psychological levels (Round numbers +/- 150 points)
        zones.extend(
            self.sr_engine.compute_psychological_levels(
                current_price=candle.close,
                atr_value=atr_val,
                range_points=150.0,
            )
        )

        # Structural swing clusters
        if confirmed_swings:
            zones.extend(
                self.sr_engine.compute_swing_clusters(
                    swings=confirmed_swings,
                    atr_value=atr_val,
                    as_of_time=candle.timestamp,
                )
            )

        # VWAP zone
        if vwap_val is not None:
            zones.append(
                self.sr_engine.create_zone(
                    level_type=LevelType.VWAP,
                    nominal_price=vwap_val,
                    atr_value=atr_val,
                    description="Session Anchored VWAP",
                    weight=2.0,
                )
            )

        # Find immediate nearest support and resistance
        nearest_sup, nearest_res = self.sr_engine.find_nearest_zones(candle.close, zones)

        # 6. Evaluate Breakout State Machine
        eval_zones: List[PriceZone] = []
        if nearest_sup:
            eval_zones.append(nearest_sup)
        if nearest_res:
            eval_zones.append(nearest_res)

        # Also include any zone that intersects the candle range [low, high]
        for z in zones:
            if z not in eval_zones and (z.lower_bound <= candle.high and z.upper_bound >= candle.low):
                eval_zones.append(z)

        active_setups: List[BreakoutContext] = []
        terminal_states = {
            BreakoutState.SETUP_ELIGIBLE,
            BreakoutState.FALSE_BREAKOUT,
            BreakoutState.RETEST_FAILED,
            BreakoutState.SETUP_EXPIRED,
            BreakoutState.INVALIDATED,
        }

        # Step active contexts forward with the latest candle
        for lvl_id, ctx in list(self.active_breakouts.items()):
            matching_zone = next((z for z in zones if f"{z.level_type.value}_{z.nominal_price:.1f}" == lvl_id), None)
            if not matching_zone:
                continue

            updated_ctx = self.breakout_machine.update_candle(
                candle=candle,
                candle_index=n - 1,
                zone=matching_zone,
                atr_value=atr_val,
                futures_rvol=rvol_val,
                context=ctx,
            )
            self.active_breakouts[lvl_id] = updated_ctx

            if updated_ctx.current_state == BreakoutState.SETUP_ELIGIBLE:
                active_setups.append(updated_ctx)
                if on_setup_eligible:
                    snapshot_preview = TechnicalSnapshot(
                        symbol=self.symbol,
                        timestamp=candle.timestamp,
                        timeframe="1m",
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        ema_9=round(ema_9, 2),
                        ema_21=round(ema_21, 2) if ema_21 else None,
                        rsi=round(rsi_val, 1) if rsi_val else None,
                        atr=round(atr_val, 2),
                        rvol=round(rvol_val, 2),
                        trend_bias=trend_bias,
                    )
                    on_setup_eligible(updated_ctx, snapshot_preview)

            elif updated_ctx.current_state not in terminal_states:
                active_setups.append(updated_ctx)
            else:
                # Remove terminal state from active tracking
                del self.active_breakouts[lvl_id]

        # Check for fresh approaches or breakouts against nearest zones
        for z in eval_zones:
            lvl_id = f"{z.level_type.value}_{z.nominal_price:.1f}"
            if lvl_id not in self.active_breakouts:
                # Test if approaching or breaking zone
                ctx = self.breakout_machine.update_candle(
                    candle=candle,
                    candle_index=n - 1,
                    zone=z,
                    atr_value=atr_val,
                    futures_rvol=rvol_val,
                    context=None,
                )
                if ctx.current_state != BreakoutState.APPROACHING_LEVEL:
                    self.active_breakouts[lvl_id] = ctx
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
            ema_9=round(ema_9, 2),
            ema_21=round(ema_21, 2) if ema_21 is not None else None,
            ema_50=round(ema_50, 2) if ema_50 is not None else None,
            ema_200=round(ema_200, 2) if ema_200 is not None else None,
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            atr=round(atr_val, 2),
            vwap=vwap_val,
            vwap_upper=vwap_up,
            vwap_lower=vwap_dn,
            rvol=round(rvol_val, 2),
            trend_bias=trend_bias,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            active_setups=active_setups,
        )

        self.latest_snapshot = snapshot
        return snapshot

    def _update_5m_synthesizer(self, candle_1m: Candle) -> Optional[Candle]:
        """Aggregates 1m candles into 5m candles and closes on 5-minute boundaries."""
        t = candle_1m.timestamp
        # 5-minute bucket starting minute (e.g., 0, 5, 10, ..., 55)
        bucket_min = (t.minute // 5) * 5
        bucket_start = t.replace(minute=bucket_min, second=0, microsecond=0)

        closed_5m: Optional[Candle] = None

        if self._current_5m_bar is None:
            self._current_5m_bar = Candle(
                symbol=candle_1m.symbol,
                timeframe="5m",
                timestamp=bucket_start,
                open=candle_1m.open,
                high=candle_1m.high,
                low=candle_1m.low,
                close=candle_1m.close,
                volume=candle_1m.volume,
                is_closed=False,
            )
        elif self._current_5m_bar.timestamp == bucket_start:
            # Update current 5m bar with 1m tick
            b = self._current_5m_bar
            b.high = max(b.high, candle_1m.high)
            b.low = min(b.low, candle_1m.low)
            b.close = candle_1m.close
            b.volume += candle_1m.volume

            # Close 5m bar when last minute of bucket is reached (e.g. minute % 5 == 4)
            if t.minute % 5 == 4:
                b.is_closed = True
                closed_5m = b
                self.candles_5m.append(b)
                self.swing_detector_5m.on_candle(b)
                self._current_5m_bar = None
        else:
            # Timestamp moved to next 5m bucket before closing
            b = self._current_5m_bar
            b.is_closed = True
            closed_5m = b
            self.candles_5m.append(b)
            self.swing_detector_5m.on_candle(b)

            # Start new 5m bar
            self._current_5m_bar = Candle(
                symbol=candle_1m.symbol,
                timeframe="5m",
                timestamp=bucket_start,
                open=candle_1m.open,
                high=candle_1m.high,
                low=candle_1m.low,
                close=candle_1m.close,
                volume=candle_1m.volume,
                is_closed=False,
            )

        return closed_5m


class TechnicalAnalysisEngine:
    """Multi-asset orchestrator coordinating real-time technical analysis for the live pipeline."""

    def __init__(
        self,
        primary_index: str = "NSE:NIFTY50-INDEX",
        primary_futures: str = "NSE:NIFTY26NOVFUT",
        on_setup_eligible: Optional[Callable[[BreakoutContext, TechnicalSnapshot], None]] = None,
    ):
        self.primary_index = primary_index
        self.primary_futures = primary_futures
        self.on_setup_eligible = on_setup_eligible

        self.states: Dict[str, SymbolTechnicalState] = {}
        # Pre-initialize primary index and futures states
        self._get_or_create_state(primary_index, is_index=True)
        self._get_or_create_state(primary_futures, is_index=False)

    def _get_or_create_state(self, symbol: str, is_index: bool = False) -> SymbolTechnicalState:
        if symbol not in self.states:
            self.states[symbol] = SymbolTechnicalState(symbol=symbol, is_index=is_index)
        return self.states[symbol]

    def set_futures_ticker(self, futures_ticker: str) -> None:
        """Dynamically updates active futures ticker (e.g. at contract rollover)."""
        self.primary_futures = futures_ticker
        self._get_or_create_state(futures_ticker, is_index=False)

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
        """Dispatches closed candle to the corresponding symbol state and coordinates volume pairing."""
        is_idx = ("INDEX" in candle.symbol) or (candle.symbol == self.primary_index)
        state = self._get_or_create_state(candle.symbol, is_index=is_idx)

        # Cross-asset volume pairing: Spot index references paired Futures RVOL
        futures_rvol: Optional[float] = None
        if is_idx and self.primary_futures in self.states:
            futures_rvol = self.states[self.primary_futures].latest_rvol

        snapshot = state.process_1m_candle(
            candle=candle,
            futures_rvol=futures_rvol,
            on_setup_eligible=self.on_setup_eligible,
        )
        return snapshot

    def get_snapshot(self, symbol: str) -> Optional[TechnicalSnapshot]:
        """Returns the latest technical snapshot for a symbol."""
        if symbol in self.states:
            return self.states[symbol].latest_snapshot
        return None
