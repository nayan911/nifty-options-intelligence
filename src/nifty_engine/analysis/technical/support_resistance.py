"""Dynamic Support and Resistance Zones Engine.

Computes volatility-adjusted S/R zones including:
1. Daily Benchmark Levels: PDH (Prior Day High), PDL (Prior Day Low), PDC (Prior Day Close), CDO (Current Day Open).
2. Opening Range Balance: ORH and ORL (15-minute Opening Range: 09:15 - 09:30 IST).
3. Psychological Levels: Round strikes (100-point major, 50-point minor).
4. Structural Swing Levels: Horizontal clusters derived from confirmed swing highs/lows.
5. Volume Benchmark: Intraday Futures VWAP and bands.
"""

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import List, Optional, Tuple

import pytz

from nifty_engine.analysis.technical.swings import SwingPoint
from nifty_engine.core.interfaces import Candle

IST = pytz.timezone("Asia/Kolkata")


class LevelType(str, Enum):
    PDH = "PDH"
    PDL = "PDL"
    PDC = "PDC"
    CDO = "CDO"
    ORH = "ORH"
    ORL = "ORL"
    PSYCHOLOGICAL = "PSYCHOLOGICAL"
    SWING_HIGH_CLUSTER = "SWING_HIGH_CLUSTER"
    SWING_LOW_CLUSTER = "SWING_LOW_CLUSTER"
    VWAP = "VWAP"


@dataclass(frozen=True)
class PriceZone:
    """Represents a volatility-buffered support or resistance zone."""
    level_type: LevelType
    nominal_price: float
    upper_bound: float
    lower_bound: float
    description: str
    weight: float

    def contains(self, price: float) -> bool:
        """Checks if a price falls inside this zone."""
        return self.lower_bound <= price <= self.upper_bound

    def is_above(self, price: float) -> bool:
        """True if the zone is above the given price (acting as resistance)."""
        return self.lower_bound > price

    def is_below(self, price: float) -> bool:
        """True if the zone is below the given price (acting as support)."""
        return self.upper_bound < price


class SupportResistanceEngine:
    """Manages and evaluates dynamic support and resistance zones."""

    EXPECTED_OR_TIMES_1M = {time(9, 15 + i) for i in range(15)}
    EXPECTED_OR_TIMES_5M = {time(9, 15), time(9, 20), time(9, 25)}
    EXPECTED_OR_TIMES_15M = {time(9, 15)}

    def __init__(
        self,
        zone_atr_multiple: float = 0.15,
        min_zone_width_points: float = 3.0,
    ):
        self.zone_atr_multiple = zone_atr_multiple
        self.min_zone_width_points = min_zone_width_points

    def create_zone(
        self,
        level_type: LevelType,
        nominal_price: float,
        atr_value: float,
        description: str,
        weight: float = 1.0,
    ) -> PriceZone:
        """Constructs a volatility-buffered price zone."""
        buffer = max(self.zone_atr_multiple * atr_value, self.min_zone_width_points)
        return PriceZone(
            level_type=level_type,
            nominal_price=round(nominal_price, 2),
            upper_bound=round(nominal_price + buffer, 2),
            lower_bound=round(nominal_price - buffer, 2),
            description=description,
            weight=weight,
        )

    def compute_daily_levels(
        self,
        pdh: float,
        pdl: float,
        pdc: float,
        cdo: float,
        atr_value: float,
    ) -> List[PriceZone]:
        """Builds zones for Prior Day High/Low/Close and Current Day Open."""
        return [
            self.create_zone(LevelType.PDH, pdh, atr_value, "Prior Day High", weight=2.0),
            self.create_zone(LevelType.PDL, pdl, atr_value, "Prior Day Low", weight=2.0),
            self.create_zone(LevelType.PDC, pdc, atr_value, "Prior Day Close", weight=1.5),
            self.create_zone(LevelType.CDO, cdo, atr_value, "Current Day Open", weight=1.2),
        ]

    def compute_opening_range(
        self,
        candles_1m_or_5m: List[Candle],
        atr_value: float,
        or_start: time = time(9, 15),
        or_end: time = time(9, 30),
        session_date: Optional[date] = None,
    ) -> List[PriceZone]:
        """Calculates 15-minute Opening Range High (ORH) and Low (ORL).

        Enforces:
        1. Trading-session date isolation (filters exclusively for session_date).
        2. Window completion (requires candles spanning the complete 09:15-09:30 interval).
        """
        if not candles_1m_or_5m:
            return []

        target_date = session_date or candles_1m_or_5m[-1].timestamp.date()
        session_candles = [c for c in candles_1m_or_5m if c.timestamp.date() == target_date]

        or_candles = [
            c for c in session_candles
            if or_start <= c.timestamp.time() < or_end
        ]
        if not or_candles:
            return []

        timeframe = session_candles[0].timeframe
        present_times = {c.timestamp.time() for c in or_candles}

        # Strict completeness enforcement: All distinct expected closed candle intervals must be present
        if timeframe == "1m":
            if not self.EXPECTED_OR_TIMES_1M.issubset(present_times):
                return []
        elif timeframe == "5m":
            if not self.EXPECTED_OR_TIMES_5M.issubset(present_times):
                return []
        elif timeframe == "15m":
            if not self.EXPECTED_OR_TIMES_15M.issubset(present_times):
                return []
        else:
            if len(or_candles) < 3:
                return []

        orh = max(c.high for c in or_candles)
        orl = min(c.low for c in or_candles)

        return [
            self.create_zone(LevelType.ORH, orh, atr_value, "15-Min Opening Range High", weight=1.8),
            self.create_zone(LevelType.ORL, orl, atr_value, "15-Min Opening Range Low", weight=1.8),
        ]

    def compute_psychological_levels(
        self,
        current_price: float,
        atr_value: float,
        range_points: float = 200.0,
    ) -> List[PriceZone]:
        """Generates major (100-pt) and minor (50-pt) psychological round numbers."""
        zones = []
        low_bound = current_price - range_points
        high_bound = current_price + range_points

        # 50-point step
        start_round = int(low_bound // 50) * 50
        end_round = int(high_bound // 50 + 1) * 50

        for strike in range(start_round, end_round + 1, 50):
            is_century = (strike % 100 == 0)
            weight = 1.5 if is_century else 0.8
            desc = f"{'Major' if is_century else 'Minor'} Psychological Round Level ({strike})"
            zones.append(
                self.create_zone(
                    LevelType.PSYCHOLOGICAL,
                    float(strike),
                    atr_value,
                    desc,
                    weight=weight,
                )
            )
        return zones

    def compute_swing_clusters(
        self,
        swings: List[SwingPoint],
        atr_value: float,
        as_of_time: Optional[datetime] = None,
        cluster_tolerance_points: float = 8.0,
    ) -> List[PriceZone]:
        """Groups nearby confirmed swing points into horizontal support/resistance clusters."""
        valid_swings = [
            s for s in swings
            if as_of_time is None or s.confirmation_timestamp <= as_of_time
        ]
        if not valid_swings:
            return []

        zones: List[PriceZone] = []

        # Process Highs
        highs = sorted([s.price for s in valid_swings if s.is_high])
        clusters_high = self._cluster_prices(highs, cluster_tolerance_points)
        for avg_p, count in clusters_high:
            weight = 1.0 + (count * 0.4)
            zones.append(
                self.create_zone(
                    LevelType.SWING_HIGH_CLUSTER,
                    avg_p,
                    atr_value,
                    f"Swing Resistance Cluster ({count} touches at {avg_p:.1f})",
                    weight=weight,
                )
            )

        # Process Lows
        lows = sorted([s.price for s in valid_swings if s.is_low])
        clusters_low = self._cluster_prices(lows, cluster_tolerance_points)
        for avg_p, count in clusters_low:
            weight = 1.0 + (count * 0.4)
            zones.append(
                self.create_zone(
                    LevelType.SWING_LOW_CLUSTER,
                    avg_p,
                    atr_value,
                    f"Swing Support Cluster ({count} touches at {avg_p:.1f})",
                    weight=weight,
                )
            )

        return zones

    def _cluster_prices(
        self,
        sorted_prices: List[float],
        tolerance: float
    ) -> List[Tuple[float, int]]:
        if not sorted_prices:
            return []
        clusters = []
        curr_cluster = [sorted_prices[0]]

        for p in sorted_prices[1:]:
            if p - curr_cluster[-1] <= tolerance:
                curr_cluster.append(p)
            else:
                clusters.append((sum(curr_cluster) / len(curr_cluster), len(curr_cluster)))
                curr_cluster = [p]

        clusters.append((sum(curr_cluster) / len(curr_cluster), len(curr_cluster)))
        return clusters

    def find_nearest_zones(
        self,
        current_price: float,
        all_zones: List[PriceZone],
    ) -> Tuple[Optional[PriceZone], Optional[PriceZone]]:
        """Returns the immediate support (closest zone below) and resistance (closest zone above)."""
        supports = [z for z in all_zones if z.is_below(current_price)]
        resistances = [z for z in all_zones if z.is_above(current_price)]

        nearest_support = max(supports, key=lambda z: z.nominal_price) if supports else None
        nearest_resistance = min(resistances, key=lambda z: z.nominal_price) if resistances else None

        return nearest_support, nearest_resistance
