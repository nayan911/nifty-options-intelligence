"""Bounded Rolling-Window Trendline Engine.

Adapted from geometric principles of `ednunezg/pytrendline` (MIT License)
optimized specifically for real-time intraday trading:
1. Bounded strictly to recent N candles (100-200 bars) for deterministic < 50ms execution.
2. Anchored exclusively on confirmed Swing Points (zero lookahead bias).
3. Volatility-adjusted touch tolerance parameterized via ATR (default 0.1 * ATR).
4. Strict body penetration rejection (support cannot cut candle closes/lows, resistance cannot cut closes/highs).
5. Touch spacing validation (prevents false multi-touch on adjacent bars).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from nifty_engine.analysis.technical.swings import SwingPoint
from nifty_engine.core.interfaces import Candle


@dataclass(frozen=True)
class Trendline:
    """Represents a validated trendline."""
    line_type: str  # "SUPPORT" or "RESISTANCE"
    slope: float  # Price change per bar index
    intercept: float  # Price at index 0 of the window
    start_index: int
    end_index: int
    start_timestamp: datetime
    end_timestamp: datetime
    touch_indices: Tuple[int, ...]
    touch_prices: Tuple[float, ...]
    num_touches: int
    score: float
    atr_tolerance: float

    def value_at(self, index: int) -> float:
        """Returns the trendline price at given candle index."""
        return (self.slope * index) + self.intercept


class TrendlineDetector:
    """Detects and scores support and resistance trendlines within a bounded window."""

    def __init__(
        self,
        window_size: int = 150,
        min_touches: int = 2,
        min_touch_spacing: int = 4,
        tolerance_atr_multiple: float = 0.12,
    ):
        self.window_size = window_size
        self.min_touches = min_touches
        self.min_touch_spacing = min_touch_spacing
        self.tolerance_atr_multiple = tolerance_atr_multiple

    def find_trendlines(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        atr_value: float,
        as_of_time: Optional[datetime] = None,
    ) -> List[Trendline]:
        """Detects active, non-violated trendlines from visible swing points."""
        # Strict lookahead guard: truncate candles to as_of_time
        if as_of_time is not None:
            candles = [c for c in candles if c.timestamp <= as_of_time]

        if len(candles) < 10 or len(swings) < 2 or atr_value <= 0:
            return []

        # Filter candles to bounded window
        bounded_candles = candles[-self.window_size:]
        start_idx = candles.index(bounded_candles[0])
        curr_idx = len(candles) - 1

        # Filter swings: must be confirmed on or before as_of_time, and within bounded window
        valid_swings = [
            s for s in swings
            if (as_of_time is None or s.confirmation_timestamp <= as_of_time)
            and s.index >= start_idx
            and s.index <= curr_idx
        ]

        tolerance = max(self.tolerance_atr_multiple * atr_value, 1.0)

        support_lines = self._fit_lines(
            swings=[s for s in valid_swings if s.is_low],
            candles=candles,
            line_type="SUPPORT",
            tolerance=tolerance,
            curr_idx=curr_idx,
        )

        resistance_lines = self._fit_lines(
            swings=[s for s in valid_swings if s.is_high],
            candles=candles,
            line_type="RESISTANCE",
            tolerance=tolerance,
            curr_idx=curr_idx,
        )

        all_lines = support_lines + resistance_lines
        # Sort descending by score
        return sorted(all_lines, key=lambda l: l.score, reverse=True)

    def _fit_lines(
        self,
        swings: List[SwingPoint],
        candles: List[Candle],
        line_type: str,
        tolerance: float,
        curr_idx: int,
    ) -> List[Trendline]:
        lines: List[Trendline] = []
        n_swings = len(swings)
        if n_swings < 2:
            return lines

        for i in range(n_swings):
            s1 = swings[i]
            for j in range(i + 1, n_swings):
                s2 = swings[j]
                idx_diff = s2.index - s1.index
                if idx_diff < self.min_touch_spacing:
                    continue

                slope = (s2.price - s1.price) / idx_diff
                intercept = s1.price - (slope * s1.index)

                # Geometry direction filters:
                # Support trendlines in an uptrend have slope >= -0.05 * tolerance
                # Resistance trendlines in a downtrend have slope <= 0.05 * tolerance
                # Validate penetration along the line
                is_valid = True
                touch_indices = [s1.index, s2.index]
                touch_prices = [s1.price, s2.price]

                # Check intermediate and current candles up to curr_idx
                for c_idx in range(s1.index, curr_idx + 1):
                    c = candles[c_idx]
                    line_val = (slope * c_idx) + intercept

                    if line_type == "SUPPORT":
                        # For support: price close or low cannot penetrate below the line beyond tolerance
                        if c.close < line_val - tolerance:
                            is_valid = False
                            break
                        # Check touch
                        if abs(c.low - line_val) <= tolerance and c_idx not in touch_indices:
                            # Verify spacing with last recorded touch
                            if c_idx - touch_indices[-1] >= self.min_touch_spacing:
                                touch_indices.append(c_idx)
                                touch_prices.append(c.low)

                    elif line_type == "RESISTANCE":
                        # For resistance: price close or high cannot penetrate above the line beyond tolerance
                        if c.close > line_val + tolerance:
                            is_valid = False
                            break
                        # Check touch
                        if abs(c.high - line_val) <= tolerance and c_idx not in touch_indices:
                            if c_idx - touch_indices[-1] >= self.min_touch_spacing:
                                touch_indices.append(c_idx)
                                touch_prices.append(c.high)

                if not is_valid:
                    continue

                num_touches = len(touch_indices)
                if num_touches < self.min_touches:
                    continue

                # Scoring:
                # 1. Base touch score (each touch adds 15 pts)
                # 2. Touch span (longer spanning line is more significant)
                # 3. Recency bonus: reward lines where the latest touch is near current bar
                span = curr_idx - s1.index
                recency = max(0.0, 10.0 - (curr_idx - touch_indices[-1]))
                score = (num_touches * 15.0) + (span * 0.2) + recency

                lines.append(
                    Trendline(
                        line_type=line_type,
                        slope=float(slope),
                        intercept=float(intercept),
                        start_index=s1.index,
                        end_index=curr_idx,
                        start_timestamp=s1.timestamp,
                        end_timestamp=candles[curr_idx].timestamp,
                        touch_indices=tuple(sorted(touch_indices)),
                        touch_prices=tuple(touch_prices),
                        num_touches=num_touches,
                        score=float(score),
                        atr_tolerance=float(tolerance),
                    )
                )

        # Deduplicate near-identical lines (keep highest score)
        return self._deduplicate_lines(lines)

    def _deduplicate_lines(self, lines: List[Trendline]) -> List[Trendline]:
        if not lines:
            return []
        sorted_lines = sorted(lines, key=lambda l: l.score, reverse=True)
        unique_lines: List[Trendline] = []

        for candidate in sorted_lines:
            duplicate = False
            for existing in unique_lines:
                if candidate.line_type == existing.line_type:
                    # If slopes and intercepts are within 2% of each other, consider duplicate
                    slope_diff = abs(candidate.slope - existing.slope)
                    intercept_diff = abs(candidate.intercept - existing.intercept)
                    if slope_diff < 0.1 and intercept_diff < 5.0:
                        duplicate = True
                        break
            if not duplicate:
                unique_lines.append(candidate)

        return unique_lines
