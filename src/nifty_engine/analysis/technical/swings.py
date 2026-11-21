"""Pivot Swing High and Low Detection Engine.

CRITICAL DESIGN PRINCIPLE: NO LOOKAHEAD BIAS.
A pivot swing point formed at candle `i` with `right_bars = k` is ONLY confirmed and visible
to trading logic after candle `i + k` has officially closed.
Every swing point stores both `timestamp` (when the price peak/trough occurred)
and `confirmation_timestamp` (the exact moment the strategy became aware of it).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd

from nifty_engine.core.interfaces import Candle


class SwingType(str, Enum):
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


@dataclass(frozen=True)
class SwingPoint:
    """Represents a validated pivot swing point."""
    index: int
    timestamp: datetime
    confirmation_index: int
    confirmation_timestamp: datetime
    price: float
    type: SwingType
    timeframe: str
    strength: int  # left_bars + right_bars
    broken: bool = False
    broken_timestamp: Optional[datetime] = None

    @property
    def is_high(self) -> bool:
        return self.type == SwingType.SWING_HIGH

    @property
    def is_low(self) -> bool:
        return self.type == SwingType.SWING_LOW


class SwingDetector:
    """Detects swing highs and lows with strict lookahead-bias prevention.

    Supports both historical batch dataframe analysis and real-time streaming bar updates.
    """

    def __init__(
        self,
        timeframe: str = "5m",
        left_bars: int = 5,
        right_bars: int = 5,
    ):
        if left_bars < 1 or right_bars < 1:
            raise ValueError("left_bars and right_bars must be >= 1")
        self.timeframe = timeframe
        self.left_bars = left_bars
        self.right_bars = right_bars
        self._history: List[Candle] = []
        self._confirmed_swings: List[SwingPoint] = []

    def detect_vectorized(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        high_col: str = "high",
        low_col: str = "low",
    ) -> List[SwingPoint]:
        """Vectorized pivot detection across a historical DataFrame.

        Marks confirmation_timestamp strictly as the timestamp of candle i + right_bars.
        """
        n = len(df)
        window = self.left_bars + self.right_bars + 1
        if n < window:
            return []

        highs = df[high_col].values
        lows = df[low_col].values
        times = pd.to_datetime(df[time_col]).tolist()

        swings: List[SwingPoint] = []

        for i in range(self.left_bars, n - self.right_bars):
            curr_high = highs[i]
            curr_low = lows[i]

            # Swing High Check
            is_high = True
            for left in range(1, self.left_bars + 1):
                if curr_high <= highs[i - left]:
                    is_high = False
                    break
            if is_high:
                for right in range(1, self.right_bars + 1):
                    if curr_high < highs[i + right]:
                        is_high = False
                        break

            # Swing Low Check
            is_low = True
            for left in range(1, self.left_bars + 1):
                if curr_low >= lows[i - left]:
                    is_low = False
                    break
            if is_low:
                for right in range(1, self.right_bars + 1):
                    if curr_low > lows[i + right]:
                        is_low = False
                        break

            conf_idx = i + self.right_bars
            conf_time = times[conf_idx]
            pivot_time = times[i]
            strength = self.left_bars + self.right_bars

            if is_high:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pivot_time,
                        confirmation_index=conf_idx,
                        confirmation_timestamp=conf_time,
                        price=float(curr_high),
                        type=SwingType.SWING_HIGH,
                        timeframe=self.timeframe,
                        strength=strength,
                    )
                )

            if is_low:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pivot_time,
                        confirmation_index=conf_idx,
                        confirmation_timestamp=conf_time,
                        price=float(curr_low),
                        type=SwingType.SWING_LOW,
                        timeframe=self.timeframe,
                        strength=strength,
                    )
                )

        return sorted(swings, key=lambda s: s.confirmation_index)

    def on_candle(self, candle: Candle) -> Optional[SwingPoint]:
        """Incremental streaming candle processor.

        Emits a SwingPoint immediately when a new candle confirms a prior pivot.
        Returns newly confirmed SwingPoint if one was confirmed by this candle, else None.
        """
        self._history.append(candle)
        n = len(self._history)
        window = self.left_bars + self.right_bars + 1

        if n < window:
            return None

        # The candidate pivot index is exactly right_bars behind current bar
        pivot_idx = n - 1 - self.right_bars
        candidate = self._history[pivot_idx]

        curr_high = candidate.high
        curr_low = candidate.low

        # Check swing high
        is_high = True
        for left in range(1, self.left_bars + 1):
            if curr_high <= self._history[pivot_idx - left].high:
                is_high = False
                break
        if is_high:
            for right in range(1, self.right_bars + 1):
                if curr_high < self._history[pivot_idx + right].high:
                    is_high = False
                    break

        # Check swing low
        is_low = True
        for left in range(1, self.left_bars + 1):
            if curr_low >= self._history[pivot_idx - left].low:
                is_low = False
                break
        if is_low:
            for right in range(1, self.right_bars + 1):
                if curr_low > self._history[pivot_idx + right].low:
                    is_low = False
                    break

        new_swing: Optional[SwingPoint] = None
        strength = self.left_bars + self.right_bars

        if is_high:
            new_swing = SwingPoint(
                index=pivot_idx,
                timestamp=candidate.timestamp,
                confirmation_index=n - 1,
                confirmation_timestamp=candle.timestamp,
                price=float(curr_high),
                type=SwingType.SWING_HIGH,
                timeframe=self.timeframe,
                strength=strength,
            )
            self._confirmed_swings.append(new_swing)
        elif is_low:
            new_swing = SwingPoint(
                index=pivot_idx,
                timestamp=candidate.timestamp,
                confirmation_index=n - 1,
                confirmation_timestamp=candle.timestamp,
                price=float(curr_low),
                type=SwingType.SWING_LOW,
                timeframe=self.timeframe,
                strength=strength,
            )
            self._confirmed_swings.append(new_swing)

        return new_swing

    def get_visible_swings(self, as_of_time: datetime) -> List[SwingPoint]:
        """Returns only swings that were confirmed ON OR BEFORE as_of_time.

        Strictly prevents any lookahead bias in backtesting or replay.
        """
        return [
            s for s in self._confirmed_swings
            if s.confirmation_timestamp <= as_of_time
        ]

    def get_latest_high(self, as_of_time: datetime) -> Optional[SwingPoint]:
        """Gets the most recently confirmed swing high as of given time."""
        highs = [
            s for s in self._confirmed_swings
            if s.is_high and s.confirmation_timestamp <= as_of_time
        ]
        return highs[-1] if highs else None

    def get_latest_low(self, as_of_time: datetime) -> Optional[SwingPoint]:
        """Gets the most recently confirmed swing low as of given time."""
        lows = [
            s for s in self._confirmed_swings
            if s.is_low and s.confirmation_timestamp <= as_of_time
        ]
        return lows[-1] if lows else None
