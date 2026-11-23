"""Real-time Candlestick Aggregator with rigorous volume increment handling.

Strictly adheres to:
1. Isolated bar High and Low (never confusing day's cumulative high/low).
2. True incremental volume calculation (deriving increments from cumulative vol_traded_today).
3. Spot index volume handling (NIFTY/BANKNIFTY/SENSEX spot index volume is strictly 0).
4. Time-based candle finalization (closing bars even if no tick arrives immediately after interval boundary).
5. Duplicate tick rejection and out-of-order tick handling.
6. Daily volume reset handling.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pytz

from nifty_engine.core.interfaces import Candle, Tick
from nifty_engine.utils.timezone import now_ist

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class ActiveCandle:
    """Maintains running state for an unclosed candle bar."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        is_incomplete: bool = False,
        initial_tick_offset_seconds: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.is_incomplete = is_incomplete
        self.initial_tick_offset_seconds = initial_tick_offset_seconds
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: int = 0
        self.last_oi: Optional[int] = None
        self.tick_count: int = 0

    def update(self, tick: Tick) -> None:
        """Update running OHLCV with a new tick.

        TickNormalizer is the single source of truth for incremental volume.
        """
        price = tick.ltp
        # Bar High and Low are strictly calculated from ltp within this bar
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
        else:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price

        self.close = price

        # Volume Calculation:
        # TickNormalizer is the single source of truth for incremental volume.
        # Spot indices have volume = 0 strictly.
        if tick.is_index:
            self.volume = 0
        elif tick.incremental_volume is not None and tick.incremental_volume > 0:
            self.volume += tick.incremental_volume

        if tick.oi is not None:
            self.last_oi = tick.oi

        self.tick_count += 1

    def to_candle(self, is_closed: bool = False) -> Candle:
        """Convert running state to an immutable Candle model."""
        return Candle(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=self.start_time,
            open=self.open if self.open is not None else 0.0,
            high=self.high if self.high is not None else 0.0,
            low=self.low if self.low is not None else 0.0,
            close=self.close if self.close is not None else 0.0,
            volume=self.volume,
            oi=self.last_oi,
            is_closed=is_closed,
        )


class RealtimeCandleBuilder:
    """Builds and emits 1m, 5m, and 15m candles from incoming live ticks."""

    TIMEFRAMES_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
    }

    # Maximum delay allowed between bar_start and the first received tick
    # for an initial bar to be considered complete (untruncated).
    MAX_INITIAL_TICK_OFFSET_SECONDS = {
        "1m": 20.0,
        "5m": 45.0,
        "15m": 120.0,
    }

    def __init__(
        self,
        on_candle_close: Optional[Callable[[Candle], None]] = None,
        drop_incomplete_initial_bars: bool = True,
    ) -> None:
        self.on_candle_close = on_candle_close
        self.drop_incomplete_initial_bars = drop_incomplete_initial_bars
        self._lock = threading.Lock()
        # Key: (symbol, timeframe) -> ActiveCandle
        self._active_bars: Dict[tuple, ActiveCandle] = {}
        # Track last closed bar start time per (symbol, timeframe) to guarantee exactly-once closure
        self._last_closed_bar_start: Dict[tuple, datetime] = {}
        # Track last processed tick per symbol to reject duplicates: (symbol) -> (timestamp, ltp, cumulative_vol)
        self._last_tick_fingerprint: Dict[str, tuple] = {}

    def _get_bar_start_time(self, dt: datetime, interval_minutes: int) -> datetime:
        """Floor datetime to the nearest lower interval boundary."""
        minute = (dt.minute // interval_minutes) * interval_minutes
        return dt.replace(minute=minute, second=0, microsecond=0)

    def is_duplicate(self, tick: Tick) -> bool:
        """Check and reject identical consecutive ticks from network duplicates."""
        tick_time = tick.exchange_timestamp or tick.local_timestamp
        fp = (tick.symbol, tick_time, tick.ltp, tick.cumulative_volume)
        if self._last_tick_fingerprint.get(tick.symbol) == fp:
            return True
        self._last_tick_fingerprint[tick.symbol] = fp
        return False

    def process_tick(self, tick: Tick) -> List[Candle]:
        """Ingest a tick thread-safely. If a bar boundary is crossed, close old bar and emit."""
        closed_candles: List[Candle] = []
        with self._lock:
            if self.is_duplicate(tick):
                return []

            tick_time = tick.exchange_timestamp or tick.local_timestamp

            for tf, interval_mins in self.TIMEFRAMES_MINUTES.items():
                key = (tick.symbol, tf)
                bar_start = self._get_bar_start_time(tick_time, interval_mins)

                # 1. Guard against duplicate closed candle:
                # If this interval has already been finalized and closed, ignore this delayed tick
                last_closed = self._last_closed_bar_start.get(key)
                if last_closed is not None and bar_start <= last_closed:
                    continue

                active = self._active_bars.get(key)
                if active is None:
                    # First tick for this symbol/timeframe
                    offset_sec = (tick_time - bar_start).total_seconds()
                    max_offset = self.MAX_INITIAL_TICK_OFFSET_SECONDS.get(tf, 20.0)
                    is_incomplete = bool(self.drop_incomplete_initial_bars and offset_sec > max_offset)

                    active = ActiveCandle(
                        symbol=tick.symbol,
                        timeframe=tf,
                        start_time=bar_start,
                        is_incomplete=is_incomplete,
                        initial_tick_offset_seconds=offset_sec,
                    )
                    active.update(tick)
                    self._active_bars[key] = active
                elif bar_start > active.start_time:
                    # Interval boundary crossed -> close previous bar
                    self._last_closed_bar_start[key] = active.start_time

                    if not active.is_incomplete:
                        closed_candle = active.to_candle(is_closed=True)
                        closed_candle.observed_at = tick.local_timestamp or now_ist()
                        closed_candles.append(closed_candle)
                    else:
                        logger.warning(
                            "Dropped incomplete initial candle for %s (%s) starting at %s "
                            "(first tick arrived %.1fs into %dm interval). Next full bar begins at %s.",
                            active.symbol,
                            active.timeframe,
                            active.start_time.strftime("%H:%M:%S"),
                            active.initial_tick_offset_seconds,
                            interval_mins,
                            bar_start.strftime("%H:%M:%S"),
                        )

                    # Start fresh bar at the new boundary: check completeness in case feed was delayed/paused
                    offset_sec = (tick_time - bar_start).total_seconds()
                    max_offset = self.MAX_INITIAL_TICK_OFFSET_SECONDS.get(tf, 20.0)
                    is_incomplete = bool(self.drop_incomplete_initial_bars and offset_sec > max_offset)

                    new_bar = ActiveCandle(
                        symbol=tick.symbol,
                        timeframe=tf,
                        start_time=bar_start,
                        is_incomplete=is_incomplete,
                        initial_tick_offset_seconds=offset_sec,
                    )
                    new_bar.update(tick)
                    self._active_bars[key] = new_bar
                elif bar_start < active.start_time:
                    # Out-of-order tick arriving for an already closed past interval -> Ignore
                    continue
                else:
                    # Tick belongs to current active bar
                    active.update(tick)

        # Dispatch callbacks outside lock to prevent deadlocks and ensure exactly-once handling
        if self.on_candle_close and closed_candles:
            for closed in closed_candles:
                self.on_candle_close(closed)

        return closed_candles

    def flush_closed_bars(self, current_time: datetime) -> List[Candle]:
        """Finalize and emit pending bars whose interval has expired even if no tick arrived."""
        closed_candles: List[Candle] = []
        with self._lock:
            for (symbol, tf), active in list(self._active_bars.items()):
                interval_mins = self.TIMEFRAMES_MINUTES[tf]
                bar_end_time = active.start_time + timedelta(minutes=interval_mins)

                if current_time >= bar_end_time and active.tick_count > 0:
                    # 1. Record last closed bar start time to prevent duplicate recreation!
                    self._last_closed_bar_start[(symbol, tf)] = active.start_time

                    # 2. Only emit complete candles
                    if not active.is_incomplete:
                        closed_candle = active.to_candle(is_closed=True)
                        closed_candle.observed_at = current_time
                        closed_candles.append(closed_candle)
                    else:
                        logger.warning(
                            "Dropped incomplete initial candle for %s (%s) starting at %s during flush "
                            "(first tick arrived %.1fs into %dm interval).",
                            active.symbol,
                            active.timeframe,
                            active.start_time.strftime("%H:%M:%S"),
                            active.initial_tick_offset_seconds,
                            interval_mins,
                        )

                    # Remove from active
                    del self._active_bars[(symbol, tf)]

        # Dispatch callbacks outside lock to prevent deadlocks and ensure exactly-once handling
        if self.on_candle_close and closed_candles:
            for closed in closed_candles:
                self.on_candle_close(closed)

        return closed_candles

    def get_current_candle(self, symbol: str, timeframe: str = "1m") -> Optional[Candle]:
        """Peek at currently building unclosed candle."""
        with self._lock:
            active = self._active_bars.get((symbol, timeframe))
            if active:
                return active.to_candle(is_closed=False)
            return None
