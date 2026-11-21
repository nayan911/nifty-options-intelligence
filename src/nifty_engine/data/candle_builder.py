"""Real-time Candlestick Aggregator with rigorous volume increment handling.

Strictly adheres to:
1. Isolated bar High and Low (never confusing day's cumulative high/low).
2. True incremental volume calculation (deriving increments from cumulative vol_traded_today).
3. Spot index volume handling (NIFTY/BANKNIFTY/SENSEX spot index volume is strictly 0).
4. Time-based candle finalization (closing bars even if no tick arrives immediately after interval boundary).
5. Duplicate tick rejection and out-of-order tick handling.
6. Daily volume reset handling.
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set
import pytz

from nifty_engine.core.interfaces import Candle, Tick

IST = pytz.timezone("Asia/Kolkata")


class ActiveCandle:
    """Maintains running state for an unclosed candle bar."""

    def __init__(self, symbol: str, timeframe: str, start_time: datetime) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: int = 0
        self.last_oi: Optional[int] = None
        self.tick_count: int = 0
        self.last_cumulative_vol: Optional[int] = None

    def update(self, tick: Tick) -> None:
        """Update running OHLCV with a new tick."""
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
        # If spot index -> volume is 0
        if tick.is_index:
            self.volume = 0
        else:
            # Use incremental volume if already calculated, or calculate from cumulative
            if tick.incremental_volume > 0:
                self.volume += tick.incremental_volume
            elif tick.cumulative_volume > 0:
                if self.last_cumulative_vol is not None:
                    if tick.cumulative_volume >= self.last_cumulative_vol:
                        delta = tick.cumulative_volume - self.last_cumulative_vol
                        self.volume += delta
                    else:
                        # Cumulative volume reset (e.g. new session or broker counter reset)
                        self.volume += tick.cumulative_volume
                self.last_cumulative_vol = tick.cumulative_volume

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

    def __init__(
        self,
        on_candle_close: Optional[Callable[[Candle], None]] = None
    ) -> None:
        self.on_candle_close = on_candle_close
        # Key: (symbol, timeframe) -> ActiveCandle
        self._active_bars: Dict[tuple, ActiveCandle] = {}
        # Track last processed tick per symbol to reject duplicates: (symbol) -> (timestamp, ltp, cumulative_vol)
        self._last_tick_fingerprint: Dict[str, tuple] = {}
        # Track overall symbol cumulative volume baseline
        self._symbol_cum_volume: Dict[str, int] = {}

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
        """Ingest a tick. If a bar boundary is crossed, close old bar and emit."""
        if self.is_duplicate(tick):
            return []

        tick_time = tick.exchange_timestamp or tick.local_timestamp
        closed_candles: List[Candle] = []

        # Calculate incremental volume across the tick stream if cumulative volume provided
        if not tick.is_index and tick.cumulative_volume > 0 and tick.incremental_volume == 0:
            last_vol = self._symbol_cum_volume.get(tick.symbol)
            if last_vol is not None:
                if tick.cumulative_volume >= last_vol:
                    tick.incremental_volume = tick.cumulative_volume - last_vol
                else:
                    tick.incremental_volume = tick.cumulative_volume
            self._symbol_cum_volume[tick.symbol] = tick.cumulative_volume

        for tf, interval_mins in self.TIMEFRAMES_MINUTES.items():
            key = (tick.symbol, tf)
            bar_start = self._get_bar_start_time(tick_time, interval_mins)

            active = self._active_bars.get(key)
            if active is None:
                # First tick for this symbol/timeframe
                active = ActiveCandle(symbol=tick.symbol, timeframe=tf, start_time=bar_start)
                active.update(tick)
                self._active_bars[key] = active
            elif bar_start > active.start_time:
                # Interval boundary crossed -> close previous bar
                closed_candle = active.to_candle(is_closed=True)
                closed_candles.append(closed_candle)

                if self.on_candle_close:
                    self.on_candle_close(closed_candle)

                # Start fresh bar
                new_bar = ActiveCandle(symbol=tick.symbol, timeframe=tf, start_time=bar_start)
                new_bar.update(tick)
                self._active_bars[key] = new_bar
            elif bar_start < active.start_time:
                # Out-of-order tick arriving for an already closed past interval -> Ignore to prevent lookahead corruption
                continue
            else:
                # Tick belongs to current active bar
                active.update(tick)

        return closed_candles

    def flush_closed_bars(self, current_time: datetime) -> List[Candle]:
        """Finalize and emit pending bars whose interval has expired even if no tick arrived."""
        closed_candles: List[Candle] = []

        for (symbol, tf), active in list(self._active_bars.items()):
            interval_mins = self.TIMEFRAMES_MINUTES[tf]
            bar_end_time = active.start_time + timedelta(minutes=interval_mins)

            if current_time >= bar_end_time and active.tick_count > 0:
                closed_candle = active.to_candle(is_closed=True)
                closed_candles.append(closed_candle)

                if self.on_candle_close:
                    self.on_candle_close(closed_candle)

                # Remove from active
                del self._active_bars[(symbol, tf)]

        return closed_candles

    def get_current_candle(self, symbol: str, timeframe: str = "1m") -> Optional[Candle]:
        """Peek at currently building unclosed candle."""
        active = self._active_bars.get((symbol, timeframe))
        if active:
            return active.to_candle(is_closed=False)
        return None
