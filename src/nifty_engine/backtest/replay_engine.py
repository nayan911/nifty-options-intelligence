"""Historical Market Replay Engine.

Replays recorded ticks and OHLCV candles chronologically with strict lookahead isolation:
1. Multi-symbol event stream interleaving: Merges feeds from Spot Index, Futures, and Options in exact timestamp order.
2. Identical interface parity: Dispatches through standard on_tick and on_candle callbacks matching live WebSocket behavior.
3. Synchronous and asynchronous stepped execution for deterministic unit tests and backtesting.
4. Simulated clock ensuring strategy logic cannot access future events.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Union

from nifty_engine.core.interfaces import Candle, Tick
from nifty_engine.data.storage.duckdb_store import DuckDBStore


@dataclass(order=True)
class ReplayEvent:
    """Timestamp-ordered priority wrapper for chronological stream interleaving."""
    timestamp: datetime
    event_type: str  # 'TICK' or 'CANDLE'
    payload: Union[Tick, Candle] = None


class MarketReplayEngine:
    """Historical event-driven simulator with zero lookahead bias."""

    def __init__(self, store: DuckDBStore):
        self.store = store
        self._candle_subscribers: List[Callable[[Candle], None]] = []
        self._tick_subscribers: List[Callable[[Tick], None]] = []
        self._current_time: Optional[datetime] = None
        self._events_processed: int = 0

    @property
    def current_time(self) -> Optional[datetime]:
        """Current simulated exchange/local time."""
        return self._current_time

    @property
    def events_processed(self) -> int:
        return self._events_processed

    def subscribe_candle(self, callback: Callable[[Candle], None]) -> None:
        """Register a callback for candle finalization events."""
        self._candle_subscribers.append(callback)

    def subscribe_tick(self, callback: Callable[[Tick], None]) -> None:
        """Register a callback for live tick events."""
        self._tick_subscribers.append(callback)

    @staticmethod
    def get_timeframe_delta(timeframe: str) -> timedelta:
        """Derive candle duration to establish exact bar finalization/availability time."""
        tf = timeframe.lower()
        if tf.endswith("m"):
            return timedelta(minutes=int(tf[:-1]))
        elif tf in ("1d", "d"):
            return timedelta(days=1)
        return timedelta(minutes=1)

    def load_candle_stream(
        self,
        symbols: List[str],
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[ReplayEvent]:
        """Loads and sorts candles into an interleaved event list.

        CRITICAL LOOKAHEAD GUARD:
        A candle with open timestamp T and timeframe 5m is only completed and visible
        to strategy logic at T + 5m (the bar close). The ReplayEvent timestamp is strictly
        set to this availability timestamp.
        """
        all_events: List[ReplayEvent] = []
        for sym in symbols:
            candles = self.store.get_candles(sym, timeframe, start_time, end_time)
            for c in candles:
                availability_time = c.timestamp + self.get_timeframe_delta(c.timeframe)
                all_events.append(
                    ReplayEvent(
                        timestamp=availability_time,
                        event_type="CANDLE",
                        payload=c,
                    )
                )

        # Sort strictly ascending by availability timestamp; if identical, maintain deterministic order
        all_events.sort(key=lambda e: (e.timestamp, e.payload.symbol))
        return all_events
        all_events.sort(key=lambda e: (e.timestamp, e.payload.symbol))
        return all_events

    def load_tick_stream(
        self,
        symbols: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ReplayEvent]:
        """Loads and sorts ticks for multiple symbols into an interleaved event list."""
        all_events: List[ReplayEvent] = []
        for sym in symbols:
            ticks = self.store.get_ticks(sym, start_time, end_time)
            for t in ticks:
                ts = t.exchange_timestamp or t.local_timestamp
                all_events.append(
                    ReplayEvent(
                        timestamp=ts,
                        event_type="TICK",
                        payload=t,
                    )
                )

        all_events.sort(key=lambda e: (e.timestamp, e.payload.symbol))
        return all_events

    def replay_events(
        self,
        events: List[ReplayEvent],
        speed_multiplier: float = 0.0,
    ) -> int:
        """Replays pre-loaded event list sequentially.

        If speed_multiplier == 0, runs synchronously as fast as possible (for backtests).
        If speed_multiplier > 0, paces delivery proportional to actual delta-time.
        """
        self._events_processed = 0
        prev_event_time: Optional[datetime] = None

        for event in events:
            # Paced replay if speed_multiplier > 0
            if speed_multiplier > 0 and prev_event_time is not None:
                delta_sec = (event.timestamp - prev_event_time).total_seconds()
                if delta_sec > 0:
                    time.sleep(delta_sec / speed_multiplier)

            # Advance simulated clock
            self._current_time = event.timestamp
            prev_event_time = event.timestamp

            # Dispatch to subscribers
            if event.event_type == "CANDLE":
                for sub in self._candle_subscribers:
                    sub(event.payload)
            elif event.event_type == "TICK":
                for sub in self._tick_subscribers:
                    sub(event.payload)

            self._events_processed += 1

        return self._events_processed
