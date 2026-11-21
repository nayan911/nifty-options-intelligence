"""Unit Tests for Historical Market Replay Engine."""

from datetime import datetime, timedelta
import pytest
import pytz

from nifty_engine.backtest.replay_engine import MarketReplayEngine, ReplayEvent
from nifty_engine.core.interfaces import Candle, Tick
from nifty_engine.data.storage.duckdb_store import DuckDBStore

IST = pytz.timezone("Asia/Kolkata")


def test_market_replay_interleaved_candles():
    store = DuckDBStore(":memory:")
    base_time = datetime(2026, 11, 23, 9, 15, tzinfo=IST)

    # Insert candles for two symbols at alternating timestamps
    sym1 = "NSE:NIFTY50-INDEX"
    sym2 = "NSE:NIFTY26NOVFUT"

    for i in range(5):
        t1 = base_time + timedelta(minutes=5 * i)
        c1 = Candle(
            symbol=sym1, timeframe="5m", timestamp=t1,
            open=24000 + i, high=24010 + i, low=23990 + i, close=24005 + i, volume=0
        )
        store.insert_candle(c1)

        t2 = base_time + timedelta(minutes=5 * i)
        c2 = Candle(
            symbol=sym2, timeframe="5m", timestamp=t2,
            open=24050 + i, high=24060 + i, low=24040 + i, close=24055 + i, volume=5000 + i
        )
        store.insert_candle(c2)

    replay = MarketReplayEngine(store)
    events = replay.load_candle_stream(
        symbols=[sym1, sym2],
        timeframe="5m",
        start_time=base_time,
        end_time=base_time + timedelta(minutes=30),
    )

    assert len(events) == 10

    received_events = []
    seen_times = []

    def on_candle(c: Candle):
        received_events.append(c)
        seen_times.append(replay.current_time)

    replay.subscribe_candle(on_candle)
    processed = replay.replay_events(events, speed_multiplier=0)

    assert processed == 10
    assert len(received_events) == 10

    # Verify strictly monotonic non-decreasing timestamps
    for i in range(1, len(seen_times)):
        assert seen_times[i] >= seen_times[i - 1]

    # Verify final simulated time
    assert replay.current_time == base_time + timedelta(minutes=20)


def test_market_replay_ticks_and_clock_isolation():
    store = DuckDBStore(":memory:")
    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)
    sym = "NSE:NIFTY26NOVFUT"

    ticks = [
        Tick(symbol=sym, local_timestamp=base_time + timedelta(seconds=i), ltp=24000.0 + i, incremental_volume=10)
        for i in range(10)
    ]
    store.insert_ticks_bulk(ticks)

    replay = MarketReplayEngine(store)
    events = replay.load_tick_stream([sym])
    assert len(events) == 10

    observed_prices = []
    def on_tick(t: Tick):
        # Ensure simulated time strictly matches current tick's local_timestamp
        assert replay.current_time == t.local_timestamp
        observed_prices.append(t.ltp)

    replay.subscribe_tick(on_tick)
    replay.replay_events(events, speed_multiplier=0)

    assert len(observed_prices) == 10
    assert observed_prices[0] == 24000.0
    assert observed_prices[-1] == 24009.0
