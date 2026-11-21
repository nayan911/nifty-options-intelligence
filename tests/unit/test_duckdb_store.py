"""Unit tests for DuckDBStore high-performance storage engine."""

from datetime import date, datetime, timedelta

import pytest
import pytz

from nifty_engine.core.interfaces import Candle, OptionChainSnapshot, OptionChainStrike, Tick
from nifty_engine.data.storage.duckdb_store import DuckDBStore

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def store():
    # Use fast in-memory database for testing
    db = DuckDBStore(db_path=":memory:")
    yield db
    db.close()


def test_insert_and_query_candle(store):
    now = IST.localize(datetime(2026, 11, 26, 10, 0, 0))
    candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=now,
        open=25000.0,
        high=25030.0,
        low=24990.0,
        close=25020.0,
        volume=15000,
        oi=2000000,
        is_closed=True,
    )

    store.insert_candle(candle)

    # Query back
    result = store.get_candles(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        start_time=now - timedelta(minutes=5),
        end_time=now + timedelta(minutes=5)
    )

    assert len(result) == 1
    assert result[0].symbol == "NSE:NIFTY50-INDEX"
    assert result[0].high == 25030.0
    assert result[0].close == 25020.0
    assert result[0].is_closed is True


def test_bulk_tick_insert_and_resample(store):
    base_time = IST.localize(datetime(2026, 11, 26, 9, 15, 0))
    ticks = []

    # Generate 100 ticks within a 1-minute window
    prices = [25000.0, 25010.0, 24995.0, 25025.0, 25005.0]
    for i in range(100):
        t = Tick(
            symbol="NSE:NIFTY50-INDEX",
            local_timestamp=base_time + timedelta(seconds=i * 0.5),
            ltp=prices[i % len(prices)],
            incremental_volume=50,
            oi=100000,
            source="FYERS"
        )
        ticks.append(t)

    # Bulk insert
    store.insert_ticks_bulk(ticks)

    # Resample using DuckDB time_bucket
    df_ohlc = store.resample_ticks_to_ohlc(symbol="NSE:NIFTY50-INDEX", interval_minutes=1)

    assert len(df_ohlc) == 1
    assert df_ohlc.iloc[0]["open"] == 25000.0
    assert df_ohlc.iloc[0]["high"] == 25025.0
    assert df_ohlc.iloc[0]["low"] == 24995.0
    assert df_ohlc.iloc[0]["volume"] == 100 * 50


def test_save_option_chain_snapshot(store):
    now = IST.localize(datetime(2026, 11, 26, 11, 0, 0))
    snapshot = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25015.5,
        api_timestamp=now,
        expiry=date(2026, 11, 26),
        total_call_oi=12500000,
        total_put_oi=14200000,
        pcr=1.136,
        strikes=[
            OptionChainStrike(
                strike=25000.0,
                call={"ltp": 120.5, "oi": 850000, "iv": 14.2, "delta": 0.52, "theta": -12.4},
                put={"ltp": 105.0, "oi": 950000, "iv": 14.5, "delta": -0.48, "theta": -11.8},
            ),
            OptionChainStrike(
                strike=25050.0,
                call={"ltp": 92.0, "oi": 650000, "iv": 14.0, "delta": 0.44, "theta": -11.0},
                put={"ltp": 136.5, "oi": 720000, "iv": 14.8, "delta": -0.56, "theta": -12.6},
            ),
        ]
    )

    store.save_option_chain_snapshot(snapshot)

    # Verify DuckDB stores records correctly
    count_snap = store._conn.execute("SELECT COUNT(*) FROM option_chain_snapshots").fetchone()[0]
    count_strikes = store._conn.execute("SELECT COUNT(*) FROM option_chain_strikes").fetchone()[0]

    assert count_snap == 1
    assert count_strikes == 2
