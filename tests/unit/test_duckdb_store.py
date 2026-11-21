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


def test_save_option_chain_all_fields_persisted(store):
    """Verify newly added columns (volume, bid, ask, gamma, vega, oi_change) are stored."""
    now = IST.localize(datetime(2026, 11, 26, 11, 0, 0))
    snapshot = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25015.5,
        api_timestamp=now,
        expiry=date(2026, 11, 26),
        total_call_oi=1000000,
        total_put_oi=1100000,
        pcr=1.1,
        strikes=[
            OptionChainStrike(
                strike=25000.0,
                call={
                    "ltp": 120.0,
                    "oi": 50000,
                    "oi_change": 2500,
                    "volume": 75000,
                    "bid": 119.8,
                    "ask": 120.2,
                    "iv": 14.2,
                    "delta": 0.51,
                    "gamma": 0.002,
                    "theta": -10.5,
                    "vega": 15.2,
                },
                put={
                    "ltp": 95.0,
                    "oi": 60000,
                    "oi_change": -1200,
                    "volume": 85000,
                    "bid": 94.8,
                    "ask": 95.2,
                    "iv": 14.0,
                    "delta": -0.49,
                    "gamma": 0.002,
                    "theta": -9.8,
                    "vega": 14.8,
                },
            )
        ],
    )

    store.save_option_chain_snapshot(snapshot)

    row = store._conn.execute(
        """
        SELECT strike, call_volume, call_bid, call_ask, call_oi_change, call_gamma, call_vega,
               put_volume, put_bid, put_ask, put_oi_change, put_gamma, put_vega
        FROM option_chain_strikes WHERE strike = 25000.0
        """
    ).fetchone()

    assert row is not None
    assert row[0] == 25000.0
    assert row[1] == 75000  # call_volume
    assert row[2] == 119.8  # call_bid
    assert row[3] == 120.2  # call_ask
    assert row[4] == 2500   # call_oi_change
    assert row[5] == 0.002  # call_gamma
    assert row[6] == 15.2   # call_vega
    assert row[7] == 85000  # put_volume
    assert row[8] == 94.8   # put_bid
    assert row[9] == 95.2   # put_ask
    assert row[10] == -1200 # put_oi_change


def test_buffered_ticks_retained_on_failed_write(store, monkeypatch):
    """Test fault-injection: if write fails, buffer must retain ticks without loss."""
    now = IST.localize(datetime(2026, 11, 26, 9, 15, 0))
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=now, ltp=25100.0, incremental_volume=10)
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=now + timedelta(seconds=1), ltp=25105.0, incremental_volume=20)

    store.insert_tick(t1)
    store.insert_tick(t2)
    assert len(store._tick_buffer) == 2

    # Simulate database disk/connection failure during bulk insert
    def faulty_insert(ticks):
        raise RuntimeError("Simulated transient DuckDB I/O fault")

    monkeypatch.setattr(store, "_insert_ticks_bulk_unlocked", faulty_insert)

    with pytest.raises(RuntimeError, match="Simulated transient DuckDB I/O fault"):
        store.flush_ticks()

    # CRITICAL: Buffer must still contain the 2 ticks!
    assert len(store._tick_buffer) == 2

    # Now restore normal function and retry flush
    monkeypatch.undo()
    flushed = store.flush_ticks()
    assert flushed == 2
    assert len(store._tick_buffer) == 0

    # Verify rows exist in DuckDB
    count = store._conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    assert count == 2


def test_option_chain_transactional_rollback_on_failure(store, monkeypatch):
    """Transactional test: failure during strike insertion must roll back the snapshot header."""
    now = IST.localize(datetime(2026, 11, 26, 11, 0, 0))
    snapshot = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25000.0,
        api_timestamp=now,
        expiry=date(2026, 11, 26),
        total_call_oi=100000,
        total_put_oi=100000,
        pcr=1.0,
        strikes=[
            OptionChainStrike(strike=25000.0, call={"ltp": 50.0}, put={"ltp": 50.0})
        ],
    )

    original_cursor = store._get_cursor

    class FaultyCursor:
        def __init__(self, real_cur):
            self.real_cur = real_cur

        def execute(self, sql, *args, **kwargs):
            if "INSERT INTO option_chain_strikes" in sql:
                raise RuntimeError("Simulated strike insertion write error")
            return self.real_cur.execute(sql, *args, **kwargs)

        def register(self, *args, **kwargs):
            return self.real_cur.register(*args, **kwargs)

        def unregister(self, *args, **kwargs):
            return self.real_cur.unregister(*args, **kwargs)

    monkeypatch.setattr(store, "_get_cursor", lambda: FaultyCursor(original_cursor()))

    with pytest.raises(RuntimeError, match="Simulated strike insertion write error"):
        store.save_option_chain_snapshot(snapshot)

    monkeypatch.undo()

    # Transaction rolled back: header must NOT be orphaned
    count_snaps = store._conn.execute("SELECT COUNT(*) FROM option_chain_snapshots").fetchone()[0]
    count_strikes = store._conn.execute("SELECT COUNT(*) FROM option_chain_strikes").fetchone()[0]
    assert count_snaps == 0
    assert count_strikes == 0
