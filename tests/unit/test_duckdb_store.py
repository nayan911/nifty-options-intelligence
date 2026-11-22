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


def test_resample_ticks_strictly_excludes_snapshots(store):
    """Verify that off-market SNAPSHOT ticks are strictly excluded from OHLC candle resampling."""
    base_time = IST.localize(datetime(2026, 11, 26, 9, 15, 0))

    # 1. Insert 2 LIVE ticks with 100 volume each
    t1 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_time, ltp=25100.0, incremental_volume=100, event_class="LIVE")
    t2 = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_time + timedelta(seconds=10), ltp=25120.0, incremental_volume=100, event_class="LIVE")

    # 2. Insert 1 SNAPSHOT tick with anomalous price 25200 and volume 500
    t_snap = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_time + timedelta(seconds=20), ltp=25200.0, incremental_volume=500, event_class="SNAPSHOT")

    store.insert_ticks_bulk([t1, t2, t_snap])

    df = store.resample_ticks_to_ohlc(symbol="NSE:NIFTY26NOVFUT", interval_minutes=1, event_class="LIVE")

    assert len(df) == 1
    # High must be 25120 (from LIVE ticks), strictly ignoring SNAPSHOT tick 25200!
    assert df.iloc[0]["high"] == 25120.0
    # Volume must be 200, strictly ignoring SNAPSHOT volume 500!
    assert df.iloc[0]["volume"] == 200


def test_get_ticks_defaults_to_live_and_excludes_snapshots(store):
    """Verify get_ticks strictly isolates LIVE ticks from SNAPSHOT ticks by default."""
    base_time = IST.localize(datetime(2026, 11, 26, 9, 15, 0))
    t_live = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_time, ltp=25100.0, incremental_volume=100, event_class="LIVE")
    t_snap = Tick(symbol="NSE:NIFTY26NOVFUT", local_timestamp=base_time + timedelta(seconds=10), ltp=25200.0, incremental_volume=0, event_class="SNAPSHOT")
    store.insert_ticks_bulk([t_live, t_snap])

    # Default query (event_class='LIVE')
    live_ticks = store.get_ticks(symbol="NSE:NIFTY26NOVFUT")
    assert len(live_ticks) == 1
    assert live_ticks[0].ltp == 25100.0

    # Query with event_class=None (returns all)
    all_ticks = store.get_ticks(symbol="NSE:NIFTY26NOVFUT", event_class=None)
    assert len(all_ticks) == 2


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


def test_duckdb_schema_migration_from_legacy_database(tmp_path):
    """Verify that an older DuckDB file with legacy table schemas is safely migrated:
    1. Creates automatic backup file.
    2. Migrates missing columns without data loss or column ordering corruption.
    3. Successfully writes and reads new option-chain columns with explicit column mapping.
    """
    import duckdb

    legacy_db_file = tmp_path / "legacy_test.duckdb"

    # Step 1: Create a genuine legacy database with original schema (prior to commit 0436ec2)
    conn = duckdb.connect(str(legacy_db_file))
    conn.execute(
        """
        CREATE TABLE ticks (
            symbol VARCHAR NOT NULL,
            exchange_timestamp TIMESTAMP WITH TIME ZONE,
            local_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            ltp DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            bid DOUBLE,
            ask DOUBLE,
            bid_qty BIGINT,
            ask_qty BIGINT,
            oi BIGINT,
            source VARCHAR DEFAULT 'FYERS'
        );

        CREATE TABLE candles (
            symbol VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            oi BIGINT,
            is_closed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (symbol, timeframe, timestamp)
        );

        CREATE TABLE option_chain_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            underlying VARCHAR NOT NULL,
            spot_price DOUBLE NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            expiry DATE NOT NULL,
            total_call_oi BIGINT,
            total_put_oi BIGINT,
            pcr DOUBLE
        );

        CREATE TABLE option_chain_strikes (
            snapshot_id VARCHAR NOT NULL,
            strike DOUBLE NOT NULL,
            call_ltp DOUBLE,
            call_oi BIGINT,
            call_iv DOUBLE,
            call_delta DOUBLE,
            call_theta DOUBLE,
            put_ltp DOUBLE,
            put_oi BIGINT,
            put_iv DOUBLE,
            put_delta DOUBLE,
            put_theta DOUBLE,
            FOREIGN KEY (snapshot_id) REFERENCES option_chain_snapshots(snapshot_id)
        );
        """
    )
    # Insert legacy row
    conn.execute(
        "INSERT INTO ticks VALUES ('NSE:NIFTY50-INDEX', NULL, '2026-11-20 15:30:00+05:30', 24100.0, 0, NULL, NULL, NULL, NULL, NULL, 'FYERS')"
    )
    conn.close()

    # Step 2: Open with DuckDBStore - triggers backup and migration
    store = DuckDBStore(db_path=legacy_db_file)

    # Verify automated backup was created!
    bak_file = legacy_db_file.with_suffix(".duckdb.bak")
    assert bak_file.exists()
    assert bak_file.stat().st_size > 0

    # Step 3: Insert new option chain snapshot with all new columns
    now = IST.localize(datetime(2026, 11, 26, 11, 0, 0))
    snapshot = OptionChainSnapshot(
        underlying="NIFTY",
        spot_price=25000.0,
        api_timestamp=now,
        expiry=date(2026, 11, 26),
        total_call_oi=100000,
        total_put_oi=120000,
        pcr=1.2,
        strikes=[
            OptionChainStrike(
                strike=25000.0,
                call={
                    "ltp": 150.0,
                    "oi": 40000,
                    "oi_change": 3000,
                    "volume": 60000,
                    "bid": 149.5,
                    "ask": 150.5,
                    "iv": 14.0,
                    "delta": 0.55,
                    "gamma": 0.002,
                    "theta": -11.0,
                    "vega": 15.0,
                },
                put={
                    "ltp": 80.0,
                    "oi": 50000,
                    "oi_change": -1500,
                    "volume": 70000,
                    "bid": 79.5,
                    "ask": 80.5,
                    "iv": 13.8,
                    "delta": -0.45,
                    "gamma": 0.002,
                    "theta": -9.5,
                    "vega": 14.5,
                },
            )
        ],
    )
    store.save_option_chain_snapshot(snapshot)

    # Step 4: Verify columns are populated into correct semantic positions
    row = store._conn.execute(
        """
        SELECT strike, call_volume, call_oi_change, call_bid, call_ask, call_gamma, call_vega,
               put_volume, put_oi_change, put_bid, put_ask, put_gamma, put_vega
        FROM option_chain_strikes WHERE strike = 25000.0
        """
    ).fetchone()

    assert row is not None
    assert row[0] == 25000.0
    assert row[1] == 60000  # call_volume
    assert row[2] == 3000   # call_oi_change
    assert row[3] == 149.5  # call_bid
    assert row[4] == 150.5  # call_ask
    assert row[5] == 0.002  # call_gamma
    assert row[6] == 15.0   # call_vega
    assert row[7] == 70000  # put_volume
    assert row[8] == -1500  # put_oi_change
    assert row[9] == 79.5   # put_bid
    assert row[10] == 80.5  # put_ask

    store.close()


def test_get_session_opening_candle_timezone_invariant(store):
    """Verify that get_session_opening_candle works identically under UTC and Asia/Kolkata connection timezones."""
    sym = "NSE:NIFTY50-INDEX"
    t_0915 = IST.localize(datetime(2026, 11, 23, 9, 15))
    c = Candle(
        symbol=sym,
        timeframe="5m",
        timestamp=t_0915,
        open=25115.0,
        high=25130.0,
        low=25110.0,
        close=25125.0,
        volume=12000,
        oi=500000,
        is_closed=True,
    )
    store.insert_candle(c)

    # 1. Under UTC connection timezone (common on GitHub Actions and cloud hosts)
    store._conn.execute("SET TimeZone='UTC'")
    c_utc = store.get_session_opening_candle(sym, session_date=date(2026, 11, 23))
    assert c_utc is not None
    assert c_utc.open == 25115.0
    assert c_utc.high == 25130.0

    # 2. Under Asia/Kolkata connection timezone
    store._conn.execute("SET TimeZone='Asia/Kolkata'")
    c_ist = store.get_session_opening_candle(sym, session_date=date(2026, 11, 23))
    assert c_ist is not None
    assert c_ist.open == 25115.0
    assert c_ist.high == 25130.0
