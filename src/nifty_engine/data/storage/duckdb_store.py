"""DuckDB high-performance embedded storage engine."""

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import duckdb
import pandas as pd

from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import (
    Candle,
    IStorageEngine,
    OptionChainSnapshot,
    Tick,
)
from nifty_engine.data.storage.schemas import CREATE_TABLES_SQL


class DuckDBStore(IStorageEngine):
    """Local embedded columnar storage engine for ticks, candles, and option chains."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        batch_size: int = 50,
    ) -> None:
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            path = Path(db_path) if db_path else settings.DUCKDB_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)

        self.batch_size = batch_size
        self._tick_buffer: List[Tick] = []
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.db_path)
        self._init_schemas()

    def _get_cursor(self) -> duckdb.DuckDBPyConnection:
        """Internal helper to obtain a cursor (allows mocking/fault injection in tests)."""
        return self._conn.cursor()

    def _init_schemas(self) -> None:
        """Create tables and indexes if they do not exist, and migrate missing columns."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(CREATE_TABLES_SQL)
            # Idempotently migrate columns for option_chain_strikes if already existing from prior runs
            for col, dtype in [
                ("call_oi_change", "BIGINT"),
                ("call_volume", "BIGINT"),
                ("call_bid", "DOUBLE"),
                ("call_ask", "DOUBLE"),
                ("call_gamma", "DOUBLE"),
                ("call_vega", "DOUBLE"),
                ("put_oi_change", "BIGINT"),
                ("put_volume", "BIGINT"),
                ("put_bid", "DOUBLE"),
                ("put_ask", "DOUBLE"),
                ("put_gamma", "DOUBLE"),
                ("put_vega", "DOUBLE"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE option_chain_strikes ADD COLUMN IF NOT EXISTS {col} {dtype}")
                except Exception:
                    pass

    def insert_tick(self, tick: Tick) -> None:
        """Buffer a single tick for high-throughput single-writer batch insert."""
        with self._lock:
            self._tick_buffer.append(tick)
            if len(self._tick_buffer) >= self.batch_size:
                self._flush_ticks_unlocked()

    def flush_ticks(self) -> int:
        """Flush buffered ticks into DuckDB thread-safely."""
        with self._lock:
            return self._flush_ticks_unlocked()

    def _flush_ticks_unlocked(self) -> int:
        """Internal flush assuming self._lock is already held.

        Clears buffer only upon successful persistence, preventing data loss on transient write errors.
        """
        if not self._tick_buffer:
            return 0
        batch = list(self._tick_buffer)
        try:
            self._insert_ticks_bulk_unlocked(batch)
            del self._tick_buffer[: len(batch)]
            return len(batch)
        except Exception:
            # Buffer is preserved on failure
            raise

    def insert_ticks_bulk(self, ticks: List[Tick]) -> None:
        """High-performance bulk insert of ticks."""
        with self._lock:
            self._insert_ticks_bulk_unlocked(ticks)

    def _insert_ticks_bulk_unlocked(self, ticks: List[Tick]) -> None:
        if not ticks:
            return

        data = [
            {
                "symbol": t.symbol,
                "exchange_timestamp": t.exchange_timestamp,
                "local_timestamp": t.local_timestamp,
                "ltp": t.ltp,
                "volume": t.incremental_volume,
                "bid": t.bid,
                "ask": t.ask,
                "bid_qty": t.bid_qty,
                "ask_qty": t.ask_qty,
                "oi": t.oi,
                "source": t.source,
            }
            for t in ticks
        ]
        df = pd.DataFrame(data)
        cur = self._conn.cursor()
        cur.register("staging_ticks_df", df)
        try:
            cur.execute("INSERT INTO ticks SELECT * FROM staging_ticks_df")
        finally:
            cur.unregister("staging_ticks_df")

    def insert_candle(self, candle: Candle) -> None:
        """Upsert a candle (replaces if existing bar with same key is updated)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO candles (
                    symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    candle.symbol,
                    candle.timeframe,
                    candle.timestamp,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.oi,
                    candle.is_closed,
                ]
            )

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Query OHLCV candles within a time range, ordered by timestamp ascending."""
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(query, [symbol, timeframe, start_time, end_time]).fetchall()
        return [
            Candle(
                symbol=r[0],
                timeframe=r[1],
                timestamp=r[2],
                open=r[3],
                high=r[4],
                low=r[5],
                close=r[6],
                volume=r[7],
                oi=r[8],
                is_closed=r[9],
            )
            for r in rows
        ]

    def get_ticks(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tick]:
        """Query raw ticks in ascending order of local_timestamp."""
        conditions = ["symbol = ?"]
        params: List[Any] = [symbol]
        if start_time:
            conditions.append("local_timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("local_timestamp <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT symbol, exchange_timestamp, local_timestamp, ltp, volume,
                   bid, ask, bid_qty, ask_qty, oi, source
            FROM ticks
            WHERE {where_clause}
            ORDER BY local_timestamp ASC
        """
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(query, params).fetchall()
        return [
            Tick(
                symbol=r[0],
                exchange_timestamp=r[1],
                local_timestamp=r[2],
                ltp=r[3],
                incremental_volume=r[4],
                bid=r[5],
                ask=r[6],
                bid_qty=r[7],
                ask_qty=r[8],
                oi=r[9],
                source=r[10],
            )
            for r in rows
        ]

    def resample_ticks_to_ohlc(
        self,
        symbol: str,
        interval_minutes: int = 1,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Vectorized SQL aggregation of ticks into OHLC bars using DuckDB time_bucket."""
        query = f"""
            SELECT
                symbol,
                time_bucket(INTERVAL '{interval_minutes} minutes', local_timestamp) AS timestamp,
                FIRST(ltp) AS open,
                MAX(ltp) AS high,
                MIN(ltp) AS low,
                LAST(ltp) AS close,
                SUM(volume) AS volume,
                LAST(oi) AS oi
            FROM ticks
            WHERE symbol = ?
            {"AND local_timestamp >= ?" if start_time else ""}
            {"AND local_timestamp <= ?" if end_time else ""}
            GROUP BY symbol, timestamp
            ORDER BY timestamp ASC
        """
        params = [symbol]
        if start_time:
            params.append(start_time)
        if end_time:
            params.append(end_time)

        with self._lock:
            cur = self._conn.cursor()
            return cur.execute(query, params).df()

    def save_option_chain_snapshot(self, snapshot: OptionChainSnapshot) -> None:
        """Store header and strike-level details of an option chain snapshot transactionally."""
        snap_id = str(uuid.uuid4())

        rows = []
        for s in snapshot.strikes:
            c = s.call or {}
            p = s.put or {}
            rows.append({
                "snapshot_id": snap_id,
                "strike": s.strike,
                "call_ltp": c.get("ltp"),
                "call_oi": c.get("oi"),
                "call_oi_change": c.get("oi_change"),
                "call_volume": c.get("volume"),
                "call_bid": c.get("bid"),
                "call_ask": c.get("ask"),
                "call_iv": c.get("iv"),
                "call_delta": c.get("delta"),
                "call_gamma": c.get("gamma"),
                "call_theta": c.get("theta"),
                "call_vega": c.get("vega"),
                "put_ltp": p.get("ltp"),
                "put_oi": p.get("oi"),
                "put_oi_change": p.get("oi_change"),
                "put_volume": p.get("volume"),
                "put_bid": p.get("bid"),
                "put_ask": p.get("ask"),
                "put_iv": p.get("iv"),
                "put_delta": p.get("delta"),
                "put_gamma": p.get("gamma"),
                "put_theta": p.get("theta"),
                "put_vega": p.get("vega"),
            })

        with self._lock:
            cur = self._get_cursor()
            cur.execute("BEGIN TRANSACTION")
            try:
                cur.execute(
                    """
                    INSERT INTO option_chain_snapshots (
                        snapshot_id, underlying, spot_price, timestamp, expiry,
                        total_call_oi, total_put_oi, pcr
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        snap_id,
                        snapshot.underlying,
                        snapshot.spot_price,
                        snapshot.api_timestamp,
                        snapshot.expiry,
                        snapshot.total_call_oi,
                        snapshot.total_put_oi,
                        snapshot.pcr,
                    ],
                )

                if rows:
                    df = pd.DataFrame(rows)
                    cur.register("staging_strikes_df", df)
                    try:
                        cur.execute("INSERT INTO option_chain_strikes SELECT * FROM staging_strikes_df")
                    finally:
                        cur.unregister("staging_strikes_df")

                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Flush any pending buffered ticks and close connection."""
        self.flush_ticks()
        self._conn.close()
