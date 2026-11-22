"""DuckDB high-performance embedded storage engine."""

import shutil
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        """Create tables and indexes if they do not exist, and migrate missing columns with automated backup."""
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            if db_file.exists() and db_file.stat().st_size > 0:
                bak_file = db_file.with_suffix(".duckdb.bak")
                try:
                    shutil.copy2(db_file, bak_file)
                except Exception:
                    pass

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(CREATE_TABLES_SQL)

            # Idempotently migrate event_class for ticks
            try:
                cur.execute("ALTER TABLE ticks ADD COLUMN IF NOT EXISTS event_class VARCHAR DEFAULT 'LIVE'")
            except Exception:
                pass

            # Idempotently migrate observed_at for candles
            try:
                cur.execute("ALTER TABLE candles ADD COLUMN IF NOT EXISTS observed_at TIMESTAMP WITH TIME ZONE")
            except Exception:
                pass

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

            # Create index on event_class now that column is guaranteed to exist
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ticks_provenance ON ticks(symbol, event_class, local_timestamp)")
            except Exception:
                pass

            # Idempotently migrate columns for paper_positions if table exists from prior runs
            for col, dtype in [
                ("initial_stop_loss_premium", "DOUBLE"),
                ("last_quote_time", "TIMESTAMP WITH TIME ZONE"),
                ("last_candle_time", "TIMESTAMP WITH TIME ZONE"),
                ("last_bid_time", "TIMESTAMP WITH TIME ZONE"),
                ("last_ask_time", "TIMESTAMP WITH TIME ZONE"),
                ("current_ltp", "DOUBLE"),
                ("current_bid", "DOUBLE"),
                ("current_ask", "DOUBLE"),
                ("is_exit_triggered", "BOOLEAN DEFAULT FALSE"),
                ("pending_exit_reason", "VARCHAR"),
                ("is_t1_hit", "BOOLEAN DEFAULT FALSE"),
                ("is_breakeven_sl_active", "BOOLEAN DEFAULT FALSE"),
                ("is_estimated_exit", "BOOLEAN DEFAULT FALSE"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS {col} {dtype}")
                except Exception:
                    pass

            # Idempotently migrate columns for paper_orders if table exists from prior runs
            for col, dtype in [
                ("is_estimated_fill", "BOOLEAN DEFAULT FALSE"),
                ("reason", "VARCHAR"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS {col} {dtype}")
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
                "event_class": getattr(t, "event_class", "LIVE"),
            }
            for t in ticks
        ]
        df = pd.DataFrame(data)
        cur = self._conn.cursor()
        cur.register("staging_ticks_df", df)
        try:
            cur.execute(
                """
                INSERT INTO ticks (
                    symbol, exchange_timestamp, local_timestamp, ltp, volume,
                    bid, ask, bid_qty, ask_qty, oi, source, event_class
                ) SELECT
                    symbol, exchange_timestamp, local_timestamp, ltp, volume,
                    bid, ask, bid_qty, ask_qty, oi, source, event_class
                FROM staging_ticks_df
                """
            )
        finally:
            cur.unregister("staging_ticks_df")

    def insert_candle(self, candle: Candle) -> None:
        """Upsert a candle (replaces if existing bar with same key is updated)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO candles (
                    symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    candle.observed_at,
                ]
            )

    def insert_candles_batch(self, candles: List[Candle]) -> int:
        """Efficiently batch-inserts candles using DuckDB parameter binding."""
        if not candles:
            return 0
        rows = [
            (
                c.symbol,
                c.timeframe,
                c.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.oi,
                c.is_closed,
                c.observed_at,
            )
            for c in candles
        ]
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO candles (
                    symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_latest_candle_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Returns the latest candle timestamp for a symbol and timeframe, or None if empty."""
        query = """
            SELECT MAX(timestamp)
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND is_closed = TRUE
        """
        with self._lock:
            cur = self._conn.cursor()
            res = cur.execute(query, [symbol, timeframe]).fetchone()
            if res and res[0] is not None:
                ts = res[0]
                if isinstance(ts, datetime):
                    return ts
        return None

    def get_latest_completed_daily_candle(self, symbol: str, before_date: date) -> Optional[Candle]:
        """Returns the latest completed Daily candle strictly before the given date."""
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
            FROM candles
            WHERE symbol = ? AND timeframe IN ('D', '1D', 'd') AND CAST(timestamp AS DATE) < ? AND is_closed = TRUE
            ORDER BY timestamp DESC
            LIMIT 1
        """
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(query, [symbol, before_date]).fetchone()
            if row:
                return Candle(
                    symbol=row[0],
                    timeframe=row[1],
                    timestamp=row[2],
                    open=row[3],
                    high=row[4],
                    low=row[5],
                    close=row[6],
                    volume=row[7],
                    oi=row[8],
                    is_closed=row[9],
                    observed_at=row[10],
                )
        return None

    def get_session_opening_candle(self, symbol: str, session_date: date) -> Optional[Candle]:
        """Returns the confirmed 09:15:00 IST opening candle for the specified session date.

        Prefers highest resolution first (1m -> 5m -> 15m).
        """
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
            FROM candles
            WHERE symbol = ?
              AND CAST(timestamp AS DATE) = ?
              AND EXTRACT(HOUR FROM timestamp) = 9 AND EXTRACT(MINUTE FROM timestamp) = 15
              AND is_closed = TRUE
            ORDER BY CASE timeframe WHEN '1m' THEN 1 WHEN '5m' THEN 2 WHEN '15m' THEN 3 ELSE 4 END ASC
            LIMIT 1
        """
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(query, [symbol, session_date]).fetchone()
            if row:
                return Candle(
                    symbol=row[0],
                    timeframe=row[1],
                    timestamp=row[2],
                    open=row[3],
                    high=row[4],
                    low=row[5],
                    close=row[6],
                    volume=row[7],
                    oi=row[8],
                    is_closed=row[9],
                    observed_at=row[10],
                )
        return None

    def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        as_of: Optional[datetime] = None,
    ) -> List[Candle]:
        """Queries the N most recent candles up to as_of, returned in ascending chronological order."""
        if as_of is None:
            query = """
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
                FROM candles
                WHERE symbol = ? AND timeframe = ? AND is_closed = TRUE
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [symbol, timeframe, limit]
        else:
            query = """
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
                FROM candles
                WHERE symbol = ? AND timeframe = ? AND timestamp <= ? AND is_closed = TRUE
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [symbol, timeframe, as_of, limit]
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(query, params).fetchall()

        candles = [
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
                observed_at=r[10],
            )
            for r in rows
        ]
        candles.reverse()
        return candles

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Query OHLCV candles within a time range, ordered by timestamp ascending."""
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, oi, is_closed, observed_at
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
                observed_at=r[10],
            )
            for r in rows
        ]

    def get_ticks(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_class: Optional[str] = "LIVE",
    ) -> List[Tick]:
        """Query raw ticks in ascending order of local_timestamp.

        Defaults to event_class='LIVE' to strictly isolate live market observations from
        holiday/weekend snapshots during backtesting and strategy replay.
        Pass event_class=None to retrieve all ticks including snapshots for diagnostics.
        """
        conditions = ["symbol = ?"]
        params: List[Any] = [symbol]
        if start_time:
            conditions.append("local_timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("local_timestamp <= ?")
            params.append(end_time)
        if event_class is not None:
            conditions.append("event_class = ?")
            params.append(event_class)

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT symbol, exchange_timestamp, local_timestamp, ltp, volume,
                   bid, ask, bid_qty, ask_qty, oi, source, event_class
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
                event_class=r[11] if len(r) > 11 else "LIVE",
            )
            for r in rows
        ]

    def resample_ticks_to_ohlc(
        self,
        symbol: str,
        interval_minutes: int = 1,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_class: str = "LIVE",
    ) -> pd.DataFrame:
        """Vectorized SQL aggregation of ticks into OHLC bars using DuckDB time_bucket.

        Strictly filters on event_class (defaults to 'LIVE') so off-market holiday
        snapshots never contaminate candlestick reconstruction.
        """
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
            WHERE symbol = ? AND event_class = ?
            {"AND local_timestamp >= ?" if start_time else ""}
            {"AND local_timestamp <= ?" if end_time else ""}
            GROUP BY symbol, timestamp
            ORDER BY timestamp ASC
        """
        params = [symbol, event_class]
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
                        cur.execute(
                            """
                            INSERT INTO option_chain_strikes (
                                snapshot_id, strike,
                                call_ltp, call_oi, call_oi_change, call_volume, call_bid, call_ask,
                                call_iv, call_delta, call_gamma, call_theta, call_vega,
                                put_ltp, put_oi, put_oi_change, put_volume, put_bid, put_ask,
                                put_iv, put_delta, put_gamma, put_theta, put_vega
                            ) SELECT
                                snapshot_id, strike,
                                call_ltp, call_oi, call_oi_change, call_volume, call_bid, call_ask,
                                call_iv, call_delta, call_gamma, call_theta, call_vega,
                                put_ltp, put_oi, put_oi_change, put_volume, put_bid, put_ask,
                                put_iv, put_delta, put_gamma, put_theta, put_vega
                            FROM staging_strikes_df
                            """
                        )
                    finally:
                        cur.unregister("staging_strikes_df")

                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def save_paper_order(self, order: Dict[str, Any]) -> None:
        """Inserts or updates a paper trading order."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO paper_orders (
                    order_id, position_id, signal_id, symbol, side, order_type,
                    status, quantity, requested_price, filled_price, slippage,
                    is_estimated_fill, reason, created_at, filled_at, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    order.get("order_id"),
                    order.get("position_id"),
                    order.get("signal_id"),
                    order.get("symbol"),
                    order.get("side"),
                    order.get("order_type"),
                    order.get("status"),
                    order.get("quantity"),
                    order.get("requested_price"),
                    order.get("filled_price"),
                    order.get("slippage", 0.0),
                    order.get("is_estimated_fill", False),
                    order.get("reason"),
                    order.get("created_at"),
                    order.get("filled_at"),
                    order.get("rejection_reason"),
                ],
            )

    def save_paper_position(self, pos: Dict[str, Any]) -> None:
        """Inserts or updates a paper trading position record in the trade journal."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO paper_positions (
                    position_id, signal_id, symbol, underlying, option_type,
                    strike, expiry, lot_size, initial_quantity, remaining_quantity,
                    entry_price, exit_price, target_premium_t1, target_premium_t2,
                    stop_loss_premium, initial_stop_loss_premium, spot_entry,
                    spot_target_t1, spot_target_t2, spot_sl, entry_time,
                    exit_time, last_quote_time, last_candle_time, last_bid_time, last_ask_time,
                    current_ltp, current_bid,
                    current_ask, is_exit_triggered, pending_exit_reason,
                    is_t1_hit, is_breakeven_sl_active,
                    status, exit_reason, gross_pnl, net_pnl, total_costs,
                    realized_rr, is_estimated_exit, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    pos.get("position_id"),
                    pos.get("signal_id"),
                    pos.get("symbol"),
                    pos.get("underlying"),
                    pos.get("option_type"),
                    pos.get("strike"),
                    pos.get("expiry"),
                    pos.get("lot_size"),
                    pos.get("initial_quantity"),
                    pos.get("remaining_quantity"),
                    pos.get("entry_price"),
                    pos.get("exit_price"),
                    pos.get("target_premium_t1"),
                    pos.get("target_premium_t2"),
                    pos.get("stop_loss_premium"),
                    pos.get("initial_stop_loss_premium", pos.get("stop_loss_premium")),
                    pos.get("spot_entry"),
                    pos.get("spot_target_t1"),
                    pos.get("spot_target_t2"),
                    pos.get("spot_sl"),
                    pos.get("entry_time"),
                    pos.get("exit_time"),
                    pos.get("last_quote_time"),
                    pos.get("last_candle_time"),
                    pos.get("last_bid_time"),
                    pos.get("last_ask_time"),
                    pos.get("current_ltp"),
                    pos.get("current_bid"),
                    pos.get("current_ask"),
                    pos.get("is_exit_triggered", False),
                    pos.get("pending_exit_reason"),
                    pos.get("is_t1_hit", False),
                    pos.get("is_breakeven_sl_active", False),
                    pos.get("status"),
                    pos.get("exit_reason"),
                    pos.get("gross_pnl", 0.0),
                    pos.get("net_pnl", 0.0),
                    pos.get("total_costs", 0.0),
                    pos.get("realized_rr"),
                    pos.get("is_estimated_exit", False),
                    pos.get("notes"),
                ],
            )

    def get_paper_positions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries paper positions from DuckDB, optionally filtered by status."""
        with self._lock:
            cur = self._conn.cursor()
            if status:
                res = cur.execute(
                    "SELECT * FROM paper_positions WHERE status = ? ORDER BY entry_time DESC",
                    [status],
                ).fetchall()
            else:
                res = cur.execute(
                    "SELECT * FROM paper_positions ORDER BY entry_time DESC"
                ).fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in res]

    def get_paper_orders(self, position_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries paper orders, optionally filtered by position_id."""
        with self._lock:
            cur = self._conn.cursor()
            if position_id:
                res = cur.execute(
                    "SELECT * FROM paper_orders WHERE position_id = ? ORDER BY created_at ASC",
                    [position_id],
                ).fetchall()
            else:
                res = cur.execute(
                    "SELECT * FROM paper_orders ORDER BY created_at ASC"
                ).fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in res]

    def close(self) -> None:
        """Flush any pending buffered ticks and close connection."""
        self.flush_ticks()
        self._conn.close()
