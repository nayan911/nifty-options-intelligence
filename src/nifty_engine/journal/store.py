"""Journal tables in the live DuckDB file (the runner's single writer connection).

journal_events     one row per live SETUP / CANDIDATE (+ entry decision) / PAPER_FILL / PAPER_EXIT,
                   with the strategy identity and live-only context (readiness, feed age, option quote)
journal_tracking   option contracts pinned for premium tracking after a candidate (subscription kept
                   until track_until; per-minute bid/ask is rebuilt from the stored ticks at end of day)

Rows are insert-only with deterministic ids, so a restart that re-emits an event cannot duplicate it.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS journal_events (
        event_id VARCHAR PRIMARY KEY,
        day DATE NOT NULL,
        event_at TIMESTAMP WITH TIME ZONE NOT NULL,
        recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
        kind VARCHAR NOT NULL,
        symbol VARCHAR,
        timeframe VARCHAR,
        direction VARCHAR,
        level_id VARCHAR,
        candidate_id VARCHAR,
        decision_code VARCHAR,
        decision_detail VARCHAR,
        strategy_id VARCHAR,
        strategy_version INTEGER,
        strategy_fingerprint VARCHAR,
        readiness VARCHAR,
        worst_feed_symbol VARCHAR,
        worst_feed_age_s DOUBLE,
        payload VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS journal_tracking (
        candidate_id VARCHAR PRIMARY KEY,
        day DATE NOT NULL,
        contract VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        decided_at TIMESTAMP WITH TIME ZONE NOT NULL,
        track_until TIMESTAMP WITH TIME ZONE NOT NULL,
        source VARCHAR NOT NULL,
        quote_at_signal VARCHAR
    )""",
]


class JournalStore:
    def __init__(self, store: Any) -> None:
        """store: the runner's DuckDBStore (its connection and lock are reused; DuckDB allows one writer)."""
        self.store = store
        with store._lock:
            cur = store._conn.cursor()
            for ddl in SCHEMA:
                cur.execute(ddl)

    def _exec(self, sql: str, params: list) -> None:
        with self.store._lock:
            cur = self.store._conn.cursor()
            cur.execute(sql, params)
            cur.fetchall()

    def add_event(self, row: Dict[str, Any]) -> None:
        cols = ["event_id", "day", "event_at", "recorded_at", "kind", "symbol", "timeframe", "direction", "level_id",
                "candidate_id", "decision_code", "decision_detail", "strategy_id", "strategy_version",
                "strategy_fingerprint", "readiness", "worst_feed_symbol", "worst_feed_age_s", "payload"]
        vals = [row.get(c) for c in cols]
        vals[-1] = json.dumps(row.get("payload") or {}, sort_keys=True, default=str)
        self._exec(f"INSERT OR IGNORE INTO journal_events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                   vals)

    def add_tracking(self, candidate_id: str, day, contract: str, direction: str, decided_at: datetime,
                     track_until: datetime, source: str, quote: Optional[Dict[str, Any]]) -> None:
        self._exec("""INSERT OR IGNORE INTO journal_tracking VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   [candidate_id, day, contract, direction, decided_at, track_until, source,
                    json.dumps(quote or {}, default=str)])

    def active_tracking(self, now: datetime) -> List[str]:
        with self.store._lock:
            rows = self.store._conn.cursor().execute(
                "SELECT DISTINCT contract FROM journal_tracking WHERE track_until > ?", [now]).fetchall()
        return [r[0] for r in rows]

    def events(self, day) -> List[Dict[str, Any]]:
        with self.store._lock:
            cur = self.store._conn.cursor()
            cur.execute("SELECT * FROM journal_events WHERE day = ? ORDER BY event_at, kind, event_id", [day])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def tracking(self, day) -> List[Dict[str, Any]]:
        with self.store._lock:
            cur = self.store._conn.cursor()
            cur.execute("SELECT * FROM journal_tracking WHERE day = ? ORDER BY decided_at", [day])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
