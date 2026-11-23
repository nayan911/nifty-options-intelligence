"""Historical bar store for research/backtesting (data/research.duckdb).

Design rules (see Phase 1 plan):
- Futures are stored per real contract ("OWN" series: actual symbol, unadjusted price, volume,
  OI) whenever FYERS still serves that contract. FYERS rejects expired contract symbols on the
  normal history API, so older history comes from FYERS' stitched continuous series ("CONT").
- The stitched series switches contracts at expiry: OI collapses ~90% and price/basis jump at
  those boundaries (verified on Aug/Sep/Oct 2026 data). CONT bars therefore carry a
  `segment_id`; any change computed across a segment boundary is invalid and must be masked.
- Index "volume" returned by FYERS history is stored but must never be used as a feature: the
  live feed has no index volume, and features must be identical live and in backtests.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import duckdb
import pandas as pd

from nifty_engine.core.interfaces import Candle

logger = logging.getLogger(__name__)

DEFAULT_RESEARCH_DB = Path("data/research.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS hist_bars (
    instrument VARCHAR NOT NULL,        -- logical series, e.g. NIFTY_SPOT, NIFTY_FUT, INDIAVIX
    symbol VARCHAR NOT NULL,            -- real FYERS symbol (contract for futures)
    source VARCHAR NOT NULL,            -- SPOT | OWN (real contract) | CONT (stitched near month) | CONT_NEXT (stitched next month)
    resolution VARCHAR NOT NULL,        -- '1' (minute) or 'D'
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    contract_expiry DATE,               -- OWN futures only
    segment_id INTEGER,                 -- CONT futures: contiguous single-contract segment
    PRIMARY KEY (symbol, source, resolution, ts)
);
"""


@dataclass(frozen=True)
class QualityIssue:
    instrument: str
    source: str
    resolution: str
    day: Optional[date]
    kind: str
    detail: str


class ResearchHistoryStore:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_RESEARCH_DB) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        # Same reason as the live store: session logic relies on IST wall-clock times.
        self._write("SET GLOBAL TimeZone = 'Asia/Kolkata'")
        self._write(SCHEMA)

    def _write(self, sql: str, params: Optional[list] = None) -> None:
        """Execute a statement and fully consume its result (DuckDB refuses further work on a
        connection that still holds an unfinished result)."""
        cur = self.conn.execute(sql, params) if params is not None else self.conn.execute(sql)
        try:
            cur.fetchall()
        except Exception:
            pass

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ writes
    def upsert_bars(
        self,
        instrument: str,
        symbol: str,
        source: str,
        resolution: str,
        candles: Sequence[Candle],
        contract_expiry: Optional[date] = None,
    ) -> int:
        rows = [
            (instrument, symbol, source, resolution, c.timestamp, float(c.open), float(c.high), float(c.low),
             float(c.close), int(c.volume or 0), c.oi, contract_expiry, None)
            for c in candles
            if c.is_closed
        ]
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=[
            "instrument", "symbol", "source", "resolution", "ts", "open", "high", "low", "close",
            "volume", "oi", "contract_expiry", "segment_id",
        ])
        self.conn.register("incoming", df)
        try:
            self._write("INSERT OR REPLACE INTO hist_bars SELECT * FROM incoming")
        finally:
            self.conn.unregister("incoming")
        return len(rows)

    def assign_cont_segments(
        self,
        instrument: str,
        resolution: str = "1",
        oi_ratio_threshold: float = 0.5,
        source: str = "CONT",
    ) -> List[Dict]:
        """OI-ratio-only segmentation (heuristic; kept for diagnostics).

        Production segmentation uses research.segments (2-of-3 vote: expiry calendar, OI ratio,
        basis jump): the OI rule alone missed a real NIFTY switch (Feb 2026) and flagged others.

        A switch is detected between consecutive trading days when OI at the day's first bar is
        below `oi_ratio_threshold` x (or above its inverse) the previous day's last-bar OI. Returns
        the detected boundaries for reporting.
        """
        df = self.conn.execute(
            """SELECT ts, oi FROM hist_bars WHERE instrument = ? AND source = ? AND resolution = ?
               ORDER BY ts""",
            [instrument, source, resolution],
        ).df()
        if df.empty:
            return []
        df["day"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata").dt.date
        firsts = df.groupby("day").first()
        lasts = df.groupby("day").last()
        days = list(firsts.index)
        seg = 0
        seg_by_day: Dict[date, int] = {}
        boundaries: List[Dict] = []
        for i, d in enumerate(days):
            if i > 0:
                prev_oi = lasts.loc[days[i - 1], "oi"]
                cur_oi = firsts.loc[d, "oi"]
                if prev_oi and cur_oi and pd.notna(prev_oi) and pd.notna(cur_oi):
                    ratio = float(cur_oi) / float(prev_oi)
                    if ratio < oi_ratio_threshold or ratio > 1.0 / oi_ratio_threshold:
                        seg += 1
                        boundaries.append({"day": d, "prev_day": days[i - 1], "oi_ratio": round(ratio, 3)})
            seg_by_day[d] = seg
        mapping = pd.DataFrame({"day": list(seg_by_day.keys()), "segment_id": list(seg_by_day.values())})
        self.conn.register("segmap", mapping)
        try:
            self._write(
                """UPDATE hist_bars SET segment_id = segmap.segment_id FROM segmap
                   WHERE hist_bars.instrument = ? AND hist_bars.source = ? AND hist_bars.resolution = ?
                     AND CAST(hist_bars.ts AS DATE) = segmap.day""",
                [instrument, source, resolution],
            )
        finally:
            self.conn.unregister("segmap")
        return boundaries

    # ------------------------------------------------------------------ reads
    def coverage(self) -> pd.DataFrame:
        return self.conn.execute(
            """SELECT instrument, source, resolution, symbol,
                      min(ts) AS first_bar, max(ts) AS last_bar,
                      count(*) AS bars, count(DISTINCT CAST(ts AS DATE)) AS days,
                      count(oi) AS bars_with_oi, max(segment_id) AS max_segment
               FROM hist_bars GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"""
        ).df()

    def has_day(self, symbol: str, source: str, resolution: str, day: date) -> bool:
        n = self.conn.execute(
            """SELECT count(*) FROM hist_bars WHERE symbol = ? AND source = ? AND resolution = ?
               AND CAST(ts AS DATE) = ?""",
            [symbol, source, resolution, day],
        ).fetchall()[0][0]
        return n > 0

    def minutes_by_day(self, symbol: str, source: str, resolution: str, lo: date, hi: date) -> Dict[date, set]:
        """Stored bar start times per day in [lo, hi] (for per-day completeness checks)."""
        df = self.conn.execute(
            """SELECT ts FROM hist_bars WHERE symbol = ? AND source = ? AND resolution = ?
               AND CAST(ts AS DATE) BETWEEN ? AND ?""",
            [symbol, source, resolution, lo, hi],
        ).df()
        out: Dict[date, set] = {}
        if df.empty:
            return out
        ts = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata")
        for t in ts:
            py = t.to_pydatetime()
            out.setdefault(py.date(), set()).add(py)
        return out

    def first_day(self, symbol: str, source: str, resolution: str) -> Optional[date]:
        v = self.conn.execute(
            "SELECT min(CAST(ts AS DATE)) FROM hist_bars WHERE symbol = ? AND source = ? AND resolution = ?",
            [symbol, source, resolution],
        ).fetchall()[0][0]
        return v

    def bars(self, instrument: str, source: str, resolution: str = "1",
             start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        q = "SELECT * FROM hist_bars WHERE instrument = ? AND source = ? AND resolution = ?"
        args: list = [instrument, source, resolution]
        if start is not None:
            q += " AND ts >= ?"
            args.append(start)
        if end is not None:
            q += " AND ts <= ?"
            args.append(end)
        return self.conn.execute(q + " ORDER BY ts", args).df()

    # ------------------------------------------------------------------ quality
    def quality_report(self, trading_days: Optional[Sequence[date]] = None,
                       min_intraday_bars: int = 360) -> List[QualityIssue]:
        """Checks: OHLC consistency, non-positive prices, thin/missing intraday days."""
        issues: List[QualityIssue] = []
        bad = self.conn.execute(
            """SELECT instrument, source, resolution, CAST(ts AS DATE) AS d, count(*) n FROM hist_bars
               WHERE low > least(open, close) OR high < greatest(open, close) OR low <= 0 OR high < low
               GROUP BY 1, 2, 3, 4"""
        ).fetchall()
        for inst, src, res, d, n in bad:
            issues.append(QualityIssue(inst, src, res, d, "BAD_OHLC", f"{n} bars with inconsistent OHLC"))

        per_day = self.conn.execute(
            """SELECT instrument, source, symbol, CAST(ts AS DATE) AS d, count(*) n FROM hist_bars
               WHERE resolution = '1' GROUP BY 1, 2, 3, 4"""
        ).df()
        per_day["d"] = pd.to_datetime(per_day["d"]).dt.date
        for (inst, src, sym), g in per_day.groupby(["instrument", "source", "symbol"]):
            for _, r in g[g["n"] < min_intraday_bars].iterrows():
                issues.append(QualityIssue(inst, src, "1", r["d"], "THIN_DAY", f"{sym}: only {r['n']} 1m bars"))
            if trading_days:
                have = set(g["d"])
                lo, hi = min(have), max(have)
                for d in trading_days:
                    if lo <= d <= hi and d not in have:
                        issues.append(QualityIssue(inst, src, "1", d, "MISSING_DAY", f"{sym}: no bars"))
        return issues
