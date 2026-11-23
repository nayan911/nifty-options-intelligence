"""Contract identity for stitched futures series: which real contract each segment holds.

Why this exists: FYERS stitches by the requested contract's position *at request time*, and that
meaning shifts around a rollover. On 23-Nov-2026 a routine refresh of the latest chunk (requested
via DEC after NOV had expired) silently replaced the NEXT-month series with near-month data from
25-Sep onward. Bars alone can't reveal that; contract identity checks can.

For a day d:   near contract = 1st monthly expiry >= d,   next contract = 2nd monthly expiry >= d.

`segment_contract_table` records, per stitched segment, the inferred contract expiry and a
confidence:
  VERIFIED   >= 99% of overlapping bars equal the stored real (OWN) contract (close and OI)
  CONFLICT   overlap exists but bars do not match -> data error
  INFERRED   no stored real contract overlaps (older history); identity from the calendar only
`identical_days` flags days where NEAR and NEXT are the same series (must never happen).
`repair_next_from_own` rebuilds NEXT days from stored real contracts where possible and deletes
NEXT days that duplicate NEAR and cannot be repaired.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.sessions import monthly_expiries

POSITION = {"CONT": 0, "CONT_NEXT": 1}


def contract_expiry_for(cal: ExchangeCalendar, day: date, source: str) -> Optional[date]:
    exps = monthly_expiries(cal, day, day + timedelta(days=100))
    idx = POSITION[source]
    return exps[idx] if len(exps) > idx else None


def _cont_symbol(underlying: str, source: str) -> str:
    return f"CONT:{underlying}:{'NEAR' if source == 'CONT' else 'NEXT'}"


def _match_stats(store: ResearchHistoryStore, instrument: str, cont_symbol: str, source: str, expiry: date,
                 lo: date, hi: date) -> Dict:
    r = store.conn.execute(
        """SELECT count(*), sum(CASE WHEN abs(a.close - b.close) < 0.01 AND a.oi = b.oi THEN 1 ELSE 0 END),
                  any_value(b.symbol)
           FROM hist_bars a JOIN hist_bars b
             ON b.ts = a.ts AND b.source = 'OWN' AND b.resolution = '1' AND b.contract_expiry = ?
                AND b.instrument = ?
           WHERE a.symbol = ? AND a.source = ? AND a.resolution = '1'
             AND CAST(a.ts AS DATE) BETWEEN ? AND ?""",
        [expiry, instrument, cont_symbol, source, lo, hi],
    ).fetchall()[0]
    return {"checked": int(r[0] or 0), "matched": int(r[1] or 0), "own_symbol": r[2]}


def segment_contract_table(store: ResearchHistoryStore, cal: ExchangeCalendar, instrument: str,
                           source: str, underlying: str) -> List[Dict]:
    sym = _cont_symbol(underlying, source)
    segs = store.conn.execute(
        """SELECT segment_id, min(CAST(ts AS DATE)), max(CAST(ts AS DATE)) FROM hist_bars
           WHERE symbol = ? AND source = ? AND resolution = '1' AND segment_id IS NOT NULL
           GROUP BY 1 ORDER BY 1""",
        [sym, source],
    ).fetchall()
    rows = []
    for sid, lo, hi in segs:
        expiry = contract_expiry_for(cal, lo, source)
        stats = _match_stats(store, instrument, sym, source, expiry, lo, hi) if expiry else {"checked": 0, "matched": 0,
                                                                                 "own_symbol": None}
        if stats["checked"] == 0:
            conf = "INFERRED"
        elif stats["matched"] >= 0.99 * stats["checked"]:
            conf = "VERIFIED"
        else:
            conf = "CONFLICT"
        rows.append({
            "instrument": instrument, "source": source, "segment_id": int(sid), "start_day": lo, "end_day": hi,
            "contract_expiry": expiry, "contract_symbol": stats["own_symbol"] if conf == "VERIFIED" else None,
            "confidence": conf, "checked_bars": stats["checked"], "matched_bars": stats["matched"],
        })
    return rows


def write_segment_table(store: ResearchHistoryStore, rows: List[Dict]) -> None:
    store._write("""CREATE TABLE IF NOT EXISTS cont_segments (
        instrument VARCHAR, source VARCHAR, segment_id INTEGER, start_day DATE, end_day DATE,
        contract_expiry DATE, contract_symbol VARCHAR, confidence VARCHAR,
        checked_bars INTEGER, matched_bars INTEGER, PRIMARY KEY (instrument, source, segment_id))""")
    if not rows:
        return
    keys = {(r["instrument"], r["source"]) for r in rows}
    for inst, src in keys:
        store._write("DELETE FROM cont_segments WHERE instrument = ? AND source = ?", [inst, src])
    df = pd.DataFrame(rows)[["instrument", "source", "segment_id", "start_day", "end_day", "contract_expiry",
                             "contract_symbol", "confidence", "checked_bars", "matched_bars"]]
    store.conn.register("segrows", df)
    try:
        store._write("INSERT INTO cont_segments SELECT * FROM segrows")
    finally:
        store.conn.unregister("segrows")


def identical_days(store: ResearchHistoryStore, underlying: str) -> List[date]:
    """Days where the NEAR and NEXT stitched series are bar-for-bar the same contract."""
    rows = store.conn.execute(
        """SELECT CAST(a.ts AS DATE) d, count(*) n,
                  sum(CASE WHEN a.close = b.close AND a.oi = b.oi THEN 1 ELSE 0 END) same
           FROM hist_bars a JOIN hist_bars b ON b.ts = a.ts AND b.symbol = ? AND b.resolution = '1'
           WHERE a.symbol = ? AND a.resolution = '1'
           GROUP BY 1 HAVING n > 0""",
        [_cont_symbol(underlying, "CONT_NEXT"), _cont_symbol(underlying, "CONT")],
    ).fetchall()
    return sorted(d for d, n, same in rows if same >= 0.99 * n)


def repair_next_from_own(store: ResearchHistoryStore, cal: ExchangeCalendar, instrument: str,
                         underlying: str) -> Dict:
    """Rebuild NEXT-month days from stored real contracts when the NEXT series duplicates NEAR.

    For each duplicated day: if the real next-month contract (2nd expiry >= day) is stored, its
    bars replace the NEXT bars exactly; otherwise the duplicated NEXT bars are deleted (missing is
    honest; a copy of the near month is wrong data).
    """
    next_sym = _cont_symbol(underlying, "CONT_NEXT")
    repaired, deleted = [], []
    for d in identical_days(store, underlying):
        expiry = contract_expiry_for(cal, d, "CONT_NEXT")
        store._write("DELETE FROM hist_bars WHERE symbol = ? AND source = 'CONT_NEXT' AND CAST(ts AS DATE) = ?",
                     [next_sym, d])
        n = store.conn.execute(
            """SELECT count(*) FROM hist_bars WHERE source = 'OWN' AND resolution = '1' AND instrument = ?
               AND contract_expiry = ? AND CAST(ts AS DATE) = ?""",
            [instrument, expiry, d],
        ).fetchall()[0][0]
        if n:
            store._write(
                """INSERT INTO hist_bars
                   SELECT ?, ?, 'CONT_NEXT', resolution, ts, open, high, low, close, volume, oi, contract_expiry, NULL
                   FROM hist_bars WHERE source = 'OWN' AND resolution = '1' AND instrument = ?
                     AND contract_expiry = ? AND CAST(ts AS DATE) = ?""",
                [instrument, next_sym, instrument, expiry, d],
            )
            repaired.append(d)
        else:
            deleted.append(d)
    return {"repaired": repaired, "deleted_unrecoverable": deleted}
