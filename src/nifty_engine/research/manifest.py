"""Validated dataset manifest: the gate between downloaded history and any backtest.

For every series and trading day the stored 1m bars are compared with the exact minutes the
exchange calendar says should exist (research.sessions.expected_minutes):

  COMPLETE      every expected minute present
  SPARSE        a real futures contract (OWN) missing <= sparse_tolerance minutes: those minutes
                simply had no trades in an illiquid next-month contract (not a download gap)
  INCOMPLETE    expected minutes missing
  MISSING       no bars at all on a trading day
  NOT_LISTED    OWN contract day before its first available bar (FYERS serves a contract only
                from listing; expected absence, never an error)

Bars outside the session window are counted as `out_of_session` (warning: features must filter
them). Stitched-futures contract switches come from research.segments (2-of-3 vote of expiry
calendar, OI ratio and basis jump); single-signal candidates are reported as warnings. The manifest is `ok` only when every REQUIRED series is
COMPLETE on every trading day. Backtests must refuse to run
otherwise (see `require_valid_manifest`).
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.sessions import CASH, FNO, expected_minutes

DEFAULT_MANIFEST = Path("data/research_manifest.json")

# (instrument, source) pairs that must be complete for strategy research
REQUIRED_SERIES: Tuple[Tuple[str, str], ...] = (
    ("NIFTY_SPOT", "SPOT"),
    ("NIFTY_FUT", "CONT"),
    ("INDIAVIX", "SPOT"),
    ("BANKNIFTY_SPOT", "SPOT"),
    ("BANKNIFTY_FUT", "CONT"),
)


class ManifestError(RuntimeError):
    pass


@dataclass
class DayStatus:
    day: date
    status: str
    expected: int
    present: int
    missing: int
    out_of_session: int


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def dataset_fingerprint(store: ResearchHistoryStore) -> str:
    """Content fingerprint of the research dataset plus the calendar config it was validated with.

    Any added/removed/changed bar (count, time range, price or OI totals per series) or any change
    to the holiday/session config yields a different fingerprint.
    """
    rows = store.conn.execute(
        """SELECT instrument, source, symbol, resolution, count(*), min(ts)::VARCHAR, max(ts)::VARCHAR,
                  round(sum(close), 4), sum(coalesce(oi, 0)), sum(volume), sum(coalesce(segment_id, -1))
           FROM hist_bars GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"""
    ).fetchall()
    h = hashlib.sha256(repr(rows).encode())
    from nifty_engine.core.config import settings
    for cfg in (settings.HOLIDAYS_CONFIG_PATH, settings.SESSIONS_CONFIG_PATH):
        p = Path(cfg)
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:24]


def _integrity_errors(store: ResearchHistoryStore, cal: ExchangeCalendar, trading_days: Sequence[date],
                      required: Sequence[Tuple[str, str]]) -> List[str]:
    """Checks beyond 1m coverage: daily bars, futures OI, segment ids, contract identity."""
    errors: List[str] = []
    tset = set(trading_days)
    lo, hi = (min(trading_days), max(trading_days)) if trading_days else (None, None)
    for inst, src in required:
        have = {r[0] for r in store.conn.execute(
            """SELECT DISTINCT CAST(ts AS DATE) FROM hist_bars WHERE instrument = ? AND source = ?
               AND resolution = 'D' AND CAST(ts AS DATE) BETWEEN ? AND ?""", [inst, src, lo, hi]).fetchall()}
        miss = sorted(tset - have)
        if miss:
            errors.append(f"{inst}/{src}: {len(miss)} trading day(s) without a daily bar, e.g. {miss[:3]}")
    for inst, src, sym, n in store.conn.execute(
        """SELECT instrument, source, symbol, count(*) FROM hist_bars
           WHERE source IN ('CONT', 'CONT_NEXT', 'OWN') AND oi IS NULL GROUP BY 1, 2, 3"""
    ).fetchall():
        errors.append(f"{inst}/{src} {sym}: {n} futures bars without OI")
    for inst, src, n in store.conn.execute(
        """SELECT instrument, source, count(*) FROM hist_bars
           WHERE source IN ('CONT', 'CONT_NEXT') AND resolution = '1' AND segment_id IS NULL GROUP BY 1, 2"""
    ).fetchall():
        errors.append(f"{inst}/{src}: {n} stitched bars not assigned to a contract segment")
    has_table = store.conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'cont_segments'").fetchall()[0][0]
    if has_table:
        for inst, src, sid, lo_d, hi_d, m, c in store.conn.execute(
            """SELECT instrument, source, segment_id, start_day, end_day, matched_bars, checked_bars
               FROM cont_segments WHERE confidence = 'CONFLICT'"""
        ).fetchall():
            errors.append(f"{inst}/{src} segment {sid} ({lo_d}..{hi_d}): contract identity CONFLICT ({m}/{c} bars "
                          "match the expected real contract)")
    from nifty_engine.research.contracts import identical_days
    for und in ("NIFTY", "BANKNIFTY"):
        same = identical_days(store, und)
        if same:
            errors.append(f"{und}: NEAR and NEXT stitched series are the same contract on {len(same)} day(s), "
                          f"e.g. {same[:3]}")
    return errors


def classify_series_days(
    cal: ExchangeCalendar,
    minutes_by_day: Dict[date, set],
    trading_days: Sequence[date],
    kind: str,
    source: str,
    sparse_tolerance: int = 5,
) -> List[DayStatus]:
    first_day = min(minutes_by_day) if minutes_by_day else None
    out: List[DayStatus] = []
    for d in trading_days:
        exp = expected_minutes(cal, d, kind)
        exp_set = set(exp)
        have = minutes_by_day.get(d, set())
        present = len(have & exp_set)
        missing = len(exp_set) - present
        extra = len(have - exp_set)
        if source == "OWN" and (first_day is None or d < first_day):
            status = "NOT_LISTED"
        elif not have:
            status = "MISSING"
        elif missing == 0:
            status = "COMPLETE"
        elif source in ("OWN", "CONT_NEXT") and missing <= sparse_tolerance:
            status = "SPARSE"
        else:
            status = "INCOMPLETE"
        out.append(DayStatus(d, status, len(exp_set), present, missing, extra))
    return out


def build_manifest(
    store: ResearchHistoryStore,
    cal: ExchangeCalendar,
    start: date,
    end: date,
    required: Sequence[Tuple[str, str]] = REQUIRED_SERIES,
    sparse_tolerance: int = 5,
    segment_reports: Optional[Dict[str, Dict]] = None,
) -> Dict:
    trading_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    trading_days = [d for d in trading_days if cal.is_trading_day(d)]

    df = store.conn.execute(
        """SELECT instrument, source, symbol, ts FROM hist_bars
           WHERE resolution = '1' AND CAST(ts AS DATE) BETWEEN ? AND ?""",
        [start, end],
    ).df()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata")
        df["day"] = df["ts"].dt.date

    series_out = []
    errors: List[str] = []
    warnings: List[str] = []
    present_series = set()
    for (inst, src, sym), g in (df.groupby(["instrument", "source", "symbol"]) if not df.empty else []):
        present_series.add((inst, src))
        minutes_by_day: Dict[date, set] = {}
        for d, gd in g.groupby("day"):
            minutes_by_day[d] = set(ts.to_pydatetime() for ts in gd["ts"])
        kind = FNO if src in ("CONT", "CONT_NEXT", "OWN") else CASH
        days = classify_series_days(cal, minutes_by_day, trading_days, kind, src, sparse_tolerance)
        counts: Dict[str, int] = {}
        for ds in days:
            counts[ds.status] = counts.get(ds.status, 0) + 1
        problem_days = [
            {"day": ds.day.isoformat(), "status": ds.status, "expected": ds.expected, "present": ds.present,
             "missing": ds.missing}
            for ds in days if ds.status in ("INCOMPLETE", "MISSING", "SPARSE")
        ]
        oos = [{"day": ds.day.isoformat(), "bars": ds.out_of_session} for ds in days if ds.out_of_session]
        required_series = (inst, src) in required
        if required_series and any(ds.status in ("INCOMPLETE", "MISSING") for ds in days):
            bad = [p["day"] for p in problem_days if p["status"] in ("INCOMPLETE", "MISSING")]
            errors.append(f"{inst}/{src} {sym}: {len(bad)} incomplete/missing day(s), e.g. {bad[:5]}")
        if oos:
            warnings.append(f"{inst}/{src} {sym}: out-of-session bars on {[o['day'] for o in oos][:5]} "
                            "(excluded from features)")
        series_out.append({
            "instrument": inst, "source": src, "symbol": sym, "required": required_series,
            "status_counts": counts, "problem_days": problem_days, "out_of_session": oos,
        })

    for inst, src in required:
        if (inst, src) not in present_series:
            errors.append(f"{inst}/{src}: required series has no data in {start}..{end}")

    rollovers = {}
    for key, rep in (segment_reports or {}).items():
        def ser(items):
            return [{**b, "day": b["day"].isoformat(), "prev_day": b["prev_day"].isoformat()} for b in items]
        rollovers[key] = {"accepted": ser(rep["accepted"]), "disputed": ser(rep["disputed"])}
        for b in rep["disputed"]:
            warnings.append(f"{key}: single-signal possible contract switch {b['prev_day']}->{b['day']} "
                            f"votes={b['votes']} (not treated as a boundary)")

    errors.extend(_integrity_errors(store, cal, trading_days, required))
    contract_ids = []
    has_table = store.conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'cont_segments'").fetchall()[0][0]
    if has_table:
        for r in store.conn.execute(
            """SELECT instrument, source, segment_id, start_day, end_day, contract_expiry, contract_symbol,
                      confidence, matched_bars, checked_bars FROM cont_segments ORDER BY 1, 2, 3"""
        ).fetchall():
            contract_ids.append({"instrument": r[0], "source": r[1], "segment_id": r[2], "start": str(r[3]),
                                 "end": str(r[4]), "contract_expiry": str(r[5]), "contract_symbol": r[6],
                                 "confidence": r[7], "matched_bars": r[8], "checked_bars": r[9]})

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_fingerprint": dataset_fingerprint(store),
        "contract_segments": contract_ids,
        "range": {"start": start.isoformat(), "end": end.isoformat(), "trading_days": len(trading_days)},
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rollovers": rollovers,
        "series": series_out,
    }


def write_manifest(manifest: Dict, path: Union[str, Path] = DEFAULT_MANIFEST) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    tmp.replace(p)
    return p


def require_valid_manifest(start: date, end: date, store: Optional[ResearchHistoryStore],
                           path: Union[str, Path] = DEFAULT_MANIFEST) -> Dict:
    """Gate for backtests: raise unless a manifest exists, is ok, covers [start, end], and still
    describes the data in `store` (fingerprint unchanged since validation)."""
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"No dataset manifest at {p}; run scripts/build_history.py first")
    m = json.loads(p.read_text())
    if not m.get("ok"):
        raise ManifestError(f"Dataset manifest is not valid: {m.get('errors', [])[:3]}")
    r = m["range"]
    if date.fromisoformat(r["start"]) > start or date.fromisoformat(r["end"]) < end:
        raise ManifestError(f"Manifest covers {r['start']}..{r['end']}, backtest needs {start}..{end}")
    if store is None:
        raise ManifestError("A store is required to verify the manifest still matches the data")
    current = dataset_fingerprint(store)
    if m.get("dataset_fingerprint") != current:
        raise ManifestError(f"Dataset changed since validation (manifest {m.get('dataset_fingerprint')}, "
                            f"data now {current}); re-run scripts/build_history.py --validate-only")
    return m


def manifest_day_statuses(manifest: Dict, instrument: str, source: str) -> Optional[Dict[str, str]]:
    for s in manifest.get("series", []):
        if s["instrument"] == instrument and s["source"] == source:
            return {p["day"]: p["status"] for p in s["problem_days"]}
    return None
