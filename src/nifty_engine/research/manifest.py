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

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
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


def require_valid_manifest(start: date, end: date, path: Union[str, Path] = DEFAULT_MANIFEST) -> Dict:
    """Gate for backtests: raise unless a manifest exists, is ok, and covers [start, end]."""
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"No dataset manifest at {p}; run scripts/build_history.py first")
    m = json.loads(p.read_text())
    if not m.get("ok"):
        raise ManifestError(f"Dataset manifest is not valid: {m.get('errors', [])[:3]}")
    r = m["range"]
    if date.fromisoformat(r["start"]) > start or date.fromisoformat(r["end"]) < end:
        raise ManifestError(f"Manifest covers {r['start']}..{r['end']}, backtest needs {start}..{end}")
    return m


def manifest_day_statuses(manifest: Dict, instrument: str, source: str) -> Optional[Dict[str, str]]:
    for s in manifest.get("series", []):
        if s["instrument"] == instrument and s["source"] == source:
            return {p["day"]: p["status"] for p in s["problem_days"]}
    return None
