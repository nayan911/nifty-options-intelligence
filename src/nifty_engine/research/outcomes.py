"""Phase 3A: label historical S000 candidates with what NIFTY spot actually did next.

Definitions (fixed before looking at results):
- Opportunity: candidates on the same day, direction and decision minute are ONE opportunity
  (several levels confluent on one bar). The representative is the highest-confluence candidate
  (ties: lowest candidate_id); its stop is used for the "S000 stop" variant.
- Entry: the OPEN of the first in-session 1m spot bar starting at/after the decision time
  (decision_at = close of the trigger bar). No entry if the session ends first.
- Outcome per (target, stop, horizon), walking 1m bars from the entry bar inclusive:
    TARGET    target touched (high/low) before the stop
    STOP      stop touched first (also if the entry open is already beyond the S000 stop level)
    AMBIGUOUS both touched inside the same 1m bar: order unknown, NEVER counted as a win
    TIMEOUT   neither within the horizon (or before the session's last bar)
  Targets are NIFTY points from the entry. Stops: S000's own stop level, or fixed points.
- One position at a time: for each configuration, a "trade" is an opportunity taken only when no
  earlier trade is still open at its entry (the realistic sequence for a single manual trader).
- Hit rates use the Wilson score interval (95%). Samples under 30 are flagged.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

TARGETS = (30.0, 40.0)
STOPS: Tuple[Any, ...] = ("S000", 15.0, 20.0)
HORIZONS = (45, 60)
MIN_SAMPLE = 30


@dataclass
class Opportunity:
    day: Any
    decision_at: datetime
    direction: str             # BULLISH / BEARISH
    candidates: List[Dict[str, Any]]
    rep: Dict[str, Any]        # representative candidate detail
    tier: Optional[str] = None
    entry_at: Optional[datetime] = None
    entry: Optional[float] = None
    outcomes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.direction == "BULLISH" else -1


def group_opportunities(candidates: Sequence[Any]) -> List[Opportunity]:
    """candidates: ReplaySignal(kind='CANDIDATE') with detail incl. decision_at, entry, stop_loss."""
    groups: Dict[Tuple, List[Any]] = {}
    for c in candidates:
        dec = pd.Timestamp(c.detail["decision_at"]).to_pydatetime()
        groups.setdefault((c.day, dec, c.direction), []).append(c)
    out = []
    for (day, dec, direction), cs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        rep = sorted(cs, key=lambda c: (-float(c.detail.get("confluence") or 0), c.detail["candidate_id"]))[0]
        out.append(Opportunity(day=day, decision_at=dec, direction=direction,
                               candidates=[c.detail for c in cs], rep=rep.detail))
    return out


def label(op: Opportunity, bars: pd.DataFrame, target: float, stop: Any, horizon_min: int) -> Dict[str, Any]:
    """bars: that day's in-session 1m spot bars (ts, open, high, low, close), ascending."""
    after = bars[bars["ts"] >= pd.Timestamp(op.decision_at)]
    if after.empty:
        return {"outcome": "NO_ENTRY"}
    entry = float(after.iloc[0]["open"])
    entry_at = after.iloc[0]["ts"]
    window = after[after["ts"] < entry_at + pd.Timedelta(minutes=horizon_min)]
    s = op.sign
    tgt_px = entry + s * target
    if stop == "S000":
        stop_px = float(op.rep["stop_loss"])
        stop_pts = s * (entry - stop_px)
    else:
        stop_pts = float(stop)
        stop_px = entry - s * stop_pts
    res: Dict[str, Any] = {"entry": entry, "entry_at": entry_at.isoformat(), "stop_points": round(stop_pts, 2)}
    mfe = mae = 0.0
    if stop_pts <= 0:  # opened beyond the S000 stop level
        res.update(outcome="STOP", exit_at=entry_at.isoformat(), minutes=0, mfe=0.0, mae=round(-stop_pts, 2))
        return res
    for r in window.itertuples(index=False):
        fav = s * ((r.high if s > 0 else r.low) - entry)
        adv = s * (entry - (r.low if s > 0 else r.high))
        hit_t = fav >= target
        hit_s = adv >= stop_pts
        mfe, mae = max(mfe, fav), max(mae, adv)
        minutes = int((r.ts - entry_at).total_seconds() // 60)
        if hit_t and hit_s:
            res.update(outcome="AMBIGUOUS", exit_at=r.ts.isoformat(), minutes=minutes)
            break
        if hit_t:
            res.update(outcome="TARGET", exit_at=r.ts.isoformat(), minutes=minutes)
            break
        if hit_s:
            res.update(outcome="STOP", exit_at=r.ts.isoformat(), minutes=minutes)
            break
    else:
        last = window.iloc[-1] if not window.empty else after.iloc[0]
        res.update(outcome="TIMEOUT", exit_at=last["ts"].isoformat(),
                   minutes=int((last["ts"] - entry_at).total_seconds() // 60),
                   close_points=round(s * (float(last["close"]) - entry), 2))
    res.update(mfe=round(mfe, 2), mae=round(mae, 2), target_px=round(tgt_px, 2), stop_px=round(stop_px, 2))
    return res


def config_key(target: float, stop: Any, horizon: int) -> str:
    return f"T{int(target)}_S{stop if stop == 'S000' else int(stop)}_H{horizon}"


def label_all(ops: List[Opportunity], bars_by_day: Dict[Any, pd.DataFrame]) -> None:
    for op in ops:
        bars = bars_by_day.get(op.day)
        for t in TARGETS:
            for s in STOPS:
                for h in HORIZONS:
                    op.outcomes[config_key(t, s, h)] = (
                        label(op, bars, t, s, h) if bars is not None else {"outcome": "NO_DATA"})
        first = next(iter(op.outcomes.values()))
        if "entry" in first:
            op.entry, op.entry_at = first["entry"], pd.Timestamp(first["entry_at"]).to_pydatetime()


def one_position_at_a_time(ops: List[Opportunity], key: str) -> List[Opportunity]:
    taken, busy_until = [], None
    for op in sorted(ops, key=lambda o: o.decision_at):
        o = op.outcomes.get(key, {})
        if "entry_at" not in o:
            continue
        start = pd.Timestamp(o["entry_at"])
        if busy_until is not None and start <= busy_until:
            continue
        taken.append(op)
        busy_until = pd.Timestamp(o["exit_at"])
    return taken


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def summarize(ops: List[Opportunity], key: str) -> Dict[str, Any]:
    rows = [op.outcomes[key] for op in ops if op.outcomes.get(key, {}).get("outcome") not in (None, "NO_ENTRY", "NO_DATA")]
    n = len(rows)
    counts = {k: sum(r["outcome"] == k for r in rows) for k in ("TARGET", "STOP", "AMBIGUOUS", "TIMEOUT")}
    lo, hi = wilson(counts["TARGET"], n)
    hits = [r["minutes"] for r in rows if r["outcome"] == "TARGET"]
    decided = counts["TARGET"] + counts["STOP"]
    target = float(key.split("_")[0][1:]) if key.startswith("T") else None
    pts = [points(r, target) for r in rows] if target is not None else []
    mean_pts = float(pd.Series(pts).mean()) if pts else None
    se_pts = float(pd.Series(pts).std(ddof=0) / math.sqrt(len(pts))) if len(pts) > 1 else None
    return {
        "n": n, **counts,
        "hit_rate": round(100 * counts["TARGET"] / n, 1) if n else None,
        "hit_rate_ci95": [lo, hi],
        "target_vs_stop": round(100 * counts["TARGET"] / decided, 1) if decided else None,
        "median_minutes_to_target": float(pd.Series(hits).median()) if hits else None,
        "median_mfe": float(pd.Series([r["mfe"] for r in rows]).median()) if rows else None,
        "median_mae": float(pd.Series([r["mae"] for r in rows]).median()) if rows else None,
        "expectancy_points": round(mean_pts, 2) if mean_pts is not None else None,
        "expectancy_se": round(se_pts, 2) if se_pts is not None else None,
        "small_sample": n < MIN_SAMPLE,
    }


def points(r: Dict[str, Any], target: float) -> float:
    """NIFTY points for one labelled outcome: +target, -stop (ambiguous counted as a stop), or the
    timeout close. Gross of all costs."""
    if r["outcome"] == "TARGET":
        return target
    if r["outcome"] in ("STOP", "AMBIGUOUS"):
        return -float(r["stop_points"])
    return float(r.get("close_points", 0.0))


def hour_bucket(ts: datetime) -> str:
    m = ts.hour * 60 + ts.minute
    if m < 10 * 60:
        return "09:15-10:00"
    if m < 11 * 60 + 30:
        return "10:00-11:30"
    if m < 13 * 60 + 30:
        return "11:30-13:30"
    return "13:30-close"


def confluence_bucket(score: Optional[float]) -> str:
    s = float(score or 0)
    return "80+" if s >= 80 else ("70-79" if s >= 70 else "60-69")


def breakdowns(ops: List[Opportunity], key: str) -> Dict[str, Dict[str, Any]]:
    dims = {
        "contract_tier": lambda o: o.tier or "UNVALIDATED",
        "direction": lambda o: o.direction,
        "month": lambda o: o.decision_at.strftime("%Y-%m"),
        "time_of_day": lambda o: hour_bucket(o.decision_at),
        "confluence": lambda o: confluence_bucket(o.rep.get("confluence")),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in dims.items():
        groups: Dict[str, List[Opportunity]] = {}
        for o in ops:
            groups.setdefault(fn(o), []).append(o)
        out[name] = {g: summarize(v, key) for g, v in sorted(groups.items())}
    return out


def bars_by_day(df1m: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
    df = df1m.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    return {d: g.sort_values("ts").reset_index(drop=True) for d, g in df.groupby(df["ts"].dt.date)}


def report(ops: List[Opportunity], n_candidates: int) -> Dict[str, Any]:
    configs = {}
    for t in TARGETS:
        for s in STOPS:
            for h in HORIZONS:
                k = config_key(t, s, h)
                seq = one_position_at_a_time(ops, k)
                configs[k] = {"all_opportunities": summarize(ops, k), "one_position_at_a_time": summarize(seq, k)}
    s000_stops = [o.outcomes[config_key(30, "S000", 60)].get("stop_points") for o in ops
                  if "stop_points" in o.outcomes.get(config_key(30, "S000", 60), {})]
    return {
        "counts": {
            "candidates": n_candidates,
            "opportunities": len(ops),
            "merged_same_bar": n_candidates - len(ops),
            "days_with_opportunity": len({o.day for o in ops}),
        },
        "s000_stop_points": {
            "median": round(float(pd.Series(s000_stops).median()), 1) if s000_stops else None,
            "p10": round(float(pd.Series(s000_stops).quantile(0.1)), 1) if s000_stops else None,
            "p90": round(float(pd.Series(s000_stops).quantile(0.9)), 1) if s000_stops else None,
        },
        "configs": configs,
        "breakdowns_T30_S20_H60": breakdowns(ops, config_key(30, 20.0, 60)),
        "breakdowns_T30_S000_H60": breakdowns(ops, config_key(30, "S000", 60)),
        "opportunities": [
            {"day": str(o.day), "decision_at": o.decision_at.isoformat(), "direction": o.direction,
             "tier": o.tier, "n_candidates": len(o.candidates), "level": o.rep.get("level_id"),
             "confluence": o.rep.get("confluence"), "s000_stop": o.rep.get("stop_loss"),
             "outcomes": o.outcomes}
            for o in ops
        ],
    }
