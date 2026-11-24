"""Matched random-entry benchmark: does S000 pick better moments than chance?

For each S000 opportunity, draw `draws` random entries that share its minute of day and direction
on OTHER trading days (same session kind, entry minute must exist), with the same stop rule (S000's
own stop distance is reused as a fixed distance; fixed stops unchanged). Every entry is labelled with
exactly the same rules as S000 (outcomes.label: next 1m open, ambiguous never a win, horizon).

Comparison: resample one matched entry per opportunity many times -> the distribution of hit rate
and mean points a "random S000-shaped" set of the same size achieves. Reported: S000's value, the
benchmark mean, and the share of resamples at least as good as S000 (a one-sided p-value).
"""

import random
from datetime import timedelta
from typing import Any, Dict, List

import pandas as pd

from nifty_engine.research.outcomes import EXECUTED, Opportunity, label, points


def _shift(op: Opportunity, day, bars: pd.DataFrame, stop_points: float) -> Opportunity:
    dec = pd.Timestamp(op.decision_at)
    t = dec.tz_convert("Asia/Kolkata")
    new_dec = pd.Timestamp(bars["ts"].iloc[0]).normalize() + timedelta(hours=t.hour, minutes=t.minute)
    s = 1 if op.direction == "BULLISH" else -1
    rep = {"candidate_id": f"bench_{day}_{t:%H%M}", "stop_loss": None, "confluence": None,
           "_stop_points": stop_points, "_sign": s}
    return Opportunity(day=day, decision_at=new_dec.to_pydatetime(), direction=op.direction, candidates=[], rep=rep)


def benchmark(ops: List[Opportunity], bars_by_day: Dict[Any, pd.DataFrame], configs: List[tuple],
              draws: int = 50, resamples: int = 10000, seed: int = 7) -> Dict[str, Any]:
    """configs: [(target, stop, horizon)] with stop 'S000' or points."""
    rng = random.Random(seed)
    days = sorted(bars_by_day)
    out: Dict[str, Any] = {"draws_per_opportunity": draws, "resamples": resamples, "seed": seed, "configs": {}}
    for target, stop, horizon in configs:
        key = f"T{int(target)}_S{stop if stop == 'S000' else int(stop)}_H{horizon}"
        s000_pts, s000_hits, bench = [], [], []
        for op in ops:
            res = op.outcomes.get(key, {})
            if res.get("outcome") not in EXECUTED:
                continue
            s000_hits.append(res["outcome"] == "TARGET")
            s000_pts.append(points(res, target))
            stop_pts = float(res["stop_points"]) if stop == "S000" else float(stop)
            t = pd.Timestamp(op.decision_at).tz_convert("Asia/Kolkata")
            pool = [d for d in days if d != op.day
                    and ((bars_by_day[d]["ts"].dt.hour == t.hour) & (bars_by_day[d]["ts"].dt.minute == t.minute)).any()]
            sample = [rng.choice(pool) for _ in range(draws)] if pool else []
            labelled = []
            for d in sample:
                b = bars_by_day[d]
                fake = _shift(op, d, b, stop_pts)
                r = label(fake, b, target, stop_pts, horizon)
                if r.get("outcome") in EXECUTED:
                    labelled.append((r["outcome"] == "TARGET", points(r, target)))
            bench.append(labelled)
        n = len(s000_hits)
        valid = [b for b in bench if b]
        flat = [x for b in valid for x in b]
        s_hit = 100 * sum(s000_hits) / n if n else None
        s_pts = sum(s000_pts) / n if n else None
        rates, means = [], []
        for _ in range(resamples):
            pick = [rng.choice(b) for b in valid]
            rates.append(100 * sum(h for h, _ in pick) / len(pick))
            means.append(sum(p for _, p in pick) / len(pick))
        out["configs"][key] = {
            "n": n,
            "s000_hit_pct": round(s_hit, 1) if s_hit is not None else None,
            "benchmark_hit_pct": round(100 * sum(h for h, _ in flat) / len(flat), 1) if flat else None,
            "p_hit_at_least_s000": round(sum(r >= s_hit for r in rates) / len(rates), 4) if rates else None,
            "s000_avg_points": round(s_pts, 2) if s_pts is not None else None,
            "benchmark_avg_points": round(sum(p for _, p in flat) / len(flat), 2) if flat else None,
            "p_points_at_least_s000": round(sum(m >= s_pts for m in means) / len(means), 4) if means else None,
            "benchmark_hit_pct_p5_p95": ([round(pd.Series(rates).quantile(q), 1) for q in (0.05, 0.95)]
                                         if rates else None),
        }
    return out


def same_day_benchmark(ops: List[Opportunity], bars_by_day: Dict[Any, pd.DataFrame], target: float, stop: Any,
                       horizon: int, draws: int = 50, seed: int = 11) -> Dict[str, Any]:
    """Random entries on the SAME day as each opportunity (same direction and stop distance, any other
    session minute). If S000 still beats these, its edge is in WHEN it enters within a day, not only in
    which days it trades. Returns per-opportunity matched draws for paired analysis."""
    rng = random.Random(seed)
    key = f"T{int(target)}_S{stop if stop == 'S000' else int(stop)}_H{horizon}"
    rows = []
    for op in ops:
        res = op.outcomes.get(key, {})
        if res.get("outcome") not in EXECUTED:
            continue
        stop_pts = float(res["stop_points"]) if stop == "S000" else float(stop)
        b = bars_by_day[op.day]
        own = pd.Timestamp(op.decision_at)
        minutes = [t for t in b["ts"] if t != own]
        labelled = []
        for t in (rng.choice(minutes) for _ in range(draws)):
            fake = Opportunity(day=op.day, decision_at=t.to_pydatetime(), direction=op.direction, candidates=[],
                               rep={"stop_loss": None})
            r = label(fake, b, target, stop_pts, horizon)
            if r.get("outcome") in EXECUTED:
                labelled.append((r["outcome"] == "TARGET", points(r, target)))
        rows.append({"day": op.day, "s000_hit": res["outcome"] == "TARGET", "s000_points": points(res, target),
                     "random": labelled})
    return {"key": key, "rows": rows}


def day_block_bootstrap(rows: List[Dict[str, Any]], resamples: int = 10000, seed: int = 13) -> Dict[str, Any]:
    """Paired S000-minus-matched-random difference, resampling whole DAYS (opportunities on one day share
    market conditions, so they are not treated as independent). rows: s000_hit, s000_points, random."""
    rng = random.Random(seed)
    by_day: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        if r["random"]:
            by_day.setdefault(r["day"], []).append(r)
    days = sorted(by_day)

    def diff(sel):
        rs = [r for d in sel for r in by_day[d]]
        hit = sum(r["s000_hit"] for r in rs) / len(rs) - sum(
            sum(h for h, _ in r["random"]) / len(r["random"]) for r in rs) / len(rs)
        pts = sum(r["s000_points"] for r in rs) / len(rs) - sum(
            sum(p for _, p in r["random"]) / len(r["random"]) for r in rs) / len(rs)
        return 100 * hit, pts

    obs_hit, obs_pts = diff(days)
    boots = [diff([rng.choice(days) for _ in days]) for _ in range(resamples)]
    hits, pts = pd.Series([b[0] for b in boots]), pd.Series([b[1] for b in boots])
    return {
        "days": len(days), "opportunities": sum(len(v) for v in by_day.values()),
        "hit_diff_pp": round(obs_hit, 1),
        "hit_diff_ci95": [round(float(hits.quantile(0.025)), 1), round(float(hits.quantile(0.975)), 1)],
        "p_hit_diff_le_0": round(float((hits <= 0).mean()), 4),
        "points_diff": round(obs_pts, 2),
        "points_diff_ci95": [round(float(pts.quantile(0.025)), 2), round(float(pts.quantile(0.975)), 2)],
        "p_points_diff_le_0": round(float((pts <= 0).mean()), 4),
    }


def matched_rows(ops: List[Opportunity], bars_by_day: Dict[Any, pd.DataFrame], target: float, stop: Any,
                 horizon: int, draws: int = 50, seed: int = 7) -> List[Dict[str, Any]]:
    """Other-day matched draws (same minute of day, direction, stop distance) as rows for day_block_bootstrap."""
    rng = random.Random(seed)
    key = f"T{int(target)}_S{stop if stop == 'S000' else int(stop)}_H{horizon}"
    days = sorted(bars_by_day)
    rows = []
    for op in ops:
        res = op.outcomes.get(key, {})
        if res.get("outcome") not in EXECUTED:
            continue
        stop_pts = float(res["stop_points"]) if stop == "S000" else float(stop)
        t = pd.Timestamp(op.decision_at).tz_convert("Asia/Kolkata")
        pool = [d for d in days if d != op.day
                and ((bars_by_day[d]["ts"].dt.hour == t.hour) & (bars_by_day[d]["ts"].dt.minute == t.minute)).any()]
        labelled = []
        for d in (rng.choice(pool) for _ in range(draws)) if pool else []:
            b = bars_by_day[d]
            r = label(_shift(op, d, b, stop_pts), b, target, stop_pts, horizon)
            if r.get("outcome") in EXECUTED:
                labelled.append((r["outcome"] == "TARGET", points(r, target)))
        rows.append({"day": op.day, "s000_hit": res["outcome"] == "TARGET", "s000_points": points(res, target),
                     "random": labelled})
    return rows
