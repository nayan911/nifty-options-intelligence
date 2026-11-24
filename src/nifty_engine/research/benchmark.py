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
