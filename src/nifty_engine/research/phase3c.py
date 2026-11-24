"""Phase 3C: evaluate pre-registered indicator hypotheses (research/preregistration/P3C_v1.yaml)."""

import random
from typing import Any, Dict, List

import pandas as pd

from nifty_engine.research.outcomes import EXECUTED, wilson

MIN_GATE_GROUP = 10
MIN_HOLDOUT_GROUP = 5
PERMUTATIONS = 10000
SEED = 2026


def masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Hypothesis pass masks (NaN where the indicator is unavailable). Must match P3C_v1.yaml."""
    bull, bear = df["direction"] == "BULLISH", df["direction"] == "BEARISH"
    known_dt = df["day_type"].notna() & (df["day_type"] != "UNKNOWN")
    out = {
        "H1_day_type_aligned": ((bull & (df["day_type"] == "TRENDING_UP")) | (bear & (df["day_type"] == "TRENDING_DOWN")))
        .where(known_dt),
        "H2_rvol_above_normal": (df["rvol_tod_5m"] >= 1.0).where(df["rvol_tod_5m"].notna()),
        "H3_near_expiry": (df["days_to_expiry"] <= 2).where(df["days_to_expiry"].notna()),
        "H4_heavyweights_confirm": ((bull & (df["hw_up_count"] >= 4)) | (bear & (df["hw_up_count"] <= 1)))
        .where(df["hw_up_count"].notna()),
    }
    return out


PREDICTS = {"H1_day_type_aligned": "higher", "H2_rvol_above_normal": "higher", "H3_near_expiry": "lower",
            "H4_heavyweights_confirm": "higher"}


def _group(hit: pd.Series, pts: pd.Series) -> Dict[str, Any]:
    n = int(len(hit))
    k = int(hit.sum())
    lo, hi = wilson(k, n)
    return {"n": n, "hits": k, "hit_pct": round(100 * k / n, 1) if n else None, "ci95": [lo, hi],
            "avg_points": round(float(pts.mean()), 2) if n else None}


def evaluate(df: pd.DataFrame, primary: str, target: float) -> Dict[str, Dict[str, Any]]:
    """Pass vs fail for each hypothesis. df: opportunity rows with out_<primary> and stop points."""
    executed = df[df[f"out_{primary}"].isin(EXECUTED)].copy()
    hit = executed[f"out_{primary}"] == "TARGET"
    pts = executed.apply(lambda r: target if r[f"out_{primary}"] == "TARGET" else (
        -float(r[f"stop_{primary}"]) if r[f"out_{primary}"] in ("STOP", "AMBIGUOUS") else float(r[f"close_{primary}"])),
        axis=1)
    res = {}
    for name, m in masks(executed).items():
        known = m.notna()
        mk = m[known].astype(bool)
        h, p = hit[known], pts[known]
        g_pass, g_fail = _group(h[mk], p[mk]), _group(h[~mk], p[~mk])
        diff = (g_pass["hit_pct"] - g_fail["hit_pct"]) if g_pass["n"] and g_fail["n"] else None
        res[name] = {"predicts": PREDICTS[name], "pass": g_pass, "fail": g_fail,
                     "hit_diff_pp": round(diff, 1) if diff is not None else None,
                     "unavailable": int((~known).sum()), "_hits": h.tolist(), "_mask": mk.tolist()}
    return res


def permutation_p(hits: List[bool], mask: List[bool], predicts: str, n: int = PERMUTATIONS, seed: int = SEED) -> float:
    """One-sided p-value for the pass-minus-fail hit-rate difference in the predicted direction."""
    rng = random.Random(seed)

    def diff(m):
        a = [h for h, x in zip(hits, m) if x]
        b = [h for h, x in zip(hits, m) if not x]
        return (sum(a) / len(a) - sum(b) / len(b)) if a and b else 0.0

    obs = diff(mask)
    sign = 1 if predicts == "higher" else -1
    m = list(mask)
    count = 0
    for _ in range(n):
        rng.shuffle(m)
        if sign * diff(m) >= sign * obs - 1e-12:
            count += 1
    return count / n


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 4)
    return out


def verdict(r: Dict[str, Any], p_holm: float) -> str:
    if r["pass"]["n"] < MIN_HOLDOUT_GROUP or r["fail"]["n"] < MIN_HOLDOUT_GROUP:
        return "INSUFFICIENT"
    d = r["hit_diff_pp"] if r["predicts"] == "higher" else -r["hit_diff_pp"]
    if p_holm < 0.05 and d > 0:
        return "CONFIRMED"
    if d >= 10:
        return "SUPPORTED"
    return "NOT_SUPPORTED"


def gate(r: Dict[str, Any]) -> bool:
    if r["pass"]["n"] < MIN_GATE_GROUP or r["fail"]["n"] < MIN_GATE_GROUP or r["hit_diff_pp"] is None:
        return False
    return r["hit_diff_pp"] > 0 if r["predicts"] == "higher" else r["hit_diff_pp"] < 0
