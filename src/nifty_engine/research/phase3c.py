"""Phase 3C: evaluate pre-registered indicator hypotheses (research/preregistration/P3C_v1.yaml)."""

import random
from typing import Any, Dict, List

import pandas as pd

from nifty_engine.research.outcomes import EXECUTED, wilson

# Inference methods a pre-registration may declare. Every new pre-registration MUST declare one;
# only the archived P3C_v1 protocol (registered before the field existed) defaults to its original method.
INFERENCE_METHODS = ("permutation_opportunity", "day_cluster_bootstrap_ci")
LEGACY_INFERENCE = {"P3C_v1": "permutation_opportunity"}
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
                     "unavailable": int((~known).sum()), "_hits": h.tolist(), "_mask": mk.tolist(),
                     "_days": executed.loc[known, "day"].astype(str).tolist() if "day" in executed else []}
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


def inference_method(reg: Dict[str, Any]) -> str:
    m = reg.get("inference")
    if m is None:
        if reg.get("id") in LEGACY_INFERENCE:
            return LEGACY_INFERENCE[reg["id"]]
        raise ValueError("the pre-registration must declare `inference` (one of "
                         f"{', '.join(INFERENCE_METHODS)})")
    if m not in INFERENCE_METHODS:
        raise ValueError(f"unsupported inference method {m!r}; allowed: {', '.join(INFERENCE_METHODS)}")
    return m


def _day_resamples(hits, mask, days, n, seed):
    rng = random.Random(seed)
    by_day: Dict[str, List[tuple]] = {}
    for h, m, d in zip(hits, mask, days):
        by_day.setdefault(d, []).append((h, m))
    keys = sorted(by_day)
    for _ in range(n):
        rows = [x for d in (rng.choice(keys) for _ in keys) for x in by_day[d]]
        a = [h for h, m in rows if m]
        b = [h for h, m in rows if not m]
        if a and b:
            yield 100 * (sum(a) / len(a) - sum(b) / len(b))


def day_bootstrap_ci(hits: List[bool], mask: List[bool], days: List[str], level: float = 0.95,
                     n: int = PERMUTATIONS, seed: int = SEED) -> List[float]:
    """Percentile confidence interval (pp) for the pass-minus-fail hit-rate difference, resampling whole
    trading DAYS (same-day opportunities share market conditions). Used for decisions in day-cluster
    studies: an effect counts only if the (multiplicity-adjusted) interval excludes zero."""
    diffs = pd.Series(list(_day_resamples(hits, mask, days, n, seed)))
    if diffs.empty:
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [round(float(diffs.quantile(a)), 1), round(float(diffs.quantile(1 - a)), 1)]


def day_bootstrap_tail(hits: List[bool], mask: List[bool], days: List[str], predicts: str, n: int = PERMUTATIONS,
                       seed: int = SEED) -> float:
    """Share of whole-day bootstrap resamples whose pass-minus-fail difference is NOT in the predicted
    direction. A descriptive directional bootstrap probability that respects same-day dependence - NOT a
    calibrated null-hypothesis p-value (there is no null distribution here); decisions use
    day_bootstrap_ci."""
    sign = 1 if predicts == "higher" else -1
    diffs = list(_day_resamples(hits, mask, days, n, seed))
    return sum(sign * d <= 0 for d in diffs) / len(diffs) if diffs else 1.0


def verdict_ci(r: Dict[str, Any], ci: List[float]) -> str:
    """Verdict for day-cluster studies: CONFIRMED only if the multiplicity-adjusted day-bootstrap interval
    excludes zero in the predicted direction."""
    if r["pass"]["n"] < MIN_HOLDOUT_GROUP or r["fail"]["n"] < MIN_HOLDOUT_GROUP:
        return "INSUFFICIENT"
    d = r["hit_diff_pp"] if r["predicts"] == "higher" else -r["hit_diff_pp"]
    excludes = (ci[0] > 0) if r["predicts"] == "higher" else (ci[1] < 0)
    if excludes and d > 0:
        return "CONFIRMED"
    if d >= 10:
        return "SUPPORTED"
    return "NOT_SUPPORTED"

def check_selection_report(sel: Dict[str, Any], reg: Dict[str, Any], prereg_commit: str,
                           feature_manifest: Dict[str, Any]) -> List[str]:
    """The holdout may only use a selection report produced from this exact pre-registration, data,
    strategy, primary outcome and selection range."""
    m = sel.get("meta", {})
    problems = []
    if sel.get("stage") != "selection":
        problems.append("not a selection report")
    if m.get("prereg_commit") != prereg_commit:
        problems.append(f"selection used pre-registration {m.get('prereg_commit')}, current is {prereg_commit}")
    if m.get("primary") != reg["outcome"]["primary"]:
        problems.append(f"primary outcome {m.get('primary')} != {reg['outcome']['primary']}")
    fm = m.get("feature_manifest", {})
    if fm.get("dataset_fingerprint") != reg["dataset_fingerprint"] or fm.get("feature_version") != reg["feature_version"]:
        problems.append("selection feature dataset/version differs from the pre-registration")
    if fm.get("dataset_fingerprint") != feature_manifest.get("dataset_fingerprint"):
        problems.append("selection feature dataset differs from the current one")
    if "strategy_fingerprint" in m:
        if m["strategy_fingerprint"] != reg["strategy"]["fingerprint"]:
            problems.append(f"selection strategy {m['strategy_fingerprint']} != {reg['strategy']['fingerprint']}")
    elif reg.get("id") in LEGACY_INFERENCE:
        # Documented compatibility path for the archived P3C_v1 selection report, written before the field
        # existed (the report itself is never rewritten): its feature dataset matched above, and that dataset's
        # manifest records the strategy it was built for.
        fm_strategy = (feature_manifest.get("strategy") or {}).get("fingerprint")
        if fm_strategy != reg["strategy"]["fingerprint"]:
            problems.append(f"legacy selection report: feature dataset strategy {fm_strategy} != "
                            f"{reg['strategy']['fingerprint']}")
    else:
        problems.append("selection report has no strategy_fingerprint")
    if [str(x) for x in sel.get("range", [])] != [str(x) for x in reg["split"]["selection"]]:
        problems.append(f"selection range {sel.get('range')} != pre-registered {reg['split']['selection']}")
    return problems
