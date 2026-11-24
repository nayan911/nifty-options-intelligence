"""Phase 3C: evaluate the pre-registered hypotheses in research/preregistration/P3C_v1.yaml.

  --stage selection   gate each hypothesis on Jan-Aug (direction + group sizes only; nothing tuned)
  --stage holdout     evaluate gated hypotheses ONCE on Sep-Nov (refuses if already done; --force re-runs
                      and the report says so)
Guards: the pre-registration file must be committed and unchanged; the feature dataset must match its
dataset fingerprint, strategy fingerprint and feature version. Reports go to reports/phase3c/.
"""

import argparse
import json
import subprocess
from datetime import date, datetime
from pathlib import Path

import duckdb
import yaml

from nifty_engine.research import phase3c as p3
from nifty_engine.research.replay import _git_commit

PREREG = Path("research/preregistration/P3C_v1.yaml")
OUT = Path("reports/phase3c")


def prereg_commit() -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PREREG)], capture_output=True, text=True).stdout
    commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(PREREG)], capture_output=True,
                            text=True).stdout.strip()
    if dirty.strip() or not commit:
        raise SystemExit(f"{PREREG} must be committed and unchanged before any evaluation")
    return commit


def load(reg):
    fv = reg["feature_version"]
    man = json.loads(Path(f"data/features/manifest_v{fv}.json").read_text())
    if man["dataset_fingerprint"] != reg["dataset_fingerprint"] or man["strategy"]["fingerprint"] != reg["strategy"]["fingerprint"]:
        raise SystemExit(f"feature dataset v{fv} does not match the pre-registration (dataset/strategy fingerprint)")
    db = duckdb.connect(f"data/features/features_v{fv}.duckdb", read_only=True)
    df = db.execute("SELECT * FROM opportunities").df()
    df["day"] = df["day"].astype(str)
    return df, man


def table(res, extra=None):
    L = ["| Hypothesis | Predicts | Pass: hit (95%) n | Fail: hit (95%) n | Diff pp | Pass / fail avg pts | Unavail |"
         + ("".join(f" {h} |" for h in extra) if extra else ""),
         "|---|---|---|---|---|---|---|" + ("---|" * len(extra) if extra else "")]
    for k, r in res.items():
        a, b = r["pass"], r["fail"]
        row = (f"| {k} | {r['predicts']} | {a['hit_pct']}% {a['ci95']} n={a['n']} | {b['hit_pct']}% {b['ci95']} n={b['n']} | "
               f"{r['hit_diff_pp']} | {a['avg_points']} / {b['avg_points']} | {r['unavailable']} |")
        if extra:
            row += "".join(f" {r.get(h)} |" for h in extra)
        L.append(row)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("selection", "holdout"), required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    reg = yaml.safe_load(PREREG.read_text())
    try:
        method = p3.inference_method(reg)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    commit = prereg_commit()
    df, man = load(reg)
    primary = reg["outcome"]["primary"]
    target = float(primary.split("_")[0][1:])
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"preregistration": str(PREREG), "prereg_commit": commit, "code_commit": _git_commit(),
            "feature_manifest": {k: man[k] for k in ("dataset_fingerprint", "feature_version", "git_commit")},
            "generated_at": datetime.now().isoformat(timespec="seconds"), "primary": primary,
            "strategy_fingerprint": man["strategy"]["fingerprint"],
            "inference": method}

    if a.stage == "selection":
        lo, hi = (str(date.fromisoformat(str(x))) for x in reg["split"]["selection"])
        res = p3.evaluate(df[(df["day"] >= lo) & (df["day"] <= hi)], primary, target)
        for r in res.values():
            r["gate_passed"] = p3.gate(r)
        out = {"meta": meta, "stage": "selection", "range": [lo, hi],
               "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}}
        (OUT / "P3C_v1_selection.json").write_text(json.dumps(out, indent=1, default=str))
        L = [f"# P3C_v1 selection gate ({lo}..{hi})", "", f"Pre-registration commit `{commit}`, code `{meta['code_commit']}`, "
             f"primary outcome {primary}. Gate: predicted sign and >= {p3.MIN_GATE_GROUP} per group. Nothing tuned.", ""]
        L += table(out["results"], extra=["gate_passed"])
        (OUT / "P3C_v1_selection.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    sel_path = OUT / "P3C_v1_selection.json"
    if not sel_path.exists():
        raise SystemExit("run --stage selection first (and commit its report)")
    hold_path = OUT / "P3C_v1_holdout.json"
    if hold_path.exists() and not a.force:
        raise SystemExit(f"holdout already evaluated ({hold_path}); it is evaluated once. --force re-runs and is recorded.")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(sel_path)], capture_output=True).returncode == 0
    sel_dirty = subprocess.run(["git", "status", "--porcelain", "--", str(sel_path)], capture_output=True,
                               text=True).stdout.strip()
    if not tracked or sel_dirty:
        raise SystemExit(f"{sel_path} must be committed and unchanged before the holdout is opened")
    sel = json.loads(sel_path.read_text())
    problems = p3.check_selection_report(sel, reg, commit, man)
    if problems:
        raise SystemExit("selection report does not belong to this pre-registration: " + "; ".join(problems))
    gated = [k for k, v in sel["results"].items() if v["gate_passed"]]
    lo, hi = (str(date.fromisoformat(str(x))) for x in reg["split"]["holdout"])
    res = p3.evaluate(df[(df["day"] >= lo) & (df["day"] <= hi)], primary, target)
    res = {k: v for k, v in res.items() if k in gated}
    # The inference method is part of the pre-registration (validated above).
    testable = {k: v for k, v in res.items() if v["pass"]["n"] and v["fail"]["n"]}
    if method == "day_cluster_bootstrap_ci":
        # Bonferroni-adjusted day-cluster bootstrap intervals; a descriptive directional tail is also reported
        level = 1 - 0.05 / max(1, len(testable))
        for k, v in res.items():
            ci = p3.day_bootstrap_ci(v["_hits"], v["_mask"], v["_days"], level) if k in testable else [None, None]
            v["ci_level"] = round(level, 4)
            v["diff_ci_day_bootstrap"] = ci
            v["bootstrap_tail"] = (p3.day_bootstrap_tail(v["_hits"], v["_mask"], v["_days"], v["predicts"])
                                   if k in testable else None)
            v["verdict"] = p3.verdict_ci(v, ci) if k in testable else "INSUFFICIENT"
        extra = ["ci_level", "diff_ci_day_bootstrap", "bootstrap_tail", "verdict"]
    else:
        pvals = {k: p3.permutation_p(v["_hits"], v["_mask"], v["predicts"]) for k, v in testable.items()}
        adj = p3.holm(pvals) if pvals else {}
        for k, v in res.items():
            v["p_one_sided"] = pvals.get(k)
            v["p_holm"] = adj.get(k)
            v["verdict"] = p3.verdict(v, adj.get(k, 1.0))
        extra = ["p_one_sided", "p_holm", "verdict"]
    out = {"meta": {**meta, "forced_rerun": bool(hold_path.exists() and a.force)}, "stage": "holdout",
           "range": [lo, hi], "gated": gated, "not_gated": [k for k in sel["results"] if k not in gated],
           "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}}
    hold_path.write_text(json.dumps(out, indent=1, default=str))
    L = [f"# P3C_v1 holdout evaluation ({lo}..{hi})", "",
         f"Pre-registration commit `{commit}`, code `{meta['code_commit']}`, primary {primary}. Evaluated once"
         + (" (FORCED RE-RUN)" if out["meta"]["forced_rerun"] else "") + ".",
         f"Gated in selection: {gated or 'none'}; not gated: {out['not_gated'] or 'none'}.", ""]
    L += table(out["results"], extra=extra)
    L += ["", "SUPPORTED = worth re-testing on new data, not a rule to trade. NIFTY points, gross of costs."]
    (OUT / "P3C_v1_holdout.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
