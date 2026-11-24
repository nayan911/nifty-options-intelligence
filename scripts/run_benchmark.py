"""Matched random-entry benchmark for the S000 v1 baseline (see research/benchmark.py).

Reads reports/baseline/S000_v1.json, draws matched random entries (same minute of day, same
direction, other days, same stop distance), labels them with the identical rules and writes
reports/baseline/S000_v1_benchmark.{json,md}.
Usage: python scripts/run_benchmark.py [--draws 50] [--resamples 10000]
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research import outcomes as oc
from nifty_engine.research.bars import in_session_1m
from nifty_engine.research.benchmark import benchmark
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.manifest import dataset_fingerprint
from nifty_engine.research.replay import _git_commit
from nifty_engine.research.sessions import CASH

BASE = Path("reports/baseline/S000_v1.json")
CONFIGS = [(30.0, 20.0, 60), (30.0, "S000", 60), (30.0, 15.0, 60), (40.0, 20.0, 60)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--db", default="data/research.duckdb")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    rep = json.loads(BASE.read_text())
    ops = [oc.Opportunity(day=date.fromisoformat(o["day"]), decision_at=pd.Timestamp(o["decision_at"]).to_pydatetime(),
                          direction=o["direction"], candidates=[], rep={"stop_loss": o["s000_stop"]},
                          tier=o["tier"], outcomes=o["outcomes"]) for o in rep["opportunities"]]
    st = ResearchHistoryStore(a.db)
    if dataset_fingerprint(st) != rep["meta"]["dataset_fingerprint"]:
        raise SystemExit("dataset changed since the baseline was generated; regenerate the baseline first")
    spot = st.conn.execute(
        "SELECT ts, open, high, low, close, volume, oi FROM hist_bars WHERE instrument = 'NIFTY_SPOT' "
        "AND source = 'SPOT' AND resolution = '1' ORDER BY ts").df()
    bars = oc.bars_by_day(in_session_1m(spot, ExchangeCalendar(), CASH))
    res = benchmark(ops, bars, CONFIGS, draws=a.draws, resamples=a.resamples)
    res["meta"] = {"generated_at": datetime.now().isoformat(timespec="seconds"), "git_commit": _git_commit(),
                   "baseline_commit": rep["meta"]["git_commit"], "dataset_fingerprint": rep["meta"]["dataset_fingerprint"]}
    out = BASE.with_name("S000_v1_benchmark.json")
    out.write_text(json.dumps(res, indent=1))
    L = ["# S000 v1 vs matched random entries", "",
         f"Generated {res['meta']['generated_at']} | commit `{res['meta']['git_commit']}` | {a.draws} matched draws "
         f"per opportunity (same minute of day, same direction, other days, same stop distance) | "
         f"{a.resamples} resamples", "",
         "p = share of random same-size sets at least as good as S000 (one-sided). Gross NIFTY points.", "",
         "| Config | n | S000 hit | Random hit (5-95% of sets) | p(hit) | S000 avg pts | Random avg pts | p(pts) |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in res["configs"].items():
        lo, hi = v["benchmark_hit_pct_p5_p95"]
        L.append(f"| {k} | {v['n']} | {v['s000_hit_pct']}% | {v['benchmark_hit_pct']}% ({lo}-{hi}) | "
                 f"{v['p_hit_at_least_s000']} | {v['s000_avg_points']:+.2f} | {v['benchmark_avg_points']:+.2f} | "
                 f"{v['p_points_at_least_s000']} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
