"""Phase 3B: build the research feature dataset (10 indicators at bar, setup and opportunity level).

Writes data/features/features_v<FV>.duckdb (not committed; rebuildable) with tables:
  minutes        per-minute feature table of every session
  bars           every closed 5-minute bar
  setups         every raw S000 setup (requires a replay, ~16 min; skip with --no-setups)
  opportunities  S000 v1 opportunities + labelled outcomes (from reports/baseline/S000_v1.json)
and data/features/manifest_v<FV>.json (provenance: dataset fingerprint, commit, feature version, counts).
Usage: python scripts/build_features.py [--no-setups]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from nifty_engine.features.library import FEATURE_VERSION
from nifty_engine.research import feature_dataset as fd
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.manifest import dataset_fingerprint, require_valid_manifest
from nifty_engine.research.replay import ReplayDriver, _git_commit
from nifty_engine.strategy.registry import load_registry

OUT = Path("data/features")
BASELINE = Path("reports/baseline/S000_v1.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-setups", action="store_true")
    ap.add_argument("--db", default="data/research.duckdb")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    st = ResearchHistoryStore(a.db)
    first, last = st.conn.execute(
        "SELECT min(CAST(ts AS DATE)), max(CAST(ts AS DATE)) FROM hist_bars "
        "WHERE instrument = 'NIFTY_SPOT' AND resolution = '1'").fetchall()[0]
    require_valid_manifest(first, last, st)
    baseline = json.loads(BASELINE.read_text())
    fp = dataset_fingerprint(st)
    strat = load_registry().get(baseline["meta"]["strategy"]["id"])
    try:
        fd.check_baseline_compatible(baseline, fp, strat)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    b = fd.FeatureDatasetBuilder(st)
    b.load()
    tables = b.minute_tables(first, last)
    OUT.mkdir(parents=True, exist_ok=True)
    v = FEATURE_VERSION
    db_path = OUT / f"features_v{v}.duckdb"
    db_path.unlink(missing_ok=True)
    db = duckdb.connect(str(db_path))
    db.execute("SET GLOBAL TimeZone = 'Asia/Kolkata'")

    def save(name, df):
        db.register("_df", df)
        db.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        db.unregister("_df")

    minutes = pd.concat([t.assign(day=d) for d, t in tables.items()]).reset_index()
    save("minutes", minutes)
    bars = fd.bar_level(tables, b.spot_5m_closes())
    save("bars", bars)
    opps = fd.opportunity_level(baseline, tables)
    save("opportunities", opps)
    counts = {"days": len(tables), "minutes": len(minutes), "bars": len(bars), "opportunities": len(opps)}
    if not a.no_setups:
        res = ReplayDriver(st).run(first, last)
        setups = fd.setup_level([s for s in res.signals if s.kind == "SETUP"], tables)
        save("setups", setups)
        counts["setups"] = len(setups)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "feature_version": v,
        "dataset_fingerprint": fp, "git_commit": _git_commit(),
        "strategy": {"id": strat.id, "version": strat.version, "fingerprint": strat.fingerprint()},
        "baseline_report": str(BASELINE), "baseline_commit": baseline["meta"]["git_commit"],
        "range": [str(first), str(last)], "counts": counts,
        "unavailable_share_opportunities": {
            c: round(float(opps[c].isna().mean()), 3) for c in opps.columns if c in fd.FEATURE_COLUMNS},
    }
    db.close()
    (OUT / f"manifest_v{v}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
