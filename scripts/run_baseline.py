"""Phase 3A: S000 baseline over the validated history.

Replays S000 (from strategies/registry.yaml) through the live engine, groups same-bar candidates into
opportunities, labels what NIFTY spot did next (+30/+40 before S000's stop or fixed -15/-20, within
45/60 minutes; ambiguous bars never count as wins) and writes:
  reports/baseline/<ID>_v<version>.json   full report incl. every opportunity and its outcomes
  reports/baseline/<ID>_v<version>.md     readable summary

These are NIFTY index outcomes, not option P&L (spread, slippage, delta and decay are not modelled).
Usage: python scripts/run_baseline.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--order tick|flush]
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research import outcomes as oc
from nifty_engine.research.bars import in_session_1m
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.replay import ReplayDriver
from nifty_engine.research.sessions import CASH

OUT_DIR = Path("reports/baseline")


def fmt(s):
    if not s["n"]:
        return "n=0"
    lo, hi = s["hit_rate_ci95"]
    flag = " (small sample)" if s["small_sample"] else ""
    exp = s.get("expectancy_points")
    exp_s = f"; avg {exp:+.1f} pts (se {s['expectancy_se']})" if exp is not None else ""
    return (f"**{s['hit_rate']}%** [{lo}-{hi}] n={s['n']}{flag}; T {s['TARGET']}, S {s['STOP']}, "
            f"amb {s['AMBIGUOUS']}, t/o {s['TIMEOUT']}{exp_s}")


def write_markdown(rep, path):
    m = rep["meta"]
    c = rep["counts"]
    L = [f"# {m['strategy']['id']} v{m['strategy']['version']} baseline (Phase 3A)", "",
         f"Generated {m['generated_at']} | range {m['range'][0]}..{m['range'][1]} | "
         f"{m['days_replayed']} days replayed | delivery order {m['delivery_order']}",
         f"Strategy fingerprint `{m['strategy']['fingerprint']}` | dataset `{m['dataset_fingerprint']}` | "
         f"commit `{m['git_commit']}`", "",
         "> NIFTY spot outcomes only. Not option P&L: spread, slippage, delta and premium decay are not modelled.",
         "", "## Counts", "",
         f"- Candidates: **{c['candidates']}**",
         f"- Opportunities (same-bar candidates merged): **{c['opportunities']}** "
         f"({c['merged_same_bar']} merged) on {c['days_with_opportunity']} days",
         f"- Contract tiers of replayed days: {m['contract_confidence_days']}",
         f"- S000's own stop distance: median {rep['s000_stop_points']['median']} pts "
         f"(10th-90th pct {rep['s000_stop_points']['p10']}-{rep['s000_stop_points']['p90']})", "",
         "## Hit rate: target reached before stop (95% Wilson interval)", "",
         "avg = mean NIFTY points per opportunity (target, minus stop, or timeout close; ambiguous counted "
         "as a stop), gross of all costs; se = its standard error.", "",
         "| Target | Stop | Horizon | All opportunities | One position at a time |", "|---|---|---|---|---|"]
    for k, v in rep["configs"].items():
        t, s, h = k.split("_")
        L.append(f"| +{t[1:]} | {'S000 stop' if s == 'SS000' else '-' + s[1:]} | {h[1:]} min | "
                 f"{fmt(v['all_opportunities'])} | {fmt(v['one_position_at_a_time'])} |")
    for title, key in (("+30, fixed -20 stop, 60 min", "breakdowns_T30_S20_H60"),
                       ("+30, S000 stop, 60 min", "breakdowns_T30_S000_H60")):
        L += ["", f"## Breakdowns: {title} (all opportunities)", ""]
        for dim, groups in rep[key].items():
            L += [f"**{dim}**", "", "| Group | Result | Median min to target | Median MFE / MAE |", "|---|---|---|---|"]
            for g, s in groups.items():
                L.append(f"| {g} | {fmt(s)} | {s['median_minutes_to_target']} | "
                         f"{round(s['median_mfe'], 1)} / {round(s['median_mae'], 1)} |")
            L.append("")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--order", choices=("tick", "flush"), default="tick")
    ap.add_argument("--db", default="data/research.duckdb")
    a = ap.parse_args()
    logging.disable(logging.WARNING)

    st = ResearchHistoryStore(a.db)
    cal = ExchangeCalendar()
    first, last = st.conn.execute(
        "SELECT min(CAST(ts AS DATE)), max(CAST(ts AS DATE)) FROM hist_bars "
        "WHERE instrument = 'NIFTY_SPOT' AND resolution = '1'").fetchall()[0]
    start, end = a.start or first, a.end or last

    drv = ReplayDriver(st, delivery_order=a.order)
    res = drv.run(start, end)
    cands = res.candidates()
    ops = oc.group_opportunities(cands)
    tiers = drv.contract_tiers(res.days)
    for o in ops:
        o.tier = tiers.get(o.day)

    spot = st.conn.execute(
        "SELECT ts, open, high, low, close, volume, oi FROM hist_bars WHERE instrument = 'NIFTY_SPOT' "
        "AND source = 'SPOT' AND resolution = '1' ORDER BY ts").df()
    oc.label_all(ops, oc.bars_by_day(in_session_1m(spot, cal, CASH)))

    rep = oc.report(ops, len(cands))
    prov = res.provenance
    rep["meta"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "range": [start.isoformat(), end.isoformat()], "days_replayed": len(res.days),
        "skipped_days": [[str(d), r] for d, r in res.skipped_days],
        "delivery_order": a.order, "strategy": prov["strategy"], "dataset_fingerprint": prov["dataset_fingerprint"],
        "git_commit": prov["git_commit"], "manifest": prov["manifest"],
        "contract_confidence_days": prov["contract_confidence_days"],
        "definitions": oc.__doc__,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sid, ver = prov["strategy"]["id"], prov["strategy"]["version"]
    jp, mp = OUT_DIR / f"{sid}_v{ver}.json", OUT_DIR / f"{sid}_v{ver}.md"
    jp.write_text(json.dumps(rep, indent=1, default=str))
    write_markdown(rep, mp)
    print(mp.read_text())
    print(f"wrote {jp} and {mp}")


if __name__ == "__main__":
    main()
