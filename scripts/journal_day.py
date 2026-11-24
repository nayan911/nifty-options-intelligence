"""End-of-day live journal report (Phase 4.3/4.4). Run after the market-data pipeline has stopped.

  1. --refresh: download a validated FYERS history window for the day into data/daily/ (separate from
     the frozen research dataset) with scripts/build_history.py (needs today's FYERS login)
  2. NIFTY outcomes of every live candidate (backtest rules, on FYERS's validated bars)
  3. option premium path of each tracked contract from stored live ticks (buy at ask, value at bid)
  4. the 10 research indicators at each decision
  5. bar parity (live-built vs FYERS 1m bars) and decision parity (replay vs live candidates)
Writes journal/YYYY/MM/DD.md and DD.json.

Usage: python scripts/journal_day.py [--day YYYY-MM-DD] [--refresh]
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.features.library import FEATURE_COLUMNS, compute_day_features, features_at
from nifty_engine.journal import daily
from nifty_engine.research.bars import in_session_1m
from nifty_engine.research.feature_dataset import FeatureDatasetBuilder
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.manifest import dataset_fingerprint
from nifty_engine.research.outcomes import bars_by_day
from nifty_engine.research.replay import ReplayDriver
from nifty_engine.research.sessions import CASH
from nifty_engine.strategy.registry import load_registry
from nifty_engine.utils.timezone import now_ist

DAILY_DIR = Path("data/daily")
OUT_DIR = Path("journal")
WINDOW_DAYS = 30


def refresh(day: date) -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/build_history.py", "--start", (day - timedelta(days=WINDOW_DAYS)).isoformat(),
           "--end", day.isoformat(), "--db", str(DAILY_DIR / "history.duckdb"),
           "--manifest", str(DAILY_DIR / "manifest.json")]
    print("refreshing history window:", " ".join(cmd[1:]))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"history refresh failed (exit {rc}); is today's FYERS login done?")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", type=date.fromisoformat, default=now_ist().date())
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--live-db", default="data/market.duckdb")
    ap.add_argument("--history-db", default=str(DAILY_DIR / "history.duckdb"))
    ap.add_argument("--history-manifest", default=str(DAILY_DIR / "manifest.json"))
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    d = a.day
    if a.refresh:
        refresh(d)
    hist_path = Path(a.history_db)
    if not hist_path.exists():
        raise SystemExit("no history window for the day: run with --refresh (needs today's FYERS login)")
    try:
        live = duckdb.connect(a.live_db, read_only=True)
    except duckdb.IOException as e:
        raise SystemExit(f"cannot open {a.live_db} read-only (is the pipeline still running?): {e}")
    live.execute("SET TimeZone = 'Asia/Kolkata'")
    has = {r[0] for r in live.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    events = []
    tracking = []
    if "journal_events" in has:
        cur = live.execute("SELECT * FROM journal_events WHERE day = ? ORDER BY event_at", [d])
        cols = [c[0] for c in cur.description]
        events = [dict(zip(cols, r)) for r in cur.fetchall()]
    if "journal_tracking" in has:
        cur = live.execute("SELECT * FROM journal_tracking WHERE day = ?", [d])
        cols = [c[0] for c in cur.description]
        tracking = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

    hist = ResearchHistoryStore(hist_path)
    cal = ExchangeCalendar()
    spot = hist.conn.execute("SELECT ts, open, high, low, close, volume, oi FROM hist_bars WHERE instrument = 'NIFTY_SPOT' "
                             "AND source = 'SPOT' AND resolution = '1' ORDER BY ts").df()
    spot_days = bars_by_day(in_session_1m(spot, cal, CASH))
    if d not in spot_days:
        raise SystemExit(f"the history window has no NIFTY bars for {d}")

    # NIFTY outcomes (backtest rules)
    signals = daily.candidate_signals(events)
    ops = daily.nifty_outcomes(signals, spot_days)

    # indicators at each decision
    fb = FeatureDatasetBuilder(hist, cal)
    fb.load()
    inp = fb.day_inputs(d, sorted(fb._by_day["spot"]))
    ftable = compute_day_features(inp) if inp is not None else None

    report_ops = []
    for op in ops:
        cids = [c["candidate_id"] for c in op.candidates]
        tr = next((tracking[c] for c in cids if c in tracking), None)
        prem = None
        if tr is not None:
            ticks = live.execute("SELECT local_timestamp, ltp, bid, ask FROM ticks WHERE symbol = ? AND "
                                 "local_timestamp BETWEEN ? AND ? ORDER BY local_timestamp",
                                 [tr["contract"], pd.Timestamp(tr["decided_at"]) - pd.Timedelta(minutes=2),
                                  tr["track_until"]]).df()
            prem = daily.premium_summary(daily.premium_path(ticks, tr["decided_at"]), op.outcomes)
        frow = features_at(ftable, op.decision_at) if ftable is not None else None
        report_ops.append({
            "decision_at": op.decision_at.isoformat(), "direction": op.direction, "n_candidates": len(cids),
            "candidate_ids": cids, "level": op.rep.get("level_id"),
            "decision_codes": sorted({c.get("decision_code") for c in op.candidates}),
            "nifty": op.outcomes, "contract": tr["contract"] if tr else None,
            "contract_source": tr["source"] if tr else None, "premium": prem,
            "features": ({c: (None if pd.isna(frow[c]) else (round(float(frow[c]), 3) if isinstance(frow[c], float)
                                                             else frow[c])) for c in FEATURE_COLUMNS}
                         if frow is not None else None),
        })

    # bar parity
    parity = {}
    lb = live.execute("SELECT symbol, timestamp AS ts, open, high, low, close FROM candles WHERE timeframe = '1m' "
                      "AND CAST(timestamp AS DATE) = ? AND (symbol = 'NSE:NIFTY50-INDEX' OR symbol LIKE 'NSE:NIFTY%FUT')",
                      [d]).df()
    hist_fut = hist.conn.execute("SELECT ts, open, high, low, close FROM hist_bars WHERE instrument = 'NIFTY_FUT' AND "
                                 "source = 'CONT' AND symbol = 'CONT:NIFTY:NEAR' AND resolution = '1' "
                                 "AND CAST(ts AS DATE) = ? ORDER BY ts", [d]).df()
    if not lb.empty:
        parity["NSE:NIFTY50-INDEX"] = daily.bar_parity(lb[lb["symbol"] == "NSE:NIFTY50-INDEX"], spot_days[d])
        futs = {s: daily.bar_parity(g, hist_fut) for s, g in lb[lb["symbol"].str.endswith("FUT")].groupby("symbol")}
        if futs:
            near = min(futs, key=lambda s: futs[s]["mismatched_count"] - futs[s]["common"])
            parity[f"{near} (near month)"] = futs[near]

    # decision parity
    rep_dp = None
    try:
        rd = ReplayDriver(hist).run(d, d, manifest_path=Path(a.history_manifest))
        rep_dp = daily.decision_parity([e["candidate_id"] for e in events if e["kind"] == "CANDIDATE"],
                                       [s.detail["candidate_id"] for s in rd.candidates()])
    except Exception as e:
        rep_dp = {"error": str(e), "identical": False, "journal": None, "replay": None, "both": None}

    s000 = load_registry().get("S000")
    codes = {}
    for e in events:
        if e["kind"] == "CANDIDATE":
            codes[e["decision_code"]] = codes.get(e["decision_code"], 0) + 1
    rep = {
        "day": d.isoformat(), "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": f"{s000.id} v{s000.version} {s000.fingerprint()}", "live_db": a.live_db,
        "history_fingerprint": dataset_fingerprint(hist),
        "summary": {"setups": sum(e["kind"] == "SETUP" for e in events),
                    "candidates": sum(e["kind"] == "CANDIDATE" for e in events), "opportunities": len(ops),
                    "decision_codes": codes, "paper_fills": sum(e["kind"] == "PAPER_FILL" for e in events),
                    "paper_exits": sum(e["kind"] == "PAPER_EXIT" for e in events)},
        "decision_parity": rep_dp if "error" not in rep_dp else None,
        "decision_parity_error": rep_dp.get("error"),
        "bar_parity": parity, "opportunities": report_ops,
    }
    out = OUT_DIR / f"{d:%Y}" / f"{d:%m}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{d:%d}.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / f"{d:%d}.md").write_text(daily.render_markdown(rep))
    print((out / f"{d:%d}.md").read_text())


if __name__ == "__main__":
    main()
