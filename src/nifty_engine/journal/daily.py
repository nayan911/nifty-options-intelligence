"""Phase 4.3/4.4: end-of-day processing of the live journal.

Inputs: the live journal (decisions, option ticks) and a validated FYERS history window for the day
(data/daily/history.duckdb, same loader/validation as research, kept separate so the frozen research
dataset and its fingerprints never change).

Outputs per day:
- NIFTY outcomes of every live candidate, with exactly the backtest's rules (outcomes.py: same-bar
  candidates merged, next-1m-open entry, +30/+40 vs S000 stop / -15 / -20, 45/60 min, ambiguous never
  a win) on FYERS's validated 1m bars
- the option premium path of each tracked contract, rebuilt per minute from stored live ticks:
  buy at the ASK at the decision, value at the BID afterwards (what a manual trader really gets)
- the 10 research indicators at each decision (causal library on the validated bars)
- bar parity: live-built 1m bars vs FYERS 1m bars
- decision parity: replay of the day through the live engine vs the candidates journaled live
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from nifty_engine.research.outcomes import config_key, group_opportunities, label_all
from nifty_engine.research.replay import ReplaySignal

PREMIUM_CONFIGS = [(30.0, 20.0, 60), (40.0, 20.0, 60), (30.0, "S000", 60)]
MAX_TICK_AGE_S = 120.0
PRICE_TOL = 0.051  # points: live vs FYERS bar values within a tick size count as equal


def candidate_signals(events: List[Dict[str, Any]]) -> List[ReplaySignal]:
    """Journal CANDIDATE rows -> the ReplaySignal shape the backtest outcome code consumes."""
    out = []
    for e in events:
        if e["kind"] != "CANDIDATE":
            continue
        p = json.loads(e["payload"]) if isinstance(e["payload"], str) else (e["payload"] or {})
        out.append(ReplaySignal("CANDIDATE", pd.Timestamp(e["event_at"]).date(), pd.Timestamp(p["bar_at"]).to_pydatetime(),
                                e["symbol"], "1m", e["direction"], {
                                    "candidate_id": e["candidate_id"], "level_id": e["level_id"],
                                    "decision_at": p["decision_at"], "entry": p["entry"], "stop_loss": p["stop"],
                                    "confluence": p.get("confluence"), "decision_code": e["decision_code"]}))
    return out


def nifty_outcomes(signals: List[ReplaySignal], spot_bars_by_day: Dict[Any, pd.DataFrame]):
    ops = group_opportunities(signals)
    label_all(ops, spot_bars_by_day)
    return ops


def premium_path(ticks: pd.DataFrame, decided_at: datetime, minutes: int = 60) -> pd.DataFrame:
    """Per-minute option quote after a decision from stored ticks (columns local_timestamp, ltp, bid, ask).
    Row k = the latest tick at or before decided_at + k minutes, if no older than MAX_TICK_AGE_S."""
    t = ticks.copy() if not ticks.empty else pd.DataFrame(columns=["local_timestamp", "ltp", "bid", "ask"])
    t["ts"] = pd.to_datetime(t["local_timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    t = t.sort_values("ts")
    rows = []
    for k in range(minutes + 1):
        at = pd.Timestamp(decided_at) + pd.Timedelta(minutes=k)
        prior = t[t["ts"] <= at]
        if prior.empty or (at - prior["ts"].iloc[-1]).total_seconds() > MAX_TICK_AGE_S:
            rows.append({"k": k, "at": at, "bid": None, "ask": None, "ltp": None})
            continue
        r = prior.iloc[-1]
        rows.append({"k": k, "at": at, "bid": _pos(r["bid"]), "ask": _pos(r["ask"]), "ltp": _pos(r["ltp"])})
    return pd.DataFrame(rows)


def _pos(x) -> Optional[float]:
    return float(x) if x is not None and not pd.isna(x) and float(x) > 0 else None


def premium_summary(path: pd.DataFrame, nifty: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Premium economics of a long option bought at the decision (ask) and valued at the bid."""
    if path.empty or pd.isna(path.iloc[0]["ask"]):   # missing quotes are NaN inside the frame
        return {"status": "NO_ENTRY_QUOTE"}
    entry_ask = float(path.iloc[0]["ask"])
    entry_bid = None if pd.isna(path.iloc[0]["bid"]) else float(path.iloc[0]["bid"])
    bids = path.dropna(subset=["bid"])
    gains = bids["bid"] - entry_ask
    out: Dict[str, Any] = {
        "status": "OK", "entry_ask": entry_ask, "entry_bid": entry_bid,
        "entry_spread": round(entry_ask - entry_bid, 2) if entry_bid is not None else None,
        "quoted_minutes": int(len(bids)), "missing_minutes": int(len(path) - len(bids)),
        "max_gain_at_bid": round(float(gains.max()), 2) if len(gains) else None,
        "max_gain_minute": int(bids.loc[gains.idxmax(), "k"]) if len(gains) else None,
        "max_drawdown_at_bid": round(float(-gains.min()), 2) if len(gains) else None,
        "reached_plus15": bool((gains >= 15).any()), "reached_plus20": bool((gains >= 20).any()),
        "at_60m": round(float(bids.iloc[-1]["bid"] - entry_ask), 2) if len(bids) else None,
    }
    for t, s, h in PREMIUM_CONFIGS:
        key = config_key(t, s, h)
        res = nifty.get(key, {})
        if "exit_at" not in res:
            out[f"at_nifty_exit_{key}"] = None
            continue
        exit_at = pd.Timestamp(res["exit_at"])
        after = bids[bids["at"] >= exit_at]
        out[f"at_nifty_exit_{key}"] = ({"nifty_outcome": res["outcome"],
                                        "premium_points": round(float(after.iloc[0]["bid"] - entry_ask), 2),
                                        "minute": int(after.iloc[0]["k"])} if len(after) else
                                       {"nifty_outcome": res["outcome"], "premium_points": None})
    return out


def bar_parity(live: pd.DataFrame, hist: pd.DataFrame) -> Dict[str, Any]:
    """Live-built 1m bars vs FYERS 1m bars for one symbol-day (columns ts, open, high, low, close)."""
    a = live.set_index(pd.to_datetime(live["ts"]))[["open", "high", "low", "close"]]
    b = hist.set_index(pd.to_datetime(hist["ts"]))[["open", "high", "low", "close"]]
    both = a.index.intersection(b.index)
    diff = (a.loc[both] - b.loc[both]).abs()
    mismatched = diff[(diff > PRICE_TOL).any(axis=1)]
    return {
        "fyers_bars": int(len(b)), "live_bars": int(len(a)), "common": int(len(both)),
        "missing_live": [t.strftime("%H:%M") for t in b.index.difference(a.index)][:20],
        "missing_live_count": int(len(b.index.difference(a.index))),
        "extra_live_count": int(len(a.index.difference(b.index))),
        "mismatched_count": int(len(mismatched)),
        "mismatched_examples": [{"at": t.strftime("%H:%M"), **{c: round(float(v), 2) for c, v in r.items()}}
                                for t, r in mismatched.head(5).iterrows()],
        "max_abs_diff": round(float(diff.max().max()), 2) if len(both) else None,
    }


def decision_parity(journal_ids: List[str], replay_ids: List[str]) -> Dict[str, Any]:
    j, r = set(journal_ids), set(replay_ids)
    return {"journal": len(j), "replay": len(r), "both": len(j & r),
            "only_live": sorted(j - r), "only_replay": sorted(r - j), "identical": j == r}


def render_markdown(rep: Dict[str, Any]) -> str:
    d = rep["day"]
    L = [f"# Journal {d}", "",
         f"Strategy {rep['strategy']} | live DB `{rep['live_db']}` | history window manifest "
         f"`{rep.get('history_fingerprint')}` | generated {rep['generated_at']}", "",
         "> Live S000 decisions, what NIFTY did next (backtest rules) and what the option premium did "
         "(bought at the ask, valued at the bid). Gross of brokerage and taxes.", "",
         "## Summary", ""]
    s = rep["summary"]
    L += [f"- Setups (NIFTY spot): {s['setups']}", f"- Candidates: {s['candidates']} -> opportunities: {s['opportunities']}",
          f"- Entry decisions: {s['decision_codes'] or 'none'}",
          f"- Paper fills / exits: {s['paper_fills']} / {s['paper_exits']}"]
    dp = rep.get("decision_parity")
    if dp:
        L.append(f"- Decision parity (replay vs live): {'IDENTICAL' if dp['identical'] else 'DIFFERENT'} "
                 f"(live {dp['journal']}, replay {dp['replay']}, both {dp['both']})")
    for sym, bp in (rep.get("bar_parity") or {}).items():
        L.append(f"- Bar parity {sym}: {bp['common']}/{bp['fyers_bars']} minutes compared, "
                 f"{bp['mismatched_count']} differ, {bp['missing_live_count']} missing live")
    L += ["", "## Opportunities", ""]
    if not rep["opportunities"]:
        L.append("No S000 candidates today.")
    for o in rep["opportunities"]:
        L += [f"### {o['decision_at'][11:16]} {o['direction']} ({o['n_candidates']} candidate(s), level {o['level']})", "",
              f"- Entry decision(s): {', '.join(o['decision_codes'])}",
              "- NIFTY (next 1m open): " + "; ".join(
                  f"{k}: {v.get('outcome')}" + (f" in {v['minutes']}m" if v.get("minutes") is not None else "")
                  for k, v in o["nifty"].items() if k in ("T30_S20_H60", "T40_S20_H60", "T30_SS000_H60"))]
        pm = o.get("premium")
        if pm and pm.get("status") == "OK":
            L.append(f"- Option {o['contract']} ({o['contract_source']}): ask {pm['entry_ask']} / bid {pm['entry_bid']} "
                     f"(spread {pm['entry_spread']}); best +{pm['max_gain_at_bid']} at {pm['max_gain_minute']}m, "
                     f"worst -{pm['max_drawdown_at_bid']}; +15 {'yes' if pm['reached_plus15'] else 'no'}, "
                     f"+20 {'yes' if pm['reached_plus20'] else 'no'}; at 60m {pm['at_60m']:+}")
            x = pm.get("at_nifty_exit_T30_S20_H60")
            if x and x.get("premium_points") is not None:
                L.append(f"- When NIFTY resolved +30/-20 ({x['nifty_outcome']}): premium {x['premium_points']:+} points")
        elif o.get("contract"):
            L.append(f"- Option {o['contract']}: {pm.get('status') if pm else 'no quotes'}")
        f = o.get("features") or {}
        if f:
            L.append("- Indicators: " + ", ".join(f"{k}={v}" for k, v in f.items() if k in (
                "day_type", "rvol_tod_5m", "vwap_z", "or_width_atr", "fut_oi_chg_open_pct", "basis",
                "vix", "days_to_expiry", "hw_up_count")))
        L.append("")
    return "\n".join(L) + "\n"
