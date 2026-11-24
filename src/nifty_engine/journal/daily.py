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


ENTRY_MAX_DELAY_S = 60.0     # an executable entry quote must arrive within this long after the decision
EXIT_MAX_AGE_S = 120.0       # an exit/valuation quote must be at most this old at the time it is needed


def _quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """Usable quotes only: event_class LIVE (UNVERIFIED/SNAPSHOT ticks are never executable or used for
    valuation), positive prices, and a consistent book (a crossed quote with bid > ask is dropped).
    An ask with a recorded ask quantity of 0 is not executable."""
    cols = ["local_timestamp", "ltp", "bid", "ask", "event_class", "bid_qty", "ask_qty"]
    t = ticks.copy() if not ticks.empty else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in t:
            t[c] = None
    t["ts"] = pd.to_datetime(t["local_timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    t = t[t["event_class"] == "LIVE"].copy()   # fail closed: a missing classification is not LIVE
    for c in ("bid", "ask", "ltp", "bid_qty", "ask_qty"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ("bid", "ask", "ltp"):
        t.loc[t[c] <= 0, c] = float("nan")   # a zero/negative price is no quote
    crossed = t["bid"].notna() & t["ask"].notna() & (t["bid"] > t["ask"])
    t.loc[crossed, ["bid", "ask"]] = float("nan")
    t.loc[t["ask_qty"] == 0, "ask"] = float("nan")
    t.loc[t["bid_qty"] == 0, "bid"] = float("nan")
    return t.sort_values("ts").reset_index(drop=True)


def _quote_at(q: pd.DataFrame, at: pd.Timestamp, col: str,
              not_before: Optional[pd.Timestamp] = None) -> Optional[Dict[str, Any]]:
    """Latest valid `col` quote at or before `at`, if no older than EXIT_MAX_AGE_S and (when given) recorded
    at or after `not_before` - a position can only be valued with quotes from after it was opened."""
    prior = q[(q["ts"] <= at) & q[col].notna()]
    if not_before is not None:
        prior = prior[prior["ts"] >= not_before]
    if prior.empty or (at - prior["ts"].iloc[-1]).total_seconds() > EXIT_MAX_AGE_S:
        return None
    return {"price": float(prior[col].iloc[-1]), "at": prior["ts"].iloc[-1]}


def premium_analysis(ticks: pd.DataFrame, decided_at: datetime, nifty: Dict[str, Dict[str, Any]],
                     horizon_min: int = 60, lot_size: Optional[int] = None,
                     cost_model: Optional[Any] = None) -> Dict[str, Any]:
    """Option premium economics of a long option for one decision, kept in four separate layers:
      signal_snapshot  latest quote AT the decision, with its age (context only, not tradable)
      entry            first valid ASK at/after the decision within ENTRY_MAX_DELAY_S (executable paper
                       entry); without it there is no paper P&L ("NO_EXECUTABLE_ENTRY")
      gross            bid-side value against the entry ask, over EVERY recorded quote (not minute samples);
                       a valuation exists only if a fresh quote (<= EXIT_MAX_AGE_S) exists at that time
      net_estimate     gross minus statutory costs and brokerage for one lot (when a cost model is given)
    """
    q = _quotes(ticks)
    dec = pd.Timestamp(decided_at)
    excluded = 0 if ticks.empty else (int((ticks["event_class"] != "LIVE").sum()) if "event_class" in ticks
                                      else len(ticks))
    snap = q[(q["ts"] <= dec) & (q["ask"].notna() | q["bid"].notna())]
    out: Dict[str, Any] = {"signal_snapshot": None if snap.empty else {
        "bid": None if pd.isna(snap["bid"].iloc[-1]) else float(snap["bid"].iloc[-1]),
        "ask": None if pd.isna(snap["ask"].iloc[-1]) else float(snap["ask"].iloc[-1]),
        "age_s": round((dec - snap["ts"].iloc[-1]).total_seconds(), 1)},
        "non_live_quotes_excluded": excluded}
    entry = q[(q["ts"] >= dec) & q["ask"].notna()]
    if entry.empty or (entry["ts"].iloc[0] - dec).total_seconds() > ENTRY_MAX_DELAY_S:
        out["status"] = "NO_EXECUTABLE_ENTRY"
        return out
    e = entry.iloc[0]
    entry_at, entry_ask = e["ts"], float(e["ask"])
    out.update({"status": "OK", "entry_at": entry_at.isoformat(), "entry_delay_s": round((entry_at - dec).total_seconds(), 1),
                "entry_ask": entry_ask, "entry_bid": None if pd.isna(e["bid"]) else float(e["bid"])})
    out["entry_spread"] = round(entry_ask - out["entry_bid"], 2) if out["entry_bid"] is not None else None
    end = entry_at + pd.Timedelta(minutes=horizon_min)
    path = q[(q["ts"] >= entry_at) & (q["ts"] <= end) & q["bid"].notna()]
    gains = path["bid"] - entry_ask
    out["bid_quotes"] = int(len(path))
    if len(path):
        i_max, i_min = gains.idxmax(), gains.idxmin()
        out["max_gain"] = round(float(gains[i_max]), 2)
        out["max_gain_after_min"] = round((path.loc[i_max, "ts"] - entry_at).total_seconds() / 60, 1)
        out["max_drawdown"] = round(float(-gains[i_min]), 2)
        for x in (15, 20):
            hit = path[gains >= x]
            out[f"first_plus{x}_after_min"] = (round((hit["ts"].iloc[0] - entry_at).total_seconds() / 60, 1)
                                               if len(hit) else None)
    else:
        out.update({"max_gain": None, "max_gain_after_min": None, "max_drawdown": None,
                    "first_plus15_after_min": None, "first_plus20_after_min": None})
    q60 = _quote_at(q, end, "bid", not_before=entry_at)
    out["gross_at_horizon"] = round(q60["price"] - entry_ask, 2) if q60 else None
    for t, s, h in PREMIUM_CONFIGS:
        key = config_key(t, s, h)
        res = nifty.get(key, {})
        if "exit_at" not in res:
            out[f"at_nifty_exit_{key}"] = None
            continue
        exit_at = pd.Timestamp(res["exit_at"])
        if exit_at < entry_at:   # NIFTY resolved before the option could be bought: nothing to value
            out[f"at_nifty_exit_{key}"] = {"nifty_outcome": res["outcome"], "gross_points": None,
                                           "unavailable": "nifty_exit_before_option_entry"}
            continue
        qx = _quote_at(q, exit_at, "bid", not_before=entry_at)
        gross = round(qx["price"] - entry_ask, 2) if qx else None
        rec = {"nifty_outcome": res["outcome"], "gross_points": gross}
        if qx is None:
            rec["unavailable"] = "no_fresh_bid_after_entry"
        if gross is not None and cost_model is not None and lot_size:
            costs = (cost_model.compute_fill_costs("BUY", entry_ask, lot_size)["total_costs"]
                     + cost_model.compute_fill_costs("SELL", qx["price"], lot_size)["total_costs"])
            rec["est_cost_points"] = round(costs / lot_size, 2)
            rec["est_net_points"] = round(gross - costs / lot_size, 2)
        out[f"at_nifty_exit_{key}"] = rec
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


DETAIL_FIELDS = ("direction", "decision_at", "level_id", "entry", "stop_loss")


def decision_parity(journal: Dict[str, Dict[str, Any]], replay: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """journal/replay: candidate_id -> decision details. Same ids AND same direction, decision time,
    level, entry and stop are required for parity."""
    j, r = set(journal), set(replay)
    mismatched = []
    for cid in sorted(j & r):
        diffs = {f: (journal[cid].get(f), replay[cid].get(f)) for f in DETAIL_FIELDS
                 if not _same(journal[cid].get(f), replay[cid].get(f))}
        if diffs:
            mismatched.append({"candidate_id": cid, "differences": diffs})
    return {"journal": len(j), "replay": len(r), "both": len(j & r), "only_live": sorted(j - r),
            "only_replay": sorted(r - j), "detail_mismatches": mismatched,
            "identical": j == r and not mismatched}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    try:
        return pd.Timestamp(a) == pd.Timestamp(b)
    except (ValueError, TypeError):
        return str(a) == str(b)


def _fmt(x, signed=False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:+}" if signed else f"{x}"


def _premium_lines(o: Dict[str, Any], pm: Optional[Dict[str, Any]]) -> List[str]:
    if not o.get("contract"):
        return ["- Option: none tracked"]
    if not pm:
        return [f"- Option {o['contract']}: no quotes"]
    snap = pm.get("signal_snapshot")
    snap_s = (f"signal snapshot bid {_fmt(snap['bid'])} / ask {_fmt(snap['ask'])} ({snap['age_s']}s old)"
              if snap else "no quote before the signal")
    match = o.get("contract_match")
    match_s = "" if match in (None, "representative") else f" [FALLBACK: {match}, not the representative's contract]"
    lines = [f"- Option {o['contract']} ({o.get('contract_source')}){match_s}: {snap_s}"]
    if pm.get("status") != "OK":
        return lines + [f"- Paper entry: {pm.get('status')} (no executable ask within {ENTRY_MAX_DELAY_S:.0f}s)"]
    lines.append(f"- Executable entry: ask {pm['entry_ask']} after {pm['entry_delay_s']}s "
                 f"(bid {_fmt(pm['entry_bid'])}, spread {_fmt(pm['entry_spread'])}); eligible for a paper trade: "
                 f"{'yes' if 'ACCEPTED' in o.get('decision_codes', []) else 'no'}")
    lines.append(f"- Gross at bid ({pm['bid_quotes']} quotes): best {_fmt(pm['max_gain'], True)} "
                 f"after {_fmt(pm['max_gain_after_min'])}m, worst {_fmt(-pm['max_drawdown'] if pm['max_drawdown'] is not None else None, True)}; "
                 f"+15 after {_fmt(pm['first_plus15_after_min'])}m, +20 after {_fmt(pm['first_plus20_after_min'])}m; "
                 f"at 60m {_fmt(pm['gross_at_horizon'], True)}")
    x = pm.get("at_nifty_exit_T30_S20_H60")
    if x:
        net = f", est. net {_fmt(x.get('est_net_points'), True)}" if "est_net_points" in x else ""
        lines.append(f"- When NIFTY resolved +30/-20 ({x['nifty_outcome']}): gross {_fmt(x['gross_points'], True)}{net} "
                     "(quote-based estimate, not a proven fill)")
    return lines


def render_markdown(rep: Dict[str, Any]) -> str:
    d = rep["day"]
    L = [f"# Journal {d}", "",
         f"Strategy {rep['strategy']} | live DB `{rep['live_db']}` | history window manifest "
         f"`{rep.get('history_fingerprint')}` | generated {rep['generated_at']}", "",
         "> Four separate layers: what NIFTY did (backtest rules); what the tracked option's quotes did "
         "(gross, bought at the first executable ask after the decision, valued at the bid); whether a "
         "paper trade was eligible (entry decision code); and an estimated net after statutory costs.", "",
         "## Summary", ""]
    s = rep["summary"]
    L += [f"- Setups (NIFTY spot): {s['setups']}", f"- Candidates: {s['candidates']} -> opportunities: {s['opportunities']}",
          f"- Entry decisions: {s['decision_codes'] or 'none'}",
          f"- Paper fills / exits: {s['paper_fills']} / {s['paper_exits']}"]
    dp = rep.get("decision_parity")
    if dp:
        L.append(f"- Decision parity (replay vs live): {'IDENTICAL' if dp['identical'] else 'DIFFERENT'} "
                 f"(live {dp['journal']}, replay {dp['replay']}, both {dp['both']}, "
                 f"detail mismatches {len(dp.get('detail_mismatches', []))})")
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
        L += _premium_lines(o, pm)
        f = o.get("features") or {}
        if f:
            L.append("- Indicators: " + ", ".join(f"{k}={v}" for k, v in f.items() if k in (
                "day_type", "rvol_tod_5m", "vwap_z", "or_width_atr", "fut_oi_chg_open_pct", "basis",
                "vix", "days_to_expiry", "hw_up_count")))
        L.append("")
    return "\n".join(L) + "\n"


def tracked_contract(op: Any, tracking: Dict[str, Dict[str, Any]]) -> "tuple":
    """The tracked option for an opportunity: the REPRESENTATIVE candidate's own contract (the one whose
    stop and level the NIFTY outcome uses). Falls back to another merged candidate's contract only if the
    representative has none, and says so."""
    rep = op.rep.get("candidate_id")
    if rep in tracking:
        return tracking[rep], "representative"
    for c in op.candidates:
        if c["candidate_id"] in tracking:
            return tracking[c["candidate_id"]], f"other_candidate:{c['candidate_id']}"
    return None, "none"
