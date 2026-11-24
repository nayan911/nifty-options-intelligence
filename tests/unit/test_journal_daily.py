"""Phase 4.3/4.4 end-of-day: premium path from ticks, premium economics, parity checks, outcomes."""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytz

from nifty_engine.journal import daily

IST = pytz.timezone("Asia/Kolkata")
T0 = IST.localize(datetime(2026, 11, 25, 10, 16))


def ticks(rows):
    return pd.DataFrame([{"local_timestamp": T0 + timedelta(seconds=s), "ltp": l, "bid": b, "ask": a}
                         for s, l, b, a in rows])


def test_premium_path_uses_latest_fresh_tick_each_minute():
    t = ticks([(-5, 100.0, 99.5, 100.5), (65, 104.0, 103.5, 104.5), (130, 110.0, 109.0, 111.0)])
    p = daily.premium_path(t, T0, minutes=5)
    assert p.loc[0, "ask"] == 100.5 and p.loc[1, "bid"] == 99.5          # minute 1: last tick is still the -5s one
    assert p.loc[2, "bid"] == 103.5 and p.loc[3, "bid"] == 109.0
    assert pd.isna(p.loc[5, "bid"])                                       # 170 s since the last tick: stale
    zero_bid = daily.premium_path(ticks([(-1, 100.0, 0.0, 100.5)]), T0, minutes=0)
    assert pd.isna(zero_bid.loc[0, "bid"])                                # a zero bid is no quote
    no_ask = daily.premium_path(ticks([(-1, 100.0, 99.0, 0.0)]), T0, minutes=2)
    assert daily.premium_summary(no_ask, {})["status"] == "NO_ENTRY_QUOTE"   # NaN ask is not a price


def test_premium_summary_buys_at_ask_and_values_at_bid():
    t = ticks([(-1, 100.0, 99.0, 101.0), (60, 110.0, 109.0, 111.0), (180, 125.0, 123.0, 126.0),
               (300, 95.0, 94.0, 96.0)])
    path = daily.premium_path(t, T0, minutes=5)
    nifty = {"T30_S20_H60": {"outcome": "TARGET", "exit_at": (T0 + timedelta(minutes=3)).isoformat()}}
    s = daily.premium_summary(path, nifty)
    assert s["entry_ask"] == 101.0 and s["entry_spread"] == 2.0
    assert s["max_gain_at_bid"] == 22.0 and s["max_gain_minute"] == 3        # 123 bid - 101 ask
    assert s["reached_plus15"] and s["reached_plus20"]
    assert s["max_drawdown_at_bid"] == 7.0                                   # 101 - 94
    assert s["at_nifty_exit_T30_S20_H60"] == {"nifty_outcome": "TARGET", "premium_points": 22.0, "minute": 3}
    assert s["at_60m"] == -7.0
    assert daily.premium_summary(daily.premium_path(ticks([]), T0, 3), nifty)["status"] == "NO_ENTRY_QUOTE"


def test_bar_parity_reports_missing_and_mismatched_minutes():
    def bars(rows):
        return pd.DataFrame([{"ts": T0 + timedelta(minutes=i), "open": o, "high": o + 2, "low": o - 2, "close": o + 1}
                             for i, o in rows])
    fyers = bars([(0, 100.0), (1, 101.0), (2, 102.0), (3, 103.0)])
    live = bars([(0, 100.0), (1, 101.04), (3, 108.0)])
    r = daily.bar_parity(live, fyers)
    assert r["common"] == 3 and r["missing_live_count"] == 1 and r["missing_live"] == [(T0 + timedelta(minutes=2)).strftime("%H:%M")]
    assert r["mismatched_count"] == 1 and r["max_abs_diff"] == 5.0            # 0.04 is within tolerance


def test_decision_parity():
    r = daily.decision_parity(["a", "b"], ["b", "c"])
    assert r == {"journal": 2, "replay": 2, "both": 1, "only_live": ["a"], "only_replay": ["c"], "identical": False}
    assert daily.decision_parity(["a"], ["a"])["identical"]


def test_journal_candidates_get_backtest_outcomes_and_render():
    payload = {"bar_at": T0 - timedelta(minutes=1), "decision_at": T0, "entry": 25100.0, "stop": 25088.0,
               "confluence": 70.0}
    ev = [{"kind": "CANDIDATE", "event_at": T0, "symbol": "NSE:NIFTY50-INDEX", "direction": "BULLISH",
           "candidate_id": "C1", "level_id": "PDH", "decision_code": "NET_RR_BELOW_MIN",
           "payload": json.dumps(payload, default=str)},
          {"kind": "CANDIDATE", "event_at": T0, "symbol": "NSE:NIFTY50-INDEX", "direction": "BULLISH",
           "candidate_id": "C2", "level_id": "SWING", "decision_code": "SAME_DIRECTION_OPEN",
           "payload": json.dumps({**payload, "confluence": 80.0}, default=str)},
          {"kind": "SETUP", "event_at": T0, "payload": "{}"}]
    sigs = daily.candidate_signals(ev)
    assert len(sigs) == 2
    bars = pd.DataFrame([{"ts": pd.Timestamp(T0 + timedelta(minutes=i)), "open": 25100 + 2 * i, "high": 25103 + 2 * i,
                          "low": 25099 + 2 * i, "close": 25102 + 2 * i} for i in range(30)])
    ops = daily.nifty_outcomes(sigs, {T0.date(): bars})
    assert len(ops) == 1 and ops[0].rep["candidate_id"] == "C2"             # same bar merged; highest confluence
    assert ops[0].outcomes["T30_S20_H60"]["outcome"] == "TARGET"
    md = daily.render_markdown({"day": "2026-11-25", "strategy": "S000 v1", "live_db": "x", "generated_at": "now",
                                "summary": {"setups": 1, "candidates": 2, "opportunities": 1,
                                            "decision_codes": {"NET_RR_BELOW_MIN": 1}, "paper_fills": 0,
                                            "paper_exits": 0},
                                "decision_parity": {"identical": True, "journal": 2, "replay": 2, "both": 2},
                                "bar_parity": {}, "opportunities": [{
                                    "decision_at": T0.isoformat(), "direction": "BULLISH", "n_candidates": 2,
                                    "level": "SWING", "decision_codes": ["NET_RR_BELOW_MIN", "SAME_DIRECTION_OPEN"],
                                    "nifty": ops[0].outcomes, "contract": None, "premium": None, "features": {}}]})
    assert "Decision parity (replay vs live): IDENTICAL" in md and "T30_S20_H60: TARGET" in md
