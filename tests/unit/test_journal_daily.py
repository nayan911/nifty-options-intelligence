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


def test_entry_is_the_first_executable_ask_after_the_decision_not_an_older_quote():
    t = ticks([(-75, 100.0, 99.5, 100.5), (3, 101.0, 100.8, 101.2), (65, 104.0, 103.5, 104.5)])
    r = daily.premium_analysis(t, T0, {})
    assert r["signal_snapshot"] == {"bid": 99.5, "ask": 100.5, "age_s": 75.0}   # context only
    assert r["status"] == "OK" and r["entry_ask"] == 101.2 and r["entry_delay_s"] == 3.0
    late = daily.premium_analysis(ticks([(-5, 100.0, 99.5, 100.5), (75, 101.0, 100.5, 101.5)]), T0, {})
    assert late["status"] == "NO_EXECUTABLE_ENTRY" and late["signal_snapshot"]["ask"] == 100.5
    assert daily.premium_analysis(ticks([]), T0, {})["status"] == "NO_EXECUTABLE_ENTRY"


def test_targets_use_every_quote_and_horizon_needs_a_fresh_quote():
    # a spike between minute samples (+90 s) reaches +16 and falls back; quotes stop after minute 34
    t = ticks([(1, 100.0, 99.0, 101.0), (90, 117.0, 117.0, 118.0), (120, 103.0, 102.0, 104.0),
               (34 * 60, 105.0, 104.0, 106.0)])
    r = daily.premium_analysis(t, T0, {})
    assert r["first_plus15_after_min"] == 1.5 and r["first_plus20_after_min"] is None
    assert r["max_gain"] == 16.0 and r["max_drawdown"] == 2.0
    assert r["gross_at_horizon"] is None             # the minute-34 quote is not a 60-minute value
    t2 = ticks([(1, 100.0, 99.0, 101.0), (60 * 60, 110.0, 109.0, 111.0)])
    assert daily.premium_analysis(t2, T0, {})["gross_at_horizon"] == 8.0


def test_valuation_at_nifty_exit_with_estimated_net():
    from nifty_engine.analysis.derivatives.cost_model import PremiumCostModel

    t = ticks([(1, 100.0, 99.0, 101.0), (170, 125.0, 123.0, 126.0)])
    nifty = {"T30_S20_H60": {"outcome": "TARGET", "exit_at": (T0 + timedelta(minutes=3)).isoformat()},
             "T40_S20_H60": {"outcome": "STOP", "exit_at": (T0 + timedelta(minutes=30)).isoformat()}}
    r = daily.premium_analysis(t, T0, nifty, lot_size=65, cost_model=PremiumCostModel())
    x = r["at_nifty_exit_T30_S20_H60"]
    assert x["gross_points"] == 22.0 and 0 < x["est_cost_points"] < 2 and x["est_net_points"] < 22.0
    assert r["at_nifty_exit_T40_S20_H60"]["gross_points"] is None   # no fresh quote at that exit


def test_ask_without_any_bid_renders_without_crashing():
    r = daily.premium_analysis(ticks([(1, 100.0, 0.0, 101.0)]), T0, {})
    assert r["status"] == "OK" and r["max_gain"] is None and r["gross_at_horizon"] is None
    md = daily.render_markdown({
        "day": "2026-11-25", "strategy": "S000 v1", "live_db": "x", "generated_at": "now",
        "summary": {"setups": 0, "candidates": 1, "opportunities": 1, "decision_codes": {}, "paper_fills": 0,
                    "paper_exits": 0}, "decision_parity": None, "bar_parity": {},
        "opportunities": [{"decision_at": T0.isoformat(), "direction": "BULLISH", "n_candidates": 1, "level": "L",
                           "decision_codes": ["NET_RR_BELOW_MIN"], "nifty": {}, "contract": "NSE:X",
                           "contract_source": "atm_fallback", "premium": r, "features": {}}]})
    assert "best n/a" in md and "at 60m n/a" in md and "eligible for a paper trade: no" in md


def test_bar_parity_reports_missing_and_mismatched_minutes():
    def bars(rows):
        return pd.DataFrame([{"ts": T0 + timedelta(minutes=i), "open": o, "high": o + 2, "low": o - 2, "close": o + 1}
                             for i, o in rows])
    fyers = bars([(0, 100.0), (1, 101.0), (2, 102.0), (3, 103.0)])
    live = bars([(0, 100.0), (1, 101.04), (3, 108.0)])
    r = daily.bar_parity(live, fyers)
    assert r["common"] == 3 and r["missing_live_count"] == 1 and r["missing_live"] == [(T0 + timedelta(minutes=2)).strftime("%H:%M")]
    assert r["mismatched_count"] == 1 and r["max_abs_diff"] == 5.0            # 0.04 is within tolerance


def test_decision_parity_compares_details_not_just_ids():
    base = {"direction": "BULLISH", "decision_at": T0.isoformat(), "level_id": "PDH", "entry": 25100.0,
            "stop_loss": 25088.0}
    r = daily.decision_parity({"a": base, "b": base}, {"b": base, "c": base})
    assert (r["only_live"], r["only_replay"], r["identical"]) == (["a"], ["c"], False)
    same = daily.decision_parity({"a": base}, {"a": {**base, "decision_at": T0}})   # str vs datetime: same time
    assert same["identical"] and same["detail_mismatches"] == []
    moved = daily.decision_parity({"a": base}, {"a": {**base, "stop_loss": 25080.0}})
    assert not moved["identical"] and moved["detail_mismatches"][0]["differences"] == {"stop_loss": (25088.0, 25080.0)}


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


def _ticks2(rows):
    return pd.DataFrame([{"local_timestamp": T0 + timedelta(seconds=sec), "ltp": a, "bid": b, "ask": a,
                          "event_class": ec, "bid_qty": bq, "ask_qty": aq} for sec, b, a, ec, bq, aq in rows])


def test_only_live_consistent_quotes_are_executable():
    t = _ticks2([(2, 90.0, 91.0, "UNVERIFIED", 50, 50),     # artificially favourable, unverified
                 (4, 92.0, 93.0, "SNAPSHOT", 50, 50),
                 (6, 99.0, 98.0, "LIVE", 50, 50),           # crossed book: bid > ask
                 (8, 99.0, 100.0, "LIVE", 50, 0),           # no ask quantity
                 (10, 100.0, 101.0, "LIVE", 50, 50)])
    r = daily.premium_analysis(t, T0, {})
    assert r["entry_ask"] == 101.0 and r["entry_delay_s"] == 10.0
    assert r["non_live_quotes_excluded"] == 2
    # a favourable non-live quote is never used for valuation either
    t2 = _ticks2([(1, 100.0, 101.0, "LIVE", 50, 50), (60, 140.0, 141.0, "UNVERIFIED", 50, 50)])
    assert daily.premium_analysis(t2, T0, {})["max_gain"] == -1.0


def test_premium_uses_the_representative_candidates_own_contract():
    from nifty_engine.research.outcomes import group_opportunities
    from nifty_engine.research.replay import ReplaySignal

    def sig(cid, conf):
        return ReplaySignal("CANDIDATE", T0.date(), T0 - timedelta(minutes=1), "NSE:NIFTY50-INDEX", "1m", "BULLISH",
                            {"candidate_id": cid, "decision_at": T0.isoformat(), "entry": 25100.0,
                             "stop_loss": 25088.0, "confluence": conf})
    op = group_opportunities([sig("low", 65.0), sig("high", 85.0)])[0]   # higher confluence listed second
    tracking = {"low": {"contract": "NSE:NIFTY26D0125100CE"}, "high": {"contract": "NSE:NIFTY26D0125150CE"}}
    tr, how = daily.tracked_contract(op, tracking)
    assert op.rep["candidate_id"] == "high" and tr["contract"] == "NSE:NIFTY26D0125150CE" and how == "representative"
    tr, how = daily.tracked_contract(op, {"low": tracking["low"]})
    assert tr["contract"] == "NSE:NIFTY26D0125100CE" and how == "other_candidate:low"
    assert daily.tracked_contract(op, {}) == (None, "none")
