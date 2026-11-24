"""Phase 3A outcome labelling: entry, target/stop order, ambiguity, timeouts, grouping, Wilson CI."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytz

from nifty_engine.research.outcomes import (
    group_opportunities,
    label,
    one_position_at_a_time,
    summarize,
    wilson,
)
from nifty_engine.research.replay import ReplaySignal

IST = pytz.timezone("Asia/Kolkata")
D = date(2026, 6, 10)
T0 = IST.localize(datetime(2026, 6, 10, 10, 0))


def bars(path):
    """path: list of (open, high, low, close) per minute starting 10:00."""
    return pd.DataFrame([{"ts": pd.Timestamp(T0 + timedelta(minutes=i)), "open": o, "high": h, "low": l, "close": c}
                         for i, (o, h, l, c) in enumerate(path)])


def cand(direction="BULLISH", stop=25080.0, decision=T0, conf=65.0, cid="a"):
    return ReplaySignal("CANDIDATE", D, decision - timedelta(minutes=1), "NSE:NIFTY50-INDEX", "1m", direction,
                        {"candidate_id": cid, "decision_at": decision.isoformat(), "entry": 25100.0,
                         "stop_loss": stop, "confluence": conf, "level_id": cid})


def op_for(c):
    return group_opportunities([c])[0]


def test_entry_is_the_next_bar_open_and_target_first_wins():
    b = bars([(25100, 25110, 25095, 25108), (25108, 25131, 25105, 25130)])
    r = label(op_for(cand()), b, 30, 20.0, 60)
    assert r["entry"] == 25100 and r["outcome"] == "TARGET" and r["mfe"] == 31
    assert r["minutes"] == 2 and r["exit_at"] == (T0 + timedelta(minutes=2)).isoformat()  # end of the hit bar


def test_stop_first_and_bearish_direction():
    b = bars([(25100, 25104, 25090, 25092), (25092, 25125, 25060, 25070)])
    assert label(op_for(cand()), b, 30, 15.0, 60)["outcome"] == "STOP"       # bar 2: +25 < 30 but -40
    assert label(op_for(cand()), b, 20, 15.0, 60)["outcome"] == "AMBIGUOUS"  # both inside bar 2
    assert label(op_for(cand()), b, 30, 10.0, 60)["outcome"] == "STOP"       # -10 hit in bar 1
    bear = bars([(25100, 25102, 25069, 25071)])
    r = label(op_for(cand("BEARISH", stop=25120.0)), bear, 30, "S000", 60)
    assert r["outcome"] == "TARGET" and r["stop_points"] == 20.0


def test_ambiguous_bar_is_never_a_win():
    b = bars([(25100, 25135, 25075, 25110)])
    r = label(op_for(cand()), b, 30, 20.0, 60)
    assert r["outcome"] == "AMBIGUOUS"
    s = summarize([_with(op_for(cand()), "K", r)], "K")
    assert s["TARGET"] == 0 and s["hit_rate"] == 0.0


def test_timeout_respects_horizon_and_session_end():
    b = bars([(25100, 25110, 25095, 25105)] * 50 + [(25105, 25140, 25100, 25138)])
    r = label(op_for(cand()), b, 30, 20.0, 45)
    assert r["outcome"] == "TIMEOUT" and r["close_points"] == 5
    assert r["minutes"] == 45 and r["exit_at"] == (T0 + timedelta(minutes=45)).isoformat()  # 10:44 bar closes 10:45
    assert label(op_for(cand()), b, 30, 20.0, 60)["outcome"] == "TARGET"
    short = bars([(25100, 25110, 25095, 25105)] * 3)
    assert label(op_for(cand()), short, 30, 20.0, 60)["outcome"] == "TIMEOUT"
    assert label(op_for(cand(decision=T0 + timedelta(minutes=10))), short, 30, 20.0, 60)["outcome"] == "NO_ENTRY"


def test_open_beyond_s000_stop_is_no_trade_and_never_adds_points():
    b = bars([(25075, 25140, 25070, 25130)])
    r = label(op_for(cand(stop=25080.0)), b, 30, "S000", 60)
    assert r["outcome"] == "INVALID_AT_OPEN" and r["stop_points"] <= 0
    k = "T30_SS000_H60"
    s = summarize([_with(op_for(cand(cid="x", stop=25080.0)), k, r)], k)
    assert s["invalid_at_open"] == 1 and s["n"] == 0 and s["expectancy_points"] is None
    import pytest

    from nifty_engine.research.outcomes import points
    with pytest.raises(ValueError):
        points({"outcome": "STOP", "stop_points": -5.0}, 30.0)  # a stopped trade can never add points


def test_same_bar_candidates_are_one_opportunity_with_highest_confluence_rep():
    ops = group_opportunities([cand(conf=65, cid="b", stop=25090), cand(conf=80, cid="a", stop=25070),
                               cand("BEARISH", cid="c"), cand(decision=T0 + timedelta(minutes=1), cid="d")])
    assert len(ops) == 3
    first = [o for o in ops if o.direction == "BULLISH" and o.decision_at == T0][0]
    assert len(first.candidates) == 2 and first.rep["candidate_id"] == "a" and first.rep["stop_loss"] == 25070


def _with(op, key, res):
    op.outcomes[key] = res
    return op


def test_one_position_at_a_time_skips_overlapping_entries():
    a = _with(op_for(cand(cid="a")), "K", {"outcome": "STOP", "entry_at": T0.isoformat(),
                                           "exit_at": (T0 + timedelta(minutes=5)).isoformat(), "minutes": 5})
    b = _with(op_for(cand(decision=T0 + timedelta(minutes=3), cid="b")), "K",
              {"outcome": "TARGET", "entry_at": (T0 + timedelta(minutes=3)).isoformat(),
               "exit_at": (T0 + timedelta(minutes=9)).isoformat(), "minutes": 6})
    c = _with(op_for(cand(decision=T0 + timedelta(minutes=6), cid="c")), "K",
              {"outcome": "TARGET", "entry_at": (T0 + timedelta(minutes=6)).isoformat(),
               "exit_at": (T0 + timedelta(minutes=8)).isoformat(), "minutes": 2})
    assert [o.rep["candidate_id"] for o in one_position_at_a_time([a, b, c], "K")] == ["a", "c"]


def test_wilson_interval():
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(50, 100)
    assert lo == 40.4 and hi == 59.6
    lo, hi = wilson(9, 10)
    assert lo < 90 < hi and hi <= 100


def test_expectancy_counts_ambiguous_as_a_stop_and_timeouts_at_close():
    k = "T30_S20_H60"
    ops = [_with(op_for(cand(cid=c)), k, r) for c, r in (
        ("a", {"outcome": "TARGET", "minutes": 3, "mfe": 31, "mae": 2, "stop_points": 20.0}),
        ("b", {"outcome": "STOP", "minutes": 3, "mfe": 1, "mae": 20, "stop_points": 20.0}),
        ("c", {"outcome": "AMBIGUOUS", "minutes": 1, "mfe": 30, "mae": 20, "stop_points": 20.0}),
        ("d", {"outcome": "TIMEOUT", "minutes": 59, "mfe": 9, "mae": 5, "stop_points": 20.0, "close_points": 6.0}))]
    s = summarize(ops, k)
    assert s["expectancy_points"] == round((30 - 20 - 20 + 6) / 4, 2) and s["TARGET"] == 1 and s["hit_rate"] == 25.0


def test_exit_at_bar_end_lets_the_next_open_enter():
    a = _with(op_for(cand(cid="a")), "K", {"outcome": "TARGET", "entry_at": T0.isoformat(),
                                           "exit_at": (T0 + timedelta(minutes=3)).isoformat(), "minutes": 3})
    b = _with(op_for(cand(decision=T0 + timedelta(minutes=3), cid="b")), "K",
              {"outcome": "STOP", "entry_at": (T0 + timedelta(minutes=3)).isoformat(),
               "exit_at": (T0 + timedelta(minutes=5)).isoformat(), "minutes": 2})
    assert [o.rep["candidate_id"] for o in one_position_at_a_time([a, b], "K")] == ["a", "b"]


def test_invalid_at_open_neither_crashes_nor_blocks_the_next_trade():
    k = "T30_SS000_H60"
    invalid = op_for(cand(stop=25080.0, cid="bad"))
    invalid.outcomes[k] = label(invalid, bars([(25075, 25140, 25070, 25130)]), 30, "S000", 60)
    assert invalid.outcomes[k]["outcome"] == "INVALID_AT_OPEN" and "entry_at" in invalid.outcomes[k]
    valid = _with(op_for(cand(decision=T0, cid="ok")), k, {
        "outcome": "TARGET", "entry_at": T0.isoformat(), "exit_at": (T0 + timedelta(minutes=4)).isoformat(),
        "minutes": 4, "mfe": 31, "mae": 2, "stop_points": 12.0})
    taken = one_position_at_a_time([invalid, valid], k)
    assert [o.rep["candidate_id"] for o in taken] == ["ok"]
    s = summarize([invalid, valid], k)
    assert s["n"] == 1 and s["TARGET"] == 1 and s["invalid_at_open"] == 1


def test_matched_benchmark_uses_same_minute_direction_and_rules():
    from nifty_engine.research.benchmark import benchmark

    up = [(25100, 25110, 25098, 25108), (25108, 25135, 25105, 25132)]      # +30 reached in minute 2
    down = [(25100, 25102, 25085, 25086), (25086, 25088, 25070, 25072)]    # -20 stop first
    days = [date(2026, 6, 10), date(2026, 6, 11), date(2026, 6, 12)]

    def day_bars(d, path):
        t0 = IST.localize(datetime(d.year, d.month, d.day, 10, 0))
        return pd.DataFrame([{"ts": pd.Timestamp(t0 + timedelta(minutes=i)), "open": o, "high": h, "low": lo,
                              "close": c} for i, (o, h, lo, c) in enumerate(path)])

    op = op_for(cand())
    op.outcomes["T30_S20_H60"] = label(op, bars(up), 30, 20.0, 60)
    same = {d: day_bars(d, up) for d in days}
    r = benchmark([op], same, [(30.0, 20.0, 60)], draws=20, resamples=200)["configs"]["T30_S20_H60"]
    assert r["s000_hit_pct"] == 100.0 and r["benchmark_hit_pct"] == 100.0 and r["p_hit_at_least_s000"] == 1.0
    other = {days[0]: day_bars(days[0], up), days[1]: day_bars(days[1], down), days[2]: day_bars(days[2], down)}
    r = benchmark([op], other, [(30.0, 20.0, 60)], draws=20, resamples=200)["configs"]["T30_S20_H60"]
    assert r["benchmark_hit_pct"] == 0.0 and r["p_hit_at_least_s000"] == 0.0   # the op's own day is never drawn
    assert r["s000_avg_points"] == 30.0 and r["benchmark_avg_points"] == -20.0


def test_day_block_bootstrap_pairs_s000_with_its_random_draws_and_resamples_days():
    from nifty_engine.research.benchmark import day_block_bootstrap

    rows = [{"day": date(2026, 6, d), "s000_hit": True, "s000_points": 30.0,
             "random": [(True, 30.0), (False, -20.0)]} for d in (10, 11, 12)]
    r = day_block_bootstrap(rows, resamples=200)
    assert r["hit_diff_pp"] == 50.0 and r["points_diff"] == 25.0 and r["p_hit_diff_le_0"] == 0.0
    assert r["days"] == 3 and r["opportunities"] == 3
    worse = [{**x, "s000_hit": False, "s000_points": -20.0} for x in rows]
    assert day_block_bootstrap(worse, resamples=200)["p_hit_diff_le_0"] == 1.0


def test_same_day_benchmark_never_draws_the_opportunity_minute_or_another_day():
    from nifty_engine.research.benchmark import same_day_benchmark

    path = [(25100, 25110, 25098, 25108)] * 30
    op = op_for(cand())
    b = {D: bars(path)}
    op.outcomes["T30_S20_H60"] = label(op, b[D], 30, 20.0, 60)
    out = same_day_benchmark([op], b, 30.0, 20.0, 60, draws=40)
    assert out["key"] == "T30_S20_H60" and len(out["rows"]) == 1 and len(out["rows"][0]["random"]) > 0
    assert all(not h for h, _ in out["rows"][0]["random"])  # flat day: random same-day entries never hit +30
