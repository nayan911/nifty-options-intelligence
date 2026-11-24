"""Phase 3C evaluation mechanics (hypothesis masks, gate, permutation p, Holm, verdicts)."""

import pandas as pd

from nifty_engine.research import phase3c as p3


def _df(rows):
    base = {"day_type": "MIXED", "rvol_tod_5m": 0.5, "days_to_expiry": 9, "hw_up_count": 2,
            "stop_T30_S20_H60": 20.0, "close_T30_S20_H60": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_masks_follow_the_preregistered_rules():
    df = _df([{"direction": "BULLISH", "day_type": "TRENDING_UP", "rvol_tod_5m": 1.0, "days_to_expiry": 2, "hw_up_count": 4},
              {"direction": "BEARISH", "day_type": "TRENDING_UP", "rvol_tod_5m": 0.99, "days_to_expiry": 3, "hw_up_count": 1},
              {"direction": "BEARISH", "day_type": "UNKNOWN", "rvol_tod_5m": float("nan"), "hw_up_count": float("nan")}])
    m = p3.masks(df)
    assert m["H1_day_type_aligned"].tolist()[:2] == [True, False] and pd.isna(m["H1_day_type_aligned"][2])
    assert m["H2_rvol_above_normal"].tolist()[:2] == [True, False] and pd.isna(m["H2_rvol_above_normal"][2])
    assert m["H3_near_expiry"].tolist()[:2] == [True, False]
    assert m["H4_heavyweights_confirm"].tolist()[:2] == [True, True] and pd.isna(m["H4_heavyweights_confirm"][2])


def test_evaluate_gate_permutation_holm_and_verdicts():
    rows = []
    for i in range(24):
        aligned = i < 12
        hit = (i % 4 != 0) if aligned else (i % 4 == 0)
        rows.append({"direction": "BULLISH", "day_type": "TRENDING_UP" if aligned else "RANGE",
                     "out_T30_S20_H60": "TARGET" if hit else "STOP"})
    res = p3.evaluate(_df(rows), "T30_S20_H60", 30.0)
    h1 = res["H1_day_type_aligned"]
    assert h1["pass"]["hit_pct"] == 75.0 and h1["fail"]["hit_pct"] == 25.0 and h1["hit_diff_pp"] == 50.0
    assert h1["pass"]["avg_points"] == round((9 * 30 - 3 * 20) / 12, 2)
    assert p3.gate(h1)
    p = p3.permutation_p(h1["_hits"], h1["_mask"], "higher", n=2000)
    assert p < 0.05
    assert p3.permutation_p(h1["_hits"], h1["_mask"], "lower", n=2000) > 0.9
    adj = p3.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert p3.verdict(h1, 0.01) == "CONFIRMED" and p3.verdict(h1, 0.2) == "SUPPORTED"
    small = {**h1, "pass": {**h1["pass"], "n": 4}}
    assert p3.verdict(small, 0.01) == "INSUFFICIENT"
    assert not p3.gate({**h1, "predicts": "lower"})


def test_preregistration_file_parses_and_matches_the_code():
    from pathlib import Path

    import yaml

    reg = yaml.safe_load((Path(__file__).resolve().parents[2] / "research/preregistration/P3C_v1.yaml").read_text())
    assert set(reg["hypotheses"]) == set(p3.PREDICTS)
    assert {k: v["predicts"] for k, v in reg["hypotheses"].items()} == p3.PREDICTS
    assert reg["outcome"]["primary"] == "T30_S20_H60" and len(reg["procedure"]) == 5


def _reg():
    return {"outcome": {"primary": "T30_S20_H60"}, "dataset_fingerprint": "ds", "feature_version": 3,
            "strategy": {"fingerprint": "sf"}, "split": {"selection": ["2025-12-29", "2026-08-31"]}}


def _sel(**meta):
    m = {"prereg_commit": "abc", "primary": "T30_S20_H60", "strategy_fingerprint": "sf",
         "feature_manifest": {"dataset_fingerprint": "ds", "feature_version": 3}}
    m.update(meta)
    return {"stage": "selection", "range": ["2025-12-29", "2026-08-31"], "meta": m}


def test_holdout_only_accepts_a_selection_report_from_the_same_preregistration():
    man = {"dataset_fingerprint": "ds"}
    assert p3.check_selection_report(_sel(), _reg(), "abc", man) == []
    assert any("pre-registration" in x for x in p3.check_selection_report(_sel(), _reg(), "zzz", man))
    assert any("primary" in x for x in p3.check_selection_report(_sel(primary="T40_S20_H60"), _reg(), "abc", man))
    assert any("strategy" in x for x in p3.check_selection_report(_sel(strategy_fingerprint="old"), _reg(), "abc", man))
    assert any("current" in x for x in p3.check_selection_report(_sel(), _reg(), "abc", {"dataset_fingerprint": "new"}))
    other_range = {**_sel(), "range": ["2026-01-01", "2026-08-31"]}
    assert any("range" in x for x in p3.check_selection_report(other_range, _reg(), "abc", man))


def test_day_block_p_respects_same_day_dependence():
    # strong effect spread over many days -> small p
    hits, mask, days = [], [], []
    for d in range(20):
        hits += [True, False]
        mask += [True, False]
        days += [f"d{d}", f"d{d}"]
    assert p3.day_block_p(hits, mask, days, "higher", n=2000) < 0.01
    # no effect on 20 ordinary days, plus ONE day contributing 20 correlated "pass" wins: opportunity-level
    # permutation treats those 8 as independent evidence; resampling whole days does not
    hits, mask, days = [], [], []
    for d in range(20):
        hits += [d % 2 == 0, d % 2 == 1]
        mask += [True, False]
        days += [f"d{d}", f"d{d}"]
    hits += [True] * 20
    mask += [True] * 20
    days += ["X"] * 20
    p_perm = p3.permutation_p(hits, mask, "higher", n=4000)
    p_day = p3.day_block_p(hits, mask, days, "higher", n=4000)
    assert p_perm < 0.05 and p_day > 0.15  # 'significant' only if same-day trades were independent
