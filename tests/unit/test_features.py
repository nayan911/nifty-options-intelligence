"""Phase 2.4 research indicators: no look-ahead (truncation test), engine parity, unavailability."""

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

from nifty_engine.analysis.technical.engine import SymbolTechnicalState
from nifty_engine.core.interfaces import Candle
from nifty_engine.features.library import (
    FEATURE_COLUMNS,
    DayInputs,
    compute_day_features,
    features_at,
    volume_profile,
)

IST = pytz.timezone("Asia/Kolkata")
DAY = date(2026, 6, 10)
OPEN = IST.localize(datetime(2026, 6, 10, 9, 15))


def series(n, base, amp, period, vol=None, oi=None, shift=0.0, start=OPEN):
    rows, prev = [], base
    for i in range(n):
        c = base + amp * math.sin((i + shift) / period) + 0.15 * i
        rows.append({"ts": pd.Timestamp(start + timedelta(minutes=i)), "open": prev, "high": max(prev, c) + 1.5,
                     "low": min(prev, c) - 1.5, "close": c, "volume": (vol(i) if vol else 0),
                     "oi": (oi(i) if oi else None)})
        prev = c
    return pd.DataFrame(rows)


def inputs(n_spot=375, n_fut=385, profiles=10, special=False, with_next=True, dte=6):
    spot = series(n_spot, 25000, 40, 23)
    fut = series(n_fut, 25120, 40, 23, vol=lambda i: 1000 + (i * 53) % 700, oi=lambda i: 10_000_000 + 900 * i)
    nxt = series(n_fut, 25300, 40, 23, vol=lambda i: 200 + (i * 31) % 150, oi=lambda i: 2_000_000 + 400 * i)
    vix = series(n_spot, 13.0, 0.4, 41)
    hws = {s: series(n_spot, 1000 + 100 * k, 6 + k, 17 + 3 * k, shift=k * 7) for k, s in
           enumerate(("HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "BHARTIARTL"))}
    daily = pd.DataFrame({"high": [25100 + 10 * i for i in range(30)], "low": [24900 + 10 * i for i in range(30)],
                          "close": [25000 + 10 * i for i in range(30)]})
    prof = [volume_profile(series(n_fut, 25120, 40, 23, vol=lambda i, k=k: 900 + (i * 41 + k * 13) % 600),
                           series(n_fut, 25300, 40, 23, vol=lambda i, k=k: 150 + (i * 7 + k) % 90))
            for k in range(profiles)]
    return DayInputs(day=DAY, spot=spot, fut=fut, fut_next=nxt if with_next else None, vix=vix, heavyweights=hws,
                     prior_volume_profiles=prof, prior_spot_daily=daily, prev_vix_close=12.5,
                     days_to_expiry=dte, special_session=special)


def _truncate(inp, last_ts):
    def cut(df):
        return None if df is None else df[df["ts"] <= last_ts].reset_index(drop=True)
    return DayInputs(day=inp.day, spot=cut(inp.spot), fut=cut(inp.fut), fut_next=cut(inp.fut_next), vix=cut(inp.vix),
                     heavyweights={k: cut(v) for k, v in inp.heavyweights.items()},
                     prior_volume_profiles=inp.prior_volume_profiles, prior_spot_daily=inp.prior_spot_daily,
                     prev_vix_close=inp.prev_vix_close, days_to_expiry=inp.days_to_expiry,
                     special_session=inp.special_session)


def _same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9))


@pytest.mark.parametrize("minute", [0, 4, 13, 14, 15, 20, 44, 45, 90, 200, 374, 380])
def test_no_lookahead_values_equal_those_computed_from_bars_closed_so_far(minute):
    inp = inputs()
    full = compute_day_features(inp)
    ts = OPEN + timedelta(minutes=minute)
    part = compute_day_features(_truncate(inp, pd.Timestamp(ts)))
    row_full, row_part = full.loc[pd.Timestamp(ts)], part.loc[pd.Timestamp(ts)]
    bad = {c: (row_full[c], row_part[c]) for c in FEATURE_COLUMNS if not _same(row_full[c], row_part[c])}
    assert not bad, f"minute {minute}: future data changed {bad}"


def test_vwap_matches_the_live_engine():
    inp = inputs()
    st = SymbolTechnicalState(symbol="NSE:NIFTY-FUT-NEAR", is_index=False)
    snap = None
    for r in inp.fut.itertuples(index=False):
        snap = st.process_candle(Candle(symbol=st.symbol, timeframe="1m", timestamp=r.ts.to_pydatetime(), open=r.open,
                                        high=r.high, low=r.low, close=r.close, volume=int(r.volume), is_closed=True))
    f = compute_day_features(inp).iloc[-1]
    assert round(f["vwap"], 2) == snap.vwap
    assert round(f["vwap"] + f["vwap_sigma"], 2) == snap.vwap_upper


def test_opening_range_is_unknown_until_complete_and_day_type_staged():
    f = compute_day_features(inputs())
    t = lambda m: pd.Timestamp(OPEN + timedelta(minutes=m))  # noqa: E731
    assert pd.isna(f.loc[t(13), "or_width_atr"]) and "or_width_atr:before_or_complete" in f.loc[t(13), "unavailable"]
    assert f.loc[t(14), "or_width_atr"] > 0                  # the 09:29 bar completes the range at 09:30
    assert f.loc[t(14), "available_at"] == pd.Timestamp(OPEN + timedelta(minutes=15))
    assert f.loc[t(5), "day_type"] == "UNKNOWN"
    assert f.loc[t(40), "day_type_stage"] == "provisional" and f.loc[t(80), "day_type_stage"] == "early"
    assert f.loc[t(200), "day_type_stage"] == "established"
    assert set(f["day_type"]) <= {"UNKNOWN", "TRENDING_UP", "TRENDING_DOWN", "RANGE", "MIXED"}


def test_unavailable_inputs_are_reported_not_guessed():
    f = compute_day_features(inputs(profiles=3, with_next=False, special=True)).iloc[100]
    for col, why in (("rvol_tod_1m", "insufficient_rvol_history"), ("next_oi_chg_open_pct", "next_month_missing"),
                     ("comb_oi_chg_open_pct", "next_month_missing"), ("or_width_atr", "special_session")):
        assert pd.isna(f[col]) and f"{col}:{why}" in f["unavailable"]
    assert f["day_type"] == "UNKNOWN"


def test_values_on_known_inputs():
    inp = inputs(dte=2)
    f = compute_day_features(inp)
    r = f.iloc[60]
    fut, spot = inp.fut.iloc[60], inp.spot.iloc[60]
    assert r["basis"] == pytest.approx(fut["close"] - spot["close"])
    assert r["fut_oi_chg_open_pct"] == pytest.approx(100 * 900 * 60 / 10_000_000)
    comb0, comb = 12_000_000, 10_000_000 + 900 * 60 + 2_000_000 + 400 * 60
    assert r["comb_oi_chg_open_pct"] == pytest.approx(100 * (comb - comb0) / comb0)
    assert r["vix_chg_prev_close_pct"] == pytest.approx(100 * (inp.vix.iloc[60]["close"] / 12.5 - 1))
    assert bool(r["rollover_window"]) and r["days_to_expiry"] == 2 and not bool(r["expiry_day"])
    assert 0 <= r["hw_up_count"] <= 5 and 0 <= r["hw_agree_15m"] <= 5
    # futures minutes after the cash close: basis unknown (no carried-over spot price)
    assert pd.isna(f.iloc[-1]["basis"])
    # time-of-day RVOL: today's near+next volume over the 10-session mean at that minute
    prof = pd.concat([n.add(x, fill_value=0) for n, x in inp.prior_volume_profiles], axis=1).mean(axis=1)
    mod = 9 * 60 + 15 + 60
    assert r["rvol_basis"] == "near_next"
    assert r["rvol_tod_1m"] == pytest.approx((fut["volume"] + inp.fut_next.iloc[60]["volume"]) / prof.loc[mod])
    assert r["vwap_up2"] == pytest.approx(r["vwap"] + 2 * r["vwap_sigma"])
    assert r["vwap_dn1"] == pytest.approx(r["vwap"] - r["vwap_sigma"])
    assert r["hw_available_count"] == 5


def test_features_at_returns_only_rows_already_available():
    f = compute_day_features(inputs())
    t = OPEN + timedelta(minutes=30)
    row = features_at(f, t)
    assert row.name == pd.Timestamp(OPEN + timedelta(minutes=29))   # the 09:44 bar closed at 09:45
    assert features_at(f, OPEN) is None


def test_rvol_is_like_for_like_when_next_month_history_is_incomplete():
    inp = inputs()
    near, _ = inp.prior_volume_profiles[3]
    inp.prior_volume_profiles[3] = (near, None)          # one baseline session lacks next-month data
    r = compute_day_features(inp).iloc[60]
    prof = pd.concat([n for n, _ in inp.prior_volume_profiles], axis=1).mean(axis=1)
    assert r["rvol_basis"] == "near_only"
    assert r["rvol_tod_1m"] == pytest.approx(inp.fut.iloc[60]["volume"] / prof.loc[9 * 60 + 15 + 60])
    today_without_next = compute_day_features(inputs(with_next=False)).iloc[60]
    assert today_without_next["rvol_basis"] == "near_only"


def test_missing_heavyweight_is_unknown_not_disagreement():
    inp = inputs()
    hw = inp.heavyweights["INFY"]
    inp.heavyweights["INFY"] = hw[hw["ts"] >= pd.Timestamp(OPEN + timedelta(minutes=100))].reset_index(drop=True)
    f = compute_day_features(inp)
    early, late = f.iloc[50], f.iloc[200]
    assert early["hw_available_count"] == 4 and pd.isna(early["hw_up_count"]) and pd.isna(early["hw_agree_15m"])
    assert "hw_up_count:heavyweights_incomplete" in early["unavailable"]
    assert late["hw_available_count"] == 5 and not pd.isna(late["hw_up_count"])


def test_missing_next_month_minute_is_unknown_not_zero_volume():
    inp = inputs()
    inp.fut_next = inp.fut_next.drop(index=60).reset_index(drop=True)   # one next-month minute missing today
    f = compute_day_features(inp)
    assert f.iloc[60]["rvol_basis"] == "near_next" and pd.isna(f.iloc[60]["rvol_tod_1m"])
    assert all(pd.isna(f.iloc[i]["rvol_tod_5m"]) for i in range(60, 65))  # every 5m window containing it
    assert not pd.isna(f.iloc[59]["rvol_tod_1m"]) and not pd.isna(f.iloc[65]["rvol_tod_5m"])
    # a gap in ONE baseline session lowers that minute's baseline count instead of adding a zero
    inp2 = inputs()
    near, nxt = inp2.prior_volume_profiles[0]
    inp2.prior_volume_profiles[0] = (near, nxt.drop(index=nxt.index[60]))
    r = compute_day_features(inp2).iloc[60]
    assert r["rvol_baseline_n"] == 9
    prof = pd.concat([n.add(x) for n, x in inp2.prior_volume_profiles[1:]], axis=1).mean(axis=1)
    today = inp2.fut.iloc[60]["volume"] + inp2.fut_next.iloc[60]["volume"]
    assert r["rvol_tod_1m"] == pytest.approx(today / prof.loc[9 * 60 + 15 + 60])


def test_rvol_records_how_many_sessions_back_the_baseline():
    assert compute_day_features(inputs()).iloc[60]["rvol_baseline_n"] == 10
    short = compute_day_features(inputs(profiles=6)).iloc[60]
    assert short["rvol_baseline_n"] == 6 and not pd.isna(short["rvol_tod_1m"])
    thin = compute_day_features(inputs(profiles=4)).iloc[60]
    assert thin["rvol_baseline_n"] == 4 and pd.isna(thin["rvol_tod_1m"])
