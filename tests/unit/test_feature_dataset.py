"""Feature dataset: bar-level rows follow the live builder's emitted 5m bars; outcomes are only
joined to features from the same dataset and strategy version."""

from datetime import date, datetime

import pandas as pd
import pytest
import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.features.library import FEATURE_COLUMNS
from nifty_engine.research.bars import build_intraday_candles
from nifty_engine.research.feature_dataset import bar_level, check_baseline_compatible
from nifty_engine.research.sessions import CASH, expected_minutes
from nifty_engine.strategy.registry import load_registry

IST = pytz.timezone("Asia/Kolkata")
CAL = ExchangeCalendar()


def _spot(day, drop=()):
    mins = [m for i, m in enumerate(expected_minutes(CAL, day, CASH)) if i not in drop]
    return pd.DataFrame([{"ts": pd.Timestamp(m), "open": 25000.0, "high": 25005.0, "low": 24995.0, "close": 25001.0,
                          "volume": 0, "oi": None} for m in mins])


def test_bar_level_uses_bars_the_live_builder_emits():
    day = date(2026, 6, 10)
    df = _spot(day, drop=(45,))  # 10:00 missing -> the 10:00-10:05 bar starts late and is dropped live
    closes = [c.observed_at for c in build_intraday_candles(df, CAL, CASH, "NSE:NIFTY50-INDEX")["5m"]]
    t0 = pd.Timestamp(IST.localize(datetime(2026, 6, 10, 9, 15)))
    idx = [t0 + pd.Timedelta(minutes=i) for i in range(375)]
    table = pd.DataFrame({"available_at": [t + pd.Timedelta(minutes=1) for t in idx], "unavailable": ""}, index=idx)
    for c in FEATURE_COLUMNS:
        table[c] = 1.0
    rows = bar_level({day: table}, closes)
    times = set(rows["observed_at"])
    assert pd.Timestamp(IST.localize(datetime(2026, 6, 10, 10, 5))) not in times
    assert pd.Timestamp(IST.localize(datetime(2026, 6, 10, 10, 10))) in times
    assert len(rows) == 74  # 75 regular 5m bars minus the dropped one
    muhurat = [c.observed_at for c in build_intraday_candles(_spot(date(2026, 11, 8)), CAL, CASH, "X")["5m"]]
    assert min(muhurat).strftime("%H:%M") == "14:25" and len(muhurat) == 14


def test_outcomes_only_join_features_from_the_same_data_and_strategy():
    s = load_registry().get("S000")
    good = {"meta": {"dataset_fingerprint": "abc",
                     "strategy": {"id": "S000", "version": s.version, "fingerprint": s.fingerprint()}}}
    check_baseline_compatible(good, "abc", s)
    with pytest.raises(ValueError, match="dataset"):
        check_baseline_compatible(good, "changed", s)
    stale = {"meta": {**good["meta"], "strategy": {**good["meta"]["strategy"], "fingerprint": "old"}}}
    with pytest.raises(ValueError, match="strategy"):
        check_baseline_compatible(stale, "abc", s)
