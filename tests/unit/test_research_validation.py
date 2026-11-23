"""Phase 1.1: calendar-derived expected minutes, per-day resume, manifest gate, 2-of-3 segmentation."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle
from nifty_engine.research.history_loader import HistoryLoader, SeriesSpec
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.manifest import (
    ManifestError,
    build_manifest,
    require_valid_manifest,
    write_manifest,
)
from nifty_engine.research.segments import apply_segments, detect_switches
from nifty_engine.research.sessions import CASH, FNO, expected_minutes, monthly_expiries

IST = pytz.timezone("Asia/Kolkata")
CAL = ExchangeCalendar()


def bars_for(sym, d, kind=CASH, px=25000.0, oi=None, skip=(), extra_before=0):
    mins = expected_minutes(CAL, d, kind)
    out = []
    for i, t in enumerate(mins):
        if i in skip:
            continue
        out.append(Candle(symbol=sym, timeframe="1m", timestamp=t, open=px, high=px + 1, low=px - 1, close=px,
                          volume=10, oi=oi, is_closed=True))
    for k in range(extra_before):
        t = mins[0] - timedelta(minutes=k + 1)
        out.append(Candle(symbol=sym, timeframe="1m", timestamp=t, open=px, high=px, low=px, close=px, volume=1,
                          is_closed=True))
    return out


# ------------------------------------------------------------------ expected minutes
def test_expected_minutes_follow_calendar_and_dated_fo_close():
    assert len(expected_minutes(CAL, date(2026, 6, 10), CASH)) == 375
    assert len(expected_minutes(CAL, date(2026, 6, 10), FNO)) == 375     # before 31-Jul extension
    assert len(expected_minutes(CAL, date(2026, 10, 15), FNO)) == 385    # after: to 15:39
    assert len(expected_minutes(CAL, date(2026, 11, 8), CASH)) == 70     # Muhurat 14:20-15:29
    assert len(expected_minutes(CAL, date(2026, 11, 8), FNO)) == 80      # to 15:39
    assert len(expected_minutes(CAL, date(2026, 2, 1), FNO)) == 375      # Budget Sunday
    assert expected_minutes(CAL, date(2026, 5, 28), CASH) == []          # Bakri Id holiday


def test_monthly_expiries_move_back_for_holidays():
    exp = monthly_expiries(CAL, date(2026, 1, 1), date(2026, 12, 31))
    assert date(2026, 3, 30) in exp and date(2026, 3, 31) not in exp   # 31-Mar holiday
    assert date(2026, 11, 24) not in exp and date(2026, 11, 23) in exp  # 24-Nov holiday
    assert len(exp) == 12


# ------------------------------------------------------------------ per-day resume
class FakeRest:
    def __init__(self, bars):
        self.bars, self.calls = bars, []

    def get_historical_candles_sync(self, symbol, resolution, date_from, date_to, continuous=True, include_oi=False):
        self.calls.append((date_from, date_to))
        lo, hi = date.fromisoformat(date_from), date.fromisoformat(date_to)
        return [c for c in self.bars if lo <= c.timestamp.date() <= hi]


def _tdays(lo, n):
    return [d for d in (lo + timedelta(days=i) for i in range(n)) if CAL.is_trading_day(d)]


def test_resume_refetches_chunk_with_a_missing_middle_day_or_minute():
    days = _tdays(date(2026, 6, 1), 120)
    sym = "NSE:NIFTY50-INDEX"
    spec = SeriesSpec("NIFTY_SPOT", sym, "SPOT", resolutions=("1",))
    full = [c for d in days for c in bars_for(sym, d)]

    store = ResearchHistoryStore(":memory:")
    loader = HistoryLoader(FakeRest(full), store, calendar=CAL)
    loader.load(spec, days[0], days[-1], "1")
    again = loader.load(spec, days[0], days[-1], "1")
    assert again.skipped_chunks == 1 and again.requests == 1  # complete first chunk skipped, latest refreshed

    # Delete a middle day: endpoint checks would still pass; per-day check must re-fetch
    store._write("DELETE FROM hist_bars WHERE CAST(ts AS DATE) = ?", [days[30]])
    r = loader.load(spec, days[0], days[-1], "1")
    assert r.skipped_chunks == 0 and r.requests == 2

    # Delete ONE minute: also incomplete for a cash series
    store._write("DELETE FROM hist_bars WHERE ts = ?", [expected_minutes(CAL, days[10], CASH)[200]])
    r = loader.load(spec, days[0], days[-1], "1")
    assert r.skipped_chunks == 0


def test_real_contract_tolerates_sparse_minutes_and_skips_pre_listing_chunks():
    days = _tdays(date(2026, 3, 1), 200)
    listed_from = days[100]
    sym = "NSE:NIFTY26DECFUT"
    bars = [c for d in days if d >= listed_from for c in bars_for(sym, d, FNO, oi=1000, skip=(5, 6))]
    store = ResearchHistoryStore(":memory:")
    rest = FakeRest(bars)
    loader = HistoryLoader(rest, store, calendar=CAL)
    spec = SeriesSpec("NIFTY_FUT", sym, "OWN", resolutions=("1",))
    loader.load(spec, days[0], days[-1], "1")
    n_first = len(rest.calls)
    again = loader.load(spec, days[0], days[-1], "1")
    assert again.requests == 1  # pre-listing chunk + sparse-complete chunk skipped; latest refreshed
    assert len(rest.calls) == n_first + 1


# ------------------------------------------------------------------ manifest gate
def _fill(store, inst, src, sym, days, kind, **kw):
    store.upsert_bars(inst, sym, src, "1", [c for d in days for c in bars_for(sym, d, kind, **kw)])


def test_manifest_ok_only_when_required_series_complete(tmp_path):
    days = _tdays(date(2026, 9, 1), 10)
    store = ResearchHistoryStore(":memory:")
    _fill(store, "NIFTY_SPOT", "SPOT", "NSE:NIFTY50-INDEX", days, CASH)
    _fill(store, "NIFTY_FUT", "CONT", "CONT:NIFTY:NEAR", days, FNO, oi=1000)
    _fill(store, "INDIAVIX", "SPOT", "NSE:INDIAVIX-INDEX", days, CASH)
    _fill(store, "BANKNIFTY_SPOT", "SPOT", "NSE:NIFTYBANK-INDEX", days, CASH)
    _fill(store, "BANKNIFTY_FUT", "CONT", "CONT:BANKNIFTY:NEAR", days, FNO, oi=1000)
    m = build_manifest(store, CAL, days[0], days[-1])
    assert m["ok"], m["errors"]

    store._write("DELETE FROM hist_bars WHERE symbol = 'NSE:NIFTY50-INDEX' AND ts = ?",
                 [expected_minutes(CAL, days[3], CASH)[100]])
    m2 = build_manifest(store, CAL, days[0], days[-1])
    assert not m2["ok"] and any("NIFTY_SPOT" in e for e in m2["errors"])

    path = write_manifest(m2, tmp_path / "manifest.json")
    with pytest.raises(ManifestError):
        require_valid_manifest(days[0], days[-1], path)
    write_manifest(m, path)
    assert require_valid_manifest(days[0], days[-1], path)["ok"]
    with pytest.raises(ManifestError):
        require_valid_manifest(days[0] - timedelta(days=30), days[-1], path)  # range not covered


def test_out_of_session_bars_are_warned_not_counted():
    days = _tdays(date(2026, 9, 7), 3)
    store = ResearchHistoryStore(":memory:")
    store.upsert_bars("NIFTY_SPOT", "NSE:NIFTY50-INDEX", "SPOT", "1",
                      [c for d in days for c in bars_for("NSE:NIFTY50-INDEX", d, extra_before=5)])
    m = build_manifest(store, CAL, days[0], days[-1], required=(("NIFTY_SPOT", "SPOT"),))
    assert m["ok"]
    assert any("out-of-session" in w for w in m["warnings"])


# ------------------------------------------------------------------ 2-of-3 segmentation
def _series(store, inst, src, sym, spot_inst, plan):
    """plan: list of (day, fut_px, oi) ; spot fixed at 25000."""
    for d, fut_px, oi in plan:
        store.upsert_bars(inst, sym, src, "1", bars_for(sym, d, FNO, px=fut_px, oi=oi))
        store.upsert_bars(spot_inst, f"SPOT:{spot_inst}", "SPOT", "1", bars_for("x", d, FNO, px=25000.0))


def test_switch_accepted_when_two_signals_agree_even_if_oi_rule_misses():
    store = ResearchHistoryStore(":memory:")
    # Feb 2026 pattern: expiry 24-Feb; OI only 1.6x (OI rule silent), basis -13 -> +170
    plan = [(date(2026, 2, 23), 24990.0, 12_000_000), (date(2026, 2, 24), 24987.0, 8_100_000),
            (date(2026, 2, 25), 25170.0, 13_200_000), (date(2026, 2, 26), 25165.0, 13_000_000)]
    _series(store, "NIFTY_FUT", "CONT", "CONT:NIFTY:NEAR", "NIFTY_SPOT", plan)
    rep = detect_switches(store, CAL, "NIFTY_FUT", "CONT")
    assert [b["day"] for b in rep["accepted"]] == [date(2026, 2, 25)]
    assert rep["accepted"][0]["votes"] == {"calendar": True, "oi": False, "basis": True}
    apply_segments(store, "NIFTY_FUT", "CONT", rep)
    seg = store.conn.execute("SELECT CAST(ts AS DATE), max(segment_id) FROM hist_bars WHERE source='CONT' "
                             "GROUP BY 1 ORDER BY 1").fetchall()
    assert [s for _, s in seg] == [0, 0, 1, 1]


def test_off_calendar_switch_accepted_and_basis_only_move_disputed():
    store = ResearchHistoryStore(":memory:")
    # 25-Sep pattern (no expiry): OI 2.1x and basis 433 -> 156 => switch
    # then 29-May -> 1-Jun pattern: basis move alone => disputed, not a boundary
    plan = [(date(2026, 9, 23), 25433.0, 500_000), (date(2026, 9, 24), 25433.0, 860_000),
            (date(2026, 9, 25), 25156.0, 1_800_000), (date(2026, 9, 28), 25300.0, 1_820_000)]
    _series(store, "NIFTY_FUT", "CONT_NEXT", "CONT:NIFTY:NEXT", "NIFTY_SPOT", plan)
    rep = detect_switches(store, CAL, "NIFTY_FUT", "CONT_NEXT")
    assert [b["day"] for b in rep["accepted"]] == [date(2026, 9, 25)]
    assert [b["day"] for b in rep["disputed"]] == [date(2026, 9, 28)]


def test_continuous_series_store_under_stable_logical_symbols():
    near = SeriesSpec("NIFTY_FUT", "NSE:NIFTY26NOVFUT", "CONT", underlying="NIFTY")
    nxt = SeriesSpec("NIFTY_FUT", "NSE:NIFTY26DECFUT", "CONT_NEXT", underlying="NIFTY")
    later_near = SeriesSpec("NIFTY_FUT", "NSE:NIFTY26DECFUT", "CONT", underlying="NIFTY")
    assert near.store_symbol == later_near.store_symbol == "CONT:NIFTY:NEAR"  # rollover-proof key
    assert nxt.store_symbol == "CONT:NIFTY:NEXT" and nxt.source == "CONT_NEXT"
    own = SeriesSpec("NIFTY_FUT", "NSE:NIFTY26DECFUT", "OWN")
    assert own.store_symbol == "NSE:NIFTY26DECFUT" and own.continuous is False


def test_manifest_json_roundtrip_has_rollover_votes(tmp_path):
    days = _tdays(date(2026, 2, 23), 4)
    store = ResearchHistoryStore(":memory:")
    plan = [(days[0], 24990.0, 12_000_000), (days[1], 24987.0, 8_100_000),
            (days[2], 25170.0, 13_200_000), (days[3], 25165.0, 13_000_000)]
    _series(store, "NIFTY_FUT", "CONT", "CONT:NIFTY:NEAR", "NIFTY_SPOT", plan)
    rep = {"NIFTY_FUT/CONT": detect_switches(store, CAL, "NIFTY_FUT", "CONT")}
    m = build_manifest(store, CAL, days[0], days[-1], required=(), segment_reports=rep)
    path = write_manifest(m, tmp_path / "m.json")
    loaded = pd.read_json(path, typ="series")
    assert loaded["rollovers"]["NIFTY_FUT/CONT"]["accepted"][0]["day"] == "2026-02-25"
    assert isinstance(datetime.fromisoformat(loaded["generated_at"]), datetime)
