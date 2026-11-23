"""Phase 2: session-aware bars, live-store bulk insert, replay determinism and live-path equivalence."""

import math
from datetime import date, datetime, timedelta

import pandas as pd
import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.research.bars import in_session_1m, resample, trading_days
from nifty_engine.research.equivalence import compare_day
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.replay import ReplayDriver, signals_digest
from nifty_engine.research.sessions import CASH, FNO, expected_minutes

IST = pytz.timezone("Asia/Kolkata")
CAL = ExchangeCalendar()


# ------------------------------------------------------------------ bars
def _rows(day, kind, n_pre=0):
    mins = expected_minutes(CAL, day, kind)
    pre = [mins[0] - timedelta(minutes=k) for k in range(n_pre, 0, -1)]
    out = []
    for i, t in enumerate(pre + mins):
        out.append({"ts": pd.Timestamp(t), "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i,
                    "volume": 10, "oi": 1000 + i})
    return pd.DataFrame(out)


def test_pre_open_bars_are_excluded_and_buckets_are_clock_aligned():
    day = date(2026, 9, 8)
    df = _rows(day, CASH, n_pre=7)
    clean = in_session_1m(df, CAL, CASH)
    assert len(clean) == 375 and clean["ts"].iloc[0].strftime("%H:%M") == "09:15"
    b5 = resample(clean, "5m")
    assert len(b5) == 75 and [t.strftime("%H:%M") for t in b5["ts"][:3]] == ["09:15", "09:20", "09:25"]
    b15 = resample(clean, "15m")
    assert len(b15) == 25 and b15["ts"].iloc[1].strftime("%H:%M") == "09:30"
    first5 = b5.iloc[0]
    assert (first5.open, first5.high, first5.low, first5.close, first5.volume, first5.oi) == (
        clean.iloc[0].open, clean.iloc[:5].high.max(), clean.iloc[:5].low.min(), clean.iloc[4].close, 50,
        clean.iloc[4].oi)


def test_futures_closing_window_after_extension():
    df = _rows(date(2026, 10, 15), FNO)
    b5 = resample(in_session_1m(df, CAL, FNO), "5m")
    assert len(b5) == 77 and b5["ts"].iloc[-1].strftime("%H:%M") == "15:35"


# ------------------------------------------------------------------ live store bulk insert
def test_live_store_bulk_insert_keeps_upsert_semantics():
    s = DuckDBStore(db_path=":memory:")
    t0 = IST.localize(datetime(2026, 11, 20, 9, 15))
    mk = lambda px, oi=None, obs=None: Candle(symbol="X", timeframe="1m", timestamp=t0, open=px, high=px,  # noqa: E731
                                              low=px, close=px, volume=1, oi=oi, is_closed=True, observed_at=obs)
    s.insert_candles_batch([mk(1.0, oi=5, obs=t0 + timedelta(minutes=1)), mk(2.0, oi=7)])  # last wins
    s.insert_candles_batch([mk(3.0, oi=9, obs=t0 + timedelta(minutes=1))])                  # replaces stored
    r = s.get_candles("X", "1m", t0, t0)
    assert len(r) == 1 and r[0].close == 3.0 and r[0].oi == 9 and r[0].timestamp == t0
    assert r[0].observed_at == t0 + timedelta(minutes=1)


# ------------------------------------------------------------------ replay on a synthetic market
def _synthetic_store(days):
    """Deterministic market with real candle bodies: each bar opens at the previous close, trends
    and pullbacks cross round-number levels, and futures volume rises with the size of the move."""
    store = ResearchHistoryStore(":memory:")
    spot_prev = 25000.0
    for k, d in enumerate(days):
        spot_mins = expected_minutes(CAL, d, CASH)
        fut_mins = expected_minutes(CAL, d, FNO)
        path = []
        for i in range(len(fut_mins)):
            x = k * 400 + i
            drift = 2.2 * math.sin(x / 37.0) + 1.1 * math.sin(x / 5.0) + (0.6 if (k % 2 == 0) else -0.6)
            spot_prev += drift
            path.append(spot_prev)
        for inst, src, sym, mins, basis in (("NIFTY_SPOT", "SPOT", "NSE:NIFTY50-INDEX", spot_mins, 0.0),
                                            ("NIFTY_FUT", "CONT", "CONT:NIFTY:NEAR", fut_mins, 120.0)):
            bars, prev = [], path[0] + basis
            for i, t in enumerate(mins):
                close = path[i] + basis
                move = abs(close - prev)
                vol = 0 if src == "SPOT" else int(300 + 900 * move)
                bars.append(Candle(symbol=sym, timeframe="1m", timestamp=t, open=prev, high=max(prev, close) + 1.5,
                                   low=min(prev, close) - 1.5, close=close, volume=vol,
                                   oi=(None if src == "SPOT" else 10_000_000 + (k * 400 + i) * 50), is_closed=True))
                prev = close
            store.upsert_bars(inst, sym, src, "1", bars)
            day_ts = IST.localize(datetime(d.year, d.month, d.day))
            hi, lo = max(b.high for b in bars), min(b.low for b in bars)
            store.upsert_bars(inst, sym, src, "D", [Candle(symbol=sym, timeframe="D", timestamp=day_ts,
                                                           open=bars[0].open, high=hi, low=lo, close=bars[-1].close,
                                                           volume=sum(b.volume for b in bars), is_closed=True)])
    return store


def test_replay_is_deterministic_and_matches_the_live_runner_path():
    days = trading_days(CAL, date(2026, 6, 1), date(2026, 6, 19))
    store = _synthetic_store(days)
    r1 = ReplayDriver(store).run(days[-2], days[-1], warmup_days=10)
    r2 = ReplayDriver(store).run(days[-2], days[-1], warmup_days=10)
    assert r1.days == days[-2:]
    assert signals_digest(r1.signals) == signals_digest(r2.signals)
    assert sum(s.kind == "SETUP" for s in r1.signals) > 20  # the synthetic market really produces breakouts
    assert r1.provenance["dataset_fingerprint"] and r1.provenance["engine"].startswith("TechnicalAnalysisEngine")

    ok, a, b, n = compare_day(ReplayDriver(store), days[-1])
    assert ok, (a, b)


def test_days_without_enough_warmup_history_are_skipped_and_reported():
    days = trading_days(CAL, date(2026, 6, 1), date(2026, 6, 10))
    store = _synthetic_store(days)
    res = ReplayDriver(store).run(days[0], days[-1], warmup_days=5)
    assert [d for d, _ in res.skipped_days] == days[:5]
    assert res.days == days[5:]
