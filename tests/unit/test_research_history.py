"""Research history store/loader: chunking, resume, OI flags, contract-switch segmentation."""

from datetime import date, datetime, timedelta

import pytz

from nifty_engine.core.interfaces import Candle
from nifty_engine.research.history_loader import HistoryLoader, SeriesSpec, date_chunks
from nifty_engine.research.history_store import ResearchHistoryStore

IST = pytz.timezone("Asia/Kolkata")


def day_bars(sym, d, n=375, oi_start=None, px=25000.0, oi_step=100):
    t0 = IST.localize(datetime(d.year, d.month, d.day, 9, 15))
    out = []
    for i in range(n):
        out.append(Candle(symbol=sym, timeframe="1m", timestamp=t0 + timedelta(minutes=i), open=px, high=px + 2,
                          low=px - 2, close=px + 1, volume=100,
                          oi=(oi_start + i * oi_step) if oi_start is not None else None, is_closed=True))
    return out


class FakeRest:
    def __init__(self, bars_by_symbol):
        self.bars = bars_by_symbol
        self.calls = []

    def get_historical_candles_sync(self, symbol, resolution, date_from, date_to, continuous=True, include_oi=False):
        self.calls.append(dict(symbol=symbol, resolution=resolution, frm=date_from, to=date_to,
                               continuous=continuous, include_oi=include_oi))
        lo, hi = date.fromisoformat(date_from), date.fromisoformat(date_to)
        return [c for c in self.bars.get(symbol, []) if lo <= c.timestamp.date() <= hi]


def test_date_chunks_respect_cap():
    chunks = list(date_chunks(date(2026, 1, 1), date(2026, 11, 23), 90))
    assert chunks[0] == (date(2026, 1, 1), date(2026, 3, 31))
    assert all((hi - lo).days + 1 <= 90 for lo, hi in chunks)
    assert chunks[-1][1] == date(2026, 11, 23)


def test_series_flags_own_is_unstitched_with_oi_spot_has_no_oi():
    assert SeriesSpec("X", "NSE:NIFTY26DECFUT", "OWN").continuous is False
    assert SeriesSpec("X", "NSE:NIFTY26DECFUT", "OWN").include_oi is True
    assert SeriesSpec("X", "NSE:NIFTY26DECFUT", "CONT").continuous is True
    assert SeriesSpec("X", "NSE:NIFTY50-INDEX", "SPOT").include_oi is False


def test_load_resume_skips_complete_chunks_but_refreshes_latest():
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(200)]
    tdays = [d for d in days if d.weekday() < 5]
    rest = FakeRest({"NSE:NIFTY50-INDEX": [c for d in tdays for c in day_bars("NSE:NIFTY50-INDEX", d, n=3)]})
    store = ResearchHistoryStore(":memory:")
    loader = HistoryLoader(rest, store)
    spec = SeriesSpec("NIFTY_SPOT", "NSE:NIFTY50-INDEX", "SPOT", resolutions=("1",))
    first = loader.load(spec, days[0], days[-1], "1")
    assert first.requests == 3 and first.bars == len(tdays) * 3
    second = loader.load(spec, days[0], days[-1], "1")
    assert second.skipped_chunks == 2 and second.requests == 1  # only the latest chunk re-fetched


def test_cont_segments_split_at_oi_collapse():
    store = ResearchHistoryStore(":memory:")
    d1, d2, d3 = date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 28)
    bars = day_bars("NSE:NIFTY26DECFUT", d1, n=5, oi_start=15_000_000) \
        + day_bars("NSE:NIFTY26DECFUT", d2, n=5, oi_start=15_500_000) \
        + day_bars("NSE:NIFTY26DECFUT", d3, n=5, oi_start=1_280_000)  # switch: OI -92%
    store.upsert_bars("NIFTY_FUT", "NSE:NIFTY26DECFUT", "CONT", "1", bars)
    boundaries = store.assign_cont_segments("NIFTY_FUT", "1")
    assert [b["day"] for b in boundaries] == [d3]
    seg = store.conn.execute(
        "SELECT CAST(ts AS DATE) d, min(segment_id), max(segment_id) FROM hist_bars GROUP BY 1 ORDER BY 1").fetchall()
    assert [(r[1], r[2]) for r in seg] == [(0, 0), (0, 0), (1, 1)]


def test_quality_report_flags_thin_and_missing_days_and_bad_ohlc():
    store = ResearchHistoryStore(":memory:")
    d1, d2, d4 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 4)
    good = day_bars("NSE:NIFTY50-INDEX", d1)
    thin = day_bars("NSE:NIFTY50-INDEX", d2, n=100)
    bad = day_bars("NSE:NIFTY50-INDEX", d4)
    bad[0] = Candle(symbol=bad[0].symbol, timeframe="1m", timestamp=bad[0].timestamp, open=10, high=5, low=1,
                    close=4, volume=1, is_closed=True)
    store.upsert_bars("NIFTY_SPOT", "NSE:NIFTY50-INDEX", "SPOT", "1", good + thin + bad)
    issues = store.quality_report(trading_days=[d1, d2, date(2026, 6, 3), d4])
    kinds = {(i.kind, i.day) for i in issues}
    assert ("THIN_DAY", d2) in kinds and ("MISSING_DAY", date(2026, 6, 3)) in kinds and ("BAD_OHLC", d4) in kinds


def test_history_client_parses_oi_only_when_requested(monkeypatch):
    from unittest.mock import MagicMock

    from nifty_engine.data.fyers.rest_client import FyersRestClient

    auth = MagicMock()
    client = MagicMock()
    auth.get_client.return_value = client
    rc = FyersRestClient(auth_provider=auth, rate_limiter=MagicMock())
    ts = int(IST.localize(datetime(2026, 6, 1, 9, 15)).timestamp())
    client.history.return_value = {"s": "ok", "candles": [[ts, 1, 2, 0.5, 1.5, 100, 5_000_000]]}
    c = rc.get_historical_candles_sync("NSE:NIFTY26DECFUT", "1", "2026-06-01", "2026-06-01",
                                       continuous=False, include_oi=True)
    sent = client.history.call_args.kwargs["data"]
    assert sent["cont_flag"] == "0" and sent["oi_flag"] == "1" and c[0].oi == 5_000_000
    c2 = rc.get_historical_candles_sync("NSE:NIFTY26DECFUT", "1", "2026-06-01", "2026-06-01")
    sent2 = client.history.call_args.kwargs["data"]
    assert sent2["cont_flag"] == "1" and "oi_flag" not in sent2 and c2[0].oi is None  # live default unchanged
