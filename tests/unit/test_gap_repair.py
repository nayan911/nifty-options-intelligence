"""Feed-gap repair: detection on spot OR futures, non-blocking reconcile, and the key
correctness property - a repaired engine equals a clean restart on the same verified bars."""

import math
import threading
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.core.interfaces import Candle
from nifty_engine.data.gap_backfill import FeedGapBackfiller
from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.data.storage.duckdb_store import DuckDBStore

IST = pytz.timezone("Asia/Kolkata")
FUT = "NSE:NIFTY26NOVFUT"
DAY = date(2026, 11, 23)


def t(h, m, s=0, d=DAY):
    return IST.localize(datetime(d.year, d.month, d.day, h, m, s))


def px(ts: datetime) -> float:
    """Deterministic 'true' price path."""
    k = ts.timestamp() / 60.0
    return round(25400 + 40 * math.sin(k / 23.0) + 15 * math.sin(k / 7.0), 2)


def bar(tf, ts, close=None, high=None, low=None, vol=2000):
    c = px(ts) if close is None else close
    return Candle(symbol=FUT, timeframe=tf, timestamp=ts, open=c - 1, high=high or c + 4,
                  low=low or c - 4, close=c, volume=vol, oi=100000, is_closed=True)


def series(tf, start, end):
    step = {"1m": 1, "5m": 5, "15m": 15}[tf]
    out, cur = [], start
    while cur <= end:
        out.append(bar(tf, cur))
        cur += timedelta(minutes=step)
    return out


@pytest.fixture
def store(tmp_path):
    s = DuckDBStore(db_path=str(tmp_path / "m.duckdb"))
    fri = date(2026, 11, 20)
    s.insert_candle(Candle(symbol=FUT, timeframe="D", timestamp=t(15, 30, d=fri), open=25300, high=25500,
                           low=25250, close=25420, volume=10**6, is_closed=True))
    hist = series("15m", t(9, 15, d=fri), t(15, 15, d=fri)) + series("5m", t(9, 15, d=fri), t(15, 25, d=fri)) \
        + series("1m", t(13, 0, d=fri), t(15, 29, d=fri))
    # Today's verified bars up to 13:41 (1m), 13:35 (5m), 13:15 (15m)
    today = series("1m", t(9, 15), t(13, 41)) + series("5m", t(9, 15), t(13, 35)) + series("15m", t(9, 15), t(13, 15))
    s.insert_candles_batch(hist + today)
    return s


def new_engine():
    return TechnicalAnalysisEngine(primary_index="NSE:NIFTY50-INDEX", primary_futures=FUT,
                                   on_setup_candidate=MagicMock(), on_technical_candidate=MagicMock())


def snapshot_fingerprint(engine):
    st = engine.states[FUT]
    out = {}
    for tf in ("1m", "5m", "15m"):
        s = st.latest_snapshots.get(tf)
        out[tf] = None if s is None else (
            s.timestamp, s.close, s.ema_9, s.atr, s.vwap, s.rsi, s.trend_bias, s.is_warmed_up,
            tuple(sorted((z.level_type.value, round(z.nominal_price, 2)) for z in s.sr_zones)),
        )
    out["levels"] = dict(st.daily_reference_levels or {})
    out["n1m"] = len(st.candles_1m)
    return out


def test_repaired_engine_matches_clean_restart_after_mid_candle_outage(store, monkeypatch):
    clock = {"now": t(13, 42, 5)}
    monkeypatch.setattr("nifty_engine.data.historical_manager.now_ist", lambda: clock["now"])

    hm = HistoricalCandleManager(rest_client=MagicMock(), store=store)
    live = new_engine()
    hm.warm_up_technical_engine(live, [FUT], as_of_date=DAY)

    # Outage begins at 13:42:44, mid-candle. The builder closes the 13:42 bar at 13:43:00
    # with partial data BEFORE the 15s detector fires, and the engine consumes it.
    partial = bar("1m", t(13, 42), close=px(t(13, 42)) - 30, low=px(t(13, 42)) - 35)
    live.on_candle(partial)
    assert live.states[FUT].latest_snapshots["1m"].close == partial.close
    store.insert_candle(partial)

    # FYERS history after the outage: corrected 13:42 bar + bars through 13:47
    def fake_reconcile(symbols, target_date, timeframes, as_of):
        store.insert_candles_batch(
            series("1m", t(13, 42), t(13, 47)) + [bar("5m", t(13, 40))] + [bar("15m", t(13, 30))]
        )
        for tf in timeframes:
            hm.reconciliation_status[(FUT, tf, target_date)] = True

    monkeypatch.setattr(hm, "reconcile_session_candles", fake_reconcile)
    bf = FeedGapBackfiller(hm, live, store, symbols_provider=lambda: [FUT], gap_seconds=15)

    assert "gap detected" in bf.observe(t(13, 43, 0), {FUT: t(13, 42, 44)}, True).lower()
    assert "resumed" in bf.observe(t(13, 47, 20), {FUT: t(13, 47, 20)}, True).lower()
    clock["now"] = t(13, 48, 6)
    assert "reconciling" in bf.observe(clock["now"], {FUT: clock["now"]}, True).lower()
    bf._worker.join(timeout=5)
    msg = bf.observe(clock["now"], {FUT: clock["now"]}, True)
    assert "repaired" in msg.lower() and not bf.active

    clean = new_engine()
    hm2 = HistoricalCandleManager(rest_client=MagicMock(), store=store)
    hm2.warm_up_technical_engine(clean, [FUT], as_of_date=DAY)

    repaired_fp, clean_fp = snapshot_fingerprint(live), snapshot_fingerprint(clean)
    assert repaired_fp == clean_fp
    # And the partial bar is truly gone from indicator history
    c1342 = [c for c in live.states[FUT].candles_1m if c.timestamp == t(13, 42)]
    assert c1342 and c1342[0].close == px(t(13, 42))
    # No setups were emitted while rebuilding
    live.on_setup_candidate.assert_not_called()


def test_futures_only_stall_opens_gap_while_spot_is_healthy():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: ["SPOT", FUT], gap_seconds=15)
    now = t(11, 0, 30)
    msg = bf.observe(now, {"SPOT": now, FUT: t(11, 0, 0)}, True)
    assert msg and FUT in msg and bf.active
    # Spot alone recovering is not enough: both must be fresh to resume
    assert bf.observe(t(11, 0, 40), {"SPOT": t(11, 0, 40), FUT: t(11, 0, 0)}, True) is None
    assert "resumed" in bf.observe(t(11, 0, 50), {"SPOT": t(11, 0, 50), FUT: t(11, 0, 49)}, True).lower()


def test_reconcile_runs_off_the_maintenance_loop_and_retries_on_failure():
    release = threading.Event()
    hm = MagicMock()
    calls = {"n": 0}

    def slow_reconcile(*a, **k):
        calls["n"] += 1
        release.wait(5)
        if calls["n"] == 1:
            raise RuntimeError("FYERS timeout")

    hm.reconcile_session_candles.side_effect = slow_reconcile
    hm.is_symbol_reconciled.return_value = True
    hm.warm_up_technical_engine.return_value = {FUT: {"session_catchup_candles": 10}}
    engine = MagicMock()
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15, retry_seconds=30)

    bf.observe(t(10, 1), {FUT: t(10, 0, 30)}, True)
    bf.observe(t(10, 2, 10), {FUT: t(10, 2, 10)}, True)
    bf.observe(t(10, 3, 6), {FUT: t(10, 3, 6)}, True)  # worker started
    assert bf.state == bf.RECONCILING

    # Maintenance loop keeps ticking while the broker call hangs
    import time as _t
    start = _t.monotonic()
    for s in range(7, 12):
        assert bf.observe(t(10, 3, s), {FUT: t(10, 3, s)}, True) is None
    assert _t.monotonic() - start < 0.5

    release.set()
    bf._worker.join(timeout=5)
    msg = bf.observe(t(10, 3, 12), {FUT: t(10, 3, 12)}, True)
    assert "failed" in msg and bf.active
    engine.reset_symbols.assert_not_called()  # never rebuild from unverified data

    bf.observe(t(10, 3, 43), {FUT: t(10, 3, 43)}, True)  # retry starts
    bf._worker.join(timeout=5)
    assert "repaired" in bf.observe(t(10, 3, 44), {FUT: t(10, 3, 44)}, True).lower()
    engine.reset_symbols.assert_called_once_with([FUT])


def test_candle_closing_during_rebuild_waits_then_flows_live():
    """A bar that closes while the rebuild runs must not be silently dropped."""
    hm = MagicMock()
    hm.is_symbol_reconciled.return_value = True
    results = {}

    bf = FeedGapBackfiller(hm, MagicMock(), MagicMock(), lambda: [FUT], gap_seconds=15)

    def warm_up(**kwargs):
        th = threading.Thread(target=lambda: results.setdefault("held", bf.hold(bar("1m", t(10, 4)))))
        th.start()
        th.join(timeout=0.2)
        results["blocked_during_rebuild"] = th.is_alive()
        results["thread"] = th
        return {}

    hm.warm_up_technical_engine.side_effect = warm_up
    bf.observe(t(10, 1), {FUT: t(10, 0, 30)}, True)
    bf.observe(t(10, 2, 10), {FUT: t(10, 2, 10)}, True)
    bf.observe(t(10, 3, 6), {FUT: t(10, 3, 6)}, True)
    bf._worker.join(timeout=5)
    bf.observe(t(10, 3, 7), {FUT: t(10, 3, 7)}, True)

    results["thread"].join(timeout=5)
    assert results["blocked_during_rebuild"] is True
    assert results["held"] is False  # released after rebuild -> goes through live path


def test_no_gap_detection_outside_cash_session():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: [FUT], gap_seconds=15)
    assert bf.observe(t(15, 35), {FUT: t(15, 30)}, session_live=False) is None
    assert not bf.active


def test_symbol_never_seen_is_not_treated_as_stale():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: ["SPOT", FUT], gap_seconds=15)
    assert bf.observe(t(9, 20), {"SPOT": t(9, 20), FUT: None}, True) is None
