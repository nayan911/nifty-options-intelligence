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
from nifty_engine.data.gap_backfill import FeedGapBackfiller, affected_intervals
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
        broker = {
            "1m": series("1m", t(9, 15), t(13, 47)),
            "5m": series("5m", t(9, 15), t(13, 40)),
            "15m": series("15m", t(9, 15), t(13, 30)),
        }
        for tf in timeframes:
            store.insert_candles_batch(broker[tf])
            hm.last_broker_returned[(FUT, tf, target_date)] = {c.timestamp for c in broker[tf]}
            hm.reconciliation_status[(FUT, tf, target_date)] = True

    monkeypatch.setattr(hm, "reconcile_session_candles", fake_reconcile)
    bf = FeedGapBackfiller(hm, live, store, symbols_provider=lambda: [FUT], gap_seconds=15)

    assert "gap detected" in bf.observe(t(13, 43, 0), {FUT: t(13, 42, 44)}, True).lower()
    assert "live again" in bf.observe(t(13, 47, 20), {FUT: t(13, 47, 20)}, True).lower()
    clock["now"] = t(13, 48, 6)
    assert "repairing" in bf.observe(clock["now"], {FUT: clock["now"]}, True).lower()
    # Real DuckDB + full warmup in the worker: ~3s locally, ~7.5s under CI coverage tracing.
    # Wait generously; a genuine hang still fails loudly here.
    bf._worker.join(timeout=60)
    assert not bf._worker.is_alive(), "repair worker hung"
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


class _Everything:
    def __contains__(self, item):
        return True


class _AllReturned(dict):
    """Mock broker that returned every requested bar."""

    def get(self, key, default=None):
        return _Everything()


def mock_hm(**kw):
    hm = MagicMock()
    hm.is_symbol_reconciled.return_value = True
    hm.last_broker_returned = _AllReturned()
    hm.warm_up_technical_engine.return_value = {}
    for k, v in kw.items():
        setattr(hm, k, v)
    return hm


def drive_to_repair(bf, sym=FUT):
    bf.observe(t(10, 1), {sym: t(10, 0, 30)}, True)       # gap
    bf.observe(t(10, 2, 10), {sym: t(10, 2, 10)}, True)   # resumed -> 10:03:05
    return bf.observe(t(10, 3, 6), {sym: t(10, 3, 6)}, True)  # worker started


def test_futures_only_stall_opens_gap_while_spot_is_healthy():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: ["SPOT", FUT], gap_seconds=15)
    now = t(11, 0, 30)
    msg = bf.observe(now, {"SPOT": now, FUT: t(11, 0, 0)}, True)
    assert msg and FUT in msg and bf.active
    # Spot alone recovering is not enough: both must be fresh to resume
    assert bf.observe(t(11, 0, 40), {"SPOT": t(11, 0, 40), FUT: t(11, 0, 0)}, True) is None
    assert "live again" in bf.observe(t(11, 0, 50), {"SPOT": t(11, 0, 50), FUT: t(11, 0, 49)}, True).lower()


def test_instrument_that_never_ticked_counts_as_stale_once_others_are_live():
    """Finding A2: futures with no valid tick at all must not let the system look healthy."""
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: ["SPOT", FUT], gap_seconds=15)
    assert bf.observe(t(9, 15, 5), {"SPOT": t(9, 15, 5), FUT: None}, True) is None  # grace period
    msg = bf.observe(t(9, 15, 25), {"SPOT": t(9, 15, 25), FUT: None}, True)
    assert msg and FUT in msg and bf.active
    # Spot fresh, futures still never seen -> cannot resume
    assert bf.observe(t(9, 16), {"SPOT": t(9, 16), FUT: None}, True) is None
    assert bf.state == bf.GAP_OPEN
    assert "live again" in bf.observe(t(9, 16, 5), {"SPOT": t(9, 16, 5), FUT: t(9, 16, 4)}, True).lower()


def test_nothing_is_stale_before_any_feed_is_live():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: ["SPOT", FUT], gap_seconds=15)
    assert bf.observe(t(9, 20), {"SPOT": None, FUT: None}, True) is None and not bf.active


def test_entire_repair_runs_off_the_maintenance_loop_and_retries_on_failure():
    """Finding B: broker fetch AND indicator rebuild happen in the worker; loop stays responsive."""
    release = threading.Event()
    calls = {"n": 0}

    def slow_reconcile(*a, **k):
        calls["n"] += 1
        release.wait(5)
        if calls["n"] == 1:
            raise RuntimeError("FYERS timeout")

    def slow_warmup(**k):
        import time as _t
        _t.sleep(0.3)  # rebuild cost must not land on the maintenance thread
        return {}

    hm = mock_hm()
    hm.reconcile_session_candles.side_effect = slow_reconcile
    hm.warm_up_technical_engine.side_effect = slow_warmup
    engine = MagicMock()
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15, retry_seconds=30)
    drive_to_repair(bf)
    assert bf.state == bf.REPAIRING

    import time as _t
    start = _t.monotonic()
    for s in range(7, 12):
        assert bf.observe(t(10, 3, s), {FUT: t(10, 3, s)}, True) is None
    assert _t.monotonic() - start < 0.2

    release.set()
    bf._worker.join(timeout=30)
    msg = bf.observe(t(10, 3, 12), {FUT: t(10, 3, 12)}, True)
    assert "failed" in msg and bf.active
    engine.adopt_states.assert_not_called()  # never touch live state on failure

    bf.observe(t(10, 3, 43), {FUT: t(10, 3, 43)}, True)  # retry: worker does fetch + 0.3s rebuild
    start = _t.monotonic()
    while bf.state == bf.REPAIRING and not bf._worker_done.is_set():
        assert bf.observe(t(10, 3, 44), {FUT: t(10, 3, 44)}, True) is None
        _t.sleep(0.01)
    assert _t.monotonic() - start >= 0.25  # rebuild ran while the loop kept polling
    t_swap = _t.monotonic()
    assert "repaired" in bf.observe(t(10, 3, 45), {FUT: t(10, 3, 45)}, True).lower()
    assert _t.monotonic() - t_swap < 0.05  # main-thread part is just the swap
    engine.adopt_states.assert_called_once()


def test_rebuild_exception_does_not_crash_and_leaves_live_state_untouched():
    """Finding B: an error while rebuilding must fail closed and retry, not kill the runner."""
    hm = mock_hm()
    hm.warm_up_technical_engine.side_effect = KeyError("daily levels missing")
    engine = MagicMock()
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    drive_to_repair(bf)
    bf._worker.join(timeout=30)
    msg = bf.observe(t(10, 3, 7), {FUT: t(10, 3, 7)}, True)
    assert "failed" in msg and "daily levels missing" in msg
    assert bf.active and bf.state == bf.AWAITING_SETTLE
    engine.adopt_states.assert_not_called()


def test_swap_failure_and_state_machine_errors_never_raise():
    hm = mock_hm()
    engine = MagicMock()
    engine.adopt_states.side_effect = RuntimeError("boom")
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    drive_to_repair(bf)
    bf._worker.join(timeout=30)
    msg = bf.observe(t(10, 3, 7), {FUT: t(10, 3, 7)}, True)
    assert "swap failed" in msg and bf.active

    broken = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    msg = broken.observe(t(10, 0), {FUT: "not-a-datetime"}, True)  # garbage input
    assert "error" in msg.lower() and broken.active  # fail closed


def test_repair_rejected_when_fyers_did_not_return_outage_bars():
    """Finding C: a stored partial live bar must not satisfy coverage for the outage window."""
    hm = mock_hm()
    day = t(10, 0).date()
    full = {ts for ts in affected_intervals("1m", t(10, 0, 30), t(10, 3, 6))}
    missing_one = full - {t(10, 0)}  # the bar that was open when the outage began
    hm.last_broker_returned = {
        (FUT, "1m", day): missing_one,
        (FUT, "5m", day): {t(10, 0)},
        (FUT, "15m", day): {t(9, 45)},
    }
    engine = MagicMock()
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    drive_to_repair(bf)
    bf._worker.join(timeout=30)
    msg = bf.observe(t(10, 3, 7), {FUT: t(10, 3, 7)}, True)
    assert "did not return" in msg and "10:00" in msg and bf.active
    engine.adopt_states.assert_not_called()


def test_affected_intervals_cover_partial_bar_through_last_closed_bar():
    gap, as_of = t(13, 42, 44), t(13, 48, 6)
    assert affected_intervals("1m", gap, as_of) == [t(13, m) for m in range(42, 48)]
    assert affected_intervals("5m", gap, as_of) == [t(13, 40)]
    assert affected_intervals("15m", gap, as_of) == [t(13, 30)]  # 13:30 bar closed at 13:45
    assert affected_intervals("15m", gap, t(13, 44)) == []  # ...but still open at 13:44


def test_bar_closing_during_rebuild_is_replayed_not_lost():
    release = threading.Event()
    hm = mock_hm()
    hm.reconcile_session_candles.side_effect = lambda *a, **k: release.wait(5)
    shadow = MagicMock()
    shadow.states = {}
    engine = MagicMock()
    engine.make_shadow.return_value = shadow
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    drive_to_repair(bf)

    late = bar("1m", t(10, 3))
    assert bf.hold(late) is True  # closes while the worker is busy
    release.set()
    bf._worker.join(timeout=30)
    assert "repaired" in bf.observe(t(10, 3, 7), {FUT: t(10, 3, 7)}, True).lower()
    shadow.bootstrap_candles.assert_called_once_with([late])
    engine.adopt_states.assert_called_once_with(shadow, [FUT])
    assert bf.hold(bar("1m", t(10, 4))) is False  # live path resumes


def test_no_gap_detection_outside_cash_session():
    bf = FeedGapBackfiller(MagicMock(), MagicMock(), MagicMock(), lambda: [FUT], gap_seconds=15)
    assert bf.observe(t(15, 35), {FUT: t(15, 30)}, session_live=False) is None
    assert not bf.active


def test_rebuild_is_discarded_if_feed_died_again_during_repair():
    """A second outage while the worker rebuilds must not re-enable signals on a stale feed."""
    release = threading.Event()
    hm = mock_hm()
    hm.reconcile_session_candles.side_effect = lambda *a, **k: release.wait(5)
    engine = MagicMock()
    bf = FeedGapBackfiller(hm, engine, MagicMock(), lambda: [FUT], gap_seconds=15)
    drive_to_repair(bf)  # worker running since 10:03:06
    release.set()
    bf._worker.join(timeout=30)

    msg = bf.observe(t(10, 3, 40), {FUT: t(10, 3, 20)}, True)  # silent 20s while repairing
    assert "stale again" in msg and bf.state == bf.GAP_OPEN
    engine.adopt_states.assert_not_called()

    assert "live again" in bf.observe(t(10, 4, 10), {FUT: t(10, 4, 10)}, True).lower()
    bf.observe(t(10, 5, 6), {FUT: t(10, 5, 6)}, True)  # fresh repair covering both outages
    bf._worker.join(timeout=30)
    assert "repaired" in bf.observe(t(10, 5, 7), {FUT: t(10, 5, 7)}, True).lower()
    engine.adopt_states.assert_called_once()
