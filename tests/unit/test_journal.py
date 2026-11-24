"""Phase 4.1 live journal: stable entry-decision codes, journal rows, option tracking, runner wiring."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import pytz

from nifty_engine.analysis.technical.breakouts import BreakoutDirection
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.interfaces import SystemReadiness
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.journal.codes import EntryDecision
from nifty_engine.journal.recorder import JournalRecorder
from nifty_engine.journal.store import JournalStore
from nifty_engine.paper_trading.engine import PaperTradingEngine
from nifty_engine.strategy.registry import load_registry

IST = pytz.timezone("Asia/Kolkata")
T = IST.localize(datetime(2026, 11, 23, 10, 15))


def candidate(cid="C1", direction=BreakoutDirection.BULLISH, da=None, at=T):
    return TechnicalCandidate(
        candidate_id=cid, symbol="NSE:NIFTY50-INDEX", timestamp=at, direction=direction, setup_level="PDH_25100.0",
        nominal_price=25100.0, entry_price=25110.0, stop_loss=25095.0, target_1=25132.5, target_2=25140.0,
        risk_points=15.0, reward_points=22.5, risk_reward_ratio=1.5, confluence_score=75.0, headroom_points=None,
        timeframe_breakdown={"15m_trend": "BULLISH"}, decision_at=at + timedelta(minutes=1), derivatives_assessment=da)


def recorder():
    store = DuckDBStore(db_path=":memory:")
    j = JournalStore(store)
    return j, JournalRecorder(j, load_registry().get("S000"), clock=lambda: T + timedelta(minutes=2))


# ------------------------------------------------------------------ paper engine decision codes
def test_paper_engine_records_a_stable_code_for_each_decision():
    pe = PaperTradingEngine()
    pe.system_readiness = SystemReadiness.DEGRADED
    assert pe.process_technical_candidate(candidate("A"), current_time=T) is None
    assert pe.last_decision == {"candidate_id": "A", "code": "READINESS_NOT_READY",
                                "detail": "readiness DEGRADED", "contract": None}
    pe.system_readiness = SystemReadiness.READY
    assert pe.process_technical_candidate(candidate("B"), current_time=T) is None
    assert pe.last_decision["candidate_id"] == "B" and pe.last_decision["code"] == "NO_CONTRACT_SELECTED"
    late = IST.localize(datetime(2026, 11, 23, 15, 20))
    assert pe.process_technical_candidate(candidate("C", at=late), current_time=late) is None
    assert pe.last_decision["code"] == "AFTER_ENTRY_CUTOFF"


def test_every_code_is_stable_and_unique():
    values = [c.value for c in EntryDecision]
    assert len(values) == len(set(values)) and all(v == v.upper() for v in values)
    assert {"ACCEPTED", "NET_RR_BELOW_MIN", "SESSION_RISK_LIMIT", "SAME_DIRECTION_OPEN"} <= set(values)


# ------------------------------------------------------------------ journal rows
def test_candidate_setup_and_trade_events_are_recorded_once():
    j, rec = recorder()
    s = load_registry().get("S000")
    ctx = {"readiness": "READY", "worst_feed_symbol": "NSE:NIFTY50-INDEX", "worst_feed_age_s": 0.8}
    c = candidate()
    dec = {"candidate_id": "C1", "code": "NET_RR_BELOW_MIN", "detail": "net R:R 0.9 < 1.2", "contract": "NSE:X"}
    rec.on_candidate(c, dec, ctx)
    rec.on_candidate(c, dec, ctx)          # restart re-emit: not duplicated
    rec.on_candidate(candidate("C2"), {"candidate_id": "OTHER", "code": "ACCEPTED"}, ctx)  # stale decision ignored
    setup = SimpleNamespace(direction=BreakoutDirection.BULLISH, level_id="PDH_25100.0", nominal_price=25100.0,
                            stop_loss=25080.0, target_price=25150.0)
    snap = SimpleNamespace(symbol="NSE:NIFTY50-INDEX", timeframe="5m", timestamp=T, effective_available_at=T + timedelta(minutes=5),
                           close=25105.0, trend_bias="BULLISH")
    rec.on_setup(setup, snap, ctx)
    rec.on_trade_event({"event": "ORDER_FILLED", "position_id": "P1", "symbol": "NSE:X", "price": 101.5}, ctx)
    rows = j.events(T.date())
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["CANDIDATE", "CANDIDATE", "PAPER_FILL", "SETUP"]
    c1 = next(r for r in rows if r["candidate_id"] == "C1")
    assert c1["decision_code"] == "NET_RR_BELOW_MIN" and c1["readiness"] == "READY" and c1["worst_feed_age_s"] == 0.8
    assert (c1["strategy_id"], c1["strategy_version"], c1["strategy_fingerprint"]) == (s.id, s.version, s.fingerprint())
    assert c1["event_at"] == T + timedelta(minutes=1)          # decision time
    c2 = next(r for r in rows if r["candidate_id"] == "C2")
    assert c2["decision_code"] == "NOT_EVALUATED"
    setup_row = next(r for r in rows if r["kind"] == "SETUP")
    assert setup_row["event_at"] == T + timedelta(minutes=5)   # when the setup became known


def test_option_tracking_window():
    j, rec = recorder()
    c = candidate()
    assert rec.track_option(c, "NSE:NIFTY26D0125100CE", {"bid": 100.0, "ask": 101.0}, "paper_engine") == "NSE:NIFTY26D0125100CE"
    assert rec.track_option(c, None, None, "none") is None
    t0 = c.decision_at
    assert j.active_tracking(t0 + timedelta(minutes=59)) == ["NSE:NIFTY26D0125100CE"]
    assert j.active_tracking(t0 + timedelta(minutes=60)) == []
    tr = j.tracking(T.date())[0]
    assert tr["source"] == "paper_engine" and tr["track_until"] == t0 + timedelta(minutes=60)


def test_journal_failure_never_raises():
    rec = JournalRecorder(mock.Mock(add_event=mock.Mock(side_effect=RuntimeError("disk full"))),
                          load_registry().get("S000"), clock=lambda: T)
    rec.on_candidate(candidate(), None, {})
    rec.on_trade_event({"event": "POSITION_CLOSED", "position_id": "P"}, {})


# ------------------------------------------------------------------ runner wiring
@pytest.fixture
def runner():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_market_data import MarketDataEngine

    r = MarketDataEngine(store=DuckDBStore(db_path=":memory:"))
    r.ws_client = mock.Mock()
    r._get_live_quote = mock.Mock(return_value={"ltp": 120.0, "bid": 119.5, "ask": 120.5, "timestamp": None})
    r.master = mock.Mock()
    r.master.get_active_expiries.return_value = ["2026-12-01"]
    r.master.resolve_option.return_value = SimpleNamespace(symbol="NSE:NIFTY26D0125100CE")
    return r


def test_runner_journals_every_candidate_with_its_decision_and_tracks_an_option(runner):
    runner.paper_engine.system_readiness = SystemReadiness.DEGRADED
    runner._on_technical_candidate(candidate())
    rows = runner.journal.journal.events(T.date())
    assert [r["decision_code"] for r in rows if r["kind"] == "CANDIDATE"] == ["READINESS_NOT_READY"]
    tr = runner.journal.journal.tracking(T.date())
    assert tr[0]["contract"] == "NSE:NIFTY26D0125100CE" and tr[0]["source"] == "atm_fallback"
    runner.ws_client.subscribe.assert_called_with(["NSE:NIFTY26D0125100CE"])
    assert '"ask": 120.5' in tr[0]["quote_at_signal"]


def test_rotation_keeps_tracked_contracts_subscribed(runner):
    runner._on_technical_candidate(candidate())
    runner.current_spot_price, runner.active_atm_strike = 25300.0, 25100.0
    runner.master.resolve_option.return_value = SimpleNamespace(symbol="NSE:NEW_ATM")
    runner.subscribed_option_symbols = {"NSE:NIFTY26D0125100CE", "NSE:OLD"}
    with mock.patch("run_market_data.now_ist", return_value=T + timedelta(minutes=10)):
        runner.check_and_rotate_options()
    unsubscribed = runner.ws_client.unsubscribe.call_args[0][0]
    assert "NSE:OLD" in unsubscribed and "NSE:NIFTY26D0125100CE" not in unsubscribed


def test_next_month_futures_is_recorded_but_never_a_signal_symbol(runner):
    runner.master.resolve_future.side_effect = lambda u, expiry=None, nth=0: {
        ("NIFTY", 0): "NSE:NIFTY26DECFUT", ("NIFTY", 1): "NSE:NIFTY27JANFUT",
        ("BANKNIFTY", 0): "NSE:BANKNIFTY26DECFUT"}[(u, nth)]
    assert runner._resolve_futures()
    runner.current_spot_price = 25100.0
    runner.master.get_active_expiries.return_value = []
    wl = runner._build_watchlist()
    assert "NSE:NIFTY27JANFUT" in wl and "NSE:NIFTY26DECFUT" in wl
    assert runner._signal_symbols() == {"NSE:NIFTY50-INDEX", "NSE:NIFTY26DECFUT"}
    assert "NSE:NIFTY27JANFUT" not in runner.technical_engine.futures_index_map


def test_instrument_master_resolves_the_next_month_contract():
    import pandas as pd

    from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
    m = FyersInstrumentMaster.__new__(FyersInstrumentMaster)
    m._df = pd.DataFrame({"short_sym": ["NIFTY"] * 3, "opt": ["XX"] * 3,
                          "expiry_date": pd.to_datetime(["2099-01-27", "2099-02-24", "2099-03-30"]).date,
                          "symbol": ["NSE:NIFTY99JANFUT", "NSE:NIFTY99FEBFUT", "NSE:NIFTY99MARFUT"]})
    assert m.resolve_future("NIFTY") == "NSE:NIFTY99JANFUT"
    assert m.resolve_future("NIFTY", nth=1) == "NSE:NIFTY99FEBFUT"
    with pytest.raises(LookupError):
        m.resolve_future("NIFTY", nth=5)
