"""Regression tests for live-trading gaps found in the paper engine audit."""

import signal
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytz

from nifty_engine.analysis.derivatives.oi_analysis import OpenInterestAnalyzer
from nifty_engine.analysis.derivatives.pcr import PCREngine
from nifty_engine.core.interfaces import Candle, OptionChainSnapshot, OptionChainStrike, SystemReadiness, Tick
from nifty_engine.paper_trading.engine import PaperTradingEngine, is_underlying_spot
from nifty_engine.paper_trading.models import PaperPosition, PositionStatus

IST = pytz.timezone("Asia/Kolkata")
DAY = date(2026, 11, 25)  # regular trading Wednesday


def at(h, m, s=0, d=DAY):
    return IST.localize(datetime(d.year, d.month, d.day, h, m, s))


def make_engine(**kw):
    eng = PaperTradingEngine(store=None, **kw)
    eng.set_system_readiness(SystemReadiness.READY)
    return eng


def make_position(opt="CE", entry=at(11, 0), spot_sl=25330.0, t1=25400.0, t2=25420.0, bid=99.5, pid="p1"):
    return PaperPosition(
        position_id=pid, signal_id="s", symbol=f"NSE:NIFTY26D0125350{opt}", underlying="NIFTY", option_type=opt,
        strike=25350.0, expiry=date(2026, 12, 1), lot_size=65, initial_quantity=65, remaining_quantity=65,
        entry_price=100.0, target_premium_t1=115.0, target_premium_t2=125.0, stop_loss_premium=90.0,
        initial_stop_loss_premium=90.0, spot_entry=25375.0, spot_target_t1=t1, spot_target_t2=t2, spot_sl=spot_sl,
        entry_time=entry, last_quote_time=entry, last_bid_time=entry, last_ask_time=entry,
        status=PositionStatus.OPEN, current_ltp=100.0, current_bid=bid, current_ask=100.5,
        highest_premium_seen=100.0, lowest_premium_seen=100.0, gross_pnl=0.0, net_pnl=0.0, total_costs=0.0,
        trades=[], notes="",
    )


def make_candidate(opt="CE", cid="c1", when=at(11, 0)):
    contract = SimpleNamespace(
        symbol=f"NSE:NIFTY26D0125350{opt}", underlying="NIFTY", option_type=opt, strike=25350.0,
        expiry=date(2026, 12, 1), ltp=100.0, bid=99.5, ask=100.5, spread=1.0, quote_age_seconds=1.0,
        execution_costs=SimpleNamespace(is_cost_viable=True, net_rr_t1=2.0),
        premium_trajectory=SimpleNamespace(target_premium_t1=115.0, target_premium_t2=125.0, stop_loss_premium=90.0),
    )
    return SimpleNamespace(
        candidate_id=cid, decision_at=when, entry_price=25375.0, stop_loss=25330.0, target_1=25400.0,
        target_2=25420.0, derivatives_assessment=SimpleNamespace(selected_contract=contract),
    )


def idx_tick(sym, px, when=at(11, 1)):
    return Tick(symbol=sym, exchange_timestamp=when, local_timestamp=when, ltp=px,
                cumulative_volume=0, incremental_volume=0, is_index=True)


# ------------------------------------------------------------------ 1. spot symbol matching
def test_banknifty_tick_does_not_touch_nifty_positions():
    for opt in ("CE", "PE"):
        eng = make_engine()
        pos = make_position(opt, spot_sl=25330.0 if opt == "CE" else 25420.0,
                            t1=25400.0 if opt == "CE" else 25350.0,
                            t2=25420.0 if opt == "CE" else 25330.0)
        pos.last_bid_time = at(11, 1)  # fresh bid: a wrong trigger would fill immediately
        eng.active_positions["p1"] = pos
        assert eng.on_tick(idx_tick("NSE:NIFTYBANK-INDEX", 62585.0)) == []
        assert pos.status == PositionStatus.OPEN
        assert not pos.is_exit_triggered and pos.pending_exit_reason is None  # nor queued as pending


def test_own_spot_index_still_drives_spot_stop():
    eng = make_engine()
    pos = make_position("CE", spot_sl=25330.0)
    pos.last_bid_time = at(11, 1)  # fresh executable bid at the time of the spot breach
    eng.active_positions["p1"] = pos
    events = eng.on_tick(idx_tick("NSE:NIFTY50-INDEX", 25329.0))
    assert events and events[0][1].value == "STOP_LOSS"


def test_is_underlying_spot_is_exact():
    assert is_underlying_spot("NIFTY", "NSE:NIFTY50-INDEX")
    assert not is_underlying_spot("NIFTY", "NSE:NIFTYBANK-INDEX")
    assert not is_underlying_spot("NIFTY", "BSE:SENSEX-INDEX")
    assert is_underlying_spot("BANKNIFTY", "NSE:NIFTYBANK-INDEX")
    assert not is_underlying_spot("UNKNOWN", "NSE:NIFTY50-INDEX")


# ------------------------------------------------------------------ 3. one position per direction
def test_second_candidate_same_direction_is_rejected_opposite_allowed():
    eng = make_engine()
    assert eng.process_technical_candidate(make_candidate("CE", "a")) is not None
    # Confluent level on the same bar -> distinct candidate id, same idea: must not double up
    assert eng.process_technical_candidate(make_candidate("CE", "b")) is None
    assert len(eng.active_positions) == 1
    assert eng.process_technical_candidate(make_candidate("PE", "c")) is not None


# ------------------------------------------------------------------ 6. session risk limits
def _closed(net, exit_time, pid):
    p = make_position(pid=pid)
    p.status = PositionStatus.CLOSED
    p.net_pnl = net
    p.exit_time = exit_time
    return p


def test_consecutive_losses_trigger_cooldown_then_resume():
    eng = make_engine(max_consecutive_losses=2, loss_cooldown_minutes=30)
    eng.closed_positions = [_closed(-800, at(10, 0), "x1"), _closed(-700, at(10, 20), "x2")]
    assert eng.process_technical_candidate(make_candidate(when=at(10, 40))) is None
    assert eng.process_technical_candidate(make_candidate(cid="later", when=at(10, 51))) is not None


def test_daily_loss_and_trade_count_limits():
    eng = make_engine(max_daily_net_loss=1000, max_consecutive_losses=0)
    eng.closed_positions = [_closed(-1200, at(10, 0), "x1")]
    assert eng.process_technical_candidate(make_candidate(when=at(11, 0))) is None

    eng2 = make_engine(max_trades_per_day=2, max_consecutive_losses=0)
    eng2.closed_positions = [_closed(+500, at(10, 0), "x1"), _closed(+300, at(10, 30), "x2")]
    assert eng2.process_technical_candidate(make_candidate(when=at(11, 0))) is None

    # Yesterday's losses do not block today
    eng3 = make_engine(max_daily_net_loss=1000)
    eng3.closed_positions = [_closed(-5000, at(14, 0, d=DAY - timedelta(days=1)), "y1")]
    assert eng3.process_technical_candidate(make_candidate(when=at(11, 0))) is not None


# ------------------------------------------------------------------ 4. stale recovered positions
def test_positions_from_an_earlier_session_are_closed_on_recovery(monkeypatch):
    monkeypatch.setattr("nifty_engine.paper_trading.engine.now_ist", lambda: at(9, 10))
    eng = make_engine()
    stale = make_position(entry=at(14, 0, d=DAY - timedelta(days=1)), bid=42.0)
    today = make_position(entry=at(9, 5), pid="p2")
    eng.active_positions = {"p1": stale, "p2": today}
    eng._close_stale_recovered_positions()
    assert "p1" not in eng.active_positions and stale.status == PositionStatus.CLOSED
    assert stale.is_estimated_exit  # excluded from verified win-rate stats
    assert "p2" in eng.active_positions  # today's position untouched


# ------------------------------------------------------------------ 5. candle SL fill realism
def test_intrabar_stop_fill_never_better_than_stop_level():
    eng = make_engine()
    pos = make_position(bid=99.0)  # bar spiked through SL (90) then recovered to 99
    pos.last_bid_time = at(11, 2)
    eng.active_positions["p1"] = pos
    bar = Candle(symbol=pos.symbol, timeframe="1m", timestamp=at(11, 1), open=98, high=100, low=85,
                 close=99, volume=100, is_closed=True, observed_at=at(11, 2))
    events = eng.on_candle(bar)
    assert events and events[0][1].value == "STOP_LOSS"
    assert events[0][2] <= 90.0


# ------------------------------------------------------------------ 2. tracked expiry selection
def _chain(expiry, ts, spot=25375.0):
    strikes = [OptionChainStrike(strike=k, call={"ltp": 10.0, "oi": 1000, "volume": 1, "iv": 15.0},
                                 put={"ltp": 10.0, "oi": 900, "volume": 1, "iv": 15.0})
               for k in (25300.0, 25350.0, 25400.0)]
    return OptionChainSnapshot(underlying="NIFTY", spot_price=spot, api_timestamp=ts, expiry=expiry,
                               strikes=strikes, total_call_oi=3000, total_put_oi=2700, pcr=0.9)


def test_assessment_follows_the_expiry_being_polled_not_just_the_nearest():
    near, nxt = date(2026, 11, 25), date(2026, 12, 1)
    oi, pcr = OpenInterestAnalyzer(), PCREngine()
    for snap in (_chain(near, at(11, 28)), _chain(near, at(11, 29)), _chain(nxt, at(11, 31)), _chain(nxt, at(11, 32))):
        oi.on_option_chain(snap, as_of=snap.api_timestamp)
        pcr.on_option_chain(snap, as_of=snap.api_timestamp)
    prof = oi.get_profile(underlying="NIFTY", as_of=at(11, 35))
    assert prof.expiry == nxt and not prof.is_stale
    p = pcr.get_pcr(underlying="NIFTY", as_of=at(11, 35))
    assert p is not None and p.expiry == nxt
    # Causal: before the switch, the nearest expiry is still the tracked one
    assert oi.get_profile(underlying="NIFTY", as_of=at(11, 30)).expiry == near


# ------------------------------------------------------------------ 8. signal-safe shutdown
def test_sigterm_routes_to_clean_shutdown():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_market_data import MarketDataEngine

    prev = signal.getsignal(signal.SIGTERM)
    try:
        MarketDataEngine._install_shutdown_signals()
        handler = signal.getsignal(signal.SIGTERM)
        try:
            handler(signal.SIGTERM, None)
            raised = False
        except KeyboardInterrupt:
            raised = True
        assert raised
    finally:
        signal.signal(signal.SIGTERM, prev)
