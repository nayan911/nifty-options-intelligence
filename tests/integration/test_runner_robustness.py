"""Regression tests for issues found in the 23-Nov live log."""

import sys
import threading
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from run_market_data import MarketDataEngine  # noqa: E402

from nifty_engine.core.interfaces import Candle, Tick  # noqa: E402
from nifty_engine.data.storage.duckdb_store import DuckDBStore  # noqa: E402
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus, is_derivative_symbol  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")


def at(h, m, s=0, ms=0):
    return IST.localize(datetime(2026, 11, 23, h, m, s, ms * 1000))


def tick(sym, exch, local, ltp=25400.0, vol=1000):
    return Tick(symbol=sym, exchange_timestamp=exch, local_timestamp=local, ltp=ltp,
                cumulative_volume=vol, incremental_volume=0)


# ------------------------------------------------------------------ tick classification
def test_small_negative_clock_skew_is_still_live():
    """Live log: ~11% of NIFTY ticks had latency -37..-89ms (local clock behind exchange)."""
    n = TickNormalizer()
    now = at(11, 0, 0)
    r = n.normalize(tick("NSE:NIFTY26NOVFUT", now + timedelta(milliseconds=60), now), current_time=now)
    assert r.status == TickStatus.VALID_LIVE
    n2 = TickNormalizer()
    r2 = n2.normalize(tick("NSE:NIFTY26NOVFUT", now + timedelta(seconds=5), now), current_time=now)
    assert r2.status == TickStatus.VALID_SNAPSHOT  # beyond tolerance -> not trusted


def test_cash_instruments_are_not_live_after_1530_but_derivatives_are():
    now = at(15, 35, 0)
    n = TickNormalizer()
    assert n.normalize(tick("NSE:NIFTY50-INDEX", now, now), current_time=now).status == TickStatus.VALID_SNAPSHOT
    assert n.normalize(tick("NSE:RELIANCE-EQ", now, now), current_time=now).status == TickStatus.VALID_SNAPSHOT
    assert n.normalize(tick("NSE:NIFTY26NOVFUT", now, now), current_time=now).status == TickStatus.VALID_LIVE
    assert n.normalize(tick("NSE:NIFTY26D0125400CE", now, now, ltp=240), current_time=now).status == TickStatus.VALID_LIVE


def test_is_derivative_symbol():
    assert is_derivative_symbol("NSE:NIFTY26NOVFUT")
    assert is_derivative_symbol("NSE:NIFTY26NOV25400CE")
    assert is_derivative_symbol("NSE:NIFTY26D0125350PE")
    assert not is_derivative_symbol("NSE:NIFTY50-INDEX")
    assert not is_derivative_symbol("NSE:RELIANCE-EQ")
    assert not is_derivative_symbol("BSE:SENSEX-INDEX")


def test_purge_post_close_cash_bars_keeps_real_bars(tmp_path):
    store = DuckDBStore(db_path=str(tmp_path / "m.duckdb"))
    mk = lambda sym, tf, ts: Candle(symbol=sym, timeframe=tf, timestamp=ts, open=1, high=1, low=1,  # noqa: E731
                                    close=1, volume=0, is_closed=True)
    store.insert_candles_batch([
        mk("NSE:NIFTY50-INDEX", "1m", at(15, 29)),   # real last bar
        mk("NSE:NIFTY50-INDEX", "1m", at(15, 30)),   # synthetic
        mk("NSE:NIFTY50-INDEX", "5m", at(15, 35)),   # synthetic
        mk("NSE:RELIANCE-EQ", "15m", at(15, 30)),    # synthetic
        mk("NSE:NIFTY26NOVFUT", "1m", at(15, 35)),   # real F&O closing-window bar
        mk("NSE:NIFTY50-INDEX", "D", at(15, 30)),    # daily bar must stay
    ])
    assert store.purge_post_close_cash_bars() == 3
    assert store.purge_post_close_cash_bars() == 0  # idempotent
    left = store._conn.execute("select symbol, timeframe from candles order by 1, 2").fetchall()
    assert ("NSE:NIFTY26NOVFUT", "1m") in left and ("NSE:NIFTY50-INDEX", "D") in left
    assert len(left) == 3


# ------------------------------------------------------------------ runner behaviour
def make_engine():
    return MarketDataEngine(store=DuckDBStore(db_path=":memory:"))


def test_candle_closes_from_two_threads_are_serialized():
    """Live log showed 'Bars Closed' 55 -> 57 -> 57: two threads in the close path at once."""
    engine = make_engine()
    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def guarded(_c):
        with lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        _time.sleep(0.001)
        with lock:
            in_flight["now"] -= 1

    engine.paper_engine.on_candle = guarded
    engine.paper_engine.on_tick = guarded
    engine.evaluate_and_update_readiness = lambda: None
    engine.store.insert_candle = lambda c: None
    engine.store.flush_ticks = lambda: None

    def worker(offset):
        for i in range(40):
            engine._on_candle_closed(Candle(symbol=f"NSE:X{offset}-EQ", timeframe="1m",
                                            timestamp=at(10, 0) + timedelta(minutes=i), open=1, high=1,
                                            low=1, close=1, volume=0, is_closed=True))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine.candle_count == 160
    assert in_flight["max"] == 1


def test_confluent_levels_on_same_bar_produce_one_alert(capsys):
    engine = make_engine()
    snap = SimpleNamespace(symbol="NSE:NIFTY50-INDEX", timeframe="5m", timestamp=at(14, 30), close=25349.5,
                           trend_bias="BEARISH")
    for lvl, px in (("PSYCHOLOGICAL_25350.0", 25350.0), ("SWING_LOW_CLUSTER_25360.0", 25360.0)):
        setup = SimpleNamespace(direction=SimpleNamespace(value="BEARISH"), level_id=lvl, nominal_price=px,
                                stop_loss=25355.17, target_price=25334.5)
        engine._on_setup_candidate(setup, snap)
    out = capsys.readouterr().out
    assert out.count("[TECHNICAL_SETUP_CANDIDATE") == 1
    assert "+ confluence" in out and "SWING_LOW_CLUSTER_25360.0" in out


def test_atm_rotation_has_hysteresis():
    engine = make_engine()
    engine.master = MagicMock()
    engine.master.get_active_expiries.return_value = []  # rotation would return early after the check
    engine.ws_client = MagicMock()
    engine.active_atm_strike = 25350.0

    engine.current_spot_price = 25376.5  # live log: flipped to 25400 here
    engine.check_and_rotate_options()
    engine.master.get_active_expiries.assert_not_called()

    engine.current_spot_price = 25386.0  # clearly past midpoint + 10
    engine.check_and_rotate_options()
    engine.master.get_active_expiries.assert_called_once()


def test_shutdown_closes_database_even_if_interrupted_during_socket_close():
    engine = make_engine()
    closed = []
    engine.store.close = lambda: closed.append("db")

    def ws_close(timeout=3.0):
        raise KeyboardInterrupt  # user presses Ctrl-C again

    engine.ws_client.close = ws_close
    engine._shutdown()
    assert closed == ["db"]
    assert engine._shutting_down
    # Late socket callbacks after shutdown are ignored instead of hitting a closed DB
    engine._on_tick_received(tick("NSE:NIFTY26NOVFUT", at(10, 0), at(10, 0)))
    assert engine.total_ticks_received == 0


def test_option_chain_switches_to_next_expiry_near_expiry():
    engine = make_engine()
    engine.chain_expiries = [(date(2026, 11, 23), 111), (date(2026, 12, 1), 222)]
    assert engine._select_chain_expiry(at(10, 0)) is None        # 5.5h left: nearest
    assert engine._select_chain_expiry(at(11, 31)) == 222        # <4h left: next expiry
    assert engine._select_chain_expiry(at(15, 35)) == 222        # after expiry close
    engine.chain_expiries = [(date(2026, 11, 23), 111)]
    assert engine._select_chain_expiry(at(14, 0)) is None        # no next expiry known


def test_futures_setups_print_as_context_not_trade_alerts(capsys):
    """Futures 1m bars only confirm spot bars, so a futures setup can never become a trade."""
    engine = make_engine()
    snap = SimpleNamespace(symbol="NSE:NIFTY26NOVFUT", timeframe="5m", timestamp=at(15, 30), close=25349.5,
                           trend_bias="BEARISH")
    setup = SimpleNamespace(direction=SimpleNamespace(value="BEARISH"), level_id="PSYCHOLOGICAL_25350.0",
                            nominal_price=25350.0, stop_loss=25355.17, target_price=25334.5)
    engine._on_setup_candidate(setup, snap)
    out = capsys.readouterr().out
    assert "[TECHNICAL_SETUP_CANDIDATE" not in out and "[CONTEXT]" in out
