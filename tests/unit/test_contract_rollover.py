"""Expiry-day rollover: expired contracts, futures resolved before warmup, master refresh,
and first option-chain poll on an expired/expiring series."""

import os
import sys
import time as _time
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import pytz

from nifty_engine.core.interfaces import OptionChainSnapshot
from nifty_engine.data.fyers import instrument_master as im
from nifty_engine.data.fyers.instrument_master import NSE_FO_COLUMNS, FyersInstrumentMaster, min_active_expiry

IST = pytz.timezone("Asia/Kolkata")


def at(y, mo, d, h, m):
    return IST.localize(datetime(y, mo, d, h, m))


def epoch(d):
    return int(IST.localize(datetime(d.year, d.month, d.day, 15, 30)).timestamp())


def row(symbol, short, exp, strike=0.0, opt="XX"):
    return ["x", symbol, 14, 65, 0.05, "", "0915-1530", "2026-11-01", epoch(exp), symbol,
            10, 11, 1, short, 26000, strike, opt, "1", "None", 0, 0.0]


@pytest.fixture
def master():
    m = FyersInstrumentMaster()
    rows = [
        row("NSE:NIFTY26NOVFUT", "NIFTY", date(2026, 11, 23)),
        row("NSE:NIFTY26DECFUT", "NIFTY", date(2026, 12, 29)),
        row("NSE:BANKNIFTY26NOVFUT", "BANKNIFTY", date(2026, 11, 23)),
        row("NSE:BANKNIFTY26DECFUT", "BANKNIFTY", date(2026, 12, 29)),
        row("NSE:NIFTY26NOV25350CE", "NIFTY", date(2026, 11, 23), 25350.0, "CE"),
        row("NSE:NIFTY26D0125350CE", "NIFTY", date(2026, 12, 1), 25350.0, "CE"),
        row("NSE:NIFTY26D0825350CE", "NIFTY", date(2026, 12, 8), 25350.0, "CE"),
    ]
    m.load_from_dataframe(pd.DataFrame(rows, columns=NSE_FO_COLUMNS))
    return m


def freeze(monkeypatch, when):
    monkeypatch.setattr(im, "now_ist", lambda: when)


# ------------------------------------------------------------------ expiry cutoff
def test_min_active_expiry_rolls_at_1530_ist():
    assert min_active_expiry(at(2026, 11, 23, 15, 29)) == date(2026, 11, 23)
    assert min_active_expiry(at(2026, 11, 23, 15, 30)) == date(2026, 11, 24)


def test_contracts_expiring_today_are_dead_after_close(master, monkeypatch):
    freeze(monkeypatch, at(2026, 11, 23, 11, 0))
    assert master.resolve_future("NIFTY") == "NSE:NIFTY26NOVFUT"
    assert master.get_active_expiries("NIFTY")[0] == date(2026, 11, 23)

    freeze(monkeypatch, at(2026, 11, 23, 17, 6))  # the user's evening run on expiry day
    assert master.resolve_future("NIFTY") == "NSE:NIFTY26DECFUT"
    assert master.resolve_future("BANKNIFTY") == "NSE:BANKNIFTY26DECFUT"
    assert master.get_active_expiries("NIFTY") == [date(2026, 12, 1), date(2026, 12, 8)]  # option expiries


# ------------------------------------------------------------------ master refresh
def test_stale_cached_master_is_refreshed_on_load(tmp_path, monkeypatch):
    m = FyersInstrumentMaster(cache_dir=tmp_path)
    pd.DataFrame([row("NSE:NIFTY26DECFUT", "NIFTY", date(2026, 12, 29))]).to_csv(m.fo_cache_file, header=False, index=False)
    old = _time.time() - 3 * 86400
    os.utime(m.fo_cache_file, (old, old))
    calls = []
    monkeypatch.setattr(m, "download_master", lambda force=False: calls.append(force) or m.fo_cache_file)
    m.load_master(auto_download=True)
    assert calls == [False]  # download_master decides freshness; it is no longer skipped


# ------------------------------------------------------------------ runner startup order
def make_engine():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_market_data import MarketDataEngine

    from nifty_engine.data.storage.duckdb_store import DuckDBStore

    return MarketDataEngine(store=DuckDBStore(db_path=":memory:"))


def test_futures_resolved_before_warmup():
    engine = make_engine()
    engine.auth.load_cached_token = lambda: "token"
    engine.master = MagicMock()
    engine.master.resolve_future.side_effect = lambda u: {"NIFTY": "NSE:NIFTY26DECFUT",
                                                          "BANKNIFTY": "NSE:BANKNIFTY26DECFUT"}[u]
    seen = {}

    def capture(symbols):
        seen["warmup"] = list(symbols)
        raise KeyboardInterrupt  # stop after warmup started

    engine.historical_manager.sync_pre_market_data = capture
    with pytest.raises(KeyboardInterrupt):
        engine._run()
    assert seen["warmup"] == ["NSE:NIFTY50-INDEX", "NSE:NIFTY26DECFUT"]
    assert engine.technical_engine.primary_futures == "NSE:NIFTY26DECFUT"
    assert engine.technical_engine.index_futures_map["NSE:NIFTYBANK-INDEX"] == "NSE:BANKNIFTY26DECFUT"


def test_refuses_to_start_without_a_resolvable_futures_contract():
    engine = make_engine()
    engine.auth.load_cached_token = lambda: "token"
    engine.master = MagicMock()
    engine.master.resolve_future.side_effect = LookupError("No active futures contract found for NIFTY")
    engine.historical_manager.sync_pre_market_data = MagicMock()
    engine._run()
    engine.historical_manager.sync_pre_market_data.assert_not_called()


# ------------------------------------------------------------------ first chain poll
def test_first_chain_poll_on_expired_series_refetches_next_expiry(monkeypatch):
    engine = make_engine()
    now = at(2026, 11, 23, 17, 6)
    monkeypatch.setattr("run_market_data.now_ist", lambda: now)
    expiries = [(date(2026, 11, 23), 111), (date(2026, 12, 1), 222)]

    def snap(exp):
        return OptionChainSnapshot(underlying="NIFTY", spot_price=25380.0, api_timestamp=now, expiry=exp,
                                   strikes=[], available_expiries=expiries)

    engine.rest_client.get_option_chain = AsyncMock(side_effect=lambda **kw: snap(
        date(2026, 12, 1) if kw.get("expiry_epoch") == 222 else date(2026, 11, 23)))
    engine.store.save_option_chain_snapshot = MagicMock()
    engine._log_oi_flow = MagicMock()

    engine._poll_option_chain_if_due(is_open=False)
    epochs = [c.kwargs.get("expiry_epoch") for c in engine.rest_client.get_option_chain.call_args_list]
    assert epochs == [None, 222]
    stored = engine.store.save_option_chain_snapshot.call_args.args[0]
    assert stored.expiry == date(2026, 12, 1)


def test_first_chain_poll_on_normal_day_makes_one_call(monkeypatch):
    engine = make_engine()
    now = at(2026, 11, 25, 10, 0)
    monkeypatch.setattr("run_market_data.now_ist", lambda: now)
    expiries = [(date(2026, 12, 1), 222), (date(2026, 12, 8), 333)]
    engine.rest_client.get_option_chain = AsyncMock(return_value=OptionChainSnapshot(
        underlying="NIFTY", spot_price=25380.0, api_timestamp=now, expiry=date(2026, 12, 1), strikes=[],
        available_expiries=expiries))
    engine.store.save_option_chain_snapshot = MagicMock()
    engine._log_oi_flow = MagicMock()
    engine._poll_option_chain_if_due(is_open=True)
    assert engine.rest_client.get_option_chain.call_count == 1
