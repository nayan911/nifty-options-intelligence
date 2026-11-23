"""Strategy registry (Phase 2.3): strict validation, wiring of every parameter group, and proof that
S000 loaded from strategies/registry.yaml behaves exactly like the pre-registry engine."""

import copy
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import pytz
import yaml

from nifty_engine.analysis.technical.breakouts import BreakoutContext, BreakoutDirection
from nifty_engine.analysis.technical.engine import SymbolTechnicalState, TechnicalAnalysisEngine, TechnicalSnapshot
from nifty_engine.analysis.technical.indicators import compute_atr, compute_ema, compute_relative_volume, compute_rsi
from nifty_engine.analysis.technical.multi_timeframe import MultiTimeframeAnalyzer
from nifty_engine.core.interfaces import BreakoutState, Candle
from nifty_engine.strategy import registry as reg_mod
from nifty_engine.strategy.params import S000_FIXED, S000Params
from nifty_engine.strategy.registry import RegistryError, load_registry

IST = pytz.timezone("Asia/Kolkata")
REGISTRY = Path(__file__).resolve().parents[2] / "strategies" / "registry.yaml"


def _raw():
    return yaml.safe_load(REGISTRY.read_text())


def _load_modified(tmp_path, mutate, locks=None):
    raw = _raw()
    mutate(raw)
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    lock = REGISTRY.with_name("registry.lock.json").read_text() if locks is None else json.dumps(locks)
    (tmp_path / "registry.lock.json").write_text(lock)
    return load_registry(p)


def _params(**changes):
    """S000Params with dotted-path changes, e.g. _params(**{"entry.rsi_overbought": 55})."""
    d = copy.deepcopy(S000Params().canonical())
    for path, v in changes.items():
        node = d
        *parents, leaf = path.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = v
    return S000Params.model_validate(d)


# ------------------------------------------------------------------ the file itself
def test_registry_s000_equals_the_pre_registry_engine_values():
    e = load_registry().get("S000")
    assert e.params == S000Params()  # defaults == values hard-coded before the registry (183ef0a)
    assert e.version == 1 and e.status == "baseline"
    assert e.fixed == S000_FIXED
    assert "CONFLICT" not in e.data_requirements.contract_confidence
    assert len(e.params_hash()) == 16 and e.params_hash() == load_registry().get("S000").params_hash()


@pytest.mark.parametrize("label,mutate,match", [
    ("unknown param key", lambda r: r["strategies"]["S000"]["params"]["entry"].update(rsi_overbougt=70), "rsi_overbougt"),
    ("missing param", lambda r: r["strategies"]["S000"]["params"]["entry"].pop("min_risk_points"), "entry.min_risk_points"),
    ("missing nested timeframe", lambda r: r["strategies"]["S000"]["params"]["breakouts"].pop("5m"), "breakouts.5m"),
    ("wrong type", lambda r: r["strategies"]["S000"]["params"]["entry"].update(entry_window_minutes="fifteen"), "entry_window_minutes"),
    ("non-integer count", lambda r: r["strategies"]["S000"]["params"]["breakouts"]["5m"].update(max_retest_candles=4.5), "max_retest_candles"),
    ("out of range", lambda r: r["strategies"]["S000"]["params"]["entry"].update(rsi_overbought=40), "rsi_overbought"),
    ("targets reversed", lambda r: r["strategies"]["S000"]["params"]["targets"].update(target_2_rr=1.0), "target_2_rr"),
    ("tiers not descending", lambda r: r["strategies"]["S000"]["params"]["scoring"].update(rvol_tiers=[[1.3, 15], [2.0, 25]]), "descending"),
    ("unreachable min score", lambda r: r["strategies"]["S000"]["params"]["scoring"].update(min_score=100, rsi_points=0), "unreachable"),
    ("fixed changed", lambda r: r["strategies"]["S000"]["fixed"].update(rsi_period=9), "does not match the engine"),
    ("CONFLICT accepted", lambda r: r["strategies"]["S000"]["data_requirements"]["contract_confidence"].append("CONFLICT"), "CONFLICT"),
    ("unknown strategy id", lambda r: r["strategies"].update(S999=r["strategies"]["S000"]), "S999"),
    ("version without changelog", lambda r: r["strategies"]["S000"].update(version=2), "changelog"),
    ("bad status", lambda r: r["strategies"]["S000"].update(status="production"), "status"),
    ("schema version", lambda r: r.update(schema_version=2), "schema_version"),
    ("unknown top-level key", lambda r: r.update(defaults={}), "top-level"),
])
def test_invalid_registry_is_rejected(tmp_path, label, mutate, match):
    with pytest.raises(RegistryError, match=match):
        _load_modified(tmp_path, mutate)


def test_missing_or_broken_registry_file_is_rejected(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategies: [unclosed")
    with pytest.raises(RegistryError, match="not valid YAML"):
        load_registry(bad)


# ------------------------------------------------------------------ FIXED declarations are true
def test_fixed_indicator_definitions_match_what_the_engine_computes():
    state = SymbolTechnicalState(symbol="NSE:NIFTY-FUT-NEAR", is_index=False)
    t0 = IST.localize(datetime(2026, 6, 10, 9, 15))
    candles, snap = [], None
    for i in range(260):
        c = 25000 + 40 * math.sin(i / 9.0) + i * 0.7
        o = c - 3 * math.cos(i / 4.0)
        candle = Candle(symbol=state.symbol, timeframe="1m", timestamp=t0 + timedelta(minutes=i), open=o,
                        high=max(o, c) + 2 + (i % 5), low=min(o, c) - 2 - (i % 3), close=c,
                        volume=1000 + (i * 37) % 900, is_closed=True)
        candles.append(candle)
        snap = state.process_candle(candle)
    df = pd.DataFrame([c.model_dump() for c in candles])
    fast, mid, slow, longp = S000_FIXED["ema_periods"]
    assert snap.ema_9 == pytest.approx(compute_ema(df.close, fast).iloc[-1])
    assert snap.ema_21 == pytest.approx(compute_ema(df.close, mid).iloc[-1])
    assert snap.ema_50 == pytest.approx(compute_ema(df.close, slow).iloc[-1])
    assert snap.ema_200 == pytest.approx(compute_ema(df.close, longp).iloc[-1])
    rsi = compute_rsi(df.close, S000_FIXED["rsi_period"]).iloc[-1]
    assert snap.rsi == round(rsi, 1) != round(compute_rsi(df.close, S000_FIXED["rsi_period"] - 1).iloc[-1], 1)
    atr = compute_atr(df.high, df.low, df.close, S000_FIXED["atr_period"]).iloc[-1]  # engine rounds to 0.01
    assert snap.atr == round(atr, 2) != round(compute_atr(df.high, df.low, df.close, S000_FIXED["atr_period"] - 1).iloc[-1], 2)
    assert snap.rvol == pytest.approx(round(compute_relative_volume(df.volume, S000_FIXED["rvol_lookback_bars"]).iloc[-1], 2))
    assert state.candles_1m.maxlen == S000_FIXED["history_bars"]["1m"]
    assert state.candles_5m.maxlen == S000_FIXED["history_bars"]["5m"]
    assert state.candles_15m.maxlen == S000_FIXED["history_bars"]["15m"]


# ------------------------------------------------------------------ every parameter group is wired
def test_engine_state_is_built_from_the_params():
    p = _params(**{"levels.zone_atr_multiple": 0.3, "levels.min_zone_width_points": 6.0,
                   "swings.15m.left_bars": 4, "swings.1m.right_bars": 2,
                   "breakouts.5m.max_retest_candles": 7, "breakouts.5m.min_breakout_rvol": 1.6,
                   "breakouts.1m.stop_buffer_atr": 0.3, "breakouts.1m.min_approach_points": 6.0})
    eng = TechnicalAnalysisEngine(primary_index="NSE:NIFTY50-INDEX", primary_futures="NSE:NIFTY-FUT-NEAR",
                                  on_technical_candidate=lambda c: None, params=p)
    st = eng.states["NSE:NIFTY50-INDEX"]
    assert (st.sr_engine.zone_atr_multiple, st.sr_engine.min_zone_width_points) == (0.3, 6.0)
    assert st.swing_detector_15m.left_bars == 4 and st.swing_detector_1m.right_bars == 2
    assert st.breakout_machine_5m.max_retest_candles == 7 and st.breakout_machine_5m.min_breakout_rvol == 1.6
    assert st.breakout_machine_1m.stop_buffer_atr == 0.3 and st.breakout_machine_1m.min_approach_points == 6.0
    assert eng.multi_timeframe_analyzer.params is p
    assert eng.make_shadow().params is p  # off-thread rebuilds use the same strategy
    # a state created later (e.g. a new futures contract at rollover) also gets the params
    eng.set_futures_ticker("NSE:NIFTY-FUT-NEXT")
    assert eng.states["NSE:NIFTY-FUT-NEXT"].sr_engine.zone_atr_multiple == 0.3


def _candidate_scenario(params=None):
    """The documented 1m-entry scenario (bullish PDH breakout, RSI 60, futures RVOL 2.1)."""
    candidates = []
    eng = TechnicalAnalysisEngine(primary_index="NSE:NIFTY50-INDEX", primary_futures="NSE:NIFTY26NOVFUT",
                                  params=params)
    analyzer = MultiTimeframeAnalyzer(engine=eng, on_candidate=candidates.append)
    sym, base = "NSE:NIFTY50-INDEX", IST.localize(datetime(2026, 11, 23, 10, 0))
    st = eng.states[sym]
    s15 = TechnicalSnapshot(symbol=sym, timestamp=base - timedelta(minutes=15), timeframe="15m", open=25000.0,
                            high=25120.0, low=24980.0, close=25110.0, volume=0, trend_bias="BULLISH")
    s5 = TechnicalSnapshot(symbol=sym, timestamp=base, timeframe="5m", open=25080.0, high=25125.0, low=25075.0,
                           close=25115.0, volume=0, atr=20.0, rvol=2.1, vwap=25080.0)
    for tf, s in (("15m", s15), ("5m", s5)):
        st.latest_snapshots[tf] = s
        st.snapshot_history[tf].append(s)
    setup = BreakoutContext(level_id="PDH_25100.0", level_type="PDH", nominal_price=25100.0,
                            direction=BreakoutDirection.BULLISH, current_state=BreakoutState.SETUP_ELIGIBLE,
                            start_time=base, symbol=sym, timeframe="5m", stop_loss=25075.0, target_price=25175.0)
    if analyzer.on_5m_setup(setup, s5) is None:
        return None, analyzer
    t1 = base + timedelta(minutes=5)
    c1 = Candle(symbol=sym, timeframe="1m", timestamp=t1, open=25105.0, high=25118.0, low=25102.0, close=25115.0,
                volume=0, is_closed=True)
    s1 = TechnicalSnapshot(symbol=sym, timestamp=t1, timeframe="1m", open=25105.0, high=25118.0, low=25102.0,
                           close=25115.0, volume=0, ema_9=25108.0, vwap=25095.0, rsi=60.0)
    st.latest_snapshots["1m"] = s1
    st.snapshot_history["1m"].append(s1)
    analyzer.on_1m_candle(c1, s1)
    return (candidates[0] if candidates else None), analyzer


def test_entry_targets_scoring_and_qualification_params_change_the_decision():
    base, an = _candidate_scenario()
    assert base is not None and base.entry_price == 25115.0 and base.stop_loss == 25075.0
    assert (base.target_1, base.target_2) == (25175.0, 25195.0)  # 1.5R / 2.0R on 40 points of risk
    assert an.min_confluence_score == 60.0 and an.entry_window_minutes == 15
    score = base.confluence_score

    c, _ = _candidate_scenario(_params(**{"targets.target_1_rr": 2.0, "targets.target_2_rr": 3.0}))
    assert (c.target_1, c.target_2) == (25195.0, 25235.0)
    c, _ = _candidate_scenario(_params(**{"entry.min_risk_points": 50.0, "entry.fallback_stop_points": 60.0}))
    assert c.stop_loss == 25065.0 and c.risk_points == 50.0
    assert _candidate_scenario(_params(**{"entry.rsi_overbought": 55.0}))[0] is None        # RSI 60 now exhausted
    assert _candidate_scenario(_params(**{"scoring.min_score": score + 1}))[0] is None       # score gate
    c, _ = _candidate_scenario(_params(**{"scoring.rsi_bull_range": [61.0, 68.0]}))
    assert c.confluence_score == score - 10.0                                                 # RSI points lost
    c, _ = _candidate_scenario(_params(**{"scoring.trend_aligned": 35.0}))
    assert c.confluence_score == min(100.0, score + 5.0)
    c, an2 = _candidate_scenario(_params(**{"qualification.max_15m_macro_age_seconds": 60.0}))
    assert c is None and not an2.active_setups                                                # 15m too old


# ------------------------------------------------------------------ replay + live runner use the registry
def test_replay_with_registry_params_reproduces_the_pre_registry_engine():
    """Golden digests recorded with the frozen pre-registry code (commit 183ef0a) on the synthetic
    market, both bar-delivery orders. Any behavioural drift in the registry wiring changes them."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_replay as T

    from nifty_engine.research.replay import ReplayDriver, signals_digest

    days = T.trading_days(T.CAL, date(2026, 6, 1), date(2026, 6, 26))
    store = T._synthetic_store(days)
    golden = {"tick": "55d8a61cf40a2c7d", "flush": "5e7eac5ad63d6686"}
    for order, digest in golden.items():
        r = ReplayDriver(store, delivery_order=order).run(days[-2], days[-1], warmup_days=10, require_manifest=False)
        assert signals_digest(r.signals) == digest, order
        assert r.provenance["strategy"]["id"] == "S000" and r.provenance["strategy"]["params_override"] is False
    # the wiring is live: a different breakout threshold changes the replayed signals
    p = _params(**{"breakouts.5m.min_breakout_rvol": 1.8, "breakouts.1m.min_breakout_rvol": 1.8})
    r = ReplayDriver(store, params=p).run(days[-2], days[-1], warmup_days=10, require_manifest=False)
    assert signals_digest(r.signals) != golden["tick"] and r.provenance["strategy"]["params_override"] is True


def test_live_runner_builds_s000_from_the_registry_and_refuses_a_bad_one(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_market_data import MarketDataEngine

    from nifty_engine.data.storage.duckdb_store import DuckDBStore

    runner = MarketDataEngine(store=DuckDBStore(db_path=":memory:"))
    assert runner.strategy.id == "S000"
    assert runner.technical_engine.params == load_registry().get("S000").params
    assert runner.technical_engine.multi_timeframe_analyzer.params == runner.technical_engine.params

    bad = tmp_path / "registry.yaml"
    raw = _raw()
    raw["strategies"]["S000"]["params"]["entry"]["rsi_overbought"] = 10
    bad.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(reg_mod, "DEFAULT_REGISTRY", bad)
    with pytest.raises(RegistryError):
        MarketDataEngine(store=DuckDBStore(db_path=":memory:"))


# ------------------------------------------------------------------ strict types (no coercion)
@pytest.mark.parametrize("label,mutate,match", [
    ("numeric string", lambda r: r["strategies"]["S000"]["params"]["breakouts"]["5m"].update(min_breakout_rvol="1.3"), "min_breakout_rvol"),
    ("integer string", lambda r: r["strategies"]["S000"]["params"]["entry"].update(entry_window_minutes="15"), "entry_window_minutes"),
    ("boolean as number", lambda r: r["strategies"]["S000"]["params"]["scoring"].update(rsi_points=True), "rsi_points"),
    ("float for an integer", lambda r: r["strategies"]["S000"]["params"]["swings"]["5m"].update(left_bars=3.0), "left_bars"),
    ("tier value as string", lambda r: r["strategies"]["S000"]["params"]["scoring"].update(rvol_tiers=[["2.0", 25.0], [1.5, 20.0]]), "rvol_tiers"),
    ("version as string", lambda r: r["strategies"]["S000"].update(version="1"), "version"),
])
def test_registry_types_are_not_coerced(tmp_path, label, mutate, match):
    with pytest.raises(RegistryError, match=match):
        _load_modified(tmp_path, mutate)


# ------------------------------------------------------------------ version immutability
def test_locked_version_cannot_change_its_parameters(tmp_path):
    with pytest.raises(RegistryError, match="S000 v1 is locked"):
        _load_modified(tmp_path, lambda r: r["strategies"]["S000"]["params"]["entry"].update(rsi_overbought=72.0))


def test_new_version_must_be_locked_before_it_becomes_baseline(tmp_path):
    def bump(r, status):
        e = r["strategies"]["S000"]
        e["params"]["entry"]["rsi_overbought"] = 72.0
        e["version"], e["status"] = 2, status
        e["changelog"].append({"version": 2, "date": "2026-12-01", "change": "RSI 72"})

    with pytest.raises(RegistryError, match="no version lock"):
        _load_modified(tmp_path, lambda r: bump(r, "baseline"))
    e = _load_modified(tmp_path, lambda r: bump(r, "research")).get("S000")   # research may iterate unlocked
    assert e.version == 2 and e.params.entry.rsi_overbought == 72.0
    locks = {"S000": {"1": "1f615431ffcea0ea", "2": e.params_hash()}}
    assert _load_modified(tmp_path, lambda r: bump(r, "baseline"), locks=locks).get("S000").version == 2


def test_lock_file_is_required(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(REGISTRY.read_text())
    with pytest.raises(RegistryError, match="lock file not found"):
        load_registry(p)


def test_committed_lock_matches_s000_v1():
    locks = json.loads(REGISTRY.with_name("registry.lock.json").read_text())
    assert locks["S000"]["1"] == load_registry().get("S000").params_hash() == "1f615431ffcea0ea"


# ------------------------------------------------------------------ strategy data requirements
def _synthetic(days_range=(date(2026, 6, 1), date(2026, 6, 19))):
    sys.path.insert(0, str(Path(__file__).parent))
    import test_replay as T

    days = T.trading_days(T.CAL, *days_range)
    return T, days, T._synthetic_store(days)


def test_replay_refuses_inputs_the_strategy_does_not_declare(tmp_path):
    from nifty_engine.research.manifest import ManifestError
    from nifty_engine.research.replay import ReplayDriver

    T, days, store = _synthetic()
    no_fut_daily = _load_modified(tmp_path, lambda r: r["strategies"]["S000"]["data_requirements"]["series"][1]
                                  .update(resolutions=["1"])).get("S000")
    with pytest.raises(ManifestError, match=r"NIFTY_FUT/CONT \['D'\] not declared"):
        ReplayDriver(store, strategy=no_fut_daily).run(days[-1], days[-1], require_manifest=False)
    drv = ReplayDriver(store)
    manifest = {"series": [{"instrument": "NIFTY_SPOT", "source": "SPOT", "required": True},
                           {"instrument": "NIFTY_FUT", "source": "CONT", "required": False}]}
    with pytest.raises(ManifestError, match="NIFTY_FUT/CONT: required by S000 but not validated"):
        drv.check_data_requirements(manifest)
    manifest["series"][1]["required"] = True
    drv.check_data_requirements(manifest)  # passes


def test_replay_skips_days_whose_contract_confidence_is_not_accepted(tmp_path):
    from nifty_engine.research.replay import ReplayDriver

    T, days, store = _synthetic()
    store._write("""CREATE TABLE cont_segments (instrument VARCHAR, source VARCHAR, segment_id INTEGER,
        start_day DATE, end_day DATE, contract_expiry DATE, contract_symbol VARCHAR, confidence VARCHAR,
        checked_bars INTEGER, matched_bars INTEGER, segment_bars INTEGER, coverage DOUBLE)""")
    store._write("INSERT INTO cont_segments VALUES ('NIFTY_FUT','CONT',1,?,?,NULL,NULL,'VERIFIED',0,0,0,1.0)",
                 [days[0], days[-2]])
    store._write("INSERT INTO cont_segments VALUES ('NIFTY_FUT','CONT',2,?,?,NULL,NULL,'INFERRED',0,0,0,0.0)",
                 [days[-1], days[-1]])
    verified_only = _load_modified(tmp_path, lambda r: r["strategies"]["S000"]["data_requirements"]
                                   .update(contract_confidence=["VERIFIED"])).get("S000")
    res = ReplayDriver(store, strategy=verified_only).run(days[-2], days[-1], require_manifest=False)
    assert res.days == [days[-2]]
    assert res.skipped_days == [(days[-1], "futures contract confidence INFERRED not accepted by S000")]
    assert res.provenance["contract_confidence_days"] == {"VERIFIED": 1}
    assert res.provenance["warmup_trading_days"] == 10
