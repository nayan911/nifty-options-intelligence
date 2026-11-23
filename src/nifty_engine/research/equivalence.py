"""Equivalence check: the replay driver vs the live runner's candle path on identical bars.

The live pipeline delivers candles through MarketDataEngine._on_candle_closed (signal-symbol
filter, gap-hold, event lock, paper engine, readiness), not by calling the engine directly. This
module runs one session both ways from the same warmed-up start and compares the signals:

  replay  : ReplayDriver.replay_day()            -> engine.on_candle(c)
  runner  : MarketDataEngine._on_candle_closed(c) -> ... -> technical_engine.on_candle(c)

Identical digests prove the replay reproduces what the live system would have produced from the
same bars.
"""

import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import List, Tuple

import pytz

from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.research.replay import (
    EXPIRY_PROBES,
    FUT,
    SPOT,
    ReplayDriver,
    ReplaySignal,
    pinned_clock,
    signals_digest,
)

IST = pytz.timezone("Asia/Kolkata")


def runner_path_signals(driver: ReplayDriver, d: date) -> List[ReplaySignal]:
    """Replay session `d` through the live runner's _on_candle_closed path."""
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from run_market_data import MarketDataEngine  # live runner

    from nifty_engine.data.storage.duckdb_store import DuckDBStore

    store = driver._prepare()
    runner = MarketDataEngine(store=DuckDBStore(db_path=":memory:"))
    eng = runner.technical_engine
    eng.set_futures_ticker(FUT)
    eng.set_index_futures_mapping(SPOT, FUT)
    out: List[ReplaySignal] = []

    def on_setup(setup, snap):
        out.append(ReplaySignal("SETUP", d, snap.timestamp, snap.symbol, snap.timeframe, setup.direction.value, {
            "level_id": setup.level_id, "level": setup.nominal_price, "stop_loss": setup.stop_loss,
            "target": setup.target_price, "close": snap.close, "trend": snap.trend_bias}))

    def on_candidate(c):
        out.append(ReplaySignal("CANDIDATE", d, c.timestamp, c.symbol, "1m", c.direction.value, {
            "candidate_id": c.candidate_id, "level_id": c.setup_level, "entry": c.entry_price,
            "stop_loss": c.stop_loss, "target_1": c.target_1, "target_2": c.target_2,
            "risk_points": c.risk_points, "confluence": c.confluence_score,
            "headroom": c.headroom_points, "confirmation": c.confirmation_status,
            "decision_at": c.decision_at.isoformat() if c.decision_at else None}))

    # Same callbacks the replay records; the runner's own printing/paper paths are bypassed.
    eng.on_setup_candidate = on_setup
    if eng.multi_timeframe_analyzer is not None:
        eng.multi_timeframe_analyzer.on_candidate = on_candidate
    eng.derivatives_engine = None
    if eng.multi_timeframe_analyzer is not None:
        eng.multi_timeframe_analyzer.derivatives_engine = None
    runner.evaluate_and_update_readiness = lambda: None  # readiness side effects irrelevant here

    with pinned_clock(IST.localize(datetime.combine(d, time(9, 0)))):
        hm = HistoricalCandleManager(rest_client=None, store=store, calendar=driver.cal)
        hm.warm_up_technical_engine(engine=eng, symbols=[SPOT, FUT], as_of_date=d, reconcile_session=False)

    for close_t, candles in driver._day_events(d):
        for c in candles:
            runner._on_candle_closed(c)
        for s in EXPIRY_PROBES:
            eng.expire_pending_evaluations(close_t + timedelta(seconds=s))
    return out


def compare_day(driver: ReplayDriver, d: date) -> Tuple[bool, str, str, int]:
    logging.disable(logging.WARNING)
    try:
        replay: List[ReplaySignal] = []
        driver.replay_day(d, replay)
        runner = runner_path_signals(driver, d)
    finally:
        logging.disable(logging.NOTSET)
    a, b = signals_digest(replay), signals_digest(runner)
    return a == b, a, b, len(replay)
