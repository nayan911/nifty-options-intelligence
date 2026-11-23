"""Historical replay through the LIVE technical engine (Phase 2, option B).

S000's logic is never re-implemented: each simulated session runs the exact production path:
  1. a fresh TechnicalAnalysisEngine (+ MultiTimeframeAnalyzer), as the live pipeline restarts daily
  2. HistoricalCandleManager.warm_up_technical_engine() with the clock pinned to 09:00 IST of the
     simulated day (prior-day history, PDH/PDL/PDC, indicator bootstrap; no look-ahead)
  3. closed candles delivered at each bar close in live order: per symbol 1m -> 5m -> 15m
     (RealtimeCandleBuilder closes timeframes in that order), spot before futures
  4. engine.expire_pending_evaluations() at the times the live 1-second loop would hit its
     thresholds (+1s, +6s for the 5s 1m-pairing timeout, +31s for the 30s 5m timeout, +59s)

Futures input is the validated near-month stitched series (CONT:<UND>:NEAR), which is what the
live warmup sees too (FYERS cont_flag=1 via the nearest contract). Every run records provenance.
"""

import contextlib
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import pytz

from nifty_engine.analysis.technical.engine import TechnicalAnalysisEngine
from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle
from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.research.bars import (
    bar_build_report,
    build_intraday_candles,
    daily_candles,
    summarize_build_report,
    trading_days,
)
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.manifest import DEFAULT_MANIFEST, dataset_fingerprint, require_valid_manifest
from nifty_engine.research.sessions import CASH, FNO
from nifty_engine.strategy.params import S000Params
from nifty_engine.strategy.registry import StrategyEntry, load_registry

IST = pytz.timezone("Asia/Kolkata")
SPOT = "NSE:NIFTY50-INDEX"
FUT = "NSE:NIFTY-FUT-NEAR"   # logical name for the near-month stitched series in replay
EXPIRY_PROBES = (1, 6, 31, 59)  # seconds after a bar close at which the live loop's timeouts fire


@contextlib.contextmanager
def pinned_clock(when: datetime) -> Iterator[None]:
    """Pin the wall clock seen by the warmup/data layer (it calls now_ist() for session logic)."""
    with mock.patch("nifty_engine.data.historical_manager.now_ist", return_value=when):
        yield


@dataclass
class ReplaySignal:
    kind: str            # SETUP (raw 5m/1m breakout-retest) or CANDIDATE (multi-timeframe entry)
    day: date
    at: datetime         # bar timestamp that produced it
    symbol: str
    timeframe: str
    direction: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    start: date
    end: date
    days: List[date]
    signals: List[ReplaySignal]
    provenance: Dict[str, Any]
    skipped_days: List[Tuple[date, str]] = field(default_factory=list)
    # Why bars were excluded, per series: {"NSE:NIFTY50-INDEX": {"summary": {...}, "days": {date: {...}}}}
    validation: Dict[str, Any] = field(default_factory=dict)

    def candidates(self) -> List[ReplaySignal]:
        return [s for s in self.signals if s.kind == "CANDIDATE"]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ReplayDriver:
    def __init__(self, research: ResearchHistoryStore, calendar: Optional[ExchangeCalendar] = None,
                 spot_instrument: str = "NIFTY_SPOT", fut_instrument: str = "NIFTY_FUT",
                 fut_store_symbol: str = "CONT:NIFTY:NEAR",
                 engine_factory: Optional[Callable[..., TechnicalAnalysisEngine]] = None,
                 delivery_order: str = "tick", strategy: Optional[StrategyEntry] = None,
                 params: Optional[S000Params] = None) -> None:
        """strategy: registry entry to replay (default: S000 from strategies/registry.yaml).
        params: optional override for research experiments; provenance flags it as an override."""
        self.research = research
        self.strategy = strategy if strategy is not None else load_registry().get("S000")
        self.params = params if params is not None else self.strategy.params
        self.params_override = params is not None and params != self.strategy.params
        self.cal = calendar or ExchangeCalendar()
        self.spot_instrument = spot_instrument
        self.fut_instrument = fut_instrument
        self.fut_store_symbol = fut_store_symbol
        self.engine_factory = engine_factory or (lambda **kw: TechnicalAnalysisEngine(**kw))
        # Live emission order at a boundary depends on what closes the bars first: the next tick
        # (1m -> 5m -> 15m) or the 1-second clock flush (15m -> 5m -> 1m). Both occur live.
        if delivery_order not in ("tick", "flush"):
            raise ValueError("delivery_order must be 'tick' or 'flush'")
        self.delivery_order = delivery_order
        self._store: Optional[DuckDBStore] = None
        self._intraday: Dict[str, Dict[str, List[Candle]]] = {}

    # ------------------------------------------------------------------ data preparation
    def _prepare(self) -> DuckDBStore:
        """Load validated research bars into an in-memory live-format store (what warmup reads)."""
        if self._store is not None:
            return self._store
        store = DuckDBStore(db_path=":memory:")
        q = """SELECT ts, open, high, low, close, volume, oi FROM hist_bars
               WHERE instrument = ? AND source = ? AND resolution = ? {sym} ORDER BY ts"""
        spot1 = self.research.conn.execute(q.format(sym=""), [self.spot_instrument, "SPOT", "1"]).df()
        fut1 = self.research.conn.execute(q.format(sym="AND symbol = ?"),
                                          [self.fut_instrument, "CONT", "1", self.fut_store_symbol]).df()
        spotd = self.research.conn.execute(q.format(sym=""), [self.spot_instrument, "SPOT", "D"]).df()
        futd = self.research.conn.execute(q.format(sym="AND symbol = ?"),
                                          [self.fut_instrument, "CONT", "D", self.fut_store_symbol]).df()
        self._intraday = {
            SPOT: build_intraday_candles(spot1, self.cal, CASH, SPOT),
            FUT: build_intraday_candles(fut1, self.cal, FNO, FUT),
        }
        self._build_reports = {SPOT: bar_build_report(spot1, self.cal, CASH),
                               FUT: bar_build_report(fut1, self.cal, FNO)}
        for sym, tfs in self._intraday.items():
            for candles in tfs.values():
                store.insert_candles_batch(candles)
        store.insert_candles_batch(daily_candles(spotd, SPOT))
        store.insert_candles_batch(daily_candles(futd, FUT))
        self._store = store
        return store

    def _day_events(self, d: date) -> List[Tuple[datetime, List[Candle]]]:
        """[(bar_close_time, [candles closing then, in live delivery order])] for session day d."""
        by_close: Dict[datetime, List[Candle]] = {}
        for sym in (SPOT, FUT):
            for tf in ("1m", "5m", "15m"):
                for c in self._intraday[sym][tf]:
                    if c.timestamp.date() == d:
                        if sym == SPOT:
                            # Same data as live: the live feed has no index volume, so live-built index
                            # bars carry volume 0 (FYERS *history* reports one; warmup keeps it, as live).
                            c = c.model_copy(update={"volume": 0})
                        by_close.setdefault(c.observed_at, []).append(c)
        tf_rank = {"1m": 0, "5m": 1, "15m": 2} if self.delivery_order == "tick" else {"15m": 0, "5m": 1, "1m": 2}
        order = {(sym, tf): i * 3 + tf_rank[tf] for i, sym in enumerate((SPOT, FUT)) for tf in tf_rank}
        return [(t, sorted(cs, key=lambda c: order[(c.symbol, c.timeframe)])) for t, cs in sorted(by_close.items())]

    # ------------------------------------------------------------------ replay
    def replay_day(self, d: date, sink: List[ReplaySignal]) -> None:
        store = self._prepare()

        def on_setup(setup, snap):
            sink.append(ReplaySignal("SETUP", d, snap.timestamp, snap.symbol, snap.timeframe, setup.direction.value, {
                "level_id": setup.level_id, "level": setup.nominal_price, "stop_loss": setup.stop_loss,
                "target": setup.target_price, "close": snap.close, "trend": snap.trend_bias}))

        def on_candidate(c):
            sink.append(ReplaySignal("CANDIDATE", d, c.timestamp, c.symbol, "1m", c.direction.value, {
                "candidate_id": c.candidate_id, "level_id": c.setup_level, "entry": c.entry_price,
                "stop_loss": c.stop_loss, "target_1": c.target_1, "target_2": c.target_2,
                "risk_points": c.risk_points, "confluence": c.confluence_score,
                "headroom": c.headroom_points, "confirmation": c.confirmation_status,
                "decision_at": c.decision_at.isoformat() if c.decision_at else None}))

        engine = self.engine_factory(primary_index=SPOT, primary_futures=FUT, on_setup_candidate=on_setup,
                                     on_technical_candidate=on_candidate, params=self.params)
        start_of_day = IST.localize(datetime.combine(d, time(9, 0)))
        with pinned_clock(start_of_day):
            hm = HistoricalCandleManager(rest_client=None, store=store, calendar=self.cal)
            hm.warm_up_technical_engine(engine=engine, symbols=[SPOT, FUT], as_of_date=d, reconcile_session=False)

        for close_t, candles in self._day_events(d):
            for c in candles:
                engine.on_candle(c)
            for s in EXPIRY_PROBES:
                engine.expire_pending_evaluations(close_t + timedelta(seconds=s))

    def run(self, start: date, end: date, warmup_days: int = 10, require_manifest: bool = True,
            manifest_path=DEFAULT_MANIFEST) -> ReplayResult:
        """Replay every trading day in [start, end]. Days within `warmup_days` of the dataset start
        are skipped (not enough prior history for a live-equivalent warmup).

        Fails closed: unless `require_manifest=False` (synthetic unit tests only), the validated
        dataset manifest must be ok, cover every day the replay reads (warmup history included) and
        match the current data's fingerprint.
        """
        first = self.research.conn.execute(
            "SELECT min(CAST(ts AS DATE)) FROM hist_bars WHERE instrument = ? AND resolution = '1'",
            [self.spot_instrument],
        ).fetchall()[0][0]
        manifest = None
        if require_manifest:
            manifest = require_valid_manifest(first, end, self.research, manifest_path)
        self._prepare()
        all_days = trading_days(self.cal, first, end)
        min_day = all_days[warmup_days] if len(all_days) > warmup_days else end + timedelta(days=1)
        signals: List[ReplaySignal] = []
        days, skipped = [], []
        for d in trading_days(self.cal, start, end):
            if d < min_day:
                skipped.append((d, f"insufficient warmup history (< {warmup_days} prior sessions)"))
                continue
            self.replay_day(d, signals)
            days.append(d)
        prov = self.provenance(start, end)
        prov["manifest"] = (
            {"validated": True, "generated_at": manifest.get("generated_at"),
             "fingerprint": manifest.get("dataset_fingerprint")} if manifest else {"validated": False}
        )
        validation = {}
        for sym, rep in self._build_reports.items():
            in_range = {d: r for d, r in rep.items() if d in set(days)}
            validation[sym] = {
                "summary": summarize_build_report(in_range),
                "days_with_exclusions": {d.isoformat(): r for d, r in in_range.items()
                                         if r["out_of_session"] or r["missing_1m"] or r["5m_dropped_late_start"]
                                         or r["15m_dropped_late_start"]},
            }
        return ReplayResult(start, end, days, signals, prov, skipped, validation)

    def provenance(self, start: date, end: date) -> Dict[str, Any]:
        return {
            "dataset_fingerprint": dataset_fingerprint(self.research),
            "git_commit": _git_commit(),
            "engine": "TechnicalAnalysisEngine (live code path)",
            "strategy": {
                **self.strategy.provenance(),
                "params_override": self.params_override,
                "effective_params_hash": hashlib.sha256(
                    json.dumps(self.params.canonical(), sort_keys=True).encode()).hexdigest()[:16],
                "delivery_order": self.delivery_order,
            },
            "futures_series": self.fut_store_symbol,
            "range": [start.isoformat(), end.isoformat()],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }


def signals_digest(signals: List[ReplaySignal]) -> str:
    """Stable hash of a signal list (for equivalence comparisons)."""
    rows = [(s.kind, s.at.isoformat(), s.symbol, s.timeframe, s.direction,
             json.dumps(s.detail, sort_keys=True, default=str)) for s in signals]
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]
