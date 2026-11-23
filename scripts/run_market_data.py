#!/usr/bin/env python3
"""Main High-Reliability Market Data Ingestion & Live Analytics Engine.

Runs locally, streams WebSocket market ticks, applies TickNormalizer for volume
and out-of-order quarantine, builds 1m/5m/15m OHLCV bars, and persists data into DuckDB.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import pytz

from nifty_engine.analysis.derivatives import (
    DerivativesIntelligenceEngine,
    OptionContractSelector,
)
from nifty_engine.analysis.technical.breakouts import BreakoutContext
from nifty_engine.analysis.technical.engine import (
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
)
from nifty_engine.analysis.technical.multi_timeframe import TechnicalCandidate
from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import Candle, OptionType, SessionState, SystemReadiness, Tick
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.candle_builder import RealtimeCandleBuilder
from nifty_engine.data.freshness_monitor import DataFreshnessMonitor
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
from nifty_engine.data.fyers.rest_client import FyersRateLimitError, FyersRestClient
from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient
from nifty_engine.data.gap_backfill import FeedGapBackfiller
from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus
from nifty_engine.paper_trading.engine import PaperTradingEngine
from nifty_engine.utils.timezone import now_ist

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


class MarketDataEngine:
    """Orchestrates live market data streaming, volume normalization, and storage."""

    def __init__(self, store: Optional[DuckDBStore] = None):
        self.calendar = ExchangeCalendar()
        self.auth = FyersAuthProvider()
        self.store = store or DuckDBStore()
        self.freshness = DataFreshnessMonitor()
        self.master = FyersInstrumentMaster()
        self.normalizer = TickNormalizer(self.calendar)
        # ONE limiter for every REST call (history, option chain, quotes); usage persists per IST day
        self.rate_limiter = AsyncTokenBucketRateLimiter(usage_file=settings.API_USAGE_FILE)
        self.rest_client = FyersRestClient(self.auth, rate_limiter=self.rate_limiter)
        self.historical_manager = HistoricalCandleManager(
            rest_client=self.rest_client,
            store=self.store,
            calendar=self.calendar,
        )
        self.running = True

        self.builder = RealtimeCandleBuilder(on_candle_close=self._on_candle_closed)
        self.contract_selector = OptionContractSelector(quote_provider=self._get_live_quote)
        self.derivatives_engine = DerivativesIntelligenceEngine(
            primary_futures="NSE:NIFTY26NOVFUT",
            contract_selector=self.contract_selector,
        )
        self.technical_engine = TechnicalAnalysisEngine(
            primary_index=settings.PRIMARY_INDEX,
            primary_futures="NSE:NIFTY26NOVFUT",
            on_setup_candidate=self._on_setup_candidate,
            on_technical_candidate=self._on_technical_candidate,
            derivatives_engine=self.derivatives_engine,
        )
        self.paper_engine = PaperTradingEngine(
            store=self.store,
            instrument_master=self.master,
            quote_provider=self._get_live_quote,
            on_trade_event=self._on_paper_trade_event,
            max_trades_per_day=settings.PAPER_MAX_TRADES_PER_DAY,
            max_daily_net_loss=settings.PAPER_MAX_DAILY_NET_LOSS,
            max_consecutive_losses=settings.PAPER_MAX_CONSECUTIVE_LOSSES,
            loss_cooldown_minutes=settings.PAPER_LOSS_COOLDOWN_MINUTES,
        )
        self.ws_client = FyersWebSocketClient(
            auth_provider=self.auth,
            on_tick=self._on_tick_received,
            on_ack=self._on_ws_ack,
        )

        # Observability & diagnostic counters
        self.total_ticks_received = 0
        self.live_ticks_accepted = 0
        self.unverified_ticks_accepted = 0
        self.snapshots_accepted = 0
        self.duplicates_dropped = 0
        self.out_of_order_quarantined = 0
        self.malformed_dropped = 0
        self.ticks_persisted = 0
        self.candle_count = 0
        self.symbol_stats: Dict[str, Dict[str, int]] = {}

        self.current_spot_price: Optional[float] = None
        self.active_atm_strike: Optional[float] = None
        self.subscribed_option_symbols: Set[str] = set()
        self.active_paper_trade_symbols: Set[str] = set()

        # Telemetry: Feed Age vs Exchange Network Latency
        self.last_primary_tick_time: Optional[datetime] = None
        # Independent freshness per signal instrument (spot AND futures): either one going
        # stale opens a feed gap and blocks entries.
        self.last_signal_tick_times: Dict[str, datetime] = {}
        self.last_primary_network_latency: Optional[float] = None

        # Scheduled Option Chain Snapshot Collection & Failure Backoff
        self.last_option_chain_time: Optional[datetime] = None
        self.latest_option_chain_pcr: Optional[float] = None
        self.latest_total_call_oi: Optional[int] = None
        self.latest_total_put_oi: Optional[int] = None
        self.option_chain_failure_count: int = 0
        self.option_chain_last_error: Optional[str] = None

        # Readiness override de-duplication (log on change, not every evaluation)
        self._last_readiness_override: Optional[str] = None

        # Candle closes arrive from TWO threads: the websocket thread (a tick crosses a bar
        # boundary) and the 1s maintenance loop (clock-driven flush). The paper engine,
        # readiness state and counters are not thread-safe, so every event that can touch
        # them is serialized through this lock (re-entrant: callbacks nest).
        self._event_lock = threading.RLock()
        self._shutting_down = False
        # Raw setup alert de-duplication: (symbol, timeframe, bar_ts, direction) -> levels
        self._alerted_setups: Dict[tuple, List[str]] = {}

        # Feed-gap detection + historical backfill for the signal instruments
        self.gap_backfiller = FeedGapBackfiller(
            historical_manager=self.historical_manager,
            technical_engine=self.technical_engine,
            store=self.store,
            symbols_provider=lambda: sorted(self._signal_symbols()),
            gap_seconds=settings.FEED_GAP_SECONDS,
        )
        self.option_chain_rate_limited_until: Optional[datetime] = None
        self.chain_expiries: List[tuple] = []  # [(date, fyers_epoch)] from the last chain response
        self.banknifty_futures: Optional[str] = None
        self._chain_expiry_note: Optional[str] = None

    def _get_live_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Provides on-demand authentic broker quote snapshot for paper engine."""
        try:
            q_res = self.rest_client.get_quote(symbol, blocking=False)
            if q_res and q_res.get("s") == "ok" and q_res.get("d"):
                val = q_res["d"][0].get("v", {})
                ltp = float(val.get("lp", 0.0))
                # Authentic broker bid/ask depth only: NEVER substitute LTP for bid or ask
                raw_bid = val.get("bid")
                raw_ask = val.get("ask")
                bid = float(raw_bid) if (raw_bid is not None and float(raw_bid) > 0) else 0.0
                ask = float(raw_ask) if (raw_ask is not None and float(raw_ask) > 0) else 0.0

                # Extract authentic broker quote timestamp if available
                quote_ts = None
                tt = val.get("tt")
                if tt is not None and int(tt) > 0:
                    try:
                        quote_ts = datetime.fromtimestamp(int(tt), tz=pytz.timezone("Asia/Kolkata"))
                    except Exception:
                        quote_ts = None

                return {
                    "ltp": ltp,
                    "bid": bid,
                    "ask": ask,
                    "timestamp": quote_ts,
                }
        except Exception as e:
            logger.debug("Failed on-demand quote lookup for %s: %s", symbol, e)
        return None

    def _on_ws_ack(self, ack_msg: dict):
        """Callback on WebSocket subscription and status acknowledgements."""
        print(f"[WS ACK] {ack_msg.get('type')}: {ack_msg.get('message', 'OK')}")

    def _on_paper_trade_event(self, ev: Dict[str, Any]) -> None:
        """Callback on simulated paper execution fills and position closures."""
        if ev.get("event") == "ORDER_FILLED":
            print(
                f"\n{'#'*78}\n"
                f"  [PAPER EXECUTION: BUY ORDER FILLED]\n"
                f"  Position ID: {ev.get('position_id')}\n"
                f"  Contract: {ev.get('symbol')} | Quantity: {ev.get('quantity')} | Fill Price: ₹{ev.get('price', 0.0):.2f}\n"
                f"  Statutory Friction: ₹{ev.get('costs', 0.0):.2f} (Brokerage, STT, Exchange, GST)\n"
                f"{'#'*78}\n"
            )
        elif ev.get("event") == "POSITION_CLOSED":
            gross = ev.get("gross_pnl", 0.0)
            net = ev.get("net_pnl", 0.0)
            status_tag = "WIN" if net > 0 else "LOSS"
            print(
                f"\n{'#'*78}\n"
                f"  [PAPER EXECUTION: POSITION CLOSED ({status_tag})]\n"
                f"  Position ID: {ev.get('position_id')} | Reason: {ev.get('reason')}\n"
                f"  Exit Price: ₹{ev.get('exit_price', 0.0):.2f} | Gross P&L: ₹{gross:+.2f} | Net P&L: ₹{net:+.2f}\n"
                f"  Total Statutory Taxes & Brokerage: ₹{ev.get('total_costs', 0.0):.2f}\n"
                f"{'#'*78}\n"
            )

    def _on_tick_received(self, raw_tick: Tick, current_time: Optional[datetime] = None):
        """Processes raw tick through normalizer, sanitizes volume, persists, and builds bars."""
        if self._shutting_down:
            return
        sym = raw_tick.symbol
        self.total_ticks_received += 1

        if sym not in self.symbol_stats:
            self.symbol_stats[sym] = {
                "received": 0,
                "live": 0,
                "snapshot": 0,
                "duplicate": 0,
                "out_of_order": 0,
                "malformed": 0,
            }
        self.symbol_stats[sym]["received"] += 1

        res = self.normalizer.normalize(raw_tick, current_time=current_time)

        if res.status == TickStatus.DUPLICATE:
            self.duplicates_dropped += 1
            self.symbol_stats[sym]["duplicate"] += 1
            return
        elif res.status == TickStatus.OUT_OF_ORDER:
            self.out_of_order_quarantined += 1
            self.symbol_stats[sym]["out_of_order"] += 1
            return
        elif res.status == TickStatus.MALFORMED:
            self.malformed_dropped += 1
            self.symbol_stats[sym]["malformed"] += 1
            return

        if not res.tick:
            return

        tick = res.tick
        self.freshness.record_tick(tick)

        # 1. Save normalized tick (with single source of truth incremental volume) to DuckDB
        self.store.insert_tick(tick)
        self.ticks_persisted += 1

        # 2. Track NIFTY spot price and telemetry
        if tick.symbol == settings.PRIMARY_INDEX:
            self.current_spot_price = tick.ltp

        # Feed freshness advances ONLY on ticks that passed live-market validation. A
        # VALID_SNAPSHOT (e.g. the stale last-quote FYERS replays on resubscribe) or an
        # UNVERIFIED quote must not make a dead feed look alive.
        if res.status == TickStatus.VALID_LIVE:
            if tick.symbol in self._signal_symbols():
                self.last_signal_tick_times[tick.symbol] = now_ist()
            if tick.symbol == settings.PRIMARY_INDEX:
                self.last_primary_tick_time = now_ist()
                if tick.exchange_timestamp:
                    self.last_primary_network_latency = (tick.local_timestamp - tick.exchange_timestamp).total_seconds()

        # 3. Only pass verified live trading ticks to CandleBuilder and PaperTradingEngine
        if res.status == TickStatus.VALID_LIVE:
            self.live_ticks_accepted += 1
            self.symbol_stats[sym]["live"] += 1
            self.builder.process_tick(tick)
            if hasattr(self, "paper_engine"):
                with self._event_lock:
                    self.paper_engine.on_tick(tick)
        elif res.status == TickStatus.UNVERIFIED:
            self.unverified_ticks_accepted += 1
            self.symbol_stats[sym].setdefault("unverified", 0)
            self.symbol_stats[sym]["unverified"] += 1
            # Quarantined: Saved to DuckDB for diagnostics, but does not form live breakout bars
        else:
            self.snapshots_accepted += 1
            self.symbol_stats[sym]["snapshot"] += 1

    def _on_candle_closed(self, candle: Candle):
        """Invoked when a 1m/5m/15m bar closes (guaranteed exactly once per closed candle).

        Called from both the websocket thread and the maintenance loop; serialized.
        """
        if self._shutting_down:
            return
        with self._event_lock:
            self._handle_candle_closed(candle)

    def _handle_candle_closed(self, candle: Candle):
        self.candle_count += 1
        self.store.insert_candle(candle)
        # Flush DuckDB buffer on candle close
        self.store.flush_ticks()
        print(
            f"[{candle.timestamp.strftime('%H:%M:%S')}] {candle.symbol} "
            f"({candle.timeframe}) C: {candle.close:.2f} | V: {candle.volume} | Bars Closed: {self.candle_count}"
        )

        # Forward closed 1m base candle to paper engine for intrabar high/low breach checks
        # Restrict strictly to 1m base candles to prevent replaying overlapping multi-timeframe boundaries (e.g. 5m, 15m)
        if hasattr(self, "paper_engine") and candle.timeframe in ("1m", "1M"):
            try:
                self.paper_engine.on_candle(candle)
            except Exception as e:
                logger.warning("Paper engine candle processing error for %s: %s", candle.symbol, e)

        # Real-time technical analysis runs ONLY on the NIFTY spot index and NIFTY futures.
        # Direction is decided from the underlying; option premiums carry theta/IV noise
        # and must never generate their own breakout signals. Options are chosen later by
        # the contract selector based on the underlying's signal.
        if candle.symbol in self._signal_symbols() and self.gap_backfiller.hold(candle):
            pass  # feed gap open: indicators resume after historical backfill repairs the series
        elif candle.symbol in self._signal_symbols():
            try:
                snapshot = self.technical_engine.on_candle(candle)
                if snapshot:
                    self._log_technical_status(snapshot)
            except Exception as e:
                print(f"[WARN] Technical analysis error for {candle.symbol}: {e}")

        # Continuous two-way operational readiness assessment
        self.evaluate_and_update_readiness()

    def _signal_symbols(self) -> Set[str]:
        """Instruments allowed to drive direction/setup detection (NIFTY spot + futures only)."""
        return {settings.PRIMARY_INDEX, self.technical_engine.primary_futures}

    def _signal_tick_times(self) -> Dict[str, Optional[datetime]]:
        return {s: self.last_signal_tick_times.get(s) for s in self._signal_symbols()}

    def _worst_signal_feed_age(self) -> "tuple[Optional[str], Optional[float]]":
        """Stalest signal instrument and its feed age in seconds (None if none seen yet)."""
        now = now_ist()
        ages = {s: (now - t).total_seconds() for s, t in self._signal_tick_times().items() if t is not None}
        if not ages:
            return None, None
        sym = max(ages, key=ages.get)
        return sym, ages[sym]

    def _log_readiness_override(self, reason: str) -> None:
        """Log a readiness override once per distinct reason instead of on every evaluation."""
        if reason != self._last_readiness_override:
            logger.info("Readiness override: %s", reason)
            print(f"[READINESS] {reason}")
            self._last_readiness_override = reason

    def evaluate_and_update_readiness(self) -> SystemReadiness:
        """Continuously re-evaluates operational readiness across data feed, indicators, and market state,

        handling both automatic upgrades (NOT_READY -> DEGRADED -> READY) and safety downgrades.
        """
        if not hasattr(self, "paper_engine"):
            return SystemReadiness.NOT_READY

        prev_readiness = self.paper_engine.system_readiness
        curr_state = self.calendar.get_session_state(is_derivative=True)
        is_live = curr_state in (SessionState.REGULAR_CASH, SessionState.DERIVATIVES_EXT, SessionState.SPECIAL_SESSION)

        # 1. Technical indicator and reference level assessment
        core_tracked = [settings.PRIMARY_INDEX, self.technical_engine.primary_futures]
        now = now_ist()
        assessed_readiness, details = self.historical_manager.assess_readiness(
            self.technical_engine,
            core_tracked,
            as_of_date=now.date(),
        )
        effective_readiness = assessed_readiness

        # 2. Live feed health & connection gating during active market hours
        override_reason: Optional[str] = None
        if is_live:
            if self.gap_backfiller.active:
                effective_readiness = SystemReadiness.NOT_READY
                override_reason = "Feed gap: waiting for historical backfill -> NOT_READY"
            elif not getattr(self.ws_client, "is_connected", False):
                effective_readiness = SystemReadiness.NOT_READY
                override_reason = "WebSocket disconnected during live market -> NOT_READY"
            else:
                worst_sym, feed_age = self._worst_signal_feed_age()
                if feed_age is not None and feed_age > 120.0:
                    effective_readiness = SystemReadiness.NOT_READY
                    override_reason = f"{worst_sym} feed stale > 120s -> NOT_READY"
                elif feed_age is not None and feed_age > 60.0:
                    if effective_readiness == SystemReadiness.READY:
                        effective_readiness = SystemReadiness.DEGRADED
                        override_reason = f"{worst_sym} feed stale > 60s -> DEGRADED"
        if override_reason:
            self._log_readiness_override(override_reason)
        elif self._last_readiness_override is not None:
            print("[READINESS] Feed override cleared")
            self._last_readiness_override = None

        # 3. Two-way state transitions (Upgrade or Downgrade)
        if effective_readiness != prev_readiness:
            self.paper_engine.set_system_readiness(effective_readiness)
            if effective_readiness == SystemReadiness.READY:
                print(
                    f"\n{'*'*78}\n"
                    f"  [SYSTEM READINESS UPGRADE: READY]\n"
                    f"  All indicators verified (1m, 5m, daily levels) and data feed fresh.\n"
                    f"  Paper trading entries ENABLED.\n"
                    f"{'*'*78}\n"
                )
            else:
                action = "DOWNGRADED" if prev_readiness == SystemReadiness.READY else "TRANSITIONED"
                print(
                    f"\n{'!'*78}\n"
                    f"  [SYSTEM READINESS {action}: {effective_readiness.value}]\n"
                    f"  State: {prev_readiness.value} -> {effective_readiness.value} (feed freshness or indicator state).\n"
                    f"  Paper trading entries DISABLED.\n"
                    f"{'!'*78}\n"
                )

        return effective_readiness

    def _log_technical_status(self, snap: TechnicalSnapshot) -> None:
        """Outputs concise quantitative technical telemetry for primary tracked assets."""
        sr_str = ""
        if snap.nearest_support:
            sr_str += f" | Sup: {snap.nearest_support.nominal_price:.1f}"
        if snap.nearest_resistance:
            sr_str += f" | Res: {snap.nearest_resistance.nominal_price:.1f}"

        vwap_str = f" | VWAP: {snap.vwap:.1f}" if snap.vwap is not None else ""
        rsi_str = f" | RSI: {snap.rsi:.1f}" if snap.rsi is not None else ""
        rvol_str = f" | RVOL: {snap.rvol:.2f}x" if snap.rvol is not None else ""
        ema9_str = f" | EMA9: {snap.ema_9:.1f}" if snap.ema_9 is not None else " | EMA9: (warmup)"
        atr_str = f" | ATR: {snap.atr:.1f}" if snap.atr is not None else " | ATR: (warmup)"

        sym_tag = snap.symbol.split(":")[-1]
        warmup_tag = "READY" if snap.is_warmed_up else "WARMING_UP"
        print(
            f"[TECH {sym_tag} {snap.timeframe}] [{warmup_tag}] C: {snap.close:.2f} | Trend: {snap.trend_bias}"
            f"{ema9_str}{atr_str}{vwap_str}{rsi_str}{rvol_str}{sr_str}"
        )

    def _on_setup_candidate(self, setup: BreakoutContext, snap: TechnicalSnapshot) -> None:
        """Invoked when a high-probability technical breakout & retest completes.

        Several levels can confirm on the same bar (e.g. a psychological round number and a
        swing-low cluster 10 pts apart). That is ONE setup with confluence, not two.
        """
        # Only NIFTY spot setups can become trades: futures 1m bars are used solely to confirm
        # spot bars, so a futures setup can never reach entry. Show it as context, not a signal.
        if snap.symbol != settings.PRIMARY_INDEX:
            print(
                f"[CONTEXT] {snap.symbol} {snap.timeframe} {setup.direction.value} structure at "
                f"{setup.level_id} ({setup.nominal_price:.1f}) @ {snap.timestamp.strftime('%H:%M')} "
                f"(confirmation only; trades come from NIFTY spot setups)"
            )
            return
        key = (snap.symbol, snap.timeframe, snap.timestamp, setup.direction.value)
        if key in self._alerted_setups:
            self._alerted_setups[key].append(setup.level_id)
            print(f"  + confluence for {snap.symbol} {snap.timeframe} {setup.direction.value} @ "
                  f"{snap.timestamp.strftime('%H:%M')}: {setup.level_id} ({setup.nominal_price:.1f})")
            return
        if len(self._alerted_setups) > 500:
            self._alerted_setups.clear()
        self._alerted_setups[key] = [setup.level_id]
        print(
            f"\n{'='*75}\n"
            f"  [TECHNICAL_SETUP_CANDIDATE: {setup.direction.value}]\n"
            f"  *** CANDIDATE ONLY: Options OI Flow, Greeks & Risk Validation Pending ***\n"
            f"{'='*75}\n"
            f"Asset: {snap.symbol} ({snap.timeframe}) | Level: {setup.level_id} ({setup.nominal_price:.1f})\n"
            f"Confirmed At: {snap.timestamp.strftime('%H:%M:%S')} | Trigger Price: {snap.close:.2f}\n"
            f"Technical SL: {setup.stop_loss} | Technical Target: {setup.target_price} | Trend: {snap.trend_bias}\n"
            f"{'='*75}\n"
        )

    def _on_technical_candidate(self, candidate: TechnicalCandidate) -> None:
        """Invoked when Phase C MultiTimeframeAnalyzer confirms a unified 15m+5m+1m technical setup."""
        headroom_str = (
            f"{candidate.headroom_points:.1f}pts"
            if candidate.headroom_points is not None
            else "N/A"
        )
        t1_rr_str = f"{candidate.target_1_rr:.2f}R" if hasattr(candidate, "target_1_rr") else "1.50R"
        t2_rr_str = f"{candidate.target_2_rr:.2f}R" if hasattr(candidate, "target_2_rr") else "2.00R"

        deriv_str = ""
        if getattr(candidate, "derivatives_assessment", None) is not None:
            da = candidate.derivatives_assessment
            reasons_summary = ", ".join(da.reasons[:2]) if da.reasons else "None"
            contra_summary = ", ".join(da.contra_reasons[:2]) if da.contra_reasons else "None"
            deriv_str = (
                f"  Derivatives Assessment: {da.assessment_type.value} "
                f"(Alignment: {da.alignment_score:.2f}, Quality: {da.data_quality_score:.2f}) | "
                f"Reasons: {reasons_summary} | Contra: {contra_summary}\n"
            )
            if getattr(da, "selected_contract", None) is not None:
                sc = da.selected_contract
                traj = sc.premium_trajectory
                costs = sc.execution_costs
                viable_str = "VIABLE" if costs.is_cost_viable else "UNVIABLE"
                deriv_str += (
                    f"  Selected Contract: {sc.symbol} (Score: {sc.selection_score:.1f}/100) | "
                    f"Δ: {sc.greeks.delta:+.2f}, IV: {sc.greeks.iv*100:.1f}%, Θ: {sc.greeks.theta:.1f}/day, Spread: {sc.spread_pct*100:.2f}%\n"
                    f"  Option Trajectory: LTP ₹{sc.ltp:.2f} -> T1: ₹{traj.target_premium_t1:.2f} (+{traj.target_premium_t1 - sc.ltp:.1f}pts) | "
                    f"SL: ₹{traj.stop_loss_premium:.2f} (-{sc.ltp - traj.stop_loss_premium:.1f}pts)\n"
                    f"  Execution Cost Model: Statutory ₹{costs.total_statutory_rupees:.2f}/lot | Friction: {costs.total_friction_points:.2f}pts | "
                    f"Gross R:R: 1:{costs.gross_rr_t1} -> Net R:R: 1:{costs.net_rr_t1} [{viable_str}]\n"
                )

        print(
            f"\n{'*'*78}\n"
            f"  [PHASE C: QUALIFIED MULTI-TIMEFRAME TECHNICAL CANDIDATE]\n"
            f"  {candidate.summary()}\n"
            f"  Macro 15m Trend: {candidate.timeframe_breakdown.get('15m_trend')} | "
            f"5m RVOL: {candidate.timeframe_breakdown.get('5m_rvol')}x | "
            f"15m Headroom: {headroom_str}\n"
            f"{deriv_str}"
            f"  Entry: {candidate.entry_price:.2f} | SL: {candidate.stop_loss:.2f} | "
            f"T1: {candidate.target_1:.2f} ({t1_rr_str}) | T2: {candidate.target_2:.2f} ({t2_rr_str}) | R:R: 1:{candidate.risk_reward_ratio}\n"
            f"{'*'*78}\n"
        )

        # Forward confirmed technical candidate to paper trading execution engine
        if hasattr(self, "paper_engine"):
            paper_pos = self.paper_engine.process_technical_candidate(candidate)
            if paper_pos:
                self.active_paper_trade_symbols.add(paper_pos.symbol)
                if paper_pos.symbol not in self.subscribed_option_symbols:
                    try:
                        self.ws_client.subscribe([paper_pos.symbol])
                        self.subscribed_option_symbols.add(paper_pos.symbol)
                        print(f"  [PAPER ENGINE] Subscribed to live WebSocket feed for {paper_pos.symbol}")
                    except Exception as e:
                        print(f"  [PAPER ENGINE] Failed to subscribe to {paper_pos.symbol}: {e}")

    def _get_initial_spot(self) -> float:
        """Determines reference NIFTY spot price from fresh DB records, live REST quotes, or master."""
        try:
            with self.store._lock:
                cur = self.store._conn.cursor()
                res = cur.execute(
                    "SELECT close, timestamp FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                    [settings.PRIMARY_INDEX],
                ).fetchall()
                if res and res[0][0] > 0:
                    cand_ts = res[0][1]
                    # Verify candle is from today if current session is regular trading
                    if cand_ts.date() == now_ist().date():
                        return float(res[0][0])

                res_tick = cur.execute(
                    "SELECT ltp, local_timestamp FROM ticks WHERE symbol = ? ORDER BY local_timestamp DESC LIMIT 1",
                    [settings.PRIMARY_INDEX],
                ).fetchall()
                if res_tick and res_tick[0][0] > 0:
                    if res_tick[0][1].date() == now_ist().date():
                        return float(res_tick[0][0])
        except Exception:
            pass

        # Try live broker quote
        try:
            q_res = self.rest_client.get_quote(settings.PRIMARY_INDEX, blocking=True)
            if q_res and q_res.get("s") == "ok" and q_res.get("d"):
                lp = q_res["d"][0].get("v", {}).get("lp")
                if lp and float(lp) > 0:
                    return float(lp)
        except Exception:
            pass

        # Calculate reference strike from instrument master's active strikes
        try:
            expiries = self.master.get_active_expiries("NIFTY")
            if expiries:
                contracts = self.master.get_option_chain_contracts("NIFTY", expiries[0])
                strikes = sorted(c.strike for c in contracts if c.strike > 0)
                if strikes:
                    return strikes[len(strikes) // 2]
        except Exception:
            pass

        raise RuntimeError("Could not resolve valid reference NIFTY spot price. Check network/instrument master.")

    def _resolve_futures(self) -> bool:
        """Point the engines at the current (unexpired, IST-aware) NIFTY/BANKNIFTY futures.

        Returns False if the NIFTY futures contract cannot be resolved.
        """
        try:
            nifty_fut = self.master.resolve_future("NIFTY")
        except Exception as e:
            print(f"[WARN] NIFTY futures resolution: {e}")
            return False
        if nifty_fut != self.technical_engine.primary_futures:
            print(f"[OK] Active NIFTY futures: {nifty_fut} (was {self.technical_engine.primary_futures})")
        self.technical_engine.set_futures_ticker(nifty_fut)
        self.technical_engine.set_index_futures_mapping(settings.PRIMARY_INDEX, nifty_fut)
        try:
            self.banknifty_futures = self.master.resolve_future("BANKNIFTY")
            self.technical_engine.set_index_futures_mapping(settings.BANKNIFTY_INDEX, self.banknifty_futures)
        except Exception as e:
            self.banknifty_futures = None
            print(f"[WARN] BANKNIFTY futures resolution: {e}")
        return True

    def _build_watchlist(self) -> List[str]:
        """Constructs 21-25 symbol watchlist within 100-symbol quota."""
        watchlist = [
            settings.PRIMARY_INDEX,
            settings.BANKNIFTY_INDEX,
            settings.SENSEX_INDEX,
            settings.INDIA_VIX,
            "NSE:HDFCBANK-EQ",
            "NSE:RELIANCE-EQ",
            "NSE:ICICIBANK-EQ",
            "NSE:INFY-EQ",
            "NSE:BHARTIARTL-EQ",
        ]

        # Futures (resolved before warmup; re-resolved here so the watchlist always matches)
        if self._resolve_futures():
            watchlist.append(self.technical_engine.primary_futures)
            if self.banknifty_futures:
                watchlist.append(self.banknifty_futures)

        # Dynamic Options: nearest expiry + next weekly expiry
        spot = self.current_spot_price or self._get_initial_spot()
        atm = round(spot / 50.0) * 50.0
        self.active_atm_strike = atm

        opt_symbols: List[str] = []
        try:
            expiries = self.master.get_active_expiries("NIFTY")
            if expiries:
                # 1. Nearest Expiry (e.g. Nov 23)
                near_exp = expiries[0]
                strikes_near = [atm - 50.0, atm, atm + 50.0]
                for s in strikes_near:
                    try:
                        ce = self.master.resolve_option("NIFTY", near_exp, s, OptionType.CE)
                        pe = self.master.resolve_option("NIFTY", near_exp, s, OptionType.PE)
                        opt_symbols.extend([ce.symbol, pe.symbol])
                    except LookupError:
                        pass

                # 2. Next Weekly Expiry (e.g. Dec 1)
                if len(expiries) > 1:
                    next_exp = expiries[1]
                    strikes_next = [atm - 50.0, atm, atm + 50.0]
                    for s in strikes_next:
                        try:
                            ce = self.master.resolve_option("NIFTY", next_exp, s, OptionType.CE)
                            pe = self.master.resolve_option("NIFTY", next_exp, s, OptionType.PE)
                            opt_symbols.extend([ce.symbol, pe.symbol])
                        except LookupError:
                            pass
        except Exception as e:
            print(f"[WARN] Options resolution: {e}")

        # Protect and subscribe all active open positions recovered from paper execution engine
        recovered_symbols: Set[str] = set()
        if hasattr(self, "paper_engine"):
            recovered_symbols = {p.symbol for p in self.paper_engine.get_open_positions()}
            self.active_paper_trade_symbols.update(recovered_symbols)

        all_opt_symbols = set(opt_symbols) | recovered_symbols
        self.subscribed_option_symbols = set(all_opt_symbols)

        for sym in all_opt_symbols:
            if sym not in watchlist:
                watchlist.append(sym)

        return watchlist

    def check_and_rotate_options(self) -> None:
        """Dynamically rotates option subscriptions when NIFTY ATM strike shifts."""
        # Always derive active paper trading symbols directly from engine open positions
        if hasattr(self, "paper_engine"):
            self.active_paper_trade_symbols = {p.symbol for p in self.paper_engine.get_open_positions()}

        if self.current_spot_price is None or self.active_atm_strike is None:
            return

        new_atm = round(self.current_spot_price / 50.0) * 50.0
        if new_atm == self.active_atm_strike:
            return
        # Hysteresis: only rotate once spot is clearly past the strike midpoint, otherwise a
        # spot hovering at e.g. 25375 flips ATM 25350<->25400 and churns 12 subscriptions.
        if abs(self.current_spot_price - self.active_atm_strike) < 25.0 + settings.ATM_ROTATION_HYSTERESIS_POINTS:
            return

        try:
            expiries = self.master.get_active_expiries("NIFTY")
            if not expiries:
                return

            new_opt_symbols: Set[str] = set()
            for exp in expiries[:2]:
                for s in [new_atm - 50.0, new_atm, new_atm + 50.0]:
                    try:
                        ce = self.master.resolve_option("NIFTY", exp, s, OptionType.CE)
                        pe = self.master.resolve_option("NIFTY", exp, s, OptionType.PE)
                        new_opt_symbols.add(ce.symbol)
                        new_opt_symbols.add(pe.symbol)
                    except LookupError:
                        pass

            to_unsubscribe = (self.subscribed_option_symbols - new_opt_symbols) - self.active_paper_trade_symbols
            to_subscribe = new_opt_symbols - self.subscribed_option_symbols

            if to_unsubscribe:
                self.ws_client.unsubscribe(list(to_unsubscribe))
                self.subscribed_option_symbols -= to_unsubscribe

            if to_subscribe:
                self.ws_client.subscribe(list(to_subscribe))
                self.subscribed_option_symbols |= to_subscribe

            # Only commit new ATM strike after subscription requests succeed
            print(f"\n[ATM ROTATION] Spot moved to {self.current_spot_price:.2f}. Shifted ATM {self.active_atm_strike} -> {new_atm}")
            self.active_atm_strike = new_atm

        except Exception as e:
            print(f"[WARN] Option rotation error (will retry next interval): {e}")

    def _poll_option_chain_if_due(self, is_open: bool) -> None:
        """Polls the NIFTY option chain (60s in market hours) through the shared rate limiter.

        Budget (FYERS Standard Data API, 5/s 50/min 5,000/day): 1 call/min during the
        375-minute session = ~375 calls/day. Outside market hours OI doesn't change, so we
        poll at most every 15 minutes. Broker limit responses pause polling for 65s so a
        per-minute violation can never repeat inside the same window.
        """
        now = now_ist()
        if self.option_chain_rate_limited_until and now < self.option_chain_rate_limited_until:
            return

        if self.last_option_chain_time is not None:
            elapsed = (now - self.last_option_chain_time).total_seconds()
            if self.option_chain_failure_count > 0:
                base = settings.OPTION_CHAIN_POLL_SECONDS
                required_wait = min(180.0, base * (2 ** min(self.option_chain_failure_count - 1, 2)))
            else:
                required_wait = settings.OPTION_CHAIN_POLL_SECONDS if is_open else 900.0
            if elapsed < required_wait:
                return

        try:
            loop = asyncio.new_event_loop()
            try:
                expiry_epoch = self._select_chain_expiry(now)
                chain = loop.run_until_complete(
                    self.rest_client.get_option_chain(
                        symbol=settings.PRIMARY_INDEX,
                        strikecount=settings.OPTION_CHAIN_STRIKECOUNT,
                        expiry_epoch=expiry_epoch,
                    )
                )
                # The expiry list is only known after the first response. If that default
                # (nearest) expiry is expired or inside the switch window, fetch the right one
                # now instead of analysing a dead chain until the next poll (1 extra call, once).
                if expiry_epoch is None and chain.available_expiries:
                    self.chain_expiries = list(chain.available_expiries)
                    better = self._select_chain_expiry(now)
                    if better is not None:
                        chain = loop.run_until_complete(
                            self.rest_client.get_option_chain(
                                symbol=settings.PRIMARY_INDEX,
                                strikecount=settings.OPTION_CHAIN_STRIKECOUNT,
                                expiry_epoch=better,
                            )
                        )
            finally:
                loop.close()

            if chain.available_expiries:
                self.chain_expiries = list(chain.available_expiries)
            self.store.save_option_chain_snapshot(chain)
            self.technical_engine.on_option_chain(chain, as_of=now)
            self.last_option_chain_time = now
            self.latest_option_chain_pcr = chain.pcr
            self.latest_total_call_oi = chain.total_call_oi
            self.latest_total_put_oi = chain.total_put_oi
            self.option_chain_failure_count = 0
            self.option_chain_last_error = None
            self._log_oi_flow(chain)
        except FyersRateLimitError as e:
            self.option_chain_failure_count += 1
            self.option_chain_last_error = str(e)
            self.last_option_chain_time = now
            self.option_chain_rate_limited_until = now + timedelta(seconds=65)
            print(f"[WARN] Option chain rate limited by FYERS ({e.scope}); pausing chain polls 65s: {e}")
        except Exception as e:
            self.option_chain_failure_count += 1
            self.option_chain_last_error = str(e)
            self.last_option_chain_time = now  # Anchor backoff timer to avoid tight loop
            if is_open:
                print(f"[WARN] Option chain poll failed (attempt {self.option_chain_failure_count}): {e}")

    def _select_chain_expiry(self, now: datetime) -> Optional[int]:
        """Expiry to analyse: the nearest one, unless it expires within
        OPTION_CHAIN_NEXT_EXPIRY_HOURS, in which case the next one (FYERS epoch).

        Returns None for "broker default (nearest)". Needs one prior response to know the
        expiry list, so the very first poll always uses the nearest expiry.
        """
        if len(self.chain_expiries) < 2:
            return None
        (near_date, _), (next_date, next_epoch) = self.chain_expiries[0], self.chain_expiries[1]
        near_close = IST.localize(datetime.combine(near_date, datetime.min.time())).replace(hour=15, minute=30)
        hours_left = (near_close - now).total_seconds() / 3600.0
        if hours_left < settings.OPTION_CHAIN_NEXT_EXPIRY_HOURS:
            note = f"{next_date}"
            if self._chain_expiry_note != note:
                print(
                    f"[OPTION CHAIN] Nearest expiry {near_date} has {max(hours_left, 0):.1f}h left; "
                    f"OI/IV analysis switched to next expiry {next_date}"
                )
                self._chain_expiry_note = note
            return int(next_epoch)
        self._chain_expiry_note = None
        return None

    def _log_oi_flow(self, chain: Any) -> None:
        """One compact line: PCR, walls, max pain, ATM IV trend and near-ATM OI flow."""
        profile = None
        try:
            analyzer = self.derivatives_engine.oi_analyzer
            profile = analyzer.latest_profiles.get((chain.underlying.upper(), chain.expiry))
        except Exception:
            profile = None

        window_pcr = (
            f"{chain.window_put_oi / chain.window_call_oi:.3f}" if getattr(chain, "window_call_oi", 0) else "-"
        )
        line = f"[OI FLOW] Exp {chain.expiry} | PCR expiry {chain.pcr:.3f} (window {window_pcr})"
        if profile is not None:
            mp = f"{profile.max_pain_strike:.0f}" if profile.max_pain_strike else "-"
            line += (
                f" | Call wall {profile.call_wall_strike:.0f} | Put wall {profile.put_wall_strike:.0f} | MaxPain(window) {mp}"
            )
            if profile.atm_iv is not None:
                line += f" | IV@{profile.atm_strike:.0f}(ATM) {profile.atm_iv:.2f}"
                if profile.atm_iv_change is not None:
                    line += f" | dIV@{profile.atm_iv_change_strike:.0f}(prev ATM) {profile.atm_iv_change:+.2f}"
            if profile.observation_interval_seconds > 0:
                def _bu(b):
                    return f" [{b[0]} x{b[1]}]" if b else ""
                line += (
                    f" | Near-ATM OI chg {profile.observation_interval_seconds:.0f}s: "
                    f"CE {profile.near_atm_call_oi_change:+,}{_bu(profile.near_atm_call_buildup)} / "
                    f"PE {profile.near_atm_put_oi_change:+,}{_bu(profile.near_atm_put_buildup)}"
                    f" | Price+OI bias {profile.aggregate_bias.value}"
                )
        print(line)

    @staticmethod
    def _install_shutdown_signals() -> None:
        """Make SIGTERM (and SIGINT when inherited as ignored) run the clean shutdown path.

        A pipeline started in the background ignores Ctrl-C (SIGINT inherited as SIG_IGN),
        and SIGTERM's default action kills the process without closing DuckDB or
        checkpointing positions. Both now raise KeyboardInterrupt -> _shutdown().
        """
        def _raise(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")

        try:
            signal.signal(signal.SIGTERM, _raise)
            if signal.getsignal(signal.SIGINT) in (signal.SIG_IGN, None):
                signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:
            pass  # not on the main thread (e.g. embedded in tests)

    def start(self):
        """Run the pipeline. The clean-shutdown guard covers the WHOLE lifecycle.

        Previously only the main loop was inside try/finally, so SIGTERM/Ctrl-C during
        token check, warmup or socket connect skipped _shutdown() and left DuckDB unclosed.
        """
        self._install_shutdown_signals()
        try:
            self._run()
        except KeyboardInterrupt:
            print("\nShutting down cleanly...")
        finally:
            self._shutdown()

    def _run(self):
        """Startup (token, warmup, subscribe, connect) followed by the maintenance loop."""
        print("=" * 72)
        print("  NIFTY Options Trading Intelligence - Market Data Pipeline")
        print("=" * 72)

        session_state = self.calendar.get_session_state(is_derivative=True)
        is_live = session_state in (SessionState.REGULAR_CASH, SessionState.DERIVATIVES_EXT, SessionState.SPECIAL_SESSION)

        print(f"Market Session: {session_state.value.upper()} (Live Trading: {'ACTIVE' if is_live else 'CLOSED'})")

        token = self.auth.load_cached_token()
        if not token:
            print("\n[ERROR] No valid FYERS access token found. Run `python scripts/fyers_login.py`.")
            sys.exit(1)

        print("[OK] FYERS access token verified.")
        print(f"[OK] DuckDB Storage ready: {self.store.db_path}")
        purged = self.store.purge_post_close_cash_bars()
        if purged:
            print(f"[OK] Purged {purged} synthetic post-15:30 index/equity bars from history")

        try:
            # Refreshes the cached NSE_FO.csv when it predates today (falls back to cache on failure)
            self.master.load_master(auto_download=True)
            print("[OK] Instrument Master loaded (NSE_FO.csv).")
        except Exception as e:
            print(f"[WARN] Master load: {e}")

        # Resolve the live futures contracts BEFORE warmup. Warmup previously ran on the
        # hard-coded constructor default (NIFTY26NOVFUT); after the monthly rollover the engine
        # then switched to a futures contract with no history -> cold indicators, NOT_READY.
        if not self._resolve_futures():
            print("[ERROR] Could not resolve the active NIFTY futures contract from the instrument master. "
                  "Futures confirmation is required for signals; refusing to start.")
            return

        # Pre-Market Historical Backfill & Technical Warmup
        print("\n[PRE-MARKET WARMUP] Initializing Historical Candle Manager...")
        try:
            core_tracked = [settings.PRIMARY_INDEX, self.technical_engine.primary_futures]
            print(f"[PRE-MARKET WARMUP] Syncing recent Daily, 15m, 5m, and 1m bars for {core_tracked}...")
            self.historical_manager.sync_pre_market_data(core_tracked)

            print("[PRE-MARKET WARMUP] Seeding prior day levels and bootstrapping indicators...")
            warmup_res = self.historical_manager.warm_up_technical_engine(
                engine=self.technical_engine,
                symbols=core_tracked,
                reconcile_session=True,
            )
            for sym, info in warmup_res.items():
                if sym.startswith("_"):
                    continue
                levels = info.get("daily_levels")
                lvl_str = f"PDH: {levels['PDH']}, PDL: {levels['PDL']}, PDC: {levels['PDC']}" if levels else "None"
                catchup_bars = info.get("session_catchup_candles", 0)
                is_5m_str = "READY" if info.get("is_5m_warmed_up") else "WARMING_UP"
                is_1m_str = "READY" if info.get("is_1m_warmed_up") else "WARMING_UP"
                print(
                    f"  - {sym}: Seeded {lvl_str} | History: {info.get('bootstrapped_candles', 0)} bars | "
                    f"Catch-up: {catchup_bars} bars | "
                    f"5m Status: {is_5m_str} | 1m Status: {is_1m_str}"
                )

            # Evaluate and set operational readiness on paper trading engine
            readiness = warmup_res.get("_overall_readiness", SystemReadiness.NOT_READY)
            if hasattr(self, "paper_engine"):
                self.paper_engine.set_system_readiness(readiness)

            banner_char = "#" if readiness == SystemReadiness.READY else ("!" if readiness == SystemReadiness.DEGRADED else "X")
            print(
                f"\n{banner_char * 78}\n"
                f"  [SYSTEM OPERATIONAL READINESS: {readiness.value}]\n"
                f"  - READY: Full history & indicators warmed (entries enabled)\n"
                f"  - DEGRADED: 5m ready, 1m warming up (entries cautious/upgrading)\n"
                f"  - NOT_READY: Critical levels/history missing (entries strictly disabled)\n"
                f"{banner_char * 78}\n"
            )
        except Exception as e:
            print(f"[WARN] Pre-market warmup encountered exception: {e}")
            if hasattr(self, "paper_engine"):
                self.paper_engine.set_system_readiness(SystemReadiness.NOT_READY)

        watchlist = self._build_watchlist()
        print(f"\nSubscribing to dynamic watchlist ({len(watchlist)} symbols within 100 quota):")
        for sym in watchlist:
            print(f"  - {sym}")

        self.ws_client.subscribe(watchlist)

        print("\nConnecting to FYERS Data Socket...")
        try:
            self.ws_client.connect()
            print("[CONNECTED] Real-time data socket active.")
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            sys.exit(1)

        loop_ticks = 0
        while self.running:
            time.sleep(1)
            loop_ticks += 1

            # 1. Clock-driven candle finalization & timeout expiry check every 1 second
            current_ist = now_ist()

            # Feed watchdog: during market hours, a silent or dropped stream triggers
            # a non-blocking reconnect with backoff (e.g. after Wi-Fi drops and returns).
            # Cash session only: the NIFTY index stops ticking at 15:30, so silence in the
            # 15:30-15:40 window is expected and must not trigger reconnects or gap repair.
            cash_live = self.calendar.get_session_state(is_derivative=True) in (
                SessionState.REGULAR_CASH, SessionState.SPECIAL_SESSION
            )
            if cash_live:
                action = self.ws_client.ensure_alive(stale_after=settings.FEED_GAP_SECONDS)
                if action:
                    print(f"[WS WATCHDOG] {action}")

            gap_msg = self.gap_backfiller.observe(current_ist, self._signal_tick_times(), cash_live)
            if gap_msg:
                print(f"[FEED GAP] {gap_msg}")

            with self._event_lock:
                self.builder.flush_closed_bars(current_ist)
                self.technical_engine.expire_pending_evaluations(current_ist)
                self.paper_engine.check_time_events(current_ist)

            # Periodic 10-second maintenance operations
            if loop_ticks % 10 != 0:
                continue

            # 2. Flush DuckDB buffer periodically
            self.store.flush_ticks()

            # 3. Dynamic option rotation if ATM shifted
            with self._event_lock:
                self.check_and_rotate_options()

            # 4. Session evaluation & status reporting
            curr_state = self.calendar.get_session_state(is_derivative=True)
            is_open = curr_state in (SessionState.REGULAR_CASH, SessionState.DERIVATIVES_EXT, SessionState.SPECIAL_SESSION)

            # 5. Scheduled Option Chain Snapshot
            self._poll_option_chain_if_due(is_open)

            ws_status = "CONNECTED" if self.ws_client.is_connected else "DISCONNECTED"
            quote_count = len(self.ws_client.quote_observed_symbols)
            sub_count = len(watchlist)

            # Differentiate Feed Age from Exchange Network Latency
            now = now_ist()
            feed_age = (now - self.last_primary_tick_time).total_seconds() if self.last_primary_tick_time else None
            net_latency = self.last_primary_network_latency

            feed_age_str = f"{feed_age:.2f}s" if feed_age is not None else "N/A"
            fut_ts = self.last_signal_tick_times.get(self.technical_engine.primary_futures)
            feed_age_str += f" | FUT Feed Age: {(now - fut_ts).total_seconds():.2f}s" if fut_ts else " | FUT Feed Age: N/A"
            net_lat_str = f"{net_latency:.3f}s" if net_latency is not None else "N/A"

            pcr_str = f" | PCR(expiry): {self.latest_option_chain_pcr:.3f}" if self.latest_option_chain_pcr is not None else ""
            api = self.rate_limiter.get_status()
            pcr_str += f" | API: {api['daily_requests_used']}/{self.rate_limiter.max_daily_requests}"
            if api["rate_limit_hits_today"]:
                pcr_str += f" (429s: {api['rate_limit_hits_today']})"

            # Paper Trading active telemetry
            with self._event_lock:
                open_pos = list(self.paper_engine.get_open_positions())
                summary = self.paper_engine.get_account_summary()
            if open_pos:
                for p in open_pos:
                    print(
                        f"  [ACTIVE PAPER POSITION: {p.symbol}] Entry: ₹{p.entry_price:.2f} | LTP: ₹{p.current_ltp:.2f} | "
                        f"Target: ₹{p.target_premium_t1:.2f} | SL: ₹{p.stop_loss_premium:.2f} | M2M Gross: ₹{p.gross_pnl:+.2f} | Net: ₹{p.net_pnl:+.2f}"
                    )
            if summary["total_trades"] > 0:
                print(
                    f"  [PAPER JOURNAL] Closed: {summary['total_trades']} (Win: {summary['win_rate_pct']}%) | "
                    f"Realized Gross: ₹{summary['total_realized_gross_pnl']:+.2f} | Realized Net: ₹{summary['total_realized_net_pnl']:+.2f} | "
                    f"Statutory Costs: ₹{summary['total_costs_paid']:.2f}"
                )

            # 6. Continuous Operational Readiness Evaluation
            with self._event_lock:
                readiness = self.evaluate_and_update_readiness()

            if not is_open:
                print(
                    f"[HOLIDAY / OFFLINE STATUS] Session: {curr_state.value.upper()} | Entries: DISABLED (market closed) | "
                    f"WS: {ws_status} | Subscribed: {sub_count} | Broker Acks: {self.ws_client.subscription_acks} | "
                    f"Quotes Seen: {quote_count}/{sub_count} | Snapshots: {self.snapshots_accepted} | "
                    f"Duplicates: {self.duplicates_dropped} | Trading Signals: DISABLED{pcr_str}"
                )
            else:
                print(
                    f"[LIVE STATUS] Session: {curr_state.value.upper()} | Readiness: {readiness.value} | "
                    f"WS: {ws_status} | Live Ticks: {self.live_ticks_accepted} | Candles: {self.candle_count} | "
                    f"NIFTY Feed Age: {feed_age_str} | Network Latency: {net_lat_str} | Quotes Seen: {quote_count}/{sub_count}{pcr_str}"
                )

    def _shutdown(self) -> None:
        """Persist state FIRST, then drop the socket. A second Ctrl-C must not skip the DB close.

        Previously the websocket close (bounded 5s wait) ran before store.close(); pressing
        Ctrl-C again during that wait aborted cleanup and left DuckDB unclosed.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        self._shutting_down = True  # tick/candle callbacks from the socket thread become no-ops
        self.running = False
        steps = [
            ("checkpoint paper positions", lambda: self.paper_engine.checkpoint_active_positions(force=True)),
            ("flush ticks", self.store.flush_ticks),
            ("close database", self.store.close),
            ("close websocket", lambda: self.ws_client.close(timeout=3.0)),
        ]
        for name, fn in steps:
            try:
                with self._event_lock:
                    fn()
            except KeyboardInterrupt:
                print(f"  (interrupt during '{name}' ignored; finishing shutdown)")
            except Exception as e:
                logger.warning("Shutdown step '%s' failed: %s", name, e)
        print("Pipeline stopped.")


if __name__ == "__main__":
    engine = MarketDataEngine()
    engine.start()
