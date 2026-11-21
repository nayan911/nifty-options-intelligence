#!/usr/bin/env python3
"""Main High-Reliability Market Data Ingestion & Live Analytics Engine.

Runs locally, streams WebSocket market ticks, applies TickNormalizer for volume
and out-of-order quarantine, builds 1m/5m/15m OHLCV bars, and persists data into DuckDB.
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

import pytz

from nifty_engine.analysis.technical.breakouts import BreakoutContext
from nifty_engine.analysis.technical.engine import (
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
)
from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import Candle, OptionType, SessionState, Tick
from nifty_engine.data.candle_builder import RealtimeCandleBuilder
from nifty_engine.data.freshness_monitor import DataFreshnessMonitor
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
from nifty_engine.data.fyers.rest_client import FyersRestClient
from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus
from nifty_engine.utils.timezone import now_ist

IST = pytz.timezone("Asia/Kolkata")


class MarketDataEngine:
    """Orchestrates live market data streaming, volume normalization, and storage."""

    def __init__(self, store: Optional[DuckDBStore] = None):
        self.calendar = ExchangeCalendar()
        self.auth = FyersAuthProvider()
        self.store = store or DuckDBStore()
        self.freshness = DataFreshnessMonitor()
        self.master = FyersInstrumentMaster()
        self.normalizer = TickNormalizer(self.calendar)
        self.rest_client = FyersRestClient(self.auth)
        self.running = True

        self.builder = RealtimeCandleBuilder(on_candle_close=self._on_candle_closed)
        self.technical_engine = TechnicalAnalysisEngine(
            primary_index=settings.PRIMARY_INDEX,
            primary_futures="NSE:NIFTY26NOVFUT",
            on_setup_candidate=self._on_setup_candidate,
        )
        self.ws_client = FyersWebSocketClient(
            auth_provider=self.auth,
            on_tick=self._on_tick_received,
            on_ack=self._on_ws_ack,
        )

        # Observability & diagnostic counters
        self.total_ticks_received = 0
        self.live_ticks_accepted = 0
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
        self.last_primary_network_latency: Optional[float] = None

        # Scheduled Option Chain Snapshot Collection
        self.last_option_chain_time: Optional[datetime] = None
        self.latest_option_chain_pcr: Optional[float] = None
        self.latest_total_call_oi: Optional[int] = None
        self.latest_total_put_oi: Optional[int] = None

    def _on_ws_ack(self, ack_msg: dict):
        """Callback on WebSocket subscription and status acknowledgements."""
        print(f"[WS ACK] {ack_msg.get('type')}: {ack_msg.get('message', 'OK')}")

    def _on_tick_received(self, raw_tick: Tick, current_time: Optional[datetime] = None):
        """Processes raw tick through normalizer, sanitizes volume, persists, and builds bars."""
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
            self.last_primary_tick_time = now_ist()
            if tick.exchange_timestamp:
                self.last_primary_network_latency = (tick.local_timestamp - tick.exchange_timestamp).total_seconds()

        # 3. Only pass live trading ticks to CandleBuilder (holiday snapshots are saved but do not form bars)
        if res.status == TickStatus.VALID_LIVE:
            self.live_ticks_accepted += 1
            self.symbol_stats[sym]["live"] += 1
            self.builder.process_tick(tick)
        else:
            self.snapshots_accepted += 1
            self.symbol_stats[sym]["snapshot"] += 1

    def _on_candle_closed(self, candle: Candle):
        """Invoked when a 1m/5m/15m bar closes (guaranteed exactly once per closed candle)."""
        self.candle_count += 1
        self.store.insert_candle(candle)
        # Flush DuckDB buffer on candle close
        self.store.flush_ticks()
        print(
            f"[{candle.timestamp.strftime('%H:%M:%S')}] {candle.symbol} "
            f"({candle.timeframe}) C: {candle.close:.2f} | V: {candle.volume} | Bars Closed: {self.candle_count}"
        )

        # Real-time quantitative technical analysis
        try:
            snapshot = self.technical_engine.on_candle(candle)
            if snapshot and candle.symbol in (settings.PRIMARY_INDEX, self.technical_engine.primary_futures):
                self._log_technical_status(snapshot)
        except Exception as e:
            print(f"[WARN] Technical analysis error for {candle.symbol}: {e}")

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
        """Invoked when a high-probability technical breakout & retest completes."""
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
            client = self.auth.get_client()
            q_res = client.quotes(data={"symbols": settings.PRIMARY_INDEX})
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

        # Futures
        try:
            nifty_fut = self.master.resolve_future("NIFTY")
            bnf_fut = self.master.resolve_future("BANKNIFTY")
            watchlist.extend([nifty_fut, bnf_fut])
            self.technical_engine.set_futures_ticker(nifty_fut)
        except Exception as e:
            print(f"[WARN] Futures resolution: {e}")

        # Dynamic Options: nearest expiry + next weekly expiry
        spot = self.current_spot_price or self._get_initial_spot()
        atm = round(spot / 50.0) * 50.0
        self.active_atm_strike = atm

        try:
            expiries = self.master.get_active_expiries("NIFTY")
            opt_symbols: List[str] = []
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

            self.subscribed_option_symbols = set(opt_symbols)
            watchlist.extend(opt_symbols)
        except Exception as e:
            print(f"[WARN] Options resolution: {e}")

        return watchlist

    def check_and_rotate_options(self) -> None:
        """Dynamically rotates option subscriptions when NIFTY ATM strike shifts."""
        if self.current_spot_price is None or self.active_atm_strike is None:
            return

        new_atm = round(self.current_spot_price / 50.0) * 50.0
        if new_atm == self.active_atm_strike:
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
        """Polls full FYERS option chain matrix and persists to DuckDB."""
        now = now_ist()
        if self.last_option_chain_time is not None:
            elapsed = (now - self.last_option_chain_time).total_seconds()
            if elapsed < 180.0:
                return

        # Fetch option chain
        try:
            loop = asyncio.new_event_loop()
            chain = loop.run_until_complete(
                self.rest_client.get_option_chain(symbol=settings.PRIMARY_INDEX, strikecount=10)
            )
            loop.close()

            self.store.save_option_chain_snapshot(chain)
            self.last_option_chain_time = now
            self.latest_option_chain_pcr = chain.pcr
            self.latest_total_call_oi = chain.total_call_oi
            self.latest_total_put_oi = chain.total_put_oi

            print(
                f"[OPTION CHAIN SNAPSHOT] Expiry: {chain.expiry} | PCR: {chain.pcr:.3f} | "
                f"Call OI: {chain.total_call_oi:,} | Put OI: {chain.total_put_oi:,}"
            )
        except Exception:
            # During weekend/holiday, broker endpoint may be unavailable
            pass

    def start(self):
        """Starts streaming and monitoring pipeline."""
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

        try:
            self.master.load_master(auto_download=False)
            print("[OK] Instrument Master loaded (NSE_FO.csv).")
        except Exception as e:
            print(f"[WARN] Master load: {e}")

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

        try:
            while self.running:
                time.sleep(10)
                # 1. Flush DuckDB buffer periodically
                self.store.flush_ticks()

                # 2. Clock-driven candle finalization (emits bars on boundary; callback handles persistence once)
                self.builder.flush_closed_bars(now_ist())

                # 3. Dynamic option rotation if ATM shifted
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
                net_lat_str = f"{net_latency:.3f}s" if net_latency is not None else "N/A"

                pcr_str = f" | PCR: {self.latest_option_chain_pcr:.3f}" if self.latest_option_chain_pcr is not None else ""

                if not is_open:
                    print(
                        f"[HOLIDAY / OFFLINE STATUS] Session: {curr_state.value.upper()} | "
                        f"WS: {ws_status} | Subscribed: {sub_count} | Broker Acks: {self.ws_client.subscription_acks} | "
                        f"Quotes Seen: {quote_count}/{sub_count} | Snapshots: {self.snapshots_accepted} | "
                        f"Duplicates: {self.duplicates_dropped} | Trading Signals: DISABLED{pcr_str}"
                    )
                else:
                    print(
                        f"[LIVE STATUS] Session: {curr_state.value.upper()} | "
                        f"WS: {ws_status} | Live Ticks: {self.live_ticks_accepted} | Candles: {self.candle_count} | "
                        f"NIFTY Feed Age: {feed_age_str} | Network Latency: {net_lat_str} | Quotes Seen: {quote_count}/{sub_count}{pcr_str}"
                    )
        except KeyboardInterrupt:
            print("\nShutting down cleanly...")
        finally:
            self.store.flush_ticks()
            self.ws_client.close()
            self.store.close()
            print("Pipeline stopped.")


if __name__ == "__main__":
    engine = MarketDataEngine()
    engine.start()
