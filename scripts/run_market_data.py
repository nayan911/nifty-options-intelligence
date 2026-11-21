#!/usr/bin/env python3
"""Main Phase 1 Market Data Ingestion & Live Candle Engine.

Runs locally, streams WebSocket market ticks, builds 1m/5m/15m OHLCV bars,
and persists data into local DuckDB (data/market.duckdb).
"""

import sys
import time

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import Candle, OptionType, Tick
from nifty_engine.data.candle_builder import RealtimeCandleBuilder
from nifty_engine.data.freshness_monitor import DataFreshnessMonitor
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient
from nifty_engine.data.storage.duckdb_store import DuckDBStore


class MarketDataEngine:
    """Orchestrates live market data streaming, candle building, and DuckDB persistence."""

    def __init__(self):
        self.calendar = ExchangeCalendar()
        self.auth = FyersAuthProvider()
        self.store = DuckDBStore()
        self.freshness = DataFreshnessMonitor()
        self.master = FyersInstrumentMaster()
        self.running = True

        # Initialize Candle Builder with persistence callback
        self.builder = RealtimeCandleBuilder(on_candle_close=self._on_candle_closed)

        # Initialize WebSocket Client
        self.ws_client = FyersWebSocketClient(
            auth_provider=self.auth,
            on_tick=self._on_tick_received
        )

        self.tick_count = 0
        self.candle_count = 0

    def _on_tick_received(self, tick: Tick):
        """Callback invoked on every incoming WebSocket tick."""
        self.tick_count += 1
        self.freshness.record_tick(tick)

        # Save tick to DuckDB
        self.store.insert_tick(tick)

        # Feed tick to candle builder
        self.builder.process_tick(tick)

    def _on_candle_closed(self, candle: Candle):
        """Callback invoked when a 1m/5m/15m candle bar closes."""
        self.candle_count += 1
        self.store.insert_candle(candle)
        print(
            f"[{candle.timestamp.strftime('%H:%M:%S')}] {candle.symbol} "
            f"({candle.timeframe}) C: {candle.close:.2f} | V: {candle.volume} | Total Closed: {self.candle_count}"
        )

    def start(self):
        """Start the live market data pipeline."""
        print("=" * 70)
        print("  NIFTY Options Trading Intelligence - Market Data Pipeline (Phase 1)")
        print("=" * 70)

        # 1. Check Session State
        session_state = self.calendar.get_session_state(is_derivative=True)
        print(f"Current Market Session Phase: {session_state.value.upper()}")

        # 2. Check Auth Token
        token = self.auth.load_cached_token()
        if not token:
            print("\n[ERROR] No valid FYERS access token found.")
            print("Please run `python scripts/fyers_login.py` first to authenticate.")
            sys.exit(1)

        print("[OK] FYERS access token verified.")
        print(f"[OK] DuckDB Storage ready: {self.store.db_path}")

        # 3. Load Instrument Master for Dynamic Contract Resolution
        try:
            self.master.load_master()
            print("[OK] Instrument Master loaded (NSE_FO.csv).")
        except Exception as e:
            print(f"[WARN] Failed to load instrument master: {e}")

        # Core Indices
        watchlist = [
            settings.PRIMARY_INDEX,     # NSE:NIFTY50-INDEX
            settings.BANKNIFTY_INDEX,   # NSE:NIFTYBANK-INDEX
            settings.SENSEX_INDEX,      # BSE:SENSEX-INDEX
            settings.INDIA_VIX,         # NSE:INDIAVIX-INDEX
        ]

        # Top 5 NIFTY Heavyweights
        constituents = [
            "NSE:HDFCBANK-EQ",
            "NSE:RELIANCE-EQ",
            "NSE:ICICIBANK-EQ",
            "NSE:INFY-EQ",
            "NSE:BHARTIARTL-EQ",
        ]
        watchlist.extend(constituents)

        # Dynamic Futures and ATM Options Resolution
        try:
            nifty_fut = self.master.resolve_future("NIFTY")
            banknifty_fut = self.master.resolve_future("BANKNIFTY")
            watchlist.extend([nifty_fut, banknifty_fut])
            print(f"[OK] Resolved Near-Month Futures: {nifty_fut}, {banknifty_fut}")

            # Resolve Options: ATM +/- 2 strikes (5 strikes x 2 types = 10 contracts)
            expiries = self.master.get_active_expiries("NIFTY")
            if expiries:
                current_expiry = expiries[0]
                # Default baseline reference strike
                base_atm = 24100.0
                strikes = [base_atm - 100, base_atm - 50, base_atm, base_atm + 50, base_atm + 100]
                opt_symbols = []
                for s in strikes:
                    try:
                        ce = self.master.resolve_option("NIFTY", current_expiry, s, OptionType.CE)
                        pe = self.master.resolve_option("NIFTY", current_expiry, s, OptionType.PE)
                        opt_symbols.extend([ce.symbol, pe.symbol])
                    except LookupError:
                        pass
                watchlist.extend(opt_symbols)
                print(f"[OK] Resolved {len(opt_symbols)} NIFTY Options for Expiry {current_expiry}: strikes {strikes}")
        except Exception as e:
            print(f"[WARN] Error resolving dynamic derivatives: {e}")

        print(f"\nSubscribing to complete watchlist ({len(watchlist)} symbols within 100-symbol quota):")
        for sym in watchlist:
            print(f"  - {sym}")

        self.ws_client.subscribe(watchlist)

        # 4. Connect WebSocket
        print("Connecting to FYERS Data Socket...")
        try:
            self.ws_client.connect()
            print("[CONNECTED] Real-time data socket active. Streaming ticks...")
        except Exception as e:
            print(f"[ERROR] WebSocket connection failed: {e}")
            sys.exit(1)

        # Run monitoring loop
        try:
            while self.running:
                time.sleep(10)
                nifty_latency = self.freshness.get_latency_seconds(f"tick:{settings.PRIMARY_INDEX}")
                latency_str = f"{nifty_latency:.2f}s" if nifty_latency is not None else "Awaiting first tick..."
                print(
                    f"[STATUS] Ticks Ingested: {self.tick_count} | "
                    f"Candles Built: {self.candle_count} | NIFTY Feed Latency: {latency_str}"
                )
        except KeyboardInterrupt:
            print("\nShutting down pipeline cleanly...")
        finally:
            self.ws_client.close()
            self.store.close()
            print("Pipeline stopped.")


if __name__ == "__main__":
    engine = MarketDataEngine()
    engine.start()
