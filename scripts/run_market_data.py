#!/usr/bin/env python3
"""Main High-Reliability Market Data Ingestion & Live Analytics Engine.

Runs locally, streams WebSocket market ticks, applies TickNormalizer for volume
and out-of-order quarantine, builds 1m/5m/15m OHLCV bars, and persists data into DuckDB.
"""

import sys
import time
from typing import List, Optional

import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import Candle, OptionType, SessionState, Tick
from nifty_engine.data.candle_builder import RealtimeCandleBuilder
from nifty_engine.data.freshness_monitor import DataFreshnessMonitor
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus

IST = pytz.timezone("Asia/Kolkata")


class MarketDataEngine:
    """Orchestrates live market data streaming, volume normalization, and storage."""

    def __init__(self):
        self.calendar = ExchangeCalendar()
        self.auth = FyersAuthProvider()
        self.store = DuckDBStore()
        self.freshness = DataFreshnessMonitor()
        self.master = FyersInstrumentMaster()
        self.normalizer = TickNormalizer(self.calendar)
        self.running = True

        self.builder = RealtimeCandleBuilder(on_candle_close=self._on_candle_closed)
        self.ws_client = FyersWebSocketClient(
            auth_provider=self.auth,
            on_tick=self._on_tick_received,
            on_ack=self._on_ws_ack,
        )

        self.tick_count = 0
        self.live_tick_count = 0
        self.snapshot_count = 0
        self.candle_count = 0
        self.current_spot_price: Optional[float] = None
        self.active_atm_strike: Optional[float] = None
        self.subscribed_option_symbols: List[str] = []

    def _on_ws_ack(self, ack_msg: dict):
        """Callback on WebSocket subscription and status acknowledgements."""
        print(f"[WS ACK] {ack_msg.get('type')}: {ack_msg.get('message', 'OK')}")

    def _on_tick_received(self, raw_tick: Tick):
        """Processes raw tick through normalizer, sanitizes volume, persists, and builds bars."""
        res = self.normalizer.normalize(raw_tick)

        if res.status in (TickStatus.DUPLICATE, TickStatus.OUT_OF_ORDER, TickStatus.MALFORMED):
            return
        if not res.tick:
            return

        tick = res.tick
        self.tick_count += 1
        self.freshness.record_tick(tick)

        # 1. Save normalized tick (with exact incremental volume) to DuckDB
        self.store.insert_tick(tick)

        # 2. Track NIFTY spot price
        if tick.symbol == settings.PRIMARY_INDEX:
            self.current_spot_price = tick.ltp

        # 3. Only pass live trading ticks to CandleBuilder (holiday snapshots are saved but do not form bars)
        if res.status == TickStatus.VALID_LIVE:
            self.live_tick_count += 1
            self.builder.process_tick(tick)
        else:
            self.snapshot_count += 1

    def _on_candle_closed(self, candle: Candle):
        """Invoked when a 1m/5m/15m bar closes."""
        self.candle_count += 1
        self.store.insert_candle(candle)
        # Flush DuckDB buffer on candle close
        self.store.flush_ticks()
        print(
            f"[{candle.timestamp.strftime('%H:%M:%S')}] {candle.symbol} "
            f"({candle.timeframe}) C: {candle.close:.2f} | V: {candle.volume} | Bars Closed: {self.candle_count}"
        )

    def _get_initial_spot(self) -> float:
        """Determines best reference NIFTY spot price from database or fallback."""
        try:
            with self.store._lock:
                cur = self.store._conn.cursor()
                res = cur.execute(
                    "SELECT close FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                    [settings.PRIMARY_INDEX],
                ).fetchall()
                if res and res[0][0] > 0:
                    return float(res[0][0])

                res_tick = cur.execute(
                    "SELECT ltp FROM ticks WHERE symbol = ? ORDER BY local_timestamp DESC LIMIT 1",
                    [settings.PRIMARY_INDEX],
                ).fetchall()
                if res_tick and res_tick[0][0] > 0:
                    return float(res_tick[0][0])
        except Exception:
            pass
        return 24100.0

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

            self.subscribed_option_symbols = opt_symbols
            watchlist.extend(opt_symbols)
        except Exception as e:
            print(f"[WARN] Options resolution: {e}")

        return watchlist

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
                # Flush DuckDB buffer periodically
                self.store.flush_ticks()

                # Session re-evaluation
                curr_state = self.calendar.get_session_state(is_derivative=True)
                is_open = curr_state in (SessionState.REGULAR_CASH, SessionState.DERIVATIVES_EXT, SessionState.SPECIAL_SESSION)

                ack_count = len(self.ws_client.acknowledged_symbols)
                sub_count = len(watchlist)

                if not is_open:
                    print(
                        f"[HOLIDAY / OFFLINE STATUS] Session: {curr_state.value.upper()} | "
                        f"WS: CONNECTED | Subscribed: {sub_count} | Acknowledged: {ack_count}/{sub_count} | "
                        f"Snapshots Stored: {self.snapshot_count} | Trading Signals: DISABLED"
                    )
                else:
                    nifty_latency = self.freshness.get_latency_seconds(f"tick:{settings.PRIMARY_INDEX}")
                    lat_str = f"{nifty_latency:.2f}s" if nifty_latency is not None else "Awaiting tick..."
                    print(
                        f"[LIVE STATUS] Session: {curr_state.value.upper()} | "
                        f"Ticks: {self.live_tick_count} | Candles: {self.candle_count} | "
                        f"NIFTY Latency: {lat_str} | Acked: {ack_count}/{sub_count}"
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
