"""FYERS WebSocket Data Socket Manager.

Encapsulates FyersDataSocket for live tick streaming, symbol budget tracking,
and normalization of raw ticks into structured Tick models.
"""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Set

import certifi
import pytz
from fyers_apiv3.FyersWebsocket.data_ws import FyersDataSocket

from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import Tick
from nifty_engine.data.fyers.auth import FyersAuthProvider

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)

# Ensure macOS Python uses certifi's verified CA certificates bundle for WebSockets
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["WEBSOCKET_CLIENT_CA_BUNDLE"] = certifi.where()

INDEX_SYMBOLS = {
    settings.PRIMARY_INDEX,
    settings.BANKNIFTY_INDEX,
    settings.SENSEX_INDEX,
    settings.INDIA_VIX,
}


class FyersWebSocketClient:
    """Manages WebSocket connection, subscription caps, and tick dispatching."""

    def __init__(
        self,
        auth_provider: Optional[FyersAuthProvider] = None,
        on_tick: Optional[Callable[[Tick], None]] = None,
        on_ack: Optional[Callable[[dict], None]] = None,
        max_symbols: int = settings.MAX_WS_SYMBOLS,
    ) -> None:
        self.auth_provider = auth_provider or FyersAuthProvider()
        self.on_tick = on_tick
        self.on_ack = on_ack
        self.max_symbols = max_symbols

        self.subscribed_symbols: Set[str] = set()
        self.quote_observed_symbols: Set[str] = set()
        self.subscription_acks: int = 0
        self.total_messages_received: int = 0
        self.last_error: Optional[str] = None
        self._socket: Optional[FyersDataSocket] = None
        self._is_connected: bool = False

        # Liveness watchdog state. The FYERS SDK runs websocket-client without ping
        # timeouts, so a dropped network leaves a half-open socket that never fires
        # on_close. We track the last inbound message ourselves and force a reconnect
        # when the stream goes silent.
        self._generation: int = 0
        self._last_message_monotonic: Optional[float] = None
        self._reconnect_failures: int = 0
        self._next_reconnect_at: float = 0.0
        self.reconnect_count: int = 0

        # Ensure local logs directory exists for FyersDataSocket logger
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def acknowledged_symbols(self) -> Set[str]:
        """Backward-compatible alias for quote_observed_symbols."""
        return self.quote_observed_symbols

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Establish WebSocket connection using cached daily access token."""
        token = self.auth_provider.load_cached_token()
        if not token:
            raise PermissionError("Cannot connect WebSocket: No valid cached FYERS token found.")

        self._generation += 1
        gen = self._generation

        # FyersDataSocket is a process-wide singleton (__new__ caches the instance).
        # Clear it so every (re)connect builds a fresh socket instead of re-running
        # __init__ on top of a half-dead one whose threads are still alive.
        FyersDataSocket._instance = None

        # Callbacks are bound to this connection generation so late events from an
        # abandoned socket cannot flip connection state or inject ticks.
        self._socket = FyersDataSocket(
            access_token=token,
            log_path=str(self.log_dir) + "/",
            write_to_file=True,
            litemode=False,
            reconnect=True,
            on_connect=lambda: self._on_open(gen),
            on_close=lambda msg: self._on_close(msg, gen),
            on_error=lambda msg: self._on_error(msg, gen),
            on_message=lambda msg: self._on_message(msg, gen),
        )
        # Start the silence clock at connect time so a connect that never delivers
        # data is also caught by the watchdog.
        self._last_message_monotonic = time.monotonic()
        self._socket.connect()

    @property
    def seconds_since_last_message(self) -> Optional[float]:
        if self._last_message_monotonic is None:
            return None
        return time.monotonic() - self._last_message_monotonic

    def ensure_alive(self, stale_after: float = 15.0) -> Optional[str]:
        """Watchdog: reconnect if the stream is disconnected or silent for too long.

        Call periodically (e.g. every second) only while the market is open, since
        silence outside trading hours is normal. Returns a short status string when
        it took action, otherwise None.
        """
        silence = self.seconds_since_last_message
        dead = (not self._is_connected) or (silence is not None and silence > stale_after)
        if not dead:
            return None

        now = time.monotonic()
        if now < self._next_reconnect_at:
            return None

        reason = "disconnected" if not self._is_connected else f"silent {silence:.0f}s"
        self._abandon_socket()
        try:
            self.connect()
            self.reconnect_count += 1
            # Grace window: give the new socket time to auth + resubscribe before
            # judging it again; backoff grows only if it keeps failing.
            self._reconnect_failures += 1
            delay = min(60.0, 5.0 * (2 ** (self._reconnect_failures - 1)))
            self._next_reconnect_at = now + max(delay, stale_after)
            return f"reconnecting ({reason}), attempt {self._reconnect_failures}"
        except Exception as e:  # network down, DNS failure, token API unreachable
            self.last_error = str(e)
            self._reconnect_failures += 1
            delay = min(60.0, 5.0 * (2 ** (self._reconnect_failures - 1)))
            self._next_reconnect_at = now + delay
            return f"reconnect failed ({reason}): {e}; retry in {delay:.0f}s"

    def _abandon_socket(self) -> None:
        """Tear down the current socket without blocking the caller.

        The SDK's close_connection() joins its ws/message/ping threads with no
        timeout and can hang forever on a dead network, so we only stop its own
        auto-reconnect and close the raw socket from a daemon thread.
        """
        old = self._socket
        self._socket = None
        self._is_connected = False
        if old is None:
            return
        try:
            old.restart_flag = False  # stop SDK's internal reconnect loop
        except Exception:
            pass
        raw_ws = getattr(old, "_FyersDataSocket__ws_object", None)
        if raw_ws is not None:
            threading.Thread(target=self._safe_close_raw, args=(raw_ws,), daemon=True).start()

    @staticmethod
    def _safe_close_raw(raw_ws) -> None:
        try:
            raw_ws.close()
        except Exception:
            pass

    def subscribe(self, symbols: List[str], data_type: str = "SymbolUpdate") -> None:
        """Subscribe to a list of symbols within the 100-symbol quota."""
        if not symbols:
            return

        new_count = len(self.subscribed_symbols.union(symbols))
        if new_count > self.max_symbols:
            raise ValueError(
                f"Subscription exceeds FYERS Standard quota ({new_count} > {self.max_symbols}). "
                "Unsubscribe unused symbols before adding new ones."
            )

        self.subscribed_symbols.update(symbols)
        if self._socket and self._is_connected:
            self._socket.subscribe(symbols=symbols, data_type=data_type)

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols to free up subscription budget."""
        for s in symbols:
            self.subscribed_symbols.discard(s)

        if self._socket and self._is_connected:
            self._socket.unsubscribe(symbols=symbols)

    def _is_current(self, gen: Optional[int]) -> bool:
        return gen is None or gen == self._generation

    def _on_open(self, gen: Optional[int] = None) -> None:
        if not self._is_current(gen):
            return
        self._is_connected = True
        self.last_error = None
        self._last_message_monotonic = time.monotonic()
        # Subscribe pending symbols upon connection
        if self.subscribed_symbols and self._socket:
            self._socket.subscribe(symbols=list(self.subscribed_symbols), data_type="SymbolUpdate")

    def _on_close(self, message: str, gen: Optional[int] = None) -> None:
        if not self._is_current(gen):
            return
        self._is_connected = False

    def _on_error(self, message: str, gen: Optional[int] = None) -> None:
        if not self._is_current(gen):
            return
        self.last_error = str(message)

    def _on_message(self, message: dict, gen: Optional[int] = None) -> None:
        """Parse raw incoming FYERS WebSocket tick or system message and dispatch."""
        if not self._is_current(gen):
            return
        if not message or not isinstance(message, dict):
            return

        self.total_messages_received += 1
        self._last_message_monotonic = time.monotonic()
        self._reconnect_failures = 0  # stream healthy again -> reset backoff

        # 1. Handle Subscription / System Acknowledgements
        msg_type = message.get("type")
        if msg_type in ("sub", "auth", "resume", "pause"):
            if msg_type == "sub":
                self.subscription_acks += 1
            if self.on_ack:
                self.on_ack(message)
            return

        # 2. Handle Market Quote Update
        symbol = message.get("symbol")
        ltp = message.get("ltp")

        if symbol and ltp is not None:
            self.quote_observed_symbols.add(symbol)
            epoch = message.get("exch_feed_time")
            exch_dt = datetime.fromtimestamp(epoch, tz=IST) if epoch else None
            is_index_asset = symbol in INDEX_SYMBOLS or "-INDEX" in symbol

            # Parse bid/ask prices supporting FYERS WebSocket field variations
            raw_bid = message.get("bid") if message.get("bid") is not None else message.get("bid_price")
            if raw_bid is None:
                raw_bid = message.get("b_price")
            bid_val = float(raw_bid) if raw_bid is not None and float(raw_bid) > 0 else None

            raw_ask = message.get("ask") if message.get("ask") is not None else message.get("ask_price")
            if raw_ask is None:
                raw_ask = message.get("a_price")
            ask_val = float(raw_ask) if raw_ask is not None and float(raw_ask) > 0 else None

            # Parse bid/ask quantities supporting FYERS WebSocket field variations
            raw_bid_qty = message.get("bid_qty") if message.get("bid_qty") is not None else message.get("b_qty")
            if raw_bid_qty is None:
                raw_bid_qty = message.get("bid_size")
            bid_qty_val = int(raw_bid_qty) if raw_bid_qty is not None and int(raw_bid_qty) >= 0 else None

            raw_ask_qty = message.get("ask_qty") if message.get("ask_qty") is not None else message.get("a_qty")
            if raw_ask_qty is None:
                raw_ask_qty = message.get("ask_size")
            ask_qty_val = int(raw_ask_qty) if raw_ask_qty is not None and int(raw_ask_qty) >= 0 else None

            tick = Tick(
                symbol=symbol,
                exchange_timestamp=exch_dt,
                local_timestamp=datetime.now(IST),
                ltp=float(ltp),
                cumulative_volume=int(message.get("vol_traded_today", 0)) if not is_index_asset else 0,
                incremental_volume=0,  # Will be calculated by TickNormalizer
                is_index=is_index_asset,
                bid=bid_val,
                ask=ask_val,
                bid_qty=bid_qty_val,
                ask_qty=ask_qty_val,
                oi=int(message.get("oi", 0)) if message.get("oi") else None,
                source="FYERS",
            )

            if self.on_tick:
                self.on_tick(tick)

    def close(self, timeout: float = 5.0) -> None:
        """Disconnect WebSocket without hanging shutdown.

        The SDK's close_connection() joins worker threads with no timeout (it hung on
        Ctrl-C in live runs), so run it in a daemon thread and give up after `timeout`.
        """
        sock = self._socket
        self._generation += 1  # ignore any callbacks that arrive during teardown
        self._socket = None
        self._is_connected = False
        if sock is None:
            return
        t = threading.Thread(target=self._safe_close_connection, args=(sock,), daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            logger.warning("FYERS socket close timed out after %.0fs; abandoning it", timeout)

    @staticmethod
    def _safe_close_connection(sock) -> None:
        try:
            sock.close_connection()
        except Exception:
            pass
