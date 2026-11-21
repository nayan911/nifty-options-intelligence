"""FYERS WebSocket Data Socket Manager.

Encapsulates FyersDataSocket for live tick streaming, symbol budget tracking,
and normalization of raw ticks into structured Tick models.
"""

import os
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
        self.acknowledged_symbols: Set[str] = set()
        self.subscription_acks: int = 0
        self._socket: Optional[FyersDataSocket] = None
        self._is_connected: bool = False

        # Ensure local logs directory exists for FyersDataSocket logger
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Establish WebSocket connection using cached daily access token."""
        token = self.auth_provider.load_cached_token()
        if not token:
            raise PermissionError("Cannot connect WebSocket: No valid cached FYERS token found.")

        # Pass raw JWT token and local logs path (with trailing slash)
        self._socket = FyersDataSocket(
            access_token=token,
            log_path=str(self.log_dir) + "/",
            write_to_file=True,
            litemode=False,
            reconnect=True,
            on_connect=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )
        self._socket.connect()

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

    def _on_open(self) -> None:
        self._is_connected = True
        # Subscribe pending symbols upon connection
        if self.subscribed_symbols and self._socket:
            self._socket.subscribe(symbols=list(self.subscribed_symbols), data_type="SymbolUpdate")

    def _on_close(self, message: str) -> None:
        self._is_connected = False

    def _on_error(self, message: str) -> None:
        pass

    def _on_message(self, message: dict) -> None:
        """Parse raw incoming FYERS WebSocket tick or system message and dispatch."""
        if not message or not isinstance(message, dict):
            return

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
            self.acknowledged_symbols.add(symbol)
            epoch = message.get("exch_feed_time")
            exch_dt = datetime.fromtimestamp(epoch, tz=IST) if epoch else None
            is_index_asset = symbol in INDEX_SYMBOLS or "-INDEX" in symbol

            tick = Tick(
                symbol=symbol,
                exchange_timestamp=exch_dt,
                local_timestamp=datetime.now(IST),
                ltp=float(ltp),
                cumulative_volume=int(message.get("vol_traded_today", 0)) if not is_index_asset else 0,
                incremental_volume=0,  # Will be calculated by TickNormalizer
                is_index=is_index_asset,
                bid=float(message.get("bid", 0)) if message.get("bid") else None,
                ask=float(message.get("ask", 0)) if message.get("ask") else None,
                oi=int(message.get("oi", 0)) if message.get("oi") else None,
                source="FYERS",
            )

            if self.on_tick:
                self.on_tick(tick)

    def close(self) -> None:
        """Disconnect WebSocket."""
        if self._socket:
            self._socket.close_connection()
            self._is_connected = False
