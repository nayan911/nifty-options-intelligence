"""Multi-Policy Data Freshness and Feed Latency Monitoring Engine.

Explicitly separates:
1. API response freshness (Did the broker return HTTP 200 recently?)
2. Exchange timestamp freshness (What was the exchange feed time?)
3. Observed OI change tracking (Unchanged OI is NOT treated as stale feed)
4. Configurable policies for Price, Volume, OI, Quotes, and External data.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import pytz

from nifty_engine.core.interfaces import Tick

IST = pytz.timezone("Asia/Kolkata")


class FeedFreshnessRecord:
    """Tracks distinct timing metadata for a single feed."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.last_api_response_time: Optional[datetime] = None
        self.last_exchange_time: Optional[datetime] = None
        self.last_receipt_time: Optional[datetime] = None
        self.last_observed_value_change: Optional[datetime] = None
        self.last_value: Optional[float] = None


class DataFreshnessMonitor:
    """Enforces distinct freshness policies across asset classes and data types."""

    # Configurable domain thresholds
    DEFAULT_POLICIES = {
        "PRICE_TICK": timedelta(seconds=5),          # High frequency price update
        "FUTURES_VOLUME": timedelta(seconds=10),     # Futures traded volume
        "OPTION_QUOTE": timedelta(seconds=30),       # Individual option bid/ask quote
        "OPTION_CHAIN_API": timedelta(seconds=90),   # REST poll response from broker
        "OI_DATA": timedelta(minutes=15),            # Exchange OI publication interval
        "EXTERNAL_DATA": timedelta(hours=1),         # Macroeconomic / news data
    }

    def __init__(self, policies: Optional[Dict[str, timedelta]] = None) -> None:
        self.policies = policies or self.DEFAULT_POLICIES
        self._records: Dict[str, FeedFreshnessRecord] = {}

    def _get_or_create(self, key: str) -> FeedFreshnessRecord:
        if key not in self._records:
            self._records[key] = FeedFreshnessRecord(key)
        return self._records[key]

    def record_tick(self, tick: Tick) -> None:
        """Record live tick with separate exchange, receipt, and price-change timestamps."""
        now = datetime.now(IST)
        rec = self._get_or_create(f"tick:{tick.symbol}")

        rec.last_receipt_time = now
        rec.last_exchange_time = tick.exchange_timestamp or now

        # Track price change
        if rec.last_value != tick.ltp:
            rec.last_observed_value_change = now
            rec.last_value = tick.ltp

        # Global tick tracker
        self._get_or_create("tick_global").last_receipt_time = now

    def record_option_chain_response(
        self,
        underlying: str,
        total_oi: int,
        exchange_timestamp: Optional[datetime] = None
    ) -> None:
        """Record option chain response, tracking API success separately from OI changes."""
        now = datetime.now(IST)
        key = f"option_chain:{underlying}"
        rec = self._get_or_create(key)

        # 1. API Response Freshness (network poll succeeded)
        rec.last_api_response_time = now
        rec.last_receipt_time = now
        if exchange_timestamp:
            rec.last_exchange_time = exchange_timestamp

        # 2. Observed OI change (only updates when underlying OI actually changes)
        if rec.last_value != float(total_oi):
            rec.last_observed_value_change = now
            rec.last_value = float(total_oi)

    def is_api_stale(self, underlying: str = "NIFTY") -> bool:
        """Check if broker REST polling has stalled (> 90 seconds)."""
        rec = self._records.get(f"option_chain:{underlying}")
        if not rec or not rec.last_api_response_time:
            return True
        now = datetime.now(IST)
        return (now - rec.last_api_response_time) > self.policies["OPTION_CHAIN_API"]

    def is_price_stale(self, symbol: str) -> bool:
        """Check if price updates have stopped (> 5 seconds during active session)."""
        rec = self._records.get(f"tick:{symbol}")
        if not rec or not rec.last_receipt_time:
            return True
        now = datetime.now(IST)
        return (now - rec.last_receipt_time) > self.policies["PRICE_TICK"]

    def is_oi_stale(self, underlying: str = "NIFTY") -> bool:
        """Check if OI data has not changed across multiple exchange publication cycles (> 15 mins).

        Note: Unchanged OI for 1-2 minutes is completely normal and NOT considered stale.
        """
        rec = self._records.get(f"option_chain:{underlying}")
        if not rec or not rec.last_observed_value_change:
            return False  # Do not declare stale without baseline
        now = datetime.now(IST)
        return (now - rec.last_observed_value_change) > self.policies["OI_DATA"]

    def get_latency_seconds(self, key: str) -> Optional[float]:
        """Return elapsed seconds since the last receipt for this feed."""
        rec = self._records.get(key)
        if not rec or not rec.last_receipt_time:
            return None
        return (datetime.now(IST) - rec.last_receipt_time).total_seconds()
