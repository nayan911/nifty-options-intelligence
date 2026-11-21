"""Core type definitions, data models, and abstract interfaces for the NIFTY Options Intelligence Engine."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nifty_engine.utils.timezone import now_ist


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class SessionState(str, Enum):
    PRE_OPEN = "pre_open"           # 09:00 - 09:15 IST
    REGULAR_CASH = "regular_cash"   # 09:15 - 15:30 IST
    DERIVATIVES_EXT = "fo_extended" # 15:30 - 15:40 IST (Derivatives closing window)
    POST_CLOSE = "post_close"       # 15:40 - 16:00 IST
    CLOSED = "closed"               # Night/Early morning
    HOLIDAY = "holiday"             # Exchange holiday / weekend
    SPECIAL_SESSION = "special"     # Muhurat trading / DR drill session


class BreakoutState(str, Enum):
    """Explicit state machine for structural level breakouts and retests."""
    APPROACHING_LEVEL = "approaching_level"
    BREAKOUT_DETECTED = "breakout_detected"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    RETEST_PENDING = "retest_pending"
    RETEST_CONFIRMED = "retest_confirmed"
    SETUP_ELIGIBLE = "setup_eligible"
    FALSE_BREAKOUT = "false_breakout"
    RETEST_FAILED = "retest_failed"
    SETUP_EXPIRED = "setup_expired"
    INVALIDATED = "invalidated"


class Tick(BaseModel):
    """Normalized tick data model with explicit incremental volume separation."""
    symbol: str
    exchange_timestamp: Optional[datetime] = None
    local_timestamp: datetime = Field(default_factory=now_ist)
    ltp: float = Field(..., gt=0, description="Last traded price")
    cumulative_volume: int = Field(default=0, ge=0, description="Cumulative daily traded volume from exchange")
    incremental_volume: int = Field(default=0, ge=0, description="Incremental traded volume since previous tick")
    is_index: bool = Field(default=False, description="True for spot indices like NIFTY50 which have no traded volume")
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_qty: Optional[int] = None
    ask_qty: Optional[int] = None
    oi: Optional[int] = None
    source: str = "FYERS"
    event_class: str = "LIVE"


class Candle(BaseModel):
    """OHLCV candlestick representation strictly isolating bar high/low from daily extremes."""
    symbol: str
    timeframe: str  # '1m', '5m', '15m', '1d'
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(default=0, description="Traded volume strictly within this candle's duration")
    oi: Optional[int] = None
    is_closed: bool = False


class OptionContract(BaseModel):
    """Normalized option contract details resolved from instrument master."""
    symbol: str  # e.g., 'NSE:NIFTY26NOV25000CE'
    fytoken: str
    underlying: str  # 'NIFTY', 'BANKNIFTY'
    expiry: date
    strike: float
    option_type: OptionType
    lot_size: int = 65
    tick_size: float = 0.05


class OptionGreeks(BaseModel):
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    iv: Optional[float] = None


class OptionChainStrike(BaseModel):
    """Both CE and PE values for a specific strike price."""
    strike: float
    call: Optional[Dict[str, Any]] = None
    put: Optional[Dict[str, Any]] = None


class OptionChainSnapshot(BaseModel):
    """Full snapshot of the option chain with separated API vs Exchange metadata."""
    underlying: str
    spot_price: float
    api_timestamp: datetime = Field(default_factory=now_ist, description="Time response received from broker")
    exchange_timestamp: Optional[datetime] = Field(default=None, description="Time reported by exchange if available")
    last_oi_change_timestamp: Optional[datetime] = Field(default=None, description="Last time strike OI values changed")
    expiry: date
    strikes: List[OptionChainStrike]
    total_call_oi: int = 0
    total_put_oi: int = 0
    pcr: float = 0.0


# Abstract Interfaces

class IStorageEngine(ABC):
    """Abstract interface for high-performance persistence (DuckDB)."""

    @abstractmethod
    def insert_tick(self, tick: Tick) -> None:
        pass

    @abstractmethod
    def insert_ticks_bulk(self, ticks: List[Tick]) -> None:
        pass

    @abstractmethod
    def insert_candle(self, candle: Candle) -> None:
        pass

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        pass

    @abstractmethod
    def save_option_chain_snapshot(self, snapshot: OptionChainSnapshot) -> None:
        pass


class IRateLimiter(ABC):
    """Abstract interface for centralized token-bucket rate limiter."""

    @abstractmethod
    async def acquire(self) -> None:
        pass


class IInstrumentMaster(ABC):
    """Abstract interface for resolving official broker symbol tickers."""

    @abstractmethod
    def load_master(self, fo_csv_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def resolve_option(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: OptionType
    ) -> OptionContract:
        pass

    @abstractmethod
    def get_atm_strike(self, underlying: str, spot_price: float) -> float:
        pass

    @abstractmethod
    def get_active_expiries(self, underlying: str) -> List[date]:
        pass
