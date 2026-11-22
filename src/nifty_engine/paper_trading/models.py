"""Data models and enums for paper trading orders, positions, and executions."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_M = "SL_M"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    EOD_SQUARE_OFF = "EOD_SQUARE_OFF"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"


@dataclass
class PaperOrder:
    """Represents a simulated execution order with realistic broker friction."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    created_at: datetime
    position_id: Optional[str] = None
    signal_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    requested_price: Optional[float] = None
    filled_price: Optional[float] = None
    slippage_points: float = 0.0
    filled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "position_id": self.position_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else str(self.side),
            "order_type": self.order_type.value if isinstance(self.order_type, OrderType) else str(self.order_type),
            "status": self.status.value if isinstance(self.status, OrderStatus) else str(self.status),
            "quantity": self.quantity,
            "requested_price": self.requested_price,
            "filled_price": self.filled_price,
            "slippage": self.slippage_points,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class PositionTrade:
    """An individual execution fill belonging to a paper position."""

    trade_id: str
    order_id: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    turnover: float
    costs: float
    reason: Optional[ExitReason] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "side": self.side.value if isinstance(self.side, OrderSide) else str(self.side),
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "turnover": round(self.turnover, 2),
            "costs": round(self.costs, 2),
            "reason": self.reason.value if isinstance(self.reason, ExitReason) else (str(self.reason) if self.reason else None),
        }


@dataclass
class PaperPosition:
    """Represents an active or historical paper-traded options position."""

    position_id: str
    signal_id: str
    symbol: str
    underlying: str
    option_type: str  # "CE" or "PE"
    strike: float
    expiry: date
    lot_size: int
    initial_quantity: int
    remaining_quantity: int
    entry_price: float
    target_premium_t1: float
    target_premium_t2: float
    stop_loss_premium: float
    spot_entry: float
    spot_target_t1: float
    spot_target_t2: float
    spot_sl: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_reason: Optional[ExitReason] = None
    is_t1_hit: bool = False
    is_breakeven_sl_active: bool = False
    current_ltp: float = 0.0
    current_bid: Optional[float] = None
    current_ask: Optional[float] = None
    highest_premium_seen: float = 0.0
    lowest_premium_seen: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_costs: float = 0.0
    realized_rr: Optional[float] = None
    trades: List[PositionTrade] = field(default_factory=list)
    notes: Optional[str] = None

    def __post_init__(self):
        if self.highest_premium_seen == 0.0 and self.entry_price > 0.0:
            self.highest_premium_seen = self.entry_price
        if self.lowest_premium_seen == 0.0 and self.entry_price > 0.0:
            self.lowest_premium_seen = self.entry_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry.isoformat() if isinstance(self.expiry, date) else str(self.expiry),
            "lot_size": self.lot_size,
            "initial_quantity": self.initial_quantity,
            "remaining_quantity": self.remaining_quantity,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price is not None else None,
            "target_premium_t1": round(self.target_premium_t1, 2),
            "target_premium_t2": round(self.target_premium_t2, 2),
            "stop_loss_premium": round(self.stop_loss_premium, 2),
            "spot_entry": round(self.spot_entry, 2),
            "spot_target_t1": round(self.spot_target_t1, 2),
            "spot_target_t2": round(self.spot_target_t2, 2),
            "spot_sl": round(self.spot_sl, 2),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status.value if isinstance(self.status, PositionStatus) else str(self.status),
            "exit_reason": self.exit_reason.value if isinstance(self.exit_reason, ExitReason) else (str(self.exit_reason) if self.exit_reason else None),
            "is_t1_hit": self.is_t1_hit,
            "is_breakeven_sl_active": self.is_breakeven_sl_active,
            "current_ltp": round(self.current_ltp, 2),
            "current_bid": round(self.current_bid, 2) if self.current_bid is not None else None,
            "current_ask": round(self.current_ask, 2) if self.current_ask is not None else None,
            "highest_premium_seen": round(self.highest_premium_seen, 2),
            "lowest_premium_seen": round(self.lowest_premium_seen, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "realized_rr": round(self.realized_rr, 2) if self.realized_rr is not None else None,
            "notes": self.notes,
        }
