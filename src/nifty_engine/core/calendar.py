"""Exchange-aware trading calendar and dynamic session manager for NSE."""

from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, Optional, Set

import yaml

from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import SessionState
from nifty_engine.utils.timezone import now_ist, to_ist


class ExchangeCalendar:
    """Manages trading days, official holidays, special sessions, and intraday market session phases."""

    def __init__(
        self,
        sessions_path: Optional[Path] = None,
        holidays_path: Optional[Path] = None
    ) -> None:
        self.sessions_path = sessions_path or settings.SESSIONS_CONFIG_PATH
        self.holidays_path = holidays_path or settings.HOLIDAYS_CONFIG_PATH

        self.holidays: Set[date] = set()
        self.special_sessions: Dict[date, Dict] = {}

        # Default fallback session timings (IST)
        self.pre_open_start = time(9, 0, 0)
        self.market_open = time(9, 15, 0)
        self.cash_close = time(15, 30, 0)
        self.derivatives_close = time(15, 40, 0)  # Official F&O closing window

        self._load_configs()

    def _load_configs(self) -> None:
        """Load session timings and gazetted holidays from YAML."""
        if self.holidays_path.exists():
            with open(self.holidays_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "holidays" in data:
                    for item in data["holidays"]:
                        dt_obj = datetime.strptime(item["date"], "%Y-%m-%d").date()
                        self.holidays.add(dt_obj)

        if self.sessions_path.exists():
            with open(self.sessions_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    std = data.get("standard", {})
                    if "cash_open" in std:
                        self.market_open = datetime.strptime(std["cash_open"], "%H:%M:%S").time()
                    if "cash_close" in std:
                        self.cash_close = datetime.strptime(std["cash_close"], "%H:%M:%S").time()
                    if "derivatives_close" in std:
                        self.derivatives_close = datetime.strptime(std["derivatives_close"], "%H:%M:%S").time()

                    # Special sessions (e.g., Muhurat trading)
                    specials = data.get("special_sessions", {})
                    for dt_str, cfg in specials.items():
                        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d").date()
                        self.special_sessions[dt_obj] = cfg

    def is_holiday(self, target_date: date) -> bool:
        """Check if date is a Saturday, Sunday, or declared exchange holiday."""
        # Weekend check: 5 = Saturday, 6 = Sunday
        if target_date.weekday() in (5, 6):
            return True
        return target_date in self.holidays

    def is_trading_day(self, target_date: date) -> bool:
        """Check if date is an active trading day (or has special session)."""
        if target_date in self.special_sessions:
            return True
        return not self.is_holiday(target_date)

    def get_session_state(
        self,
        dt: Optional[datetime] = None,
        is_derivative: bool = True
    ) -> SessionState:
        """Determine the current market session phase for a given timestamp."""
        dt_ist = to_ist(dt) if dt else now_ist()
        current_date = dt_ist.date()
        current_time = dt_ist.time()

        # Check for special session (e.g. Muhurat Trading)
        if current_date in self.special_sessions:
            cfg = self.special_sessions[current_date]
            start_t = datetime.strptime(cfg["session_start"], "%H:%M:%S").time()
            end_t = datetime.strptime(cfg["session_end"], "%H:%M:%S").time()
            if start_t <= current_time <= end_t:
                return SessionState.SPECIAL_SESSION
            return SessionState.CLOSED

        # Holiday check
        if self.is_holiday(current_date):
            return SessionState.HOLIDAY

        # Standard session progression
        if current_time < self.pre_open_start:
            return SessionState.CLOSED

        if self.pre_open_start <= current_time < self.market_open:
            return SessionState.PRE_OPEN

        if self.market_open <= current_time < self.cash_close:
            return SessionState.REGULAR_CASH

        # Derivatives extended closing window (15:30 - 15:40)
        if self.cash_close <= current_time < self.derivatives_close:
            return SessionState.DERIVATIVES_EXT if is_derivative else SessionState.POST_CLOSE

        if current_time >= self.derivatives_close:
            return SessionState.CLOSED

        return SessionState.CLOSED

    def is_market_open(
        self,
        dt: Optional[datetime] = None,
        is_derivative: bool = True
    ) -> bool:
        """Helper to quickly check if trading is currently active."""
        state = self.get_session_state(dt, is_derivative=is_derivative)
        if is_derivative:
            return state in (SessionState.REGULAR_CASH, SessionState.DERIVATIVES_EXT, SessionState.SPECIAL_SESSION)
        return state in (SessionState.REGULAR_CASH, SessionState.SPECIAL_SESSION)
