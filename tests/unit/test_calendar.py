"""Unit tests for ExchangeCalendar and market session transitions."""

from datetime import date, datetime

import pytest
import pytz

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import SessionState

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def calendar():
    return ExchangeCalendar()


def test_weekend_detection(calendar):
    # 2026-01-24 is Saturday, 2026-01-25 is Sunday
    saturday = date(2026, 1, 24)
    sunday = date(2026, 1, 25)
    monday = date(2026, 1, 27)  # Note: Jan 26 is Republic Day holiday

    assert calendar.is_holiday(saturday) is True
    assert calendar.is_holiday(sunday) is True
    assert calendar.is_holiday(monday) is False


def test_gazetted_holiday_detection(calendar):
    # 2026-01-26 is Republic Day
    republic_day = date(2026, 1, 26)
    assert calendar.is_holiday(republic_day) is True
    assert calendar.is_trading_day(republic_day) is False


def test_session_state_regular_trading(calendar):
    # 10:30 AM on a regular Tuesday (2026-01-27)
    dt = IST.localize(datetime(2026, 1, 27, 10, 30, 0))
    state = calendar.get_session_state(dt, is_derivative=True)
    assert state == SessionState.REGULAR_CASH
    assert calendar.is_market_open(dt, is_derivative=True) is True


def test_session_state_derivatives_extended_window(calendar):
    # 15:35 PM on a regular Tuesday (Derivatives active, Cash closed)
    dt = IST.localize(datetime(2026, 1, 27, 15, 35, 0))

    # Cash check -> POST_CLOSE (Cash closed at 15:30)
    cash_state = calendar.get_session_state(dt, is_derivative=False)
    assert cash_state == SessionState.POST_CLOSE
    assert calendar.is_market_open(dt, is_derivative=False) is False

    # Derivatives check -> DERIVATIVES_EXT (F&O open until 15:40)
    fo_state = calendar.get_session_state(dt, is_derivative=True)
    assert fo_state == SessionState.DERIVATIVES_EXT
    assert calendar.is_market_open(dt, is_derivative=True) is True


def test_session_state_after_derivatives_close(calendar):
    # 15:45 PM on a regular Tuesday (both Cash and Derivatives closed)
    dt = IST.localize(datetime(2026, 1, 27, 15, 45, 0))
    state = calendar.get_session_state(dt, is_derivative=True)
    assert state == SessionState.CLOSED
    assert calendar.is_market_open(dt, is_derivative=True) is False


def test_muhurat_trading_session(calendar):
    # Official NSE Muhurat trading 2026: 2026-11-08 between 18:15 and 19:15
    dt_during = IST.localize(datetime(2026, 11, 8, 18, 30, 0))
    dt_before = IST.localize(datetime(2026, 11, 8, 15, 0, 0))

    assert calendar.get_session_state(dt_during) == SessionState.SPECIAL_SESSION
    assert calendar.is_market_open(dt_during) is True

    assert calendar.get_session_state(dt_before) == SessionState.CLOSED
    assert calendar.is_market_open(dt_before) is False


def test_get_prior_trading_date(calendar):
    """Verify get_prior_trading_date skips weekends and declared exchange holidays."""
    # Monday Nov 23, 2026:
    # 1 trading day prior must be Friday Nov 20, 2026 (skipping Sat 21 and Sun 22)
    monday = date(2026, 11, 23)
    fri_20 = calendar.get_prior_trading_date(monday, trading_days_back=1)
    assert fri_20 == date(2026, 11, 20)

    # 3 trading days prior: Wed Nov 18, 2026
    wed_18 = calendar.get_prior_trading_date(monday, trading_days_back=3)
    assert wed_18 == date(2026, 11, 18)

    # Tuesday Nov 24, 2026: 1 trading day prior is Monday Nov 23
    tue = date(2026, 11, 24)
    mon_23 = calendar.get_prior_trading_date(tue, trading_days_back=1)
    assert mon_23 == date(2026, 11, 23)

    # Tuesday Jan 27, 2026: Monday Jan 26 is Republic Day holiday
    # Prior trading day must skip Jan 26, Jan 25 (Sun), Jan 24 (Sat) -> Friday Jan 23, 2026
    tue_jan27 = date(2026, 1, 27)
    fri_jan23 = calendar.get_prior_trading_date(tue_jan27, trading_days_back=1)
    assert fri_jan23 == date(2026, 1, 23)

