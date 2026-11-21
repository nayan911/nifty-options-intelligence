"""Unit tests for DataFreshnessMonitor multi-policy tracking."""

from datetime import datetime, timedelta
import pytz
import pytest

from nifty_engine.core.interfaces import Tick
from nifty_engine.data.freshness_monitor import DataFreshnessMonitor

IST = pytz.timezone("Asia/Kolkata")


def test_separated_api_and_oi_change_freshness():
    monitor = DataFreshnessMonitor()

    # 1. Option chain polled successfully
    monitor.record_option_chain_response(underlying="NIFTY", total_oi=15000000)

    # API response is fresh
    assert monitor.is_api_stale("NIFTY") is False
    # OI data is not stale
    assert monitor.is_oi_stale("NIFTY") is False


def test_unchanged_oi_is_not_stale():
    monitor = DataFreshnessMonitor()

    # Initial poll
    monitor.record_option_chain_response(underlying="NIFTY", total_oi=15000000)

    # 2 minutes later, second poll returns same OI: 15,000,000 (normal during market)
    monitor.record_option_chain_response(underlying="NIFTY", total_oi=15000000)

    # API response is fresh
    assert monitor.is_api_stale("NIFTY") is False
    # Unchanged OI after 2 mins is NOT stale (policy is 15 mins)
    assert monitor.is_oi_stale("NIFTY") is False
