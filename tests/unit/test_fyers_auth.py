"""Unit tests for FyersAuthProvider and token lifecycle."""

import os
import stat
from datetime import datetime, timedelta

import pytest
import pytz

from nifty_engine.data.fyers.auth import FyersAuthProvider

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def auth_provider(tmp_path):
    token_file = tmp_path / "token.json"
    provider = FyersAuthProvider(
        app_id="TESTAPP_100",
        secret_key="TEST_SECRET_KEY",
        redirect_uri="https://trade.fyers.in/api-login/redirect-uri/index.html",
        token_path=token_file,
    )
    return provider


def test_token_save_and_0600_permissions(auth_provider):
    token = "sample_test_jwt_access_token_12345"
    auth_provider.save_cached_token(token)

    assert auth_provider.token_file.exists()

    # Verify POSIX 0600 permissions (owner read + write only)
    file_stat = os.stat(auth_provider.token_file)
    mode = stat.S_IMODE(file_stat.st_mode)
    assert mode == 0o600

    # Verify content read back
    loaded = auth_provider.load_cached_token()
    assert loaded == token


def test_token_daily_expiry_logic(auth_provider):
    # 1. Created 10 minutes ago -> should NOT be expired
    ten_mins_ago = (datetime.now(IST) - timedelta(minutes=10)).isoformat()
    assert auth_provider.is_cached_token_expired(ten_mins_ago) is False

    # 2. Created 2 days ago -> MUST be expired
    two_days_ago = (datetime.now(IST) - timedelta(days=2)).isoformat()
    assert auth_provider.is_cached_token_expired(two_days_ago) is True
