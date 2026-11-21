"""Documented Interactive OAuth Authentication for FYERS API v3.

Follows official FYERS OAuth specifications with zero headless scraping or credential storage.
Manages daily access tokens and verifies strict user-only file permissions (0600).
"""

import json
import os
from datetime import datetime, time
from pathlib import Path
from typing import Optional

from fyers_apiv3 import fyersModel

from nifty_engine.core.config import settings
from nifty_engine.utils.timezone import now_ist

# FYERS tokens expire daily at 6:30 AM IST
FYERS_DAILY_EXPIRY_TIME = time(6, 30, 0)


class FyersAuthProvider:
    """Manages the official OAuth 2.0 authorization code flow and session persistence."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        token_path: Optional[Path] = None,
    ) -> None:
        self.app_id = app_id or settings.FYERS_APP_ID
        self.secret_key = secret_key or settings.FYERS_SECRET_KEY
        self.redirect_uri = redirect_uri or settings.FYERS_REDIRECT_URI

        self.token_dir = token_path.parent if token_path else settings.TOKEN_CACHE_DIR
        self.token_file = token_path or (self.token_dir / settings.TOKEN_CACHE_FILE)

    def _get_session_model(self) -> fyersModel.SessionModel:
        """Create a configured SessionModel instance."""
        if not self.app_id or not self.secret_key:
            raise ValueError(
                "FYERS_APP_ID and FYERS_SECRET_KEY must be configured in your .env file."
            )
        return fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )

    def generate_login_url(self) -> str:
        """Generate the official OAuth authorization link to open in browser."""
        session = self._get_session_model()
        return session.generate_authcode()

    def exchange_code_for_token(self, auth_code: str) -> str:
        """Exchange the browser redirect auth_code for a daily access_token."""
        session = self._get_session_model()
        session.set_token(auth_code.strip())
        response = session.generate_token()

        if not response or response.get("s") != "ok" or "access_token" not in response:
            error_msg = response.get("message", "Unknown error during token generation")
            raise RuntimeError(f"FYERS Token Exchange Failed: {error_msg}")

        access_token = response["access_token"]
        self.save_cached_token(access_token)
        return access_token

    def save_cached_token(self, token: str) -> None:
        """Save access token with strict POSIX 0600 (read/write only by owner) permissions."""
        self.token_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "access_token": token,
            "created_at": now_ist().isoformat(),
            "app_id": self.app_id,
        }

        # Write file atomically
        temp_file = self.token_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Enforce chmod 0600 (owner read/write only)
        os.chmod(temp_file, 0o600)
        temp_file.replace(self.token_file)

    def is_cached_token_expired(self, created_at_iso: str) -> bool:
        """Check if token has passed the daily 6:30 AM IST expiry boundary."""
        created_dt = datetime.fromisoformat(created_at_iso)
        current_dt = now_ist()

        # If created on a prior date and current time has passed 6:30 AM IST -> Expired
        if current_dt.date() > created_dt.date():
            if current_dt.time() >= FYERS_DAILY_EXPIRY_TIME:
                return True
            # Even if before 6:30 AM, if created > 24 hours ago -> Expired
            if (current_dt - created_dt).total_seconds() >= 86400:
                return True

        return False

    def load_cached_token(self) -> Optional[str]:
        """Load and return valid cached token, or None if missing/expired."""
        if not self.token_file.exists():
            return None

        try:
            with open(self.token_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            token = data.get("access_token")
            created_at = data.get("created_at")
            cached_app_id = data.get("app_id")

            if not token or not created_at:
                return None

            # Verify same app_id
            if self.app_id and cached_app_id != self.app_id:
                return None

            if self.is_cached_token_expired(created_at):
                return None

            return token
        except Exception:
            return None

    def get_client(self, access_token: Optional[str] = None) -> fyersModel.FyersModel:
        """Get an initialized FyersModel client ready for read-only market data calls."""
        token = access_token or self.load_cached_token()
        if not token:
            raise PermissionError(
                "No valid FYERS access token found. Please run login interactive flow first."
            )
        return fyersModel.FyersModel(
            client_id=self.app_id,
            token=token,
            is_async=False,
            log_path=""
        )
