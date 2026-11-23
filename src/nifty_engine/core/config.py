"""Configuration management for the NIFTY Options Intelligence Engine."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # FYERS API v3 Credentials
    FYERS_APP_ID: str = ""
    FYERS_SECRET_KEY: str = ""
    FYERS_REDIRECT_URI: str = "https://trade.fyers.in/api-login/redirect-uri/index.html"

    # Token storage path
    TOKEN_CACHE_DIR: Path = Path.home() / ".fyers"
    TOKEN_CACHE_FILE: str = "token.json"

    # Database
    DUCKDB_PATH: Path = Path("data/market.duckdb")

    # Instrument Master Cache
    SYMBOL_CACHE_DIR: Path = Path("data/symbols")

    # FYERS Data API Rate Limits (Standard plan, revised 16 Oct 2026: 5/s, 50/min, 5,000/day;
    # >3 per-minute violations in a day blocks the account for the rest of the day).
    # Prime plan users can raise these (10/s, 500/min, 500,000/day).
    RATE_LIMIT_RPS: float = 4.0      # Safety cap below official 5 req/s
    RATE_LIMIT_RPM: int = 40         # Safety cap below official 50 req/min
    RATE_LIMIT_DAILY: int = 4500     # Safety cap below official 5,000 req/day
    API_USAGE_FILE: Path = Path("data/logs/fyers_api_usage.json")  # survives restarts within a day

    # Option chain polling during market hours. 60s = ~375 calls/session (~8% of the
    # 4,500/day budget) and 1 call/min against the 40/min cap.
    OPTION_CHAIN_POLL_SECONDS: float = 60.0
    OPTION_CHAIN_STRIKECOUNT: int = 10
    # In the last hours before expiry the expiring chain's IV collapses toward 0 and its OI
    # changes are dominated by positions being closed, so OI/IV analysis switches to the next
    # expiry once the nearest one has less than this many hours left (11:30 on expiry day).
    OPTION_CHAIN_NEXT_EXPIRY_HOURS: float = 4.0

    # ATM option rotation only when spot is this many points beyond the strike midpoint
    ATM_ROTATION_HYSTERESIS_POINTS: float = 10.0

    # Feed gap handling: if the primary index stops ticking for this long during the
    # cash session, missed bars are backfilled from FYERS history after it resumes.
    FEED_GAP_SECONDS: float = 15.0

    # Trading Sessions Config
    SESSIONS_CONFIG_PATH: Path = Path("config/market_sessions.yaml")
    HOLIDAYS_CONFIG_PATH: Path = Path("config/holidays.yaml")

    # WebSocket Config
    MAX_WS_SYMBOLS: int = 100        # Standard plan cap

    # Target Underlyings
    PRIMARY_INDEX: str = "NSE:NIFTY50-INDEX"
    BANKNIFTY_INDEX: str = "NSE:NIFTYBANK-INDEX"
    SENSEX_INDEX: str = "BSE:SENSEX-INDEX"
    INDIA_VIX: str = "NSE:INDIAVIX-INDEX"

    # Logging
    LOG_LEVEL: str = "INFO"


# Global singleton instance
settings = Settings()
