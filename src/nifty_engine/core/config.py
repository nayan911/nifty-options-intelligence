"""Configuration management for the NIFTY Options Intelligence Engine."""

from pathlib import Path
from typing import Optional
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

    # FYERS Rate Limits (Standard Plan Defaults)
    RATE_LIMIT_RPS: float = 4.0      # Safety cap below official 5 req/s
    RATE_LIMIT_RPM: int = 40         # Safety cap below official 50 req/min
    RATE_LIMIT_DAILY: int = 4500     # Safety cap below official 5,000 req/day

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
