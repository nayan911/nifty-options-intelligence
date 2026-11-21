# NIFTY Options Trading Intelligence System

A real-time Indian stock market analysis and options trading decision-support platform built for macOS local execution using Python, DuckDB, and FYERS API v3.

Targeted specifically at evaluating high-probability intraday setups for capturing **15–20 points of NIFTY option premium movement** with strict risk management.

---

## 1. Quick Start in VS Code (macOS)

### Step 1: Open the Project in VS Code
Open your terminal and run:
```bash
code /Users/nayan/.gemini/antigravity/scratch/nifty-options-intelligence
```
Or open VS Code, press `Cmd + O`, and select the `/Users/nayan/.gemini/antigravity/scratch/nifty-options-intelligence` folder.

### Step 2: Configure Your FYERS API Credentials
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` in VS Code and fill in your FYERS credentials:
```env
FYERS_APP_ID=SPXXXXXX-100
FYERS_SECRET_KEY=NXXXXXXXXB
FYERS_REDIRECT_URI=https://trade.fyers.in/api-login/redirect-uri/index.html
```

### Step 3: Run the Unit Test Suite (21 Tests)
Verify that all core layers (Exchange Calendar, Rate Limiter, DuckDB Store, Candle Builder, Instrument Master, and Auth) pass:
```bash
PYTHONPATH=src .venv/bin/pytest tests/unit/ -v
```

---

## 2. Daily Workflow

### Step 1: Morning Interactive Login (OAuth v3)
FYERS access tokens expire daily at **6:30 AM IST**. Run the official login wizard once each trading morning:
```bash
PYTHONPATH=src .venv/bin/python scripts/fyers_login.py
```
1. It opens the official FYERS authorization portal in your browser.
2. Log in with your FYERS user credentials and 2FA.
3. Upon redirect, copy the `code` value from your browser URL and paste it into the terminal.
4. Your daily token is securely stored with `0600` permissions at `~/.fyers/token.json`.

### Step 2: Start the Live Market Data Pipeline
Stream live WebSocket ticks, build 1m/5m/15m OHLCV candles, and store them directly into your local DuckDB database (`data/market.duckdb`):
```bash
PYTHONPATH=src .venv/bin/python scripts/run_market_data.py
```

---

## 3. Project Architecture

```text
nifty-options-intelligence/
├── src/nifty_engine/
│   ├── core/
│   │   ├── interfaces.py       # Tick, Candle, OptionContract, and Storage interfaces
│   │   ├── config.py           # Pydantic Settings loading from .env
│   │   ├── calendar.py         # ExchangeCalendar (Cash 09:15-15:30, F&O 09:15-15:40)
│   │   └── rate_limiter.py     # Centralized Token-Bucket Limiter (4 req/s, 40 req/min)
│   ├── data/
│   │   ├── fyers/
│   │   │   ├── auth.py         # Documented OAuth flow with 0600 token cache
│   │   │   ├── instrument_master.py  # Official NSE_FO.csv resolver (no synthetic ticker guessing)
│   │   │   ├── rest_client.py  # Rate-limited REST calls for option chains & history
│   │   │   └── websocket_client.py   # WebSocket data socket manager (max 100 symbols)
│   │   ├── storage/
│   │   │   ├── schemas.py      # DuckDB SQL schemas for ticks, candles, option chains
│   │   │   └── duckdb_store.py # Embedded DuckDB columnar storage engine
│   │   ├── candle_builder.py   # Real-time tick aggregator for 1m/5m/15m bars
│   │   └── freshness_monitor.py # Staleness & feed latency detector
├── tests/unit/                 # 21 unit tests covering all components
├── config/
│   ├── market_sessions.yaml    # NSE regular & upcoming session configs
│   └── holidays.yaml           # Gazetted 2026 NSE holiday calendar
├── scripts/
│   ├── fyers_login.py          # Interactive OAuth login CLI
│   └── run_market_data.py      # Live pipeline runner
├── data/                       # Local DuckDB database & cached symbols (gitignored)
└── pyproject.toml              # Project dependencies
```

---

## 4. Key Engineering Guardrails

1. **Strictly Read-Only (V1 Constraint):** No order placement or live money trading capabilities are present.
2. **Zero Synthetic Ticker Bugs:** Option tickers are resolved against FYERS's official `NSE_FO.csv` daily master catalog.
3. **Centralized Rate Limiter:** Protects your account against FYERS Standard API 24-hour lockouts by capping requests at 4 req/sec and 40 req/min.
4. **Exchange-Aware Market Clock:** Recognizes the F&O derivatives closing window (15:40 IST) and scheduled December 2026 session revisions rather than hardcoding 3:30 PM.
