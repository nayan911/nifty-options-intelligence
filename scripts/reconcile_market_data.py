#!/usr/bin/env python3
"""Market Data Reconciliation Utility.

Reconciles intraday candles in DuckDB against verified broker historical candles (FYERS REST API).
Fixes any truncated, partial, or corrupted candles recorded during live WebSocket sessions.

Usage:
    PYTHONPATH=src:scripts .venv/bin/python scripts/reconcile_market_data.py [--date YYYY-MM-DD] [--symbols SYMBOL1 SYMBOL2]
"""

import argparse
import sys
from datetime import datetime
from typing import List

import pytz

from nifty_engine.core.config import settings
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.data.fyers.rest_client import FyersRestClient
from nifty_engine.data.historical_manager import HistoricalCandleManager
from nifty_engine.data.storage.duckdb_store import DuckDBStore
from nifty_engine.utils.timezone import now_ist

IST = pytz.timezone("Asia/Kolkata")


def main():
    parser = argparse.ArgumentParser(description="Reconcile DuckDB candles with official FYERS historical data.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target session date to reconcile (format: YYYY-MM-DD). Defaults to current IST date.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to reconcile. Defaults to core tracked indices and futures plus symbols active in DuckDB today.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["15m", "5m", "1m"],
        help="Timeframes to reconcile. Default: 15m 5m 1m.",
    )
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = now_ist().date()

    print("=" * 75)
    print("  NIFTY Options Trading Intelligence - Market Data Reconciliation")
    print(f"  Target Date: {target_date.strftime('%Y-%m-%d')} | Timeframes: {args.timeframes}")
    print("=" * 75)

    auth = FyersAuthProvider()
    token = auth.load_cached_token()
    if not token:
        print("[ERROR] No valid FYERS access token found. Run `python scripts/fyers_login.py`.")
        sys.exit(1)

    store = DuckDBStore(db_path=settings.DUCKDB_PATH)
    rest_client = FyersRestClient(auth_provider=auth)
    manager = HistoricalCandleManager(rest_client=rest_client, store=store)

    symbols: List[str] = []
    if args.symbols:
        symbols = args.symbols
    else:
        # Discover all symbols present in DuckDB for target_date plus core tracked assets
        core = [settings.PRIMARY_INDEX, "NSE:NIFTY26NOVFUT", settings.BANKNIFTY_INDEX, settings.SENSEX_INDEX]
        symbols.extend(core)

        date_start = IST.localize(datetime.combine(target_date, datetime.min.time()))
        date_end = IST.localize(datetime.combine(target_date, datetime.max.time()))

        with store._lock:
            cur = store._conn.cursor()
            db_symbols = cur.execute(
                "SELECT DISTINCT symbol FROM candles WHERE timestamp >= ? AND timestamp <= ?",
                [date_start, date_end],
            ).fetchall()

        for (s,) in db_symbols:
            if s not in symbols:
                symbols.append(s)

    print(f"[INFO] Reconciling {len(symbols)} symbols against FYERS official historical API...")
    results = manager.reconcile_session_candles(
        symbols=symbols,
        target_date=target_date,
        timeframes=args.timeframes,
    )

    print("\n" + "=" * 75)
    print("  Reconciliation Summary:")
    print("=" * 75)
    total_reconciled = 0
    for sym, tf_counts in results.items():
        counts_str = ", ".join(f"{tf}: {count}" for tf, count in tf_counts.items() if count > 0)
        total_sym = sum(tf_counts.values())
        total_reconciled += total_sym
        if total_sym > 0:
            print(f"  [OK] {sym:<25} -> {counts_str}")
        else:
            print(f"  [--] {sym:<25} -> No candles returned or updated")

    print("-" * 75)
    print(f"Total candles verified & persisted: {total_reconciled}")
    print("=" * 75)


if __name__ == "__main__":
    main()
