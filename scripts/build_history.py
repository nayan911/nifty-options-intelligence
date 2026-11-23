#!/usr/bin/env python3
"""Build / refresh the research history store (data/research.duckdb) from FYERS.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_history.py [--start 2026-01-01] [--end 2026-11-23] [--dry-run]

Downloads 1m + daily bars for NIFTY/BANKNIFTY spot, India VIX, NIFTY heavyweights, FYERS'
stitched continuous futures (with OI, segmented at contract switches) and every futures contract
FYERS still serves under its own symbol (unadjusted price, volume, OI). Re-runs resume: chunks
already stored are skipped, the latest chunk is always refreshed.
"""

import argparse
import sys
from datetime import date, datetime, timedelta

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.config import settings
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.fyers.instrument_master import FyersInstrumentMaster
from nifty_engine.data.fyers.rest_client import FyersRestClient
from nifty_engine.research.history_loader import HistoryLoader, SeriesSpec
from nifty_engine.research.history_store import DEFAULT_RESEARCH_DB, ResearchHistoryStore
from nifty_engine.utils.timezone import now_ist

SPOT_SERIES = [
    ("NIFTY_SPOT", settings.PRIMARY_INDEX),
    ("BANKNIFTY_SPOT", settings.BANKNIFTY_INDEX),
    ("INDIAVIX", settings.INDIA_VIX),
    ("HDFCBANK", "NSE:HDFCBANK-EQ"),
    ("RELIANCE", "NSE:RELIANCE-EQ"),
    ("ICICIBANK", "NSE:ICICIBANK-EQ"),
    ("INFY", "NSE:INFY-EQ"),
    ("BHARTIARTL", "NSE:BHARTIARTL-EQ"),
]


def futures_contracts(master: FyersInstrumentMaster, underlying: str, as_of: date):
    """Every listed futures contract for `underlying` expiring on/after `as_of` (includes a contract
    that expired today: FYERS still serves it until it is delisted, and it cannot be fetched later)."""
    df = master._df
    futs = df[(df["short_sym"] == underlying) & (df["opt"] == "XX") & (df["expiry_date"] >= as_of)]
    return sorted({(str(r.symbol), r.expiry_date) for r in futs.itertuples()}, key=lambda x: x[1])


def build_specs(master: FyersInstrumentMaster, today: date):
    specs = [SeriesSpec(name, sym, "SPOT") for name, sym in SPOT_SERIES]
    for underlying, name in (("NIFTY", "NIFTY_FUT"), ("BANKNIFTY", "BANKNIFTY_FUT")):
        near = master.resolve_future(underlying)
        specs.append(SeriesSpec(name, near, "CONT"))
        for sym, exp in futures_contracts(master, underlying, today):
            specs.append(SeriesSpec(name, sym, "OWN", contract_expiry=exp))
    return specs


def main() -> int:
    today = now_ist().date()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=(today - timedelta(days=330)).isoformat())
    ap.add_argument("--end", default=today.isoformat())
    ap.add_argument("--db", default=str(DEFAULT_RESEARCH_DB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    master = FyersInstrumentMaster()
    master.load_master(auto_download=True)
    specs = build_specs(master, today)

    print(f"Research history build: {start} -> {end}  db={args.db}")
    for s in specs:
        extra = f" (expiry {s.contract_expiry})" if s.contract_expiry else ""
        print(f"  - {s.instrument:15s} {s.kind:4s} {s.symbol}{extra}")
    if args.dry_run:
        return 0

    cal = ExchangeCalendar()
    store = ResearchHistoryStore(args.db)
    rest = FyersRestClient(rate_limiter=AsyncTokenBucketRateLimiter(usage_file=settings.API_USAGE_FILE))
    loader = HistoryLoader(rest, store, is_trading_day=cal.is_trading_day)

    def progress(r):
        status = "ok" if not r.errors else f"{len(r.errors)} error(s): {r.errors[0][:90]}"
        print(f"  [{datetime.now():%H:%M:%S}] {r.spec.instrument:15s} {r.spec.kind:4s} {r.spec.symbol:24s} "
              f"res={r.resolution:2s} requests={r.requests:2d} skipped={r.skipped_chunks:2d} bars={r.bars:6d} {status}",
              flush=True)

    loader.load_all(specs, start, end, progress=progress)

    print("\nContinuous-futures contract switches detected (OI ratio first bar / prev close):")
    for inst in ("NIFTY_FUT", "BANKNIFTY_FUT"):
        for b in store.assign_cont_segments(inst, "1"):
            print(f"  {inst}: {b['prev_day']} -> {b['day']}  oi_ratio={b['oi_ratio']}")

    print("\nCoverage:")
    print(store.coverage().to_string(index=False))

    tdays = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    tdays = [d for d in tdays if cal.is_trading_day(d)]
    issues = store.quality_report(trading_days=tdays)
    print(f"\nQuality issues: {len(issues)}")
    by_kind = {}
    for i in issues:
        by_kind.setdefault((i.instrument, i.kind), []).append(i)
    for (inst, kind), items in sorted(by_kind.items()):
        sample = ", ".join(f"{x.day}" for x in items[:5])
        print(f"  {inst:15s} {kind:12s} x{len(items):3d}  e.g. {sample}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
