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
from nifty_engine.research.manifest import DEFAULT_MANIFEST, build_manifest, write_manifest
from nifty_engine.research.segments import segment_all
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
        contracts = futures_contracts(master, underlying, today)
        # FYERS stitches by the requested contract's position: nearest -> near month, 2nd -> next month.
        specs.append(SeriesSpec(name, contracts[0][0], "CONT", underlying=underlying))
        if len(contracts) > 1:
            specs.append(SeriesSpec(name, contracts[1][0], "CONT_NEXT", underlying=underlying))
        for sym, exp in contracts:
            specs.append(SeriesSpec(name, sym, "OWN", contract_expiry=exp))
    return specs


def main() -> int:
    today = now_ist().date()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=(today - timedelta(days=330)).isoformat())
    ap.add_argument("--end", default=today.isoformat())
    ap.add_argument("--db", default=str(DEFAULT_RESEARCH_DB))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="skip downloading; rebuild the manifest")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    master = FyersInstrumentMaster()
    master.load_master(auto_download=True)
    specs = build_specs(master, today)

    print(f"Research history build: {start} -> {end}  db={args.db}")
    for s in specs:
        extra = f" (expiry {s.contract_expiry})" if s.contract_expiry else ""
        print(f"  - {s.instrument:15s} {s.kind:9s} {s.symbol} -> stored as {s.store_symbol}{extra}")
    if args.dry_run:
        return 0

    cal = ExchangeCalendar()
    store = ResearchHistoryStore(args.db)
    rest = FyersRestClient(rate_limiter=AsyncTokenBucketRateLimiter(usage_file=settings.API_USAGE_FILE))
    loader = HistoryLoader(rest, store, calendar=cal)

    def progress(r):
        status = "ok" if not r.errors else f"{len(r.errors)} error(s): {r.errors[0][:90]}"
        print(f"  [{datetime.now():%H:%M:%S}] {r.spec.instrument:15s} {r.spec.kind:4s} {r.spec.symbol:24s} "
              f"res={r.resolution:2s} requests={r.requests:2d} skipped={r.skipped_chunks:2d} bars={r.bars:6d} {status}",
              flush=True)

    if not args.validate_only:
        loader.load_all(specs, start, end, progress=progress)

    # Segment stitched futures by 2-of-3 vote (calendar, OI ratio, basis jump), then validate.
    segment_reports = segment_all(store, cal)
    manifest = build_manifest(store, cal, start, end, segment_reports=segment_reports)
    bad_ohlc = [i for i in store.quality_report() if i.kind == "BAD_OHLC"]
    for i in bad_ohlc:
        manifest["errors"].append(f"{i.instrument}/{i.source} {i.day}: {i.detail}")
    manifest["ok"] = not manifest["errors"]
    path = write_manifest(manifest, args.manifest)

    print(f"\nManifest: {path}  ok={manifest['ok']}  trading days={manifest['range']['trading_days']}")
    for key, rv in manifest["rollovers"].items():
        print(f"  {key}: {len(rv['accepted'])} contract switches accepted (>=2 of calendar/OI/basis), "
              f"{len(rv['disputed'])} single-signal disputed")
        for b in rv["accepted"]:
            if not all(b["votes"].values()):
                print(f"      {b['day']}: votes={b['votes']} oi_ratio={b['oi_ratio']} basis_change={b['basis_change']}")
    for srow in manifest["series"]:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(srow["status_counts"].items()))
        flag = " [required]" if srow["required"] else ""
        print(f"  {srow['instrument']:15s} {srow['source']:4s} {srow['symbol']:24s} {counts}{flag}")
    for w in manifest["warnings"]:
        print(f"  WARNING: {w}")
    for e in manifest["errors"]:
        print(f"  ERROR: {e}")
    store.close()
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
