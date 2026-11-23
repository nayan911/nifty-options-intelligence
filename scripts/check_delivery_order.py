"""Regression check: S000's decisions must not depend on the live bar-delivery order.

At a shared boundary the live builder emits bars 1m -> 5m -> 15m when the next tick closes them and
15m -> 5m -> 1m when the 1-second clock flush does; both happen live. This replays the validated
history in both orders and compares the full content of every signal as a multiset
(collections.Counter over canonical records, so a duplicated signal is detected too). Report order
is ignored on purpose.

Usage:  python scripts/check_delivery_order.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
Exit 0 when identical, 1 when any signal differs. Full history takes ~30 minutes.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date

from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.replay import ReplayDriver


def canonical(sig):
    return (sig.kind, sig.at.isoformat(), sig.symbol, sig.timeframe, sig.direction,
            json.dumps(sig.detail, sort_keys=True, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--db", default="data/research.duckdb")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    st = ResearchHistoryStore(a.db)
    first, last = st.conn.execute(
        "SELECT min(CAST(ts AS DATE)), max(CAST(ts AS DATE)) FROM hist_bars "
        "WHERE instrument = 'NIFTY_SPOT' AND resolution = '1'").fetchall()[0]
    start, end = a.start or first, a.end or last

    results = {}
    for order in ("tick", "flush"):
        res = ReplayDriver(st, delivery_order=order).run(start, end)
        results[order] = res
        print(f"{order}: {len(res.days)} days, {len(res.signals)} signals", flush=True)

    tick, flush = (Counter(canonical(s) for s in results[o].signals) for o in ("tick", "flush"))
    only_tick, only_flush = tick - flush, flush - tick
    kinds = sorted({k[0] for k in tick} | {k[0] for k in flush})
    for k in kinds:
        n_t = sum(v for key, v in tick.items() if key[0] == k)
        n_f = sum(v for key, v in flush.items() if key[0] == k)
        print(f"  {k}: tick {n_t}, flush {n_f}")
    dups = sum(v - 1 for v in tick.values() if v > 1) + sum(v - 1 for v in flush.values() if v > 1)
    print(f"  duplicate records: {dups}")
    same_days = results["tick"].days == results["flush"].days
    if only_tick or only_flush or not same_days:
        print(f"DIFFERENT: {sum(only_tick.values())} only in tick, {sum(only_flush.values())} only in flush"
              f"{'' if same_days else ', replayed days differ'}")
        for key in list(only_tick)[:5]:
            print("  tick-only :", key[:5])
        for key in list(only_flush)[:5]:
            print("  flush-only:", key[:5])
        return 1
    print(f"IDENTICAL: {sum(tick.values())} signals over {len(results['tick'].days)} days "
          f"(strategy {results['tick'].provenance['strategy']['id']} "
          f"{results['tick'].provenance['strategy']['fingerprint']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
