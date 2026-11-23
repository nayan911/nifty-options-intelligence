"""Downloads historical bars from FYERS into the research store.

FYERS limits (verified live, Standard plan): 1m history is served for at least ~11 months and
up to 100 days per request; expired futures contracts are rejected ("Invalid symbol") on the
normal history API, so per-contract history is only available while a contract is listed.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, List, Optional

from nifty_engine.research.history_store import ResearchHistoryStore

logger = logging.getLogger(__name__)

INTRADAY_CHUNK_DAYS = 90   # below FYERS' 100-day cap for minute data
DAILY_CHUNK_DAYS = 360


@dataclass(frozen=True)
class SeriesSpec:
    """One series to download.

    kind: SPOT (index/equity), CONT (FYERS stitched futures), OWN (a real futures contract)
    """
    instrument: str
    symbol: str
    kind: str
    resolutions: tuple = ("1", "D")
    contract_expiry: Optional[date] = None

    @property
    def include_oi(self) -> bool:
        return self.kind in ("CONT", "OWN")

    @property
    def continuous(self) -> bool:
        # Continuous stitching only for CONT; OWN must be the contract's own unadjusted series.
        return self.kind == "CONT"


@dataclass
class LoadResult:
    spec: SeriesSpec
    resolution: str
    requests: int = 0
    bars: int = 0
    skipped_chunks: int = 0
    errors: List[str] = field(default_factory=list)


def date_chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=days - 1))
        yield cur, stop
        cur = stop + timedelta(days=1)


class HistoryLoader:
    def __init__(self, rest_client, store: ResearchHistoryStore,
                 is_trading_day: Optional[Callable[[date], bool]] = None) -> None:
        self.rest_client = rest_client
        self.store = store
        self.is_trading_day = is_trading_day or (lambda d: d.weekday() < 5)

    def _chunk_complete(self, spec: SeriesSpec, resolution: str, lo: date, hi: date) -> bool:
        """Skip re-downloading a chunk whose first and last trading days are already stored."""
        days = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
        tdays = [d for d in days if self.is_trading_day(d)]
        if not tdays:
            return True
        source = "SPOT" if spec.kind == "SPOT" else spec.kind
        return (self.store.has_day(spec.symbol, source, resolution, tdays[0])
                and self.store.has_day(spec.symbol, source, resolution, tdays[-1]))

    def load(self, spec: SeriesSpec, start: date, end: date, resolution: str,
             resume: bool = True) -> LoadResult:
        res = LoadResult(spec=spec, resolution=resolution)
        span = DAILY_CHUNK_DAYS if resolution == "D" else INTRADAY_CHUNK_DAYS
        source = "SPOT" if spec.kind == "SPOT" else spec.kind
        for lo, hi in date_chunks(start, end, span):
            # Always refresh the most recent chunk (today's bars may have been partial before).
            if resume and hi < end and self._chunk_complete(spec, resolution, lo, hi):
                res.skipped_chunks += 1
                continue
            try:
                candles = self.rest_client.get_historical_candles_sync(
                    symbol=spec.symbol,
                    resolution=resolution,
                    date_from=lo.strftime("%Y-%m-%d"),
                    date_to=hi.strftime("%Y-%m-%d"),
                    continuous=spec.continuous,
                    include_oi=spec.include_oi,
                )
                res.requests += 1
            except Exception as e:  # keep going: one bad chunk must not abort the whole build
                res.requests += 1
                res.errors.append(f"{lo}..{hi}: {e}")
                logger.warning("History load failed for %s %s %s..%s: %s", spec.symbol, resolution, lo, hi, e)
                continue
            res.bars += self.store.upsert_bars(
                spec.instrument, spec.symbol, source, resolution, candles, contract_expiry=spec.contract_expiry
            )
        return res

    def load_all(self, specs: List[SeriesSpec], start: date, end: date,
                 progress: Optional[Callable[[LoadResult], None]] = None) -> List[LoadResult]:
        results = []
        for spec in specs:
            for resolution in spec.resolutions:
                r = self.load(spec, start, end, resolution)
                results.append(r)
                if progress:
                    progress(r)
            if spec.kind == "CONT":
                for resolution in spec.resolutions:
                    if resolution != "D":
                        self.store.assign_cont_segments(spec.instrument, resolution)
        return results
