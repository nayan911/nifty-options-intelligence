"""Downloads historical bars from FYERS into the research store.

FYERS limits (verified live, Standard plan): 1m history is served for at least ~11 months and
up to 100 days per request; expired futures contracts are rejected ("Invalid symbol") on the
normal history API, so per-contract history is only available while a contract is listed.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, List, Optional

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.sessions import expected_minutes, kind_for

logger = logging.getLogger(__name__)

INTRADAY_CHUNK_DAYS = 90   # below FYERS' 100-day cap for minute data
DAILY_CHUNK_DAYS = 360


@dataclass(frozen=True)
class SeriesSpec:
    """One series to download.

    kind: SPOT (index/equity), CONT (FYERS stitched NEAR-month futures), CONT_NEXT (stitched
    NEXT-month futures), OWN (a real futures contract, unadjusted).

    FYERS stitches by the requested contract's position: requesting the nearest listed contract
    yields the near-month series, requesting the second yields the next-month series (verified:
    via DEC on 23-Nov the "continuous" bars matched the DEC/NOV next-month contracts exactly).
    """
    instrument: str
    symbol: str          # symbol sent to FYERS
    kind: str
    resolutions: tuple = ("1", "D")
    contract_expiry: Optional[date] = None
    underlying: Optional[str] = None

    @property
    def include_oi(self) -> bool:
        return self.kind in ("CONT", "CONT_NEXT", "OWN")

    @property
    def continuous(self) -> bool:
        # Stitching only for CONT series; OWN must be the contract's own unadjusted series.
        return self.kind in ("CONT", "CONT_NEXT")

    @property
    def source(self) -> str:
        return "SPOT" if self.kind == "SPOT" else self.kind

    @property
    def store_symbol(self) -> str:
        """Stable storage key. Stitched series get a logical name so later refreshes (requested via
        a different contract after rollover) update the same rows instead of duplicating history."""
        if self.kind in ("CONT", "CONT_NEXT"):
            pos = "NEAR" if self.kind == "CONT" else "NEXT"
            return f"CONT:{self.underlying or self.instrument}:{pos}"
        return self.symbol


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
                 calendar: Optional[ExchangeCalendar] = None, sparse_tolerance: int = 5) -> None:
        self.rest_client = rest_client
        self.store = store
        self.calendar = calendar or ExchangeCalendar()
        self.sparse_tolerance = sparse_tolerance

    def is_trading_day(self, d: date) -> bool:
        return self.calendar.is_trading_day(d)

    def _chunk_complete(self, spec: SeriesSpec, resolution: str, lo: date, hi: date) -> bool:
        """A chunk may be skipped only if EVERY trading day in it is complete.

        Intraday: every minute the exchange calendar expects is stored (a real futures contract
        may miss up to `sparse_tolerance` no-trade minutes). Daily: a bar exists for every trading
        day. A real contract's chunk entirely before its first available bar is pre-listing
        (FYERS serves contracts only from listing) and is not re-requested.
        """
        tdays = [d for d in (lo + timedelta(days=i) for i in range((hi - lo).days + 1)) if self.is_trading_day(d)]
        if not tdays:
            return True
        source = spec.source
        sym = spec.store_symbol
        if spec.kind == "OWN":
            first = self.store.first_day(sym, source, resolution)
            if first is None:
                return False
            if hi < first:
                return True
            tdays = [d for d in tdays if d >= first]
        stored = self.store.minutes_by_day(sym, source, resolution, lo, hi)
        if resolution == "D":
            return all(d in stored for d in tdays)
        allowed = self.sparse_tolerance if spec.kind in ("OWN", "CONT_NEXT") else 0
        kind = kind_for(spec.kind)
        for d in tdays:
            missing = len(set(expected_minutes(self.calendar, d, kind)) - stored.get(d, set()))
            if missing > allowed:
                return False
        return True

    def _append_start(self, spec: SeriesSpec, resolution: str, start: date, end: date) -> Optional[date]:
        """Stitched series are APPEND-ONLY: return the first day to request.

        That is the day after the last complete stored day, or the last stored day itself if it is
        only partially stored (its missing minutes may be filled, but only via `_stitched_safe_bars`,
        which never overwrites a stored bar and rejects the day if FYERS now serves a different
        contract for it).

        FYERS stitches by the requested contract's position at request time, and that meaning shifts
        around each rollover (a refresh on 23-Nov via DEC returned near-month bars and overwrote the
        stored next-month series from 25-Sep). Re-requesting or back-filling stored history is
        therefore unsafe; only days after the stored range are fetched. Returns None if the series
        is empty (initial download of the whole range).
        """
        stored = self.store.minutes_by_day(spec.store_symbol, spec.source, resolution, start, end)
        if not stored:
            return None
        kind = kind_for(spec.kind)
        allowed = self.sparse_tolerance if spec.kind == "CONT_NEXT" else 0
        last_complete = None
        for d in sorted(stored, reverse=True):
            if resolution == "D":
                last_complete = d
                break
            missing = len(set(expected_minutes(self.calendar, d, kind)) - stored[d])
            if missing <= allowed:
                last_complete = d
                break
        last_stored = max(stored)
        if last_complete is None or last_complete < last_stored:
            return last_stored  # resume a partial day (insert-only, identity-checked)
        return last_complete + timedelta(days=1)

    def _stitched_safe_bars(self, spec: SeriesSpec, resolution: str, candles, res: "LoadResult"):
        """Drop every bar on a day where FYERS' answer disagrees with what is already stored.

        If any already-stored minute comes back with a different close or OI, the stitching now
        points at a different contract: the whole day is rejected (filling its gaps from another
        contract would mix contracts within a session).
        """
        stored = self.store.stored_bars_at(spec.store_symbol, spec.source, resolution, [c.timestamp for c in candles])
        bad_days = set()
        for c in candles:
            prev = stored.get(c.timestamp)
            if prev is not None and (abs(float(prev[0]) - float(c.close)) > 0.01 or prev[1] != c.oi):
                bad_days.add(c.timestamp.date())
        for d in sorted(bad_days):
            res.errors.append(f"{d}: FYERS returned a different contract than the stored bars; day not updated")
        return [c for c in candles if c.timestamp.date() not in bad_days]

    def load(self, spec: SeriesSpec, start: date, end: date, resolution: str,
             resume: bool = True) -> LoadResult:
        res = LoadResult(spec=spec, resolution=resolution)
        span = DAILY_CHUNK_DAYS if resolution == "D" else INTRADAY_CHUNK_DAYS
        if resume and spec.kind in ("CONT", "CONT_NEXT"):
            nxt = self._append_start(spec, resolution, start, end)
            if nxt is not None:
                if nxt > end:
                    res.skipped_chunks += 1
                    return res
                start = nxt
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
            stitched = spec.kind in ("CONT", "CONT_NEXT")
            if stitched:
                candles = self._stitched_safe_bars(spec, resolution, candles, res)
            res.bars += self.store.upsert_bars(
                spec.instrument, spec.store_symbol, spec.source, resolution, candles,
                contract_expiry=spec.contract_expiry, insert_only=stitched,
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
        return results
