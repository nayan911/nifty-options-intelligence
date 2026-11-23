"""Centralized sliding-window rate limiter for FYERS Data API compliance.

FYERS Standard plan Data API limits (revised 16 Oct 2026): 5 req/s, 50 req/min,
5,000 req/day. Exceeding the per-minute limit more than 3 times in a day blocks the
account for the rest of the day, so every REST call (history, option chain, quotes)
must pass through ONE shared limiter instance.

Design notes:
- True sliding windows for both the 1s and 60s limits. A token bucket with
  capacity == rate allows up to 2x the rate inside any 1s window (burst + refill),
  which is exactly how the old limiter tripped FYERS' per-second limit at warmup.
- One threading lock guards state for both sync and async callers (the pipeline
  mixes worker threads and short-lived event loops; an asyncio.Lock is bound to a
  single loop and does not exclude threads).
- Daily count resets at IST midnight (FYERS' day), not 24h after process start,
  and can be persisted so restarts during the day don't forget usage.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Optional, Union

import pytz

from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import IRateLimiter

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


def _ist_today() -> date:
    return datetime.now(IST).date()


class AsyncTokenBucketRateLimiter(IRateLimiter):
    """Guarantees outgoing requests never breach FYERS plan limits.

    Enforces three boundaries with safety margins (see settings):
    1. Per-second: at most `rate_per_sec` requests in any rolling 1s window
    2. Per-minute: at most `rate_per_min` requests in any rolling 60s window
    3. Per-day: at most `max_daily_requests` per IST calendar day
    """

    def __init__(
        self,
        rate_per_sec: float = settings.RATE_LIMIT_RPS,
        rate_per_min: int = settings.RATE_LIMIT_RPM,
        max_daily_requests: int = settings.RATE_LIMIT_DAILY,
        usage_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.rate_per_sec = rate_per_sec
        self.rate_per_min = rate_per_min
        self.max_daily_requests = max_daily_requests
        self._sec_cap = max(1, int(rate_per_sec))

        self._sec_window: Deque[float] = deque()
        self._min_window: Deque[float] = deque()

        self.daily_requests_count = 0
        self.rate_limit_hits_today = 0
        self._day: date = _ist_today()

        self._usage_file = Path(usage_file) if usage_file else None
        self._lock = threading.Lock()
        self._load_usage()

    # ------------------------------------------------------------------ persistence
    def _load_usage(self) -> None:
        if not self._usage_file or not self._usage_file.exists():
            return
        try:
            data = json.loads(self._usage_file.read_text())
            if data.get("date") == self._day.isoformat():
                self.daily_requests_count = int(data.get("requests", 0))
                self.rate_limit_hits_today = int(data.get("rate_limit_hits", 0))
        except Exception as e:  # corrupt file must never block trading startup
            logger.warning("Could not read API usage file %s: %s", self._usage_file, e)

    def _save_usage(self) -> None:
        if not self._usage_file:
            return
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._usage_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "date": self._day.isoformat(),
                        "requests": self.daily_requests_count,
                        "rate_limit_hits": self.rate_limit_hits_today,
                    }
                )
            )
            tmp.replace(self._usage_file)
        except Exception as e:
            logger.debug("Could not persist API usage: %s", e)

    # ------------------------------------------------------------------ core logic
    @property
    def minute_timestamps(self) -> list:
        """Backward-compatible view of the rolling 60s window."""
        return list(self._min_window)

    def _roll_day(self) -> None:
        today = _ist_today()
        if today != self._day:
            self._day = today
            self.daily_requests_count = 0
            self.rate_limit_hits_today = 0

    def _evict(self, now: float) -> None:
        while self._sec_window and self._sec_window[0] <= now - 1.0:
            self._sec_window.popleft()
        while self._min_window and self._min_window[0] <= now - 60.0:
            self._min_window.popleft()

    def _try_admit(self) -> float:
        """Admit one request if possible. Returns 0.0 on admission, else seconds to wait.

        Caller must hold self._lock.
        """
        self._roll_day()
        if self.daily_requests_count >= self.max_daily_requests:
            raise RuntimeError(
                f"FYERS Daily API budget exhausted ({self.daily_requests_count}/{self.max_daily_requests}). "
                "Halted to prevent account lockout."
            )

        now = time.monotonic()
        self._evict(now)

        wait = 0.0
        if len(self._sec_window) >= self._sec_cap:
            wait = max(wait, self._sec_window[0] + 1.0 - now)
        if len(self._min_window) >= self.rate_per_min:
            wait = max(wait, self._min_window[0] + 60.0 - now)
        if wait > 0:
            return wait + 0.01

        self._sec_window.append(now)
        self._min_window.append(now)
        self.daily_requests_count += 1
        self._save_usage()
        return 0.0

    def acquire_sync(self) -> None:
        """Block the calling thread until a request slot is available in all windows."""
        while True:
            with self._lock:
                wait = self._try_admit()
            if wait == 0.0:
                return
            time.sleep(min(wait, 1.0))

    async def acquire(self) -> None:
        """Await until a request slot is available in all windows."""
        while True:
            with self._lock:
                wait = self._try_admit()
            if wait == 0.0:
                return
            await asyncio.sleep(min(wait, 1.0))

    def try_acquire(self) -> bool:
        """Non-blocking admission for opportunistic calls (e.g. on-demand quotes).

        Returns False instead of waiting, so latency-sensitive paths (tick handlers)
        never stall behind the per-minute window.
        """
        with self._lock:
            try:
                return self._try_admit() == 0.0
            except RuntimeError:
                return False

    def record_rate_limit_hit(self) -> int:
        """Record a broker-side 429 / limit response. Returns today's hit count.

        FYERS blocks the account for the day after >3 per-minute violations, so
        callers should back off for a full minute and treat hits >= 2 as critical.
        """
        with self._lock:
            self._roll_day()
            self.rate_limit_hits_today += 1
            self._save_usage()
            hits = self.rate_limit_hits_today
        logger.warning("FYERS rate limit response received (%d today)", hits)
        return hits

    def get_status(self) -> dict:
        """Current rate limiter metrics."""
        with self._lock:
            self._roll_day()
            self._evict(time.monotonic())
            return {
                "daily_requests_used": self.daily_requests_count,
                "daily_requests_remaining": max(0, self.max_daily_requests - self.daily_requests_count),
                "requests_last_minute": len(self._min_window),
                "minute_budget_remaining": max(0, self.rate_per_min - len(self._min_window)),
                "requests_last_second": len(self._sec_window),
                "rate_limit_hits_today": self.rate_limit_hits_today,
            }
