"""Centralized Token Bucket Rate Limiter for FYERS Standard API compliance."""

import asyncio
import time
from typing import Optional
from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import IRateLimiter


class AsyncTokenBucketRateLimiter(IRateLimiter):
    """Guarantees outgoing requests never breach FYERS Standard plan limits.

    Enforces 3 simultaneous boundaries:
    1. Sub-second burst rate: max 4 req/sec (safety margin below 5/s)
    2. Per-minute rate: max 40 req/min (safety margin below 50/min)
    3. Daily quota: max 4,500 req/day (safety margin below 5,000/day)
    """

    def __init__(
        self,
        rate_per_sec: float = settings.RATE_LIMIT_RPS,
        rate_per_min: int = settings.RATE_LIMIT_RPM,
        max_daily_requests: int = settings.RATE_LIMIT_DAILY
    ) -> None:
        self.rate_per_sec = rate_per_sec
        self.rate_per_min = rate_per_min
        self.max_daily_requests = max_daily_requests

        # Sub-second bucket
        self.sec_capacity = rate_per_sec
        self.sec_tokens = rate_per_sec
        self.last_sec_refill = time.monotonic()

        # Minute bucket (sliding window tracker)
        self.minute_timestamps = []

        # Daily usage
        self.daily_requests_count = 0
        self.daily_reset_epoch = time.time()

        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Asynchronously wait until a request token is available in all windows."""
        async with self._lock:
            while True:
                now = time.monotonic()
                wall_now = time.time()

                # 1. Reset daily count if 24 hours have passed
                if wall_now - self.daily_reset_epoch >= 86400:
                    self.daily_requests_count = 0
                    self.daily_reset_epoch = wall_now

                if self.daily_requests_count >= self.max_daily_requests:
                    raise RuntimeError(
                        f"FYERS Daily API budget exhausted ({self.daily_requests_count}/{self.max_daily_requests}). "
                        "Halted to prevent 24-hour account lockout."
                    )

                # 2. Refill sub-second tokens
                elapsed_sec = now - self.last_sec_refill
                self.sec_tokens = min(self.sec_capacity, self.sec_tokens + elapsed_sec * self.rate_per_sec)
                self.last_sec_refill = now

                # 3. Clean up minute timestamps (> 60 seconds old)
                cutoff_time = now - 60.0
                self.minute_timestamps = [t for t in self.minute_timestamps if t > cutoff_time]

                # Check if both sub-second and per-minute constraints are satisfied
                has_sec_token = self.sec_tokens >= 1.0
                has_min_token = len(self.minute_timestamps) < self.rate_per_min

                if has_sec_token and has_min_token:
                    # Deduct tokens and grant admission
                    self.sec_tokens -= 1.0
                    self.minute_timestamps.append(now)
                    self.daily_requests_count += 1
                    return

                # Calculate required sleep time
                sleep_delay = 0.05
                if not has_min_token and self.minute_timestamps:
                    oldest = self.minute_timestamps[0]
                    sleep_delay = max(sleep_delay, (oldest + 60.0) - now + 0.01)
                elif not has_sec_token:
                    sleep_delay = max(sleep_delay, (1.0 - self.sec_tokens) / self.rate_per_sec)

                await asyncio.sleep(min(sleep_delay, 1.0))

    def get_status(self) -> dict:
        """Current rate limiter metrics."""
        return {
            "daily_requests_used": self.daily_requests_count,
            "daily_requests_remaining": max(0, self.max_daily_requests - self.daily_requests_count),
            "requests_last_minute": len(self.minute_timestamps),
            "minute_budget_remaining": max(0, self.rate_per_min - len(self.minute_timestamps)),
            "subsecond_tokens": round(self.sec_tokens, 2),
        }
