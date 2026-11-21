"""Unit tests for AsyncTokenBucketRateLimiter."""

import asyncio
import time

from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter


def test_subsecond_rate_limiting():
    async def _run():
        limiter = AsyncTokenBucketRateLimiter(rate_per_sec=4.0, rate_per_min=100, max_daily_requests=1000)

        start_time = time.monotonic()
        # Acquire 5 tokens: first 4 instant, 5th should wait ~0.25s
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start_time

        assert elapsed >= 0.20  # Proves sub-second throttling occurred
        assert limiter.daily_requests_count == 5

    asyncio.run(_run())


def test_minute_rate_limiting():
    async def _run():
        # Very small cap: 3 requests per minute
        limiter = AsyncTokenBucketRateLimiter(rate_per_sec=10.0, rate_per_min=3, max_daily_requests=100)

        for _ in range(3):
            await limiter.acquire()

        status = limiter.get_status()
        assert status["requests_last_minute"] == 3
        assert status["minute_budget_remaining"] == 0

    asyncio.run(_run())


def test_daily_budget_exhaustion():
    async def _run():
        limiter = AsyncTokenBucketRateLimiter(rate_per_sec=100.0, rate_per_min=100, max_daily_requests=2)

        await limiter.acquire()
        await limiter.acquire()

        try:
            await limiter.acquire()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Daily API budget exhausted" in str(e)

    asyncio.run(_run())
