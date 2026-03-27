"""Tests for AsyncRateLimiter."""

import asyncio
import time

import pytest

from supervisor_agent.api_adapters.rate_limiter import AsyncRateLimiter


@pytest.mark.asyncio
class TestAsyncRateLimiter:
    async def test_acquire_release(self):
        rl = AsyncRateLimiter(requests_per_second=100.0, max_concurrency=2)
        async with rl:
            pass  # Should not raise

    async def test_respects_rate(self):
        # 5 rps → 0.2s interval. 3 sequential requests should take ≥0.4s.
        rl = AsyncRateLimiter(requests_per_second=5.0, max_concurrency=4)
        t0 = time.monotonic()
        for _ in range(3):
            async with rl:
                pass
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.35, f"Expected ≥0.35s, got {elapsed:.3f}s"

    async def test_elapsed_time_aware(self):
        """If a request takes longer than the interval, no extra sleep needed."""
        rl = AsyncRateLimiter(requests_per_second=5.0, max_concurrency=4)
        async with rl:
            await asyncio.sleep(0.3)  # Longer than 0.2s interval

        t0 = time.monotonic()
        async with rl:
            pass  # Should be nearly instant
        elapsed = time.monotonic() - t0
        assert elapsed < 0.15, f"Expected near-instant, got {elapsed:.3f}s"

    async def test_concurrency_limit(self):
        rl = AsyncRateLimiter(requests_per_second=1000.0, max_concurrency=2)
        active = 0
        max_active = 0

        async def task():
            nonlocal active, max_active
            async with rl:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                active -= 1

        await asyncio.gather(*(task() for _ in range(5)))
        assert max_active <= 2
