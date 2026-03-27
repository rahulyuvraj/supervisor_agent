"""Async rate limiter: concurrency bound + minimum inter-request interval.

Uses asyncio.Semaphore for concurrency and tracks last-request time per
limiter to sleep only the remaining interval (avoids unnecessary delays
when requests take longer than the interval).
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket-style limiter using semaphore + elapsed-time tracking."""

    def __init__(self, requests_per_second: float = 5.0, max_concurrency: int = 4):
        self._interval = 1.0 / max(requests_per_second, 0.01)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            remaining = self._interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request = time.monotonic()

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()
