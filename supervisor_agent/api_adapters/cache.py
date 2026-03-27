"""Simple in-memory TTL cache with max-size LRU eviction.

Thread-safe via asyncio (single-threaded event loop).
Key = (http_method, url, sorted_params_tuple).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class TTLCache:
    """Bounded dict cache with per-entry expiry."""

    def __init__(self, max_size: int = 2048, default_ttl: int = 3600):
        self._max_size = max_size
        self._default_ttl = default_ttl
        # value: (data, expires_at)
        self._store: OrderedDict[Hashable, Tuple[Any, float]] = OrderedDict()

    # ── Public API ──

    def get(self, key: Hashable) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        data, expires_at = entry
        if time.monotonic() > expires_at:
            self._store.pop(key, None)
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return data

    def set(self, key: Hashable, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expires_at)
        self._evict()

    def invalidate(self, key: Hashable) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    # ── Internals ──

    def _evict(self) -> None:
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    @staticmethod
    def make_key(method: str, url: str, params: Optional[dict] = None) -> Hashable:
        """Deterministic cache key from request identity."""
        param_tuple = tuple(sorted((params or {}).items()))
        return (method.upper(), url, param_tuple)
