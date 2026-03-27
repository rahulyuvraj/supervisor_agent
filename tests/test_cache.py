"""Tests for TTLCache."""

import time

import pytest

from supervisor_agent.api_adapters.cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        c = TTLCache(max_size=10, default_ttl=60)
        c.set("k1", {"v": 1})
        assert c.get("k1") == {"v": 1}

    def test_missing_key_returns_none(self):
        c = TTLCache()
        assert c.get("nonexistent") is None

    def test_expiry(self):
        c = TTLCache(default_ttl=0)
        c.set("k1", "val", ttl=0)
        # Immediately expire (ttl=0 means expires at set-time)
        time.sleep(0.01)
        assert c.get("k1") is None

    def test_custom_ttl_overrides_default(self):
        c = TTLCache(default_ttl=0)
        c.set("k1", "val", ttl=300)
        assert c.get("k1") == "val"

    def test_max_size_eviction(self):
        c = TTLCache(max_size=3, default_ttl=3600)
        for i in range(5):
            c.set(f"k{i}", i)
        assert c.size == 3
        # Oldest keys (k0, k1) evicted
        assert c.get("k0") is None
        assert c.get("k1") is None
        assert c.get("k4") == 4

    def test_invalidate(self):
        c = TTLCache()
        c.set("k1", 42)
        c.invalidate("k1")
        assert c.get("k1") is None

    def test_clear(self):
        c = TTLCache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.size == 0

    def test_make_key_deterministic(self):
        k1 = TTLCache.make_key("GET", "/path", {"b": 2, "a": 1})
        k2 = TTLCache.make_key("get", "/path", {"a": 1, "b": 2})
        assert k1 == k2

    def test_make_key_without_params(self):
        k = TTLCache.make_key("POST", "/path")
        assert k == ("POST", "/path", ())

    def test_lru_refresh(self):
        c = TTLCache(max_size=2, default_ttl=3600)
        c.set("a", 1)
        c.set("b", 2)
        # Access "a" to make it recently used
        c.get("a")
        # Adding "c" should evict "b" (least recently used)
        c.set("c", 3)
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("c") == 3
