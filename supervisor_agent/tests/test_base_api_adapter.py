"""Tests for BaseAPIAdapter — retry, cache, error handling."""

import asyncio
import json

import httpx
import pytest

from supervisor_agent.api_adapters.base import (
    API_ADAPTER_REGISTRY,
    BaseAPIAdapter,
    PermanentError,
    TransientError,
)
from supervisor_agent.api_adapters.cache import TTLCache
from supervisor_agent.api_adapters.config import APIConfig


# ── Test adapter subclass ──

class _MockAdapter(BaseAPIAdapter):
    service_name = "test_mock"
    base_url = "https://mock.test"


# ── Helpers ──

def _json_response(data, status=200):
    return httpx.Response(
        status,
        json=data,
        headers={"content-type": "application/json"},
    )


def _text_response(text, status=200):
    return httpx.Response(
        status,
        text=text,
        headers={"content-type": "text/plain"},
    )


class TestAutoRegistration:
    def test_subclass_registers(self):
        assert "test_mock" in API_ADAPTER_REGISTRY
        assert API_ADAPTER_REGISTRY["test_mock"] is _MockAdapter


class TestBaseRequest:
    @pytest.fixture
    def config(self):
        return APIConfig(max_retries=2, default_timeout=5.0)

    @pytest.mark.asyncio
    async def test_get_json(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_response({"ok": True})
        )
        adapter = _MockAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        result = await adapter._request("GET", "/test")
        assert result == {"ok": True}
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_text(self, config):
        transport = httpx.MockTransport(
            lambda req: _text_response("col1\tcol2\nA\tB")
        )
        adapter = _MockAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        result = await adapter._request("GET", "/tsv")
        assert "col1" in result
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_cache_hit(self, config):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return _json_response({"n": call_count})

        transport = httpx.MockTransport(handler)
        cache = TTLCache(default_ttl=60)
        adapter = _MockAdapter(config, cache=cache)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        r1 = await adapter._request("GET", "/cached")
        r2 = await adapter._request("GET", "/cached")
        assert r1 == r2 == {"n": 1}
        assert call_count == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_skip_cache(self, config):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return _json_response({"n": call_count})

        transport = httpx.MockTransport(handler)
        adapter = _MockAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        await adapter._request("GET", "/nocache", skip_cache=True)
        await adapter._request("GET", "/nocache", skip_cache=True)
        assert call_count == 2
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_retry_on_500(self, config):
        attempts = 0

        def handler(req):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                return httpx.Response(500)
            return _json_response({"recovered": True})

        transport = httpx.MockTransport(handler)
        adapter = _MockAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        # Patch backoff to be instant
        adapter._backoff = staticmethod(lambda a: asyncio.sleep(0))
        result = await adapter._request("GET", "/flaky", skip_cache=True)
        assert result == {"recovered": True}
        assert attempts == 2
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_permanent_error_on_400(self, config):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(400, text="Bad request")
        )
        adapter = _MockAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        with pytest.raises(PermanentError, match="400"):
            await adapter._request("GET", "/bad")
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_transient(self, config):
        transport = httpx.MockTransport(lambda req: httpx.Response(503))
        adapter = _MockAdapter(APIConfig(max_retries=1))
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://mock.test"
        )
        adapter._backoff = staticmethod(lambda a: asyncio.sleep(0))
        with pytest.raises(TransientError):
            await adapter._request("GET", "/down", skip_cache=True)
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        async with _MockAdapter(config) as adapter:
            assert adapter._client is not None
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_outside_context_raises(self, config):
        adapter = _MockAdapter(config)
        with pytest.raises(RuntimeError, match="outside async context"):
            await adapter._request("GET", "/fail")
