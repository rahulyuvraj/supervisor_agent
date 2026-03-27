"""Tests for OpenFDAAdapter — label search, adverse events, API key injection."""

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.openfda import OpenFDAAdapter


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1, openfda_api_key="test-key")


@pytest.fixture
def config_no_key():
    return APIConfig(max_retries=1)


class TestOpenFDAAdapter:
    @pytest.mark.asyncio
    async def test_search_labels(self, config):
        def handler(req):
            assert "api_key=test-key" in str(req.url)
            return _json_resp({"results": [{"openfda": {"brand_name": ["HUMIRA"]}}]})

        transport = httpx.MockTransport(handler)
        adapter = OpenFDAAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://api.fda.gov")
        result = await adapter.search_labels('openfda.brand_name:"HUMIRA"')
        assert "results" in result
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_no_api_key(self, config_no_key):
        def handler(req):
            assert "api_key" not in str(req.url)
            return _json_resp({"results": []})

        transport = httpx.MockTransport(handler)
        adapter = OpenFDAAdapter(config_no_key)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://api.fda.gov")
        await adapter.search_labels('openfda.brand_name:"TEST"')
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_adverse_events(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"results": [{"serious": 1}]})
        )
        adapter = OpenFDAAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://api.fda.gov")
        result = await adapter.search_adverse_events('patient.drug.openfda.brand_name:"TEST"')
        assert len(result["results"]) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_count_adverse_events(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"results": [{"term": "NAUSEA", "count": 42}]})
        )
        adapter = OpenFDAAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://api.fda.gov")
        result = await adapter.count_adverse_events('patient.drug.openfda.brand_name:"TEST"')
        assert result["results"][0]["count"] == 42
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_ndc(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"results": [{"product_ndc": "0000-0000-00"}]})
        )
        adapter = OpenFDAAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://api.fda.gov")
        result = await adapter.search_ndc('openfda.brand_name:"TEST"')
        assert "results" in result
        await adapter._client.aclose()
