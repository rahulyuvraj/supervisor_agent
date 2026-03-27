"""Tests for STRINGAdapter — TSV parsing, batching, all mocked."""

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.string_ppi import STRINGAdapter


def _text_resp(text, status=200):
    return httpx.Response(status, text=text, headers={"content-type": "text/plain"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


_TSV_IDS = "queryItem\tstringId\tpreferredName\ttaxonId\nTP53\t9606.ENSP00000269305\tTP53\t9606"
_TSV_NETWORK = "stringId_A\tstringId_B\tscore\n9606.ENSP001\t9606.ENSP002\t0.9"
_TSV_ENRICHMENT = "category\tterm\tdescription\tnumber_of_genes\tp_value\nProcess\tGO:001\ttest\t5\t0.001"


class TestSTRINGAdapter:
    @pytest.mark.asyncio
    async def test_resolve_ids(self, config):
        transport = httpx.MockTransport(lambda req: _text_resp(_TSV_IDS))
        adapter = STRINGAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://string-db.org")
        rows = await adapter.resolve_ids(["TP53"])
        assert len(rows) == 1
        assert rows[0]["queryItem"] == "TP53"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_network(self, config):
        transport = httpx.MockTransport(lambda req: _text_resp(_TSV_NETWORK))
        adapter = STRINGAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://string-db.org")
        rows = await adapter.get_network(["TP53", "MDM2"])
        assert len(rows) == 1
        assert "score" in rows[0]
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_enrichment(self, config):
        transport = httpx.MockTransport(lambda req: _text_resp(_TSV_ENRICHMENT))
        adapter = STRINGAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://string-db.org")
        rows = await adapter.get_enrichment(["TP53", "BRCA1"])
        assert rows[0]["category"] == "Process"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_empty_response(self, config):
        transport = httpx.MockTransport(lambda req: _text_resp(""))
        adapter = STRINGAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://string-db.org")
        rows = await adapter.resolve_ids(["NONEXISTENT"])
        assert rows == []
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_batching_large_ids(self, config):
        request_count = 0

        def handler(req):
            nonlocal request_count
            request_count += 1
            return _text_resp(_TSV_IDS)

        transport = httpx.MockTransport(handler)
        adapter = STRINGAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://string-db.org")
        ids = [f"GENE{i}" for i in range(250)]
        await adapter.resolve_ids(ids)
        assert request_count == 2  # 200 + 50
        await adapter._client.aclose()
