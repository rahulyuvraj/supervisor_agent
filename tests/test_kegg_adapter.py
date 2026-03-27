"""Tests for KEGGAdapter — gating + parsing, all mocked."""

import httpx
import pytest

from supervisor_agent.api_adapters.base import AdapterDisabledError
from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.kegg import KEGGAdapter


def _text_resp(text, status=200):
    return httpx.Response(status, text=text, headers={"content-type": "text/plain"})


@pytest.fixture
def enabled_config():
    return APIConfig(kegg_enabled=True, max_retries=1)


@pytest.fixture
def disabled_config():
    return APIConfig(kegg_enabled=False, max_retries=1)


class TestKEGGGating:
    @pytest.mark.asyncio
    async def test_disabled_raises(self, disabled_config):
        adapter = KEGGAdapter(disabled_config)
        with pytest.raises(AdapterDisabledError, match="disabled"):
            await adapter.get_pathway("hsa04010")

    @pytest.mark.asyncio
    async def test_all_methods_gated(self, disabled_config):
        adapter = KEGGAdapter(disabled_config)
        gated = [
            adapter.list_pathways(),
            adapter.find_pathways_for_gene("hsa:7157"),
            adapter.get_drug("D00596"),
            adapter.find("compound", "aspirin"),
        ]
        for coro in gated:
            with pytest.raises(AdapterDisabledError):
                await coro


class TestKEGGParsing:
    @pytest.mark.asyncio
    async def test_list_pathways(self, enabled_config):
        tsv = "path:hsa04010\tMAPK signaling\npath:hsa04110\tCell cycle"
        transport = httpx.MockTransport(lambda req: _text_resp(tsv))
        adapter = KEGGAdapter(enabled_config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://rest.kegg.jp")
        result = await adapter.list_pathways("hsa")
        assert len(result) == 2
        assert result[0]["pathway_id"] == "path:hsa04010"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_pathway_flat(self, enabled_config):
        flat = "ENTRY       hsa04010\nNAME        MAPK signaling pathway\nDESCRIPTION This is a test\n///"
        transport = httpx.MockTransport(lambda req: _text_resp(flat))
        adapter = KEGGAdapter(enabled_config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://rest.kegg.jp")
        result = await adapter.get_pathway("hsa04010")
        assert result["ENTRY"] == "hsa04010"
        assert result["NAME"] == "MAPK signaling pathway"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_flat_continuation_lines(self, enabled_config):
        flat = "NAME        Long name\n            continued here\n///"
        transport = httpx.MockTransport(lambda req: _text_resp(flat))
        adapter = KEGGAdapter(enabled_config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://rest.kegg.jp")
        result = await adapter.get_pathway("test")
        assert "continued here" in result["NAME"]
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_find_drug_interactions(self, enabled_config):
        tsv = "D00596\tD00123\tinhibitor"
        transport = httpx.MockTransport(lambda req: _text_resp(tsv))
        adapter = KEGGAdapter(enabled_config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://rest.kegg.jp")
        result = await adapter.find_drug_interactions("D00596")
        assert len(result) == 1
        assert result[0]["interaction"] == "inhibitor"
        await adapter._client.aclose()
