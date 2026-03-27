"""Tests for EnsemblAdapter — gene lookup, VEP, xrefs."""

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.ensembl import EnsemblAdapter


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


class TestEnsemblAdapter:
    @pytest.mark.asyncio
    async def test_lookup_symbol(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "id": "ENSG00000141510",
                "display_name": "TP53",
                "biotype": "protein_coding",
            })
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.lookup_symbol("TP53")
        assert result["id"] == "ENSG00000141510"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_lookup_id(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "id": "ENSG00000141510",
                "display_name": "TP53",
                "species": "homo_sapiens",
            })
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.lookup_id("ENSG00000141510")
        assert result["display_name"] == "TP53"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_sequence(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "id": "ENSG00000141510",
                "seq": "ATGCGATCG...",
            })
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.get_sequence("ENSG00000141510")
        assert "seq" in result
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_vep_hgvs(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp([{
                "most_severe_consequence": "missense_variant",
                "transcript_consequences": [],
            }])
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.vep_hgvs("ENST00000269305.9:c.215C>G")
        assert result[0]["most_severe_consequence"] == "missense_variant"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_vep_region(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp([{
                "most_severe_consequence": "synonymous_variant",
            }])
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.vep_region("9:22125503-22125503:1", "C")
        assert len(result) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_xrefs_symbol(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp([
                {"id": "ENSG00000141510", "type": "gene", "dbname": "Ensembl"},
                {"id": "7157", "type": "gene", "dbname": "EntrezGene"},
            ])
        )
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        xrefs = await adapter.xrefs_symbol("TP53")
        assert len(xrefs) == 2
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_lookup_with_expand(self, config):
        def handler(req):
            assert "expand=1" in str(req.url)
            return _json_resp({"id": "ENSG00000141510", "Transcript": []})

        transport = httpx.MockTransport(handler)
        adapter = EnsemblAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://rest.ensembl.org"
        )
        result = await adapter.lookup_symbol("TP53", expand=True)
        assert "Transcript" in result
        await adapter._client.aclose()
