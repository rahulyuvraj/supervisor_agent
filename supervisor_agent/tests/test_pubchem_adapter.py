"""Tests for PubChemAdapter — compound lookup, properties, similarity."""

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.pubchem import PubChemAdapter


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


class TestPubChemAdapter:
    @pytest.mark.asyncio
    async def test_get_compound_by_name(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "PC_Compounds": [{"id": {"id": {"cid": 2244}}, "atoms": {}}]
            })
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        result = await adapter.get_compound_by_name("aspirin")
        assert result["id"]["id"]["cid"] == 2244
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_compound_by_name_not_found(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"PC_Compounds": []})
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        result = await adapter.get_compound_by_name("nonexistent_xyz")
        assert result == {}
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_compound_properties(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "PropertyTable": {
                    "Properties": [{
                        "CID": 2244,
                        "MolecularFormula": "C9H8O4",
                        "MolecularWeight": "180.16",
                    }]
                }
            })
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        props = await adapter.get_compound_properties("aspirin")
        assert props["MolecularFormula"] == "C9H8O4"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_similarity_search(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"IdentifierList": {"CID": [2244, 71616]}})
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        cids = await adapter.similarity_search("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert 2244 in cids
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_synonyms(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "InformationList": {
                    "Information": [{"Synonym": ["Aspirin", "ASA", "Acetylsalicylic acid"]}]
                }
            })
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        syns = await adapter.get_synonyms("aspirin")
        assert "Aspirin" in syns
        assert len(syns) == 3
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_synonyms_empty(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"InformationList": {"Information": []}})
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        syns = await adapter.get_synonyms("nonexistent")
        assert syns == []
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_compound_by_cid(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"PC_Compounds": [{"id": {"id": {"cid": 5090}}}]})
        )
        adapter = PubChemAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://pubchem.ncbi.nlm.nih.gov"
        )
        result = await adapter.get_compound_by_cid(5090)
        assert result["id"]["id"]["cid"] == 5090
        await adapter._client.aclose()
