"""Tests for ReactomeAdapter — all mocked, zero real API calls."""

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.reactome import ReactomeAdapter


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


class TestReactomeAdapter:
    @pytest.mark.asyncio
    async def test_get_pathway(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"stId": "R-HSA-1234", "displayName": "Test Pathway"})
        )
        adapter = ReactomeAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://reactome.org")
        result = await adapter.get_pathway("R-HSA-1234")
        assert result["stId"] == "R-HSA-1234"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_pathway_participants(self, config):
        participants = [
            {"stId": "R-HSA-100", "schemaClass": "Protein"},
            {"stId": "R-HSA-200", "schemaClass": "Drug"},
        ]
        transport = httpx.MockTransport(lambda req: _json_resp(participants))
        adapter = ReactomeAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://reactome.org")
        result = await adapter.get_pathway_participants("R-HSA-1234")
        assert len(result) == 2
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_analyse_genes(self, config):
        analysis_result = {
            "summary": {"token": "tok123"},
            "pathways": [{"stId": "R-HSA-5678", "pValue": 0.001}],
        }
        transport = httpx.MockTransport(lambda req: _json_resp(analysis_result))
        adapter = ReactomeAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://reactome.org")
        result = await adapter.analyse_genes(["TP53", "BRCA1"])
        assert result["summary"]["token"] == "tok123"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_pathway_drugs_filters(self, config):
        participants = [
            {"stId": "R-1", "schemaClass": "Protein"},
            {"stId": "R-2", "schemaClass": "ChemicalDrug"},
            {"stId": "R-3", "schemaClass": "Drug"},
            {"stId": "R-4", "schemaClass": "SmallMolecule"},
        ]
        transport = httpx.MockTransport(lambda req: _json_resp(participants))
        adapter = ReactomeAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://reactome.org")
        drugs = await adapter.get_pathway_drugs("R-HSA-1234")
        assert len(drugs) == 2
        assert all(d["schemaClass"] in {"Drug", "ChemicalDrug"} for d in drugs)
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_collect_child_pathways(self, config):
        responses = {
            "/ContentService/data/query/R-ROOT": {
                "hasEvent": [{"stId": "R-C1"}, {"stId": "R-C2"}]
            },
            "/ContentService/data/query/R-C1": {"hasEvent": [{"stId": "R-C1a"}]},
            "/ContentService/data/query/R-C2": {"hasEvent": []},
            "/ContentService/data/query/R-C1a": {"hasEvent": []},
        }

        def handler(req):
            path = req.url.path
            return _json_resp(responses.get(path, {"hasEvent": []}))

        transport = httpx.MockTransport(handler)
        adapter = ReactomeAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://reactome.org")
        children = await adapter.collect_child_pathways("R-ROOT", max_depth=3)
        assert set(children) == {"R-C1", "R-C2", "R-C1a"}
        await adapter._client.aclose()
