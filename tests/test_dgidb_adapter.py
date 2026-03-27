"""Tests for DGIdbAdapter — GraphQL batching, null score handling."""

import json

import httpx
import pytest

from supervisor_agent.api_adapters.config import APIConfig
from supervisor_agent.api_adapters.dgidb import DGIdbAdapter


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


def _gene_response(gene_name, score=None):
    return {
        "data": {
            "genes": {
                "nodes": [
                    {
                        "name": gene_name,
                        "conceptId": "hgnc:1234",
                        "interactions": [
                            {
                                "interactionScore": score,
                                "interactionTypes": [{"type": "inhibitor", "directionality": "inhibits"}],
                                "drug": {"name": "TestDrug", "conceptId": "chembl:1", "approved": True},
                                "interactionAttributes": [],
                                "publications": [{"pmid": "12345"}],
                                "sources": [{"fullName": "DGIdb"}],
                            }
                        ],
                    }
                ]
            }
        }
    }


class TestDGIdbAdapter:
    @pytest.mark.asyncio
    async def test_gene_interactions(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp(_gene_response("TP53", score=0.85))
        )
        adapter = DGIdbAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://dgidb.org")
        nodes = await adapter.get_gene_interactions(["TP53"])
        assert len(nodes) == 1
        assert nodes[0]["name"] == "TP53"
        assert nodes[0]["interactions"][0]["interactionScore"] == 0.85
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_null_score_normalised(self, config):
        """DGIdb 5.0 may return null interactionScore — we default to 0.0."""
        transport = httpx.MockTransport(
            lambda req: _json_resp(_gene_response("BRCA1", score=None))
        )
        adapter = DGIdbAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://dgidb.org")
        nodes = await adapter.get_gene_interactions(["BRCA1"])
        assert nodes[0]["interactions"][0]["interactionScore"] == 0.0
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_batching(self, config):
        """Genes exceeding batch size should be split into multiple requests."""
        request_count = 0

        def handler(req):
            nonlocal request_count
            request_count += 1
            body = json.loads(req.content)
            batch_size = len(body["variables"]["names"])
            # Return one node per gene
            nodes = [{"name": f"G{i}", "conceptId": "x", "interactions": []}
                     for i in range(batch_size)]
            return _json_resp({"data": {"genes": {"nodes": nodes}}})

        transport = httpx.MockTransport(handler)
        adapter = DGIdbAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://dgidb.org")
        # 75 genes → should be 2 batches (50 + 25)
        genes = [f"GENE{i}" for i in range(75)]
        nodes = await adapter.get_gene_interactions(genes)
        assert request_count == 2
        assert len(nodes) == 75
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_drug_interactions(self, config):
        drug_resp = {
            "data": {
                "drugs": {
                    "nodes": [
                        {
                            "name": "TestDrug",
                            "conceptId": "chembl:1",
                            "approved": True,
                            "interactions": [],
                        }
                    ]
                }
            }
        }
        transport = httpx.MockTransport(lambda req: _json_resp(drug_resp))
        adapter = DGIdbAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://dgidb.org")
        nodes = await adapter.get_drug_interactions(["TestDrug"])
        assert nodes[0]["name"] == "TestDrug"
        await adapter._client.aclose()
