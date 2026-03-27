"""Tests for ChEMBLAdapter — molecule search, mechanisms, targets."""

import httpx
import pytest

from supervisor_agent.api_adapters.chembl import ChEMBLAdapter
from supervisor_agent.api_adapters.config import APIConfig


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


class TestChEMBLAdapter:
    @pytest.mark.asyncio
    async def test_search_molecule(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"molecules": [{"pref_name": "BELIMUMAB"}]})
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        mols = await adapter.search_molecule("belimumab")
        assert len(mols) == 1
        assert mols[0]["pref_name"] == "BELIMUMAB"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_molecule_by_synonym(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"molecules": [{"pref_name": "Adalimumab"}]})
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        mols = await adapter.search_molecule_by_synonym("humira")
        assert len(mols) == 1
        # Verify icontains param was used
        assert "molecule_synonym__icontains" in str(adapter._client.build_request(
            "GET", "/chembl/api/data/molecule.json",
            params={"molecule_synonyms__molecule_synonym__icontains": "humira", "limit": 10}
        ).url)
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_mechanisms(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "mechanisms": [
                    {"mechanism_of_action": "B-lymphocyte stimulator inhibitor"}
                ]
            })
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        mechs = await adapter.get_mechanisms("CHEMBL1201572")
        assert len(mechs) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_indications(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "drug_indications": [{"mesh_heading": "Lupus Erythematosus, Systemic"}]
            })
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        indications = await adapter.get_indications("CHEMBL1201572")
        assert len(indications) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_target(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"targets": [{"pref_name": "BAFF"}]})
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        targets = await adapter.search_target("BAFF")
        assert len(targets) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_activities(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({
                "activities": [{"standard_type": "IC50", "standard_value": "5.0"}]
            })
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        acts = await adapter.get_activities("CHEMBL25", activity_type="IC50")
        assert len(acts) == 1
        assert acts[0]["standard_type"] == "IC50"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_molecule_by_id(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"molecule_chembl_id": "CHEMBL25", "pref_name": "Aspirin"})
        )
        adapter = ChEMBLAdapter(config)
        adapter._client = httpx.AsyncClient(transport=transport, base_url="https://www.ebi.ac.uk")
        mol = await adapter.get_molecule("CHEMBL25")
        assert mol["pref_name"] == "Aspirin"
        await adapter._client.aclose()
