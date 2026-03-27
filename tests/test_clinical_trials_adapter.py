"""Tests for ClinicalTrialsAdapter — search, pagination, study lookup."""

import httpx
import pytest

from supervisor_agent.api_adapters.clinical_trials import ClinicalTrialsAdapter
from supervisor_agent.api_adapters.config import APIConfig


def _json_resp(data, status=200):
    return httpx.Response(status, json=data, headers={"content-type": "application/json"})


@pytest.fixture
def config():
    return APIConfig(max_retries=1)


def _studies_page(studies, next_token=None):
    resp = {"studies": studies, "totalCount": len(studies)}
    if next_token:
        resp["nextPageToken"] = next_token
    return resp


class TestClinicalTrialsAdapter:
    @pytest.mark.asyncio
    async def test_search_studies(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp(
                _studies_page([{"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}])
            )
        )
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        result = await adapter.search_studies("lupus")
        assert len(result["studies"]) == 1
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_study(self, config):
        study = {"protocolSection": {"identificationModule": {"nctId": "NCT12345"}}}
        transport = httpx.MockTransport(lambda req: _json_resp(study))
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        result = await adapter.get_study("NCT12345")
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345"
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_all_paginated(self, config):
        """Auto-pagination should follow nextPageToken until exhausted."""
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _json_resp(
                    _studies_page(
                        [{"nctId": f"NCT{i}"} for i in range(20)],
                        next_token="page2",
                    )
                )
            else:
                return _json_resp(
                    _studies_page([{"nctId": f"NCT{i+20}"} for i in range(10)])
                )

        transport = httpx.MockTransport(handler)
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        studies = await adapter.search_all("lupus", max_results=50)
        assert call_count == 2
        assert len(studies) == 30
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_search_all_respects_max(self, config):
        """max_results should cap the returned list."""
        def handler(req):
            return _json_resp(
                _studies_page(
                    [{"nctId": f"NCT{i}"} for i in range(20)],
                    next_token="more",
                )
            )

        transport = httpx.MockTransport(handler)
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        studies = await adapter.search_all("test", max_results=15)
        assert len(studies) == 15
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_get_study_sizes(self, config):
        transport = httpx.MockTransport(
            lambda req: _json_resp({"studies": [], "totalCount": 142})
        )
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        result = await adapter.get_study_sizes("lupus")
        assert result["totalCount"] == 142
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_filter_status(self, config):
        def handler(req):
            assert "filter.overallStatus" in str(req.url)
            return _json_resp(_studies_page([]))

        transport = httpx.MockTransport(handler)
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        await adapter.search_studies(
            "test", filter_overall_status=["RECRUITING", "COMPLETED"]
        )
        await adapter._client.aclose()

    @pytest.mark.asyncio
    async def test_fields_param(self, config):
        def handler(req):
            assert "fields=" in str(req.url)
            return _json_resp(_studies_page([]))

        transport = httpx.MockTransport(handler)
        adapter = ClinicalTrialsAdapter(config)
        adapter._client = httpx.AsyncClient(
            transport=transport, base_url="https://clinicaltrials.gov"
        )
        await adapter.search_studies("test", fields=["NCTId", "BriefTitle"])
        await adapter._client.aclose()
