"""ClinicalTrials.gov API v2 adapter.

Base: https://clinicaltrials.gov/api/v2/
Pagination via ``pageToken``. Rate limit: ~50 req/min.

Docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)


class ClinicalTrialsAdapter(BaseAPIAdapter):
    service_name = "clinical_trials"
    base_url = "https://clinicaltrials.gov"

    # ── Study search ──

    async def search_studies(
        self,
        query: str,
        *,
        fields: Optional[List[str]] = None,
        page_size: int = 20,
        page_token: Optional[str] = None,
        filter_overall_status: Optional[List[str]] = None,
        sort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search clinical trials.

        ``query`` is free-text (condition, intervention, etc.).
        ``fields`` selects which data columns to return (reduces payload).
        ``filter_overall_status`` can include RECRUITING, COMPLETED, etc.
        """
        params: Dict[str, Any] = {
            "query.term": query,
            "pageSize": page_size,
        }
        if fields:
            params["fields"] = ",".join(fields)
        if page_token:
            params["pageToken"] = page_token
        if filter_overall_status:
            params["filter.overallStatus"] = ",".join(filter_overall_status)
        if sort:
            params["sort"] = sort
        return await self._request("GET", "/api/v2/studies", params=params)

    # ── Single study ──

    async def get_study(
        self,
        nct_id: str,
        *,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch a single study by NCT ID."""
        params: Dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        return await self._request(
            "GET", f"/api/v2/studies/{nct_id}", params=params
        )

    # ── Auto-paginated search ──

    async def search_all(
        self,
        query: str,
        *,
        fields: Optional[List[str]] = None,
        max_results: int = 100,
        filter_overall_status: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Paginate through all matching studies up to *max_results*."""
        collected: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        page_size = min(max_results, 20)

        while len(collected) < max_results:
            data = await self.search_studies(
                query,
                fields=fields,
                page_size=page_size,
                page_token=page_token,
                filter_overall_status=filter_overall_status,
            )
            studies = data.get("studies", [])
            if not studies:
                break
            collected.extend(studies)
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return collected[:max_results]

    # ── Study statistics ──

    async def get_study_sizes(
        self,
        query: str,
        *,
        filter_overall_status: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Total count of matching studies (without fetching full records)."""
        data = await self.search_studies(
            query,
            fields=["NCTId"],
            page_size=1,
            filter_overall_status=filter_overall_status,
        )
        return {"totalCount": data.get("totalCount", 0)}
