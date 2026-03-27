"""OpenFDA Drug API adapter.

Endpoints: label, ndc, event (adverse-event reporting).
Optional API key raises rate limits from 40/min → 240/min.

Docs: https://open.fda.gov/apis/drug/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter
from .config import APIConfig

logger = logging.getLogger(__name__)


class OpenFDAAdapter(BaseAPIAdapter):
    service_name = "openfda"
    base_url = "https://api.fda.gov"

    def __init__(self, config: APIConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._api_key = config.openfda_api_key

    def _with_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    # ── Drug labels (SPL / package inserts) ──

    async def search_labels(
        self,
        query: str,
        *,
        limit: int = 10,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """Search drug label documents (SPL).

        ``query`` follows openFDA search syntax, e.g.
        ``openfda.brand_name:"HUMIRA"`` or ``openfda.generic_name:"adalimumab"``.
        """
        return await self._request(
            "GET",
            "/drug/label.json",
            params=self._with_key({"search": query, "limit": limit, "skip": skip}),
        )

    async def get_label_by_spl(self, spl_set_id: str) -> Dict[str, Any]:
        """Retrieve a specific label by SPL set ID."""
        return await self.search_labels(f'openfda.spl_set_id:"{spl_set_id}"', limit=1)

    # ── NDC (National Drug Code) ──

    async def search_ndc(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search NDC directory."""
        return await self._request(
            "GET",
            "/drug/ndc.json",
            params=self._with_key({"search": query, "limit": limit}),
        )

    # ── Adverse events (FAERS) ──

    async def search_adverse_events(
        self,
        query: str,
        *,
        limit: int = 10,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """Search FAERS adverse-event reports.

        ``query`` example: ``patient.drug.openfda.brand_name:"HUMIRA"``
        """
        return await self._request(
            "GET",
            "/drug/event.json",
            params=self._with_key({"search": query, "limit": limit, "skip": skip}),
        )

    async def count_adverse_events(
        self,
        query: str,
        count_field: str = "patient.reaction.reactionmeddrapt.exact",
        *,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Aggregate adverse-event counts by a facet field."""
        return await self._request(
            "GET",
            "/drug/event.json",
            params=self._with_key({
                "search": query,
                "count": count_field,
                "limit": limit,
            }),
        )
