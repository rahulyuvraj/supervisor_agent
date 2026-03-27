"""Reactome REST API adapter.

Endpoints:
  ContentService  — pathway details, participants, hierarchy
  AnalysisService — gene-set enrichment (over-representation)

Docs: https://reactome.org/ContentService/ , https://reactome.org/AnalysisService/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter
from .config import APIConfig

logger = logging.getLogger(__name__)


class ReactomeAdapter(BaseAPIAdapter):
    service_name = "reactome"
    base_url = "https://reactome.org"

    # ── Pathway lookup ──

    async def get_pathway(self, pathway_id: str) -> Dict[str, Any]:
        """Fetch a pathway by stable identifier (e.g. R-HSA-1280215)."""
        return await self._request("GET", f"/ContentService/data/query/{pathway_id}")

    async def get_pathway_participants(self, pathway_id: str) -> List[Dict[str, Any]]:
        """List physical entities participating in a pathway."""
        return await self._request(
            "GET",
            f"/ContentService/data/participants/{pathway_id}",
        )

    async def get_pathway_hierarchy(self, species: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """Top-level pathway hierarchy for a species."""
        return await self._request(
            "GET",
            "/ContentService/data/pathways/top/{species}",
            params={"species": species},
        )

    # ── Enrichment analysis ──

    async def analyse_genes(
        self,
        gene_list: List[str],
        *,
        interactors: bool = False,
        page_size: int = 20,
        page: int = 1,
        p_value: float = 0.05,
        species: str = "Homo sapiens",
    ) -> Dict[str, Any]:
        """Over-representation analysis via AnalysisService (POST projection)."""
        payload = "\n".join(gene_list)
        return await self._request(
            "POST",
            "/AnalysisService/identifiers/projection",
            content=payload.encode(),
            headers={"Content-Type": "text/plain"},
            params={
                "interactors": str(interactors).lower(),
                "pageSize": page_size,
                "page": page,
                "pValue": p_value,
                "species": species,
            },
            skip_cache=True,
        )

    async def get_analysis_pathways(
        self,
        token: str,
        *,
        page_size: int = 20,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Retrieve paginated pathway results from a previous analysis token."""
        return await self._request(
            "GET",
            f"/AnalysisService/token/{token}",
            params={"pageSize": page_size, "page": page},
        )

    # ── Drug-related pathway filtering ──

    async def get_pathway_drugs(self, pathway_id: str) -> List[Dict[str, Any]]:
        """Fetch drug entities within a pathway (Drug/ChemicalDrug/ProteinDrug/RNADrug)."""
        participants = await self.get_pathway_participants(pathway_id)
        drug_types = {"Drug", "ChemicalDrug", "ProteinDrug", "RNADrug"}
        return [p for p in participants if p.get("schemaClass") in drug_types]

    # ── Hierarchy walk ──

    async def collect_child_pathways(
        self,
        pathway_id: str,
        *,
        max_depth: int = 3,
    ) -> List[str]:
        """Recursively collect child pathway IDs up to *max_depth*."""
        collected: List[str] = []
        await self._walk(pathway_id, collected, depth=0, max_depth=max_depth)
        return collected

    async def _walk(
        self,
        pid: str,
        acc: List[str],
        depth: int,
        max_depth: int,
    ) -> None:
        if depth >= max_depth:
            return
        data = await self._request(
            "GET",
            f"/ContentService/data/query/{pid}",
        )
        for child in data.get("hasEvent", []):
            child_id = child.get("stId", "")
            if child_id and child_id not in acc:
                acc.append(child_id)
                await self._walk(child_id, acc, depth + 1, max_depth)
