"""Ensembl REST API adapter.

Base: https://rest.ensembl.org
Rate limit: 15 req/s, max 10 concurrent.

Docs: https://rest.ensembl.org/documentation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)


class EnsemblAdapter(BaseAPIAdapter):
    service_name = "ensembl"
    base_url = "https://rest.ensembl.org"

    # ── Gene lookup ──

    async def lookup_symbol(
        self,
        symbol: str,
        species: str = "homo_sapiens",
        *,
        expand: bool = False,
    ) -> Dict[str, Any]:
        """Look up a gene by symbol."""
        params: Dict[str, Any] = {}
        if expand:
            params["expand"] = 1
        return await self._request(
            "GET",
            f"/lookup/symbol/{species}/{symbol}",
            params=params,
            headers={"Content-Type": "application/json"},
        )

    async def lookup_id(
        self,
        ensembl_id: str,
        *,
        expand: bool = False,
    ) -> Dict[str, Any]:
        """Look up any Ensembl stable ID (gene, transcript, etc.)."""
        params: Dict[str, Any] = {}
        if expand:
            params["expand"] = 1
        return await self._request(
            "GET",
            f"/lookup/id/{ensembl_id}",
            params=params,
            headers={"Content-Type": "application/json"},
        )

    # ── Sequence retrieval ──

    async def get_sequence(
        self,
        ensembl_id: str,
        *,
        seq_type: str = "genomic",
    ) -> Dict[str, Any]:
        """Retrieve sequence for a stable ID."""
        return await self._request(
            "GET",
            f"/sequence/id/{ensembl_id}",
            params={"type": seq_type},
            headers={"Content-Type": "application/json"},
        )

    # ── Variant Effect Predictor (VEP) ──

    async def vep_hgvs(self, hgvs_notation: str) -> List[Dict[str, Any]]:
        """Predict variant consequences from HGVS notation."""
        return await self._request(
            "GET",
            f"/vep/human/hgvs/{hgvs_notation}",
            headers={"Content-Type": "application/json"},
        )

    async def vep_region(
        self,
        region: str,
        allele: str,
    ) -> List[Dict[str, Any]]:
        """VEP by genomic region (e.g. ``9:22125503-22125503:1`` allele ``C``)."""
        return await self._request(
            "GET",
            f"/vep/human/region/{region}/{allele}",
            headers={"Content-Type": "application/json"},
        )

    # ── Cross-references / xrefs ──

    async def xrefs_symbol(
        self,
        symbol: str,
        species: str = "homo_sapiens",
    ) -> List[Dict[str, Any]]:
        """External DB cross-references for a gene symbol."""
        return await self._request(
            "GET",
            f"/xrefs/symbol/{species}/{symbol}",
            headers={"Content-Type": "application/json"},
        )

    # ── Mapping / coordinates ──

    async def map_coordinates(
        self,
        ensembl_id: str,
        start: int,
        end: int,
        *,
        coord_system: str = "GRCh38",
    ) -> Dict[str, Any]:
        """Map coordinates between assemblies."""
        return await self._request(
            "GET",
            f"/map/human/{coord_system}/{ensembl_id}:{start}..{end}/GRCh37",
            headers={"Content-Type": "application/json"},
        )
