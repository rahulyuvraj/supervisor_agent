"""STRING protein-protein interaction network API adapter.

Endpoint: POST https://string-db.org/api/tsv/
Returns tab-separated data; parsed into dicts.
Batch limit: ~200 identifiers per request, 4 req/s.

Docs: https://string-db.org/help/api/
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)

_BATCH_SIZE = 200
_DEFAULT_SPECIES = 9606  # Homo sapiens


class STRINGAdapter(BaseAPIAdapter):
    service_name = "string"
    base_url = "https://string-db.org"

    # ── ID resolution ──

    async def resolve_ids(
        self,
        identifiers: List[str],
        *,
        species: int = _DEFAULT_SPECIES,
        limit: int = 1,
    ) -> List[Dict[str, str]]:
        """Map gene symbols / accessions to STRING identifiers."""
        all_rows: List[Dict[str, str]] = []
        for i in range(0, len(identifiers), _BATCH_SIZE):
            batch = identifiers[i : i + _BATCH_SIZE]
            text = await self._request(
                "POST",
                "/api/tsv/get_string_ids",
                data={
                    "identifiers": "\r".join(batch),
                    "species": species,
                    "limit": limit,
                    "caller_identity": "supervisor_agent",
                },
                skip_cache=True,
            )
            all_rows.extend(self._parse_tsv(text))
        return all_rows

    # ── Network interactions ──

    async def get_network(
        self,
        identifiers: List[str],
        *,
        species: int = _DEFAULT_SPECIES,
        required_score: int = 400,
        network_type: str = "functional",
    ) -> List[Dict[str, str]]:
        """Interaction network edges for the given identifiers."""
        all_rows: List[Dict[str, str]] = []
        for i in range(0, len(identifiers), _BATCH_SIZE):
            batch = identifiers[i : i + _BATCH_SIZE]
            text = await self._request(
                "POST",
                "/api/tsv/network",
                data={
                    "identifiers": "\r".join(batch),
                    "species": species,
                    "required_score": required_score,
                    "network_type": network_type,
                    "caller_identity": "supervisor_agent",
                },
                skip_cache=True,
            )
            all_rows.extend(self._parse_tsv(text))
        return all_rows

    # ── Enrichment (functional) ──

    async def get_enrichment(
        self,
        identifiers: List[str],
        *,
        species: int = _DEFAULT_SPECIES,
    ) -> List[Dict[str, str]]:
        """Functional enrichment for a gene set."""
        text = await self._request(
            "POST",
            "/api/tsv/enrichment",
            data={
                "identifiers": "\r".join(identifiers[:_BATCH_SIZE]),
                "species": species,
                "caller_identity": "supervisor_agent",
            },
            skip_cache=True,
        )
        return self._parse_tsv(text)

    # ── Interaction partners ──

    async def get_interaction_partners(
        self,
        identifiers: List[str],
        *,
        species: int = _DEFAULT_SPECIES,
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        """Top interaction partners for each protein."""
        text = await self._request(
            "POST",
            "/api/tsv/interaction_partners",
            data={
                "identifiers": "\r".join(identifiers[:_BATCH_SIZE]),
                "species": species,
                "limit": limit,
                "caller_identity": "supervisor_agent",
            },
            skip_cache=True,
        )
        return self._parse_tsv(text)

    # ── TSV parser ──

    @staticmethod
    def _parse_tsv(text: str) -> List[Dict[str, str]]:
        if not text or not text.strip():
            return []
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        return list(reader)
