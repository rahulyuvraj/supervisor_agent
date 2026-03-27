"""KEGG REST API adapter — gated behind ``kegg_enabled`` config flag.

KEGG REST is free for academic use only.  All methods raise
``AdapterDisabledError`` when the flag is off.

Docs: https://rest.kegg.jp/
Response format: tab/flat-text (not JSON).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .base import AdapterDisabledError, BaseAPIAdapter
from .config import APIConfig

logger = logging.getLogger(__name__)


class KEGGAdapter(BaseAPIAdapter):
    service_name = "kegg"
    base_url = "https://rest.kegg.jp"

    def __init__(self, config: APIConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        if not config.kegg_enabled:
            self._disabled = True
        else:
            self._disabled = False

    def _gate(self) -> None:
        if self._disabled:
            raise AdapterDisabledError(
                "KEGG adapter is disabled. Set KEGG_ENABLED=true to use."
            )

    # ── Pathway lookups ──

    async def get_pathway(self, pathway_id: str) -> Dict[str, str]:
        """Fetch full pathway entry as parsed key-value dict.

        ``pathway_id`` should be like ``hsa04010`` (no ``path:`` prefix needed).
        """
        self._gate()
        raw = await self._request("GET", f"/get/{pathway_id}")
        return self._parse_flat(raw)

    async def list_pathways(self, organism: str = "hsa") -> List[Dict[str, str]]:
        """List all pathways for an organism."""
        self._gate()
        raw = await self._request("GET", f"/list/pathway/{organism}")
        return self._parse_tsv(raw, columns=["pathway_id", "name"])

    # ── Gene ↔ pathway mapping ──

    async def find_pathways_for_gene(self, gene_id: str) -> List[Dict[str, str]]:
        """Pathways containing a gene (e.g. ``hsa:7157``)."""
        self._gate()
        raw = await self._request("GET", f"/link/pathway/{gene_id}")
        return self._parse_tsv(raw, columns=["gene", "pathway_id"])

    async def find_genes_in_pathway(self, pathway_id: str) -> List[Dict[str, str]]:
        """Genes in a pathway."""
        self._gate()
        raw = await self._request("GET", f"/link/genes/{pathway_id}")
        return self._parse_tsv(raw, columns=["pathway_id", "gene"])

    # ── Drug lookups ──

    async def get_drug(self, drug_id: str) -> Dict[str, str]:
        """Drug compound details (e.g. ``D00596``)."""
        self._gate()
        raw = await self._request("GET", f"/get/{drug_id}")
        return self._parse_flat(raw)

    async def find_drug_interactions(self, drug_id: str) -> List[Dict[str, str]]:
        """DDI records for a drug."""
        self._gate()
        raw = await self._request("GET", f"/ddi/{drug_id}")
        return self._parse_tsv(raw, columns=["drug1", "drug2", "interaction"])

    # ── Converters / search ──

    async def conv_ids(self, target_db: str, source_db: str) -> List[Dict[str, str]]:
        """ID conversion between databases (e.g. genes ↔ uniprot)."""
        self._gate()
        raw = await self._request("GET", f"/conv/{target_db}/{source_db}")
        return self._parse_tsv(raw, columns=["source", "target"])

    async def find(self, database: str, query: str) -> List[Dict[str, str]]:
        """Keyword search across a KEGG database."""
        self._gate()
        raw = await self._request("GET", f"/find/{database}/{query}")
        return self._parse_tsv(raw, columns=["entry_id", "description"])

    # ── Response parsers (KEGG returns plain text) ──

    @staticmethod
    def _parse_tsv(text: str, columns: List[str]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for line in text.strip().splitlines():
            parts = line.split("\t")
            row = {columns[i]: parts[i] if i < len(parts) else "" for i in range(len(columns))}
            rows.append(row)
        return rows

    @staticmethod
    def _parse_flat(text: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        current_key = ""
        for line in text.splitlines():
            if line.startswith(" ") and current_key:
                result[current_key] += " " + line.strip()
            elif line.startswith("///"):
                break
            else:
                match = re.match(r"^(\S+)\s+(.*)", line)
                if match:
                    current_key = match.group(1)
                    result[current_key] = match.group(2).strip()
        return result
