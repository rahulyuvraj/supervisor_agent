"""Pathway enrichment adapter — discovers pathway consolidation artifacts."""

from __future__ import annotations

import logging
from typing import List

from .base import BaseModuleAdapter
from ..schemas.registry import ArtifactEntry

logger = logging.getLogger(__name__)

_FILE_PATTERNS = {
    "pathway_consolidation": r"pathway_consolidation|Pathways_Consolidated",
    "enrichment_raw": r"enrichment|enrichr|gsea_results",
    "pathway_network": r"network|interaction",
    "pathway_summary": r"summary|overview",
}

_EXPECTED_COLUMNS = {"pathway", "p_value", "combined_score", "genes"}
_COLUMN_ALIASES = {
    "term": "pathway",
    "term_name": "pathway",
    "description": "pathway",
    "pvalue": "p_value",
    "p-value": "p_value",
    "adjusted_p_value": "p_value",
    "overlap": "genes",
}


class PathwayAdapter(BaseModuleAdapter):
    module_name = "pathway_enrichment"

    def discover(self) -> List[ArtifactEntry]:
        if not self.is_available():
            return []

        entries: List[ArtifactEntry] = []
        for csv_path in self._find_csvs():
            label = self._score_match(csv_path, _FILE_PATTERNS)
            if not label:
                continue

            cols, row_count = self._read_csv_meta(csv_path)
            qc = self._check_pathway_quality(cols, label)
            entries.append(self._make_entry(
                csv_path, label, columns=cols, row_count=row_count, qc_flags=qc,
            ))
        return entries

    @staticmethod
    def _check_pathway_quality(columns: List[str], label: str) -> List[str]:
        if label != "pathway_consolidation":
            return []
        col_lower = {c.lower() for c in columns}
        normalized = set()
        for c in col_lower:
            normalized.add(_COLUMN_ALIASES.get(c, c))

        flags: List[str] = []
        if not normalized & {"pathway", "term", "term_name", "description"}:
            flags.append("missing_pathway_name")
        if not normalized & {"p_value", "pvalue", "p-value"}:
            flags.append("missing_p_value")
        return flags
