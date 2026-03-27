"""DEG module adapter — discovers differential expression output artifacts."""

from __future__ import annotations

import logging
from typing import List

from .base import BaseModuleAdapter
from ..schemas.registry import ArtifactEntry

logger = logging.getLogger(__name__)

# Structural patterns — these match file naming conventions from the DEG pipeline.
# Zero domain entities; only structural labels.
_FILE_PATTERNS = {
    "deg_table": r"_DEGs|deg_results|differential_expression",
    "gene_priorities": r"Final_Gene_Priorities|prioritized|gene_ranked",
    "volcano_plot": r"volcano",
    "ma_plot": r"ma_plot",
    "summary_stats": r"summary|statistics|stats",
}

# QC column expectations for a valid DEG table
_EXPECTED_COLUMNS = {"log2foldchange", "padj", "pvalue", "basemean"}
_EXPECTED_ALIASES = {
    "log2fc": "log2foldchange",
    "logfc": "log2foldchange",
    "p_value": "pvalue",
    "p-value": "pvalue",
    "adj_pvalue": "padj",
    "fdr": "padj",
    "adjusted_pvalue": "padj",
}


class DEGAdapter(BaseModuleAdapter):
    module_name = "deg_analysis"

    def discover(self) -> List[ArtifactEntry]:
        if not self.is_available():
            return []

        entries: List[ArtifactEntry] = []
        for csv_path in self._find_csvs():
            label = self._score_match(csv_path, _FILE_PATTERNS)
            if not label:
                continue

            cols, row_count = self._read_csv_meta(csv_path)
            qc = self._check_deg_quality(cols, label)
            entries.append(self._make_entry(
                csv_path, label, columns=cols, row_count=row_count, qc_flags=qc,
            ))

        # Also check for the gene prioritization output (often in same dir tree)
        prio_dir = self.run.output_files.get("prioritized_genes_path")
        if prio_dir:
            from pathlib import Path
            p = Path(prio_dir)
            if p.is_file() and p.suffix == ".csv":
                entries.append(self._make_entry(p, "gene_priorities"))

        return entries

    @staticmethod
    def _check_deg_quality(columns: List[str], label: str) -> List[str]:
        """Flag missing expected columns for DEG tables."""
        if label != "deg_table":
            return []
        col_lower = {c.lower() for c in columns}
        # Normalize aliases
        normalized = set()
        for c in col_lower:
            normalized.add(_EXPECTED_ALIASES.get(c, c))

        flags: List[str] = []
        if "padj" not in normalized and "pvalue" not in normalized:
            flags.append("missing_significance")
        if "log2foldchange" not in normalized:
            flags.append("missing_log2fc")
        return flags
