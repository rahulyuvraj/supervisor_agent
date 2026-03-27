"""Drug discovery adapter — discovers perturbation and drug scoring artifacts."""

from __future__ import annotations

import logging
from typing import List

from .base import BaseModuleAdapter
from ..schemas.registry import ArtifactEntry

logger = logging.getLogger(__name__)

_FILE_PATTERNS = {
    "drug_gene_pairs": r"Final_GeneDrug_Pairs|gene.?drug.?pair|drug_gene",
    "drug_scores": r"drug_scor|ranked_drugs|drug_rank",
    "depmap_results": r"depmap|essentiality|dependency",
    "l1000_results": r"l1000|perturbation_score|connectivity",
    "mechanism_summary": r"mechanism|moa|action_type",
}

# Known quality issues in the drug_agent output (immutable code):
# - Open Targets scores can be 0.0 (drug_scorer.py:558)
# - Brand/generic dedup uses string matching (drug_ranker.py:132)
# - BLK action_type may be UNKNOWN (mechanism_fetcher.py:294)
_KNOWN_QC_ISSUES = {
    "drug_scores": ["ot_zero_scores_possible", "brand_generic_dedup_inexact"],
    "mechanism_summary": ["action_type_may_be_unknown"],
}

_EXPECTED_DRUG_COLS = {"drug", "gene", "score"}
_DRUG_ALIASES = {
    "compound": "drug",
    "drug_name": "drug",
    "gene_symbol": "gene",
    "target": "gene",
    "overall_score": "score",
    "drug_score": "score",
    "confidence": "score",
}


class DrugDiscoveryAdapter(BaseModuleAdapter):
    module_name = "perturbation_analysis"

    def discover(self) -> List[ArtifactEntry]:
        if not self.is_available():
            return []

        entries: List[ArtifactEntry] = []
        for csv_path in self._find_csvs():
            label = self._score_match(csv_path, _FILE_PATTERNS)
            if not label:
                continue

            cols, row_count = self._read_csv_meta(csv_path)
            qc = self._check_drug_quality(cols, label)
            # Append known QC issues for this label
            qc.extend(_KNOWN_QC_ISSUES.get(label, []))
            entries.append(self._make_entry(
                csv_path, label, columns=cols, row_count=row_count, qc_flags=qc,
            ))
        return entries

    @staticmethod
    def _check_drug_quality(columns: List[str], label: str) -> List[str]:
        if label not in ("drug_gene_pairs", "drug_scores"):
            return []
        col_lower = {c.lower() for c in columns}
        normalized = set()
        for c in col_lower:
            normalized.add(_DRUG_ALIASES.get(c, c))

        flags: List[str] = []
        if not normalized & {"drug", "compound", "drug_name"}:
            flags.append("missing_drug_identifier")
        if not normalized & {"score", "overall_score", "confidence"}:
            flags.append("missing_score_column")
        return flags
