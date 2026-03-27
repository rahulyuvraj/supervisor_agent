"""Drug discovery section builder — populates drug/perturbation findings."""

from __future__ import annotations

from typing import List

import pandas as pd

from .base import SectionBuilder
from ...data_layer.schemas.sections import SectionBlock
from ...data_layer.schemas.evidence import Confidence, EvidenceCard


class DrugSectionBuilder(SectionBuilder):
    section_id = "drug_findings"

    def build(self, section: SectionBlock) -> SectionBlock:
        pairs_df = self._read_artifact("drug_gene_pairs")
        scores_df = self._read_artifact("drug_scores")
        depmap_df = self._read_artifact("depmap_results")
        l1000_df = self._read_artifact("l1000_results")

        parts: list[str] = []

        if pairs_df is not None:
            parts.append(
                f"Drug-gene interaction analysis identified "
                f"**{len(pairs_df)}** candidate drug-gene pairs."
            )
            section.tables.append(
                self._df_to_table(
                    pairs_df, caption="Drug-Gene Pairs",
                    label="drug_gene_pairs", max_rows=15,
                )
            )

        if scores_df is not None:
            score_col = self._find_score_column(scores_df)
            if score_col:
                sorted_df = scores_df.sort_values(score_col, ascending=False, na_position="last")
            else:
                sorted_df = scores_df
            section.tables.append(
                self._df_to_table(
                    sorted_df, caption="Ranked Drug Candidates",
                    label="drug_scores", max_rows=10,
                )
            )

        if depmap_df is not None:
            parts.append(f"\nDepMap essentiality data covers **{len(depmap_df)}** entries.")

        if l1000_df is not None:
            parts.append(f"\nL1000 perturbation signatures: **{len(l1000_df)}** entries.")

        # QC warnings for known drug_agent issues
        all_qc: list[str] = []
        for lbl in ("drug_gene_pairs", "drug_scores", "mechanism_summary"):
            all_qc.extend(self._get_qc_flags(lbl))
        if all_qc:
            unique_qc = list(dict.fromkeys(all_qc))
            parts.append(
                "\n**Quality notes:** " + "; ".join(f.replace("_", " ") for f in unique_qc)
                + ". These are known limitations of the upstream drug scoring module."
            )

        section.body = "\n\n".join(parts) if parts else "*No drug discovery data available.*"
        return section

    def collect_evidence(self, section: SectionBlock) -> List[EvidenceCard]:
        cards: list[EvidenceCard] = []
        pairs_df = self._read_artifact("drug_gene_pairs")
        if pairs_df is not None:
            qc = self._get_qc_flags("drug_gene_pairs")
            confidence = Confidence.FLAGGED if qc else Confidence.MEDIUM
            cards.append(EvidenceCard(
                finding=f"{len(pairs_df)} drug-gene pairs identified",
                module="perturbation_analysis",
                artifact_label="drug_gene_pairs",
                metric_name="drug_gene_pair_count",
                metric_value=float(len(pairs_df)),
                confidence=confidence,
                section=self.section_id,
            ))
        return cards

    @staticmethod
    def _find_score_column(df: pd.DataFrame) -> str | None:
        candidates = ("score", "overall_score", "drug_score", "confidence", "rank")
        for c in candidates:
            match = [col for col in df.columns if col.lower() == c.lower()]
            if match:
                return match[0]
        return None
