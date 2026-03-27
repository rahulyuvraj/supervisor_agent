"""Pathway section builder — populates the pathway enrichment findings section."""

from __future__ import annotations

from typing import List

import pandas as pd

from .base import SectionBuilder
from ...data_layer.schemas.sections import SectionBlock
from ...data_layer.schemas.evidence import Confidence, EvidenceCard


class PathwaySectionBuilder(SectionBuilder):
    section_id = "pathway_findings"

    def build(self, section: SectionBlock) -> SectionBlock:
        df = self._read_artifact("pathway_consolidation")
        if df is None:
            section.body = "*No pathway enrichment data available.*"
            return section

        parts: list[str] = []
        total = len(df)
        sig = self._count_significant(df)
        parts.append(
            f"Pathway enrichment analysis identified **{sig}** significant "
            f"pathways out of **{total}** tested."
        )

        # Top pathways table
        p_col = self._find_p_column(df)
        if p_col:
            sorted_df = df.sort_values(p_col, ascending=True, na_position="last")
        else:
            sorted_df = df

        section.tables.append(
            self._df_to_table(
                sorted_df, caption="Enriched Pathways (sorted by significance)",
                label="pathway_consolidation", max_rows=20,
            )
        )

        qc = self._get_qc_flags("pathway_consolidation")
        if qc:
            parts.append(
                "\n**Quality notes:** " + "; ".join(f.replace("_", " ") for f in qc) + "."
            )

        section.body = "\n\n".join(parts)
        return section

    def collect_evidence(self, section: SectionBlock) -> List[EvidenceCard]:
        cards: list[EvidenceCard] = []
        df = self._read_artifact("pathway_consolidation")
        if df is None:
            return cards

        sig = self._count_significant(df)
        cards.append(EvidenceCard(
            finding=f"{sig} significant pathways identified",
            module="pathway_enrichment",
            artifact_label="pathway_consolidation",
            metric_name="significant_pathway_count",
            metric_value=float(sig),
            confidence=Confidence.HIGH if sig > 0 else Confidence.LOW,
            section=self.section_id,
        ))
        return cards

    @staticmethod
    def _count_significant(df: pd.DataFrame, threshold: float = 0.05) -> int:
        for col in ("p_value", "pvalue", "p-value", "Adjusted_P-value", "P_value"):
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                return int((numeric < threshold).sum())
        return 0

    @staticmethod
    def _find_p_column(df: pd.DataFrame) -> str | None:
        candidates = ("p_value", "pvalue", "p-value", "P_value", "Adjusted_P-value")
        for c in candidates:
            if c in df.columns:
                return c
        return None
