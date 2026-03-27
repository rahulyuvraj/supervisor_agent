"""DEG section builder — populates the differential expression findings section."""

from __future__ import annotations

from typing import List

import pandas as pd

from .base import SectionBuilder
from ...data_layer.schemas.sections import SectionBlock
from ...data_layer.schemas.evidence import Confidence, EvidenceCard


class DEGSectionBuilder(SectionBuilder):
    section_id = "deg_findings"

    def build(self, section: SectionBlock) -> SectionBlock:
        df = self._read_artifact("deg_table")
        prio_df = self._read_artifact("gene_priorities")

        parts: list[str] = []

        if df is not None:
            sig = self._count_significant(df)
            total = len(df)
            parts.append(
                f"Differential expression analysis identified **{sig}** "
                f"significant genes out of **{total}** tested."
            )
            # Top upregulated/downregulated
            fc_col = self._find_fc_column(df)
            if fc_col:
                up = df[pd.to_numeric(df[fc_col], errors="coerce") > 0]
                down = df[pd.to_numeric(df[fc_col], errors="coerce") < 0]
                parts.append(
                    f"Of these, **{len(up)}** were upregulated and "
                    f"**{len(down)}** were downregulated."
                )
            section.tables.append(
                self._df_to_table(df, caption="Differentially Expressed Genes", label="deg_table")
            )

        if prio_df is not None:
            parts.append(
                f"\nGene prioritization ranked **{len(prio_df)}** candidates."
            )
            section.tables.append(
                self._df_to_table(
                    prio_df, caption="Prioritized Gene Candidates",
                    label="gene_priorities", max_rows=15,
                )
            )

        # QC flags
        qc = self._get_qc_flags("deg_table")
        if qc:
            parts.append(
                "\n**Quality notes:** " + "; ".join(f.replace("_", " ") for f in qc) + "."
            )

        section.body = "\n\n".join(parts) if parts else "*No DEG data available.*"
        return section

    def collect_evidence(self, section: SectionBlock) -> List[EvidenceCard]:
        cards: list[EvidenceCard] = []
        df = self._read_artifact("deg_table")
        if df is None:
            return cards

        sig = self._count_significant(df)
        cards.append(EvidenceCard(
            finding=f"{sig} differentially expressed genes identified",
            module="deg_analysis",
            artifact_label="deg_table",
            metric_name="significant_deg_count",
            metric_value=float(sig),
            confidence=Confidence.HIGH if sig > 0 else Confidence.LOW,
            section=self.section_id,
        ))
        return cards

    @staticmethod
    def _count_significant(df: pd.DataFrame, threshold: float = 0.05) -> int:
        for col in ("padj", "adj_pvalue", "fdr", "adjusted_pvalue", "Padj"):
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                return int((numeric < threshold).sum())
        return 0

    @staticmethod
    def _find_fc_column(df: pd.DataFrame) -> str | None:
        candidates = ("log2foldchange", "log2fc", "logfc", "log2FoldChange")
        for c in candidates:
            match = [col for col in df.columns if col.lower() == c.lower()]
            if match:
                return match[0]
        return None
