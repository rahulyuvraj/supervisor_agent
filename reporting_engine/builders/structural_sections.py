"""Structural section builders — module-agnostic report scaffolding.

These builders handle sections that synthesize across all modules
(executive summary, run summary, limitations, etc.) rather than
reading from a single module's output.
"""

from __future__ import annotations

from typing import List

from .base import SectionBuilder
from ...data_layer.schemas.manifest import ModuleStatus
from ...data_layer.schemas.sections import SectionBlock
from ...data_layer.schemas.evidence import EvidenceCard


class ExecutiveSummaryBuilder(SectionBuilder):
    section_id = "executive_summary"

    def build(self, section: SectionBlock) -> SectionBlock:
        completed = self.manifest.completed_modules()
        failed = [m for m in self.manifest.modules if m.status == ModuleStatus.FAILED]
        disease = self.manifest.disease_name

        parts: list[str] = []
        if disease:
            parts.append(f"This report presents the results of a multi-module "
                         f"bioinformatics analysis for **{disease}**.")
        else:
            parts.append("This report presents the results of a multi-module "
                         "bioinformatics analysis pipeline run.")

        parts.append(
            f"**{len(completed)}** analysis modules completed successfully"
            + (f", **{len(failed)}** failed" if failed else "")
            + "."
        )

        # Brief per-module one-liners
        for m in completed:
            artifacts = self.index.for_module(m.module_name)
            parts.append(f"- **{m.module_name.replace('_', ' ').title()}**: "
                         f"{len(artifacts)} output artifact(s)")

        section.body = "\n\n".join(parts)
        return section


class RunSummaryBuilder(SectionBuilder):
    section_id = "run_summary"

    def build(self, section: SectionBlock) -> SectionBlock:
        parts: list[str] = []
        parts.append("| Module | Status | Duration | Artifacts |")
        parts.append("|--------|--------|----------|-----------|")
        for m in self.manifest.modules:
            dur = f"{m.duration_s:.1f}s" if m.duration_s else "—"
            n_art = len(self.index.for_module(m.module_name))
            status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(
                m.status.value, "—"
            )
            parts.append(f"| {m.module_name.replace('_', ' ').title()} "
                         f"| {status_icon} {m.status.value} | {dur} | {n_art} |")

        if self.manifest.session_id:
            parts.append(f"\n**Session:** `{self.manifest.session_id}`")
        if self.manifest.analysis_id:
            parts.append(f"**Analysis ID:** `{self.manifest.analysis_id}`")

        section.body = "\n".join(parts)
        return section


class DataOverviewBuilder(SectionBuilder):
    section_id = "data_overview"

    def build(self, section: SectionBlock) -> SectionBlock:
        parts: list[str] = []
        if self.manifest.disease_name:
            parts.append(f"**Disease context:** {self.manifest.disease_name}")

        total_artifacts = len(self.index.artifacts)
        total_rows = sum(a.row_count for a in self.index.artifacts if a.row_count)
        parts.append(f"**Total artifacts discovered:** {total_artifacts}")
        if total_rows:
            parts.append(f"**Total data rows across all CSVs:** {total_rows:,}")

        # QC flags summary
        all_flags: list[str] = []
        for a in self.index.artifacts:
            all_flags.extend(a.qc_flags)
        if all_flags:
            unique = list(dict.fromkeys(all_flags))
            parts.append(
                f"\n**QC flags detected ({len(unique)}):** "
                + ", ".join(f.replace("_", " ") for f in unique)
            )

        section.body = "\n\n".join(parts)
        return section


class IntegratedFindingsBuilder(SectionBuilder):
    section_id = "integrated_findings"

    def build(self, section: SectionBlock) -> SectionBlock:
        completed_names = {m.module_name for m in self.manifest.completed_modules()}
        if len(completed_names) < 2:
            section.body = ("*Integrated findings require at least two completed "
                            "analysis modules.*")
            return section

        parts: list[str] = [
            "The following modules contributed to this integrated analysis: "
            + ", ".join(f"**{n.replace('_', ' ').title()}**" for n in sorted(completed_names))
            + "."
        ]
        # Cross-module gene overlap detection (if both DEG and drug artifacts exist)
        deg_df = self._read_artifact("gene_priorities")
        drug_df = self._read_artifact("drug_gene_pairs")
        if deg_df is not None and drug_df is not None:
            deg_genes = self._extract_gene_set(deg_df)
            drug_genes = self._extract_gene_set(drug_df)
            if deg_genes and drug_genes:
                overlap = deg_genes & drug_genes
                if overlap:
                    parts.append(
                        f"\n**{len(overlap)}** genes appear in both the prioritized "
                        f"gene list and drug-gene interaction results, suggesting "
                        f"converging evidence for therapeutic relevance."
                    )

        section.body = "\n\n".join(parts)
        return section

    @staticmethod
    def _extract_gene_set(df) -> set:
        for col in ("Gene", "gene", "gene_symbol", "Gene_Symbol", "target"):
            if col in df.columns:
                return set(df[col].dropna().astype(str).str.upper())
        return set()


class LimitationsBuilder(SectionBuilder):
    section_id = "limitations"

    def build(self, section: SectionBlock) -> SectionBlock:
        parts: list[str] = []
        failed = [m for m in self.manifest.modules if m.status == ModuleStatus.FAILED]
        if failed:
            parts.append("**Module failures:**")
            for m in failed:
                parts.append(f"- {m.module_name.replace('_', ' ').title()}: "
                             f"{m.error_message or 'Unknown error'}")

        # QC-flagged artifacts
        flagged = [(a.label, a.qc_flags) for a in self.index.artifacts if a.qc_flags]
        if flagged:
            parts.append("\n**Data quality warnings:**")
            for label, flags in flagged:
                parts.append(f"- **{label}**: " + "; ".join(f.replace("_", " ") for f in flags))

        if not parts:
            parts.append("No significant limitations were identified in this analysis run.")

        section.body = "\n\n".join(parts)
        return section


class NextStepsBuilder(SectionBuilder):
    section_id = "next_steps"

    def build(self, section: SectionBlock) -> SectionBlock:
        completed = {m.module_name for m in self.manifest.completed_modules()}
        suggestions: list[str] = []

        if "deg_analysis" in completed and "pathway_enrichment" not in completed:
            suggestions.append("Run pathway enrichment on the prioritized gene list.")
        if "pathway_enrichment" in completed and "perturbation_analysis" not in completed:
            suggestions.append("Run drug perturbation analysis to identify therapeutic candidates.")
        if not completed:
            suggestions.append("No modules completed — verify input data and re-run.")

        failed = [m for m in self.manifest.modules if m.status == ModuleStatus.FAILED]
        for m in failed:
            suggestions.append(f"Investigate and re-run {m.module_name.replace('_', ' ')}.")

        if not suggestions:
            suggestions.append("All primary analyses complete. Consider validation experiments.")

        section.body = "\n".join(f"1. {s}" for s in suggestions)
        return section


class AppendixBuilder(SectionBuilder):
    section_id = "appendix"

    def build(self, section: SectionBlock) -> SectionBlock:
        parts: list[str] = ["### Artifact Inventory"]
        parts.append("| Label | Module | File Type | Rows | QC Flags |")
        parts.append("|-------|--------|-----------|------|----------|")
        for a in self.index.artifacts:
            flags = ", ".join(a.qc_flags) if a.qc_flags else "—"
            parts.append(
                f"| {a.label} | {a.module} | {a.file_type} "
                f"| {a.row_count or '—'} | {flags} |"
            )
        section.body = "\n".join(parts)
        return section
