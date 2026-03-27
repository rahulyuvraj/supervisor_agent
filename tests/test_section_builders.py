"""Tests for section builders (DEG, Pathway, Drug, and structural)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ..data_layer.schemas.manifest import ModuleRun, ModuleStatus, RunManifest
from ..data_layer.schemas.registry import ArtifactEntry, ArtifactIndex
from ..data_layer.schemas.sections import ReportingConfig, SectionBlock, SectionMeta
from ..data_layer.schemas.evidence import Confidence
from ..reporting_engine.builders.base import BUILDER_REGISTRY
from ..reporting_engine.builders.deg_section import DEGSectionBuilder
from ..reporting_engine.builders.pathway_section import PathwaySectionBuilder
from ..reporting_engine.builders.drug_section import DrugSectionBuilder
from ..reporting_engine.builders.structural_sections import (
    ExecutiveSummaryBuilder, RunSummaryBuilder, DataOverviewBuilder,
    IntegratedFindingsBuilder, LimitationsBuilder, NextStepsBuilder, AppendixBuilder,
)


# ─── Builder Registry ───


class TestBuilderRegistry:
    def test_all_builders_registered(self):
        expected = {
            "deg_findings", "pathway_findings", "drug_findings",
            "executive_summary", "run_summary", "data_overview",
            "integrated_findings", "limitations", "next_steps", "appendix",
        }
        assert expected == set(BUILDER_REGISTRY.keys())


# ─── Helpers ───


def _make_builder(cls, manifest, index, config=None):
    return cls(manifest, index, config or ReportingConfig())


def _make_deg_index_from_df(deg_df: pd.DataFrame, prio_df: pd.DataFrame = None) -> ArtifactIndex:
    """Create an ArtifactIndex backed by a mock _read_artifact."""
    entries = [
        ArtifactEntry(
            path="/fake/deg.csv", module="deg_analysis", label="deg_table",
            columns=list(deg_df.columns), row_count=len(deg_df),
        ),
    ]
    if prio_df is not None:
        entries.append(ArtifactEntry(
            path="/fake/prio.csv", module="deg_analysis", label="gene_priorities",
            columns=list(prio_df.columns), row_count=len(prio_df),
        ))
    return ArtifactIndex(artifacts=entries)


# ─── DEGSectionBuilder ───


class TestDEGSectionBuilder:
    @pytest.fixture()
    def deg_df(self):
        return pd.DataFrame({
            "gene": ["A", "B", "C"],
            "log2FoldChange": [2.5, -1.8, 0.3],
            "pvalue": [0.001, 0.0001, 0.4],
            "padj": [0.01, 0.002, 0.8],
        })

    @pytest.fixture()
    def prio_df(self):
        return pd.DataFrame({"Gene": ["A", "B"], "Score": [0.95, 0.87]})

    def test_build_populates_body(self, deg_df, prio_df, full_manifest):
        index = _make_deg_index_from_df(deg_df, prio_df)
        builder = _make_builder(DEGSectionBuilder, full_manifest, index)
        section = SectionBlock(id="deg_findings", title="DEG")
        with patch.object(builder, "_read_artifact", side_effect=lambda l: {
            "deg_table": deg_df, "gene_priorities": prio_df,
        }.get(l)):
            section = builder.build(section)
        assert "significant" in section.body.lower()
        assert len(section.tables) >= 1

    def test_count_significant(self, deg_df):
        # padj < 0.05 → rows A and B
        count = DEGSectionBuilder._count_significant(deg_df)
        assert count == 2

    def test_count_significant_no_padj(self):
        df = pd.DataFrame({"gene": ["A"], "score": [1.0]})
        assert DEGSectionBuilder._count_significant(df) == 0

    def test_evidence_cards(self, deg_df, full_manifest):
        index = _make_deg_index_from_df(deg_df)
        builder = _make_builder(DEGSectionBuilder, full_manifest, index)
        section = SectionBlock(id="deg_findings", title="DEG")
        with patch.object(builder, "_read_artifact", return_value=deg_df):
            cards = builder.collect_evidence(section)
        assert len(cards) == 1
        assert cards[0].confidence == Confidence.HIGH  # 2 significant > 0

    def test_no_data_fallback(self, full_manifest, empty_artifact_index):
        builder = _make_builder(DEGSectionBuilder, full_manifest, empty_artifact_index)
        section = SectionBlock(id="deg_findings", title="DEG")
        section = builder.build(section)
        assert "no deg data" in section.body.lower()


# ─── PathwaySectionBuilder ───


class TestPathwaySectionBuilder:
    @pytest.fixture()
    def pw_df(self):
        return pd.DataFrame({
            "pathway": ["PW1", "PW2", "PW3"],
            "p_value": [0.001, 0.04, 0.12],
            "combined_score": [15, 8, 3],
        })

    def test_build_populates_body(self, pw_df, full_manifest, sample_artifact_index):
        builder = _make_builder(PathwaySectionBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="pathway_findings", title="Pathway")
        with patch.object(builder, "_read_artifact", return_value=pw_df):
            section = builder.build(section)
        assert "significant" in section.body.lower()
        assert len(section.tables) == 1

    def test_count_significant(self, pw_df):
        assert PathwaySectionBuilder._count_significant(pw_df) == 2

    def test_no_data_fallback(self, full_manifest, empty_artifact_index):
        builder = _make_builder(PathwaySectionBuilder, full_manifest, empty_artifact_index)
        section = SectionBlock(id="pathway_findings", title="Pathway")
        section = builder.build(section)
        assert "no pathway" in section.body.lower()


# ─── DrugSectionBuilder ───


class TestDrugSectionBuilder:
    @pytest.fixture()
    def pairs_df(self):
        return pd.DataFrame({
            "drug": ["D1", "D2"], "gene": ["G1", "G2"], "score": [0.9, 0.5],
        })

    def test_build_populates_body(self, pairs_df, full_manifest, sample_artifact_index):
        builder = _make_builder(DrugSectionBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="drug_findings", title="Drug")
        with patch.object(builder, "_read_artifact", side_effect=lambda l: {
            "drug_gene_pairs": pairs_df,
        }.get(l)):
            section = builder.build(section)
        assert "drug-gene" in section.body.lower()
        assert len(section.tables) >= 1

    def test_evidence_cards_flagged_when_qc(self, pairs_df, full_manifest, sample_artifact_index):
        builder = _make_builder(DrugSectionBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="drug_findings", title="Drug")
        with patch.object(builder, "_read_artifact", side_effect=lambda l: {
            "drug_gene_pairs": pairs_df,
        }.get(l)):
            cards = builder.collect_evidence(section)
        # drug_gene_pairs has qc_flags in sample_artifact_index
        assert any(c.confidence == Confidence.FLAGGED for c in cards)

    def test_no_data_fallback(self, full_manifest, empty_artifact_index):
        builder = _make_builder(DrugSectionBuilder, full_manifest, empty_artifact_index)
        section = SectionBlock(id="drug_findings", title="Drug")
        section = builder.build(section)
        assert "no drug" in section.body.lower()


# ─── Structural Builders ───


class TestStructuralBuilders:
    def test_executive_summary(self, full_manifest, sample_artifact_index):
        builder = _make_builder(ExecutiveSummaryBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="executive_summary", title="Exec")
        section = builder.build(section)
        assert "3" in section.body  # 3 completed modules
        assert "test_condition" in section.body.lower() or "test condition" in section.body.lower()

    def test_run_summary_table(self, full_manifest, sample_artifact_index):
        builder = _make_builder(RunSummaryBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="run_summary", title="Run")
        section = builder.build(section)
        assert "Module" in section.body
        assert "✅" in section.body

    def test_data_overview(self, full_manifest, sample_artifact_index):
        builder = _make_builder(DataOverviewBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="data_overview", title="Data")
        section = builder.build(section)
        assert "4" in section.body  # 4 artifacts in sample_artifact_index

    def test_limitations_with_failures(self, partial_manifest, empty_artifact_index):
        builder = _make_builder(LimitationsBuilder, partial_manifest, empty_artifact_index)
        section = SectionBlock(id="limitations", title="Limitations")
        section = builder.build(section)
        assert "timeout" in section.body.lower()

    def test_limitations_no_issues(self, full_manifest, empty_artifact_index):
        builder = _make_builder(LimitationsBuilder, full_manifest, empty_artifact_index)
        section = SectionBlock(id="limitations", title="Limitations")
        section = builder.build(section)
        assert "no significant" in section.body.lower()

    def test_next_steps_suggests_missing_pipeline(self, partial_manifest, empty_artifact_index):
        builder = _make_builder(NextStepsBuilder, partial_manifest, empty_artifact_index)
        section = SectionBlock(id="next_steps", title="Next")
        section = builder.build(section)
        # Pathway failed, drug not run → should suggest both
        assert "pathway" in section.body.lower() or "re-run" in section.body.lower()

    def test_appendix_inventory(self, full_manifest, sample_artifact_index):
        builder = _make_builder(AppendixBuilder, full_manifest, sample_artifact_index)
        section = SectionBlock(id="appendix", title="Appendix")
        section = builder.build(section)
        assert "deg_table" in section.body
        assert "pathway_consolidation" in section.body

    def test_integrated_findings_needs_two_modules(self, partial_manifest, empty_artifact_index):
        builder = _make_builder(IntegratedFindingsBuilder, partial_manifest, empty_artifact_index)
        section = SectionBlock(id="integrated_findings", title="Integrated")
        section = builder.build(section)
        assert "at least two" in section.body.lower()


# ─── Table capping ───


class TestTableCapping:
    def test_df_to_table_respects_cap(self, full_manifest, sample_artifact_index):
        builder = _make_builder(DEGSectionBuilder, full_manifest, sample_artifact_index)
        big_df = pd.DataFrame({"a": range(100), "b": range(100)})
        table = builder._df_to_table(big_df, max_rows=10)
        assert len(table.rows) == 10
        assert table.truncated is True
        assert table.total_rows == 100

    def test_df_to_table_not_truncated(self, full_manifest, sample_artifact_index):
        builder = _make_builder(DEGSectionBuilder, full_manifest, sample_artifact_index)
        small_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table = builder._df_to_table(small_df)
        assert len(table.rows) == 2
        assert table.truncated is False
