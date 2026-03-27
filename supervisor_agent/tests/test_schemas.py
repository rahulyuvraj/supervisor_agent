"""Tests for data_layer Pydantic schemas."""

from __future__ import annotations

import pytest

from ..data_layer.schemas.manifest import ModuleRun, ModuleStatus, RunManifest
from ..data_layer.schemas.registry import ArtifactEntry, ArtifactIndex
from ..data_layer.schemas.evidence import Confidence, EvidenceCard, ConflictRecord, NarrativeContext
from ..data_layer.schemas.sections import (
    NarrativeMode, ReportingConfig, SectionBlock, SectionMeta, TableBlock,
)


# ─── RunManifest ───


class TestRunManifest:
    def test_completed_modules_filters_correctly(self, full_manifest):
        assert len(full_manifest.completed_modules()) == 3

    def test_completed_modules_excludes_failed(self, partial_manifest):
        completed = partial_manifest.completed_modules()
        assert len(completed) == 1
        assert completed[0].module_name == "deg_analysis"

    def test_get_module_by_name(self, full_manifest):
        m = full_manifest.get_module("deg_analysis")
        assert m is not None
        assert m.status == ModuleStatus.COMPLETED

    def test_get_module_returns_none_for_missing(self, full_manifest):
        assert full_manifest.get_module("nonexistent") is None

    def test_empty_manifest(self, empty_manifest):
        assert empty_manifest.completed_modules() == []
        assert empty_manifest.get_module("anything") is None

    def test_module_run_defaults(self):
        m = ModuleRun(module_name="test")
        assert m.status == ModuleStatus.NOT_RUN
        assert m.output_dir is None
        assert m.output_files == {}
        assert m.params == {}


# ─── ArtifactIndex ───


class TestArtifactIndex:
    def test_get_by_label(self, sample_artifact_index):
        entry = sample_artifact_index.get("deg_table")
        assert entry is not None
        assert entry.module == "deg_analysis"

    def test_get_returns_none_for_missing(self, sample_artifact_index):
        assert sample_artifact_index.get("nonexistent") is None

    def test_for_module_filters(self, sample_artifact_index):
        deg = sample_artifact_index.for_module("deg_analysis")
        assert len(deg) == 2
        labels = {a.label for a in deg}
        assert labels == {"deg_table", "gene_priorities"}

    def test_empty_index(self, empty_artifact_index):
        assert empty_artifact_index.get("anything") is None
        assert empty_artifact_index.for_module("anything") == []

    def test_qc_flags_preserved(self, sample_artifact_index):
        drug = sample_artifact_index.get("drug_gene_pairs")
        assert "ot_zero_scores_possible" in drug.qc_flags


# ─── Evidence Schemas ───


class TestEvidenceSchemas:
    def test_confidence_enum_values(self):
        assert Confidence.HIGH.value == "high"
        assert Confidence.FLAGGED.value == "flagged"

    def test_evidence_card_defaults(self):
        card = EvidenceCard(
            finding="test", module="mod", artifact_label="label",
        )
        assert card.confidence == Confidence.MEDIUM
        assert card.rank == 0
        assert card.section == ""

    def test_conflict_record(self):
        c = ConflictRecord(
            card_a="up", card_b="down", module_a="m1", module_b="m2",
            description="direction mismatch",
        )
        assert c.module_a == "m1"

    def test_narrative_context_isolation(self):
        """NarrativeContext should have no file paths — only evidence and summaries."""
        nc = NarrativeContext(
            disease_name="test", section_title="Section",
            evidence_cards=[EvidenceCard(
                finding="x", module="m", artifact_label="l",
            )],
        )
        assert len(nc.evidence_cards) == 1
        assert nc.table_summaries == {}


# ─── Section Schemas ───


class TestSectionSchemas:
    def test_table_block_defaults(self):
        t = TableBlock(caption="test")
        assert t.rows == []
        assert not t.truncated

    def test_section_block_defaults(self):
        s = SectionBlock(id="test", title="Test")
        assert s.level == 2
        assert s.body == ""
        assert s.tables == []

    def test_reporting_config_defaults(self):
        c = ReportingConfig()
        assert c.narrative_mode == NarrativeMode.DETERMINISTIC
        assert c.table_row_cap == 25
        assert c.include_appendix is True
        assert c.score_threshold == 0.0

    def test_reporting_config_validation(self):
        # table_row_cap must be >= 5
        with pytest.raises(Exception):
            ReportingConfig(table_row_cap=2)

    def test_narrative_mode_enum(self):
        assert NarrativeMode.DETERMINISTIC.value == "deterministic"
        assert NarrativeMode.LLM_AUGMENTED.value == "llm_augmented"
