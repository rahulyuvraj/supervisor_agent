"""Tests for evidence scoring, conflict detection, and validation."""

from __future__ import annotations

import pytest

from ..data_layer.schemas.evidence import Confidence, ConflictRecord, EvidenceCard
from ..data_layer.schemas.registry import ArtifactEntry, ArtifactIndex
from ..data_layer.schemas.sections import SectionBlock, TableBlock
from ..reporting_engine.evidence import score_findings, detect_conflicts
from ..reporting_engine.validation import ValidationGuard, ValidationError


# ─── Evidence Scoring ───


class TestScoreFindings:
    def test_ranks_assigned(self, sample_evidence_cards):
        scored = score_findings(sample_evidence_cards)
        ranks = [c.rank for c in scored]
        assert ranks == [1, 2, 3]

    def test_auto_promotes_high_confidence_modules(self):
        cards = [
            EvidenceCard(
                finding="100 DEGs", module="deg_analysis",
                artifact_label="deg_table", metric_name="count",
                metric_value=100.0, confidence=Confidence.MEDIUM, section="deg",
            ),
            EvidenceCard(
                finding="low metric", module="some_other",
                artifact_label="x", metric_name="y",
                metric_value=5.0, confidence=Confidence.MEDIUM, section="other",
            ),
        ]
        scored = score_findings(cards)
        deg_card = next(c for c in scored if c.module == "deg_analysis")
        assert deg_card.confidence == Confidence.HIGH

    def test_flagged_cards_sorted_last(self):
        cards = [
            EvidenceCard(
                finding="flagged", module="m1", artifact_label="a",
                confidence=Confidence.FLAGGED, section="s",
            ),
            EvidenceCard(
                finding="high", module="deg_analysis", artifact_label="b",
                metric_value=10.0, confidence=Confidence.MEDIUM, section="s",
            ),
        ]
        scored = score_findings(cards)
        assert scored[0].finding == "high"
        assert scored[-1].finding == "flagged"

    def test_empty_cards(self):
        assert score_findings([]) == []

    def test_no_promotion_for_zero_metric(self):
        """Cards with metric_value=0 should NOT be auto-promoted."""
        card = EvidenceCard(
            finding="zero DEGs", module="deg_analysis", artifact_label="d",
            metric_name="count", metric_value=0.0, confidence=Confidence.MEDIUM, section="s",
        )
        scored = score_findings([card])
        assert scored[0].confidence == Confidence.MEDIUM

    def test_no_promotion_for_negative_metric(self):
        card = EvidenceCard(
            finding="negative", module="pathway_enrichment", artifact_label="p",
            metric_name="score", metric_value=-1.0, confidence=Confidence.MEDIUM, section="s",
        )
        scored = score_findings([card])
        assert scored[0].confidence == Confidence.MEDIUM


# ─── Conflict Detection ───


class TestDetectConflicts:
    def test_detects_directional_conflict(self):
        cards = [
            EvidenceCard(
                finding="upregulated", module="deg_analysis", artifact_label="d",
                metric_name="fc", metric_value=2.5, section="integrated",
            ),
            EvidenceCard(
                finding="downregulated", module="perturbation_analysis",
                artifact_label="p", metric_name="effect", metric_value=-1.5,
                section="integrated",
            ),
        ]
        conflicts = detect_conflicts(cards)
        assert len(conflicts) == 1
        assert "deg_analysis" in conflicts[0].module_a or "deg_analysis" in conflicts[0].module_b

    def test_no_conflict_same_direction(self):
        cards = [
            EvidenceCard(
                finding="up1", module="m1", artifact_label="a",
                metric_value=2.0, section="s",
            ),
            EvidenceCard(
                finding="up2", module="m2", artifact_label="b",
                metric_value=3.0, section="s",
            ),
        ]
        assert detect_conflicts(cards) == []

    def test_no_conflict_same_module(self):
        cards = [
            EvidenceCard(
                finding="up", module="m1", artifact_label="a",
                metric_value=2.0, section="s",
            ),
            EvidenceCard(
                finding="down", module="m1", artifact_label="b",
                metric_value=-1.0, section="s",
            ),
        ]
        assert detect_conflicts(cards) == []

    def test_no_conflict_different_sections(self):
        cards = [
            EvidenceCard(
                finding="up", module="m1", artifact_label="a",
                metric_value=2.0, section="sec1",
            ),
            EvidenceCard(
                finding="down", module="m2", artifact_label="b",
                metric_value=-1.0, section="sec2",
            ),
        ]
        assert detect_conflicts(cards) == []

    def test_no_conflict_missing_metrics(self):
        cards = [
            EvidenceCard(
                finding="up", module="m1", artifact_label="a",
                metric_value=2.0, section="s",
            ),
            EvidenceCard(
                finding="no metric", module="m2", artifact_label="b", section="s",
            ),
        ]
        assert detect_conflicts(cards) == []

    def test_empty_cards(self):
        assert detect_conflicts([]) == []


# ─── ValidationGuard ───


class TestValidationGuard:
    def test_valid_report_passes(self, sample_artifact_index, sample_sections,
                                  sample_evidence_cards):
        guard = ValidationGuard(sample_artifact_index)
        warnings = guard.validate(sample_sections, sample_evidence_cards)
        assert isinstance(warnings, list)

    def test_missing_section_id_raises(self, sample_artifact_index):
        sections = [SectionBlock(id="", title="Bad Section")]
        guard = ValidationGuard(sample_artifact_index)
        with pytest.raises(ValidationError) as exc_info:
            guard.validate(sections, [])
        assert "missing 'id'" in exc_info.value.errors[0].lower()

    def test_missing_title_raises(self, sample_artifact_index):
        sections = [SectionBlock(id="ok", title="")]
        guard = ValidationGuard(sample_artifact_index)
        with pytest.raises(ValidationError) as exc_info:
            guard.validate(sections, [])
        assert "missing title" in exc_info.value.errors[0].lower()

    def test_duplicate_id_raises(self, sample_artifact_index):
        sections = [
            SectionBlock(id="dup", title="First"),
            SectionBlock(id="dup", title="Second"),
        ]
        guard = ValidationGuard(sample_artifact_index)
        with pytest.raises(ValidationError) as exc_info:
            guard.validate(sections, [])
        assert "duplicate" in exc_info.value.errors[0].lower()

    def test_empty_finding_raises(self, sample_artifact_index):
        cards = [EvidenceCard(finding="", module="m", artifact_label="a")]
        guard = ValidationGuard(sample_artifact_index)
        with pytest.raises(ValidationError):
            guard.validate([], cards)

    def test_table_column_mismatch_warns(self, sample_artifact_index):
        section = SectionBlock(
            id="test", title="Test",
            tables=[TableBlock(
                headers=["A", "B"],
                rows=[["1", "2", "3"]],  # 3 cols, expected 2
            )],
        )
        guard = ValidationGuard(sample_artifact_index)
        warnings = guard.validate([section], [])
        assert any("cols" in w.lower() for w in warnings)

    def test_artifact_ref_missing_warns(self, empty_artifact_index):
        section = SectionBlock(
            id="test", title="Test",
            tables=[TableBlock(
                headers=["A"], rows=[["1"]],
                source_artifact="nonexistent",
            )],
        )
        guard = ValidationGuard(empty_artifact_index)
        warnings = guard.validate([section], [])
        assert any("not found" in w.lower() for w in warnings)

    def test_evidence_artifact_ref_warns(self, empty_artifact_index):
        cards = [EvidenceCard(
            finding="x", module="m", artifact_label="missing_label",
        )]
        guard = ValidationGuard(empty_artifact_index)
        warnings = guard.validate([], cards)
        assert any("missing_label" in w for w in warnings)
