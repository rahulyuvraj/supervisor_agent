"""Tests for ReportPlanner."""

from __future__ import annotations

import pytest

from ..data_layer.schemas.manifest import ModuleStatus
from ..data_layer.schemas.sections import ReportingConfig
from ..reporting_engine.planner import ReportPlanner, _SECTION_PLAN


class TestReportPlanner:
    def test_full_manifest_produces_all_sections(self, full_manifest, sample_artifact_index):
        planner = ReportPlanner(full_manifest, sample_artifact_index)
        sections = planner.plan()
        ids = [s.id for s in sections]
        expected = [sid for sid, *_ in _SECTION_PLAN]
        assert ids == expected

    def test_partial_manifest_includes_stubs(self, partial_manifest, empty_artifact_index):
        planner = ReportPlanner(partial_manifest, empty_artifact_index)
        sections = planner.plan()
        pw = next(s for s in sections if s.id == "pathway_findings")
        assert "failed" in pw.body.lower()
        drug = next(s for s in sections if s.id == "drug_findings")
        assert "not part of this run" in drug.body.lower()

    def test_empty_manifest_structural_sections_only(self, empty_manifest, empty_artifact_index):
        planner = ReportPlanner(empty_manifest, empty_artifact_index)
        sections = planner.plan()
        # Structural sections (None required_module) should always be present
        ids = {s.id for s in sections}
        for sid, _, req_mod, _ in _SECTION_PLAN:
            if req_mod is None:
                assert sid in ids
            else:
                # Module-specific sections should have stub bodies
                sec = next((s for s in sections if s.id == sid), None)
                assert sec is not None
                assert sec.meta.omitted_reason != ""

    def test_section_order_preserved(self, full_manifest, sample_artifact_index):
        planner = ReportPlanner(full_manifest, sample_artifact_index)
        sections = planner.plan()
        orders = [s.order for s in sections]
        assert orders == sorted(orders)

    def test_disabled_section_excluded(self, full_manifest, sample_artifact_index):
        config = ReportingConfig(sections_enabled={"appendix": False})
        planner = ReportPlanner(full_manifest, sample_artifact_index, config)
        sections = planner.plan()
        ids = {s.id for s in sections}
        assert "appendix" not in ids

    def test_appendix_toggle(self, full_manifest, sample_artifact_index):
        config = ReportingConfig(include_appendix=False)
        planner = ReportPlanner(full_manifest, sample_artifact_index, config)
        sections = planner.plan()
        ids = {s.id for s in sections}
        assert "appendix" not in ids

    def test_artifact_labels_attached(self, full_manifest, sample_artifact_index):
        planner = ReportPlanner(full_manifest, sample_artifact_index)
        sections = planner.plan()
        deg = next(s for s in sections if s.id == "deg_findings")
        assert "deg_table" in deg.meta.artifact_labels

    def test_skipped_module_stub(self, partial_manifest, empty_artifact_index):
        # Change pathway to SKIPPED
        for m in partial_manifest.modules:
            if m.module_name == "pathway_enrichment":
                m.status = ModuleStatus.SKIPPED
        planner = ReportPlanner(partial_manifest, empty_artifact_index)
        sections = planner.plan()
        pw = next(s for s in sections if s.id == "pathway_findings")
        assert "skipped" in pw.body.lower()
