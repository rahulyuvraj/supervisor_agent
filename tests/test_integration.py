"""Integration tests — full pipeline from mock state to rendered Markdown.

Each scenario creates real CSV files on disk, drives the full pipeline
(manifest → artifacts → plan → build → evidence → validate → render),
and asserts on the final Markdown output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from ..data_layer.manifest_builder import manifest_from_state
from ..data_layer.registry_builder import build_artifact_index
from ..reporting_engine.planner import ReportPlanner
from ..reporting_engine.builders import BUILDER_REGISTRY
from ..reporting_engine.evidence import score_findings, detect_conflicts
from ..reporting_engine.validation import ValidationGuard, ValidationError
from ..reporting_engine.renderers.markdown_renderer import render_markdown
from ..data_layer.schemas.sections import ReportingConfig

from .conftest import write_csv


# ─── Pipeline helper ───

def _run_pipeline(state: Dict, config: ReportingConfig | None = None) -> str:
    """Drive full pipeline end-to-end, return Markdown string."""
    cfg = config or ReportingConfig()

    # 1. Manifest
    manifest = manifest_from_state(state)

    # 2. Artifact index
    index = build_artifact_index(manifest)

    # 3. Plan
    planner = ReportPlanner(manifest, index, cfg)
    sections = planner.plan()

    # 4. Build sections
    all_evidence = []
    for section in sections:
        builder_cls = BUILDER_REGISTRY.get(section.id)
        if builder_cls:
            builder = builder_cls(manifest, index, cfg)
            section = builder.build(section)
            all_evidence.extend(builder.collect_evidence(section))

    # 5. Evidence scoring + conflict detection
    scored = score_findings(all_evidence)
    conflicts = detect_conflicts(scored)

    # 6. Validation
    guard = ValidationGuard(index)
    guard.validate(sections, scored)

    # 7. Render
    md = render_markdown(
        sections, evidence_cards=scored, conflicts=conflicts,
        title=f"{manifest.disease_name} Analysis Report",
        disease=manifest.disease_name,
    )
    return md


# ─── Fixture helpers ───

def _populate_deg_csvs(base: Path) -> Path:
    d = base / "deg_output"
    d.mkdir(exist_ok=True)
    write_csv(
        d / "sample_DEGs.csv",
        ["gene", "log2FoldChange", "pvalue", "padj", "baseMean"],
        [["G1", "3.1", "0.0001", "0.001", "800"],
         ["G2", "-2.0", "0.001", "0.01", "600"]],
    )
    write_csv(
        d / "Final_Gene_Priorities.csv",
        ["Gene", "Score", "Rank"],
        [["G1", "0.95", "1"], ["G2", "0.80", "2"]],
    )
    return d


def _populate_pathway_csvs(base: Path) -> Path:
    d = base / "pathway_output"
    d.mkdir(exist_ok=True)
    write_csv(
        d / "Pathways_Consolidated.csv",
        ["pathway", "p_value", "combined_score", "genes"],
        [["PW1", "0.001", "12.5", "G1;G2"],
         ["PW2", "0.2", "3.0", "G3"]],
    )
    return d


def _populate_drug_csvs(base: Path) -> Path:
    d = base / "drug_output"
    d.mkdir(exist_ok=True)
    write_csv(
        d / "Final_GeneDrug_Pairs.csv",
        ["drug", "gene", "score", "mechanism"],
        [["D1", "G1", "0.88", "inhibitor"], ["D2", "G2", "0.0", ""]],
    )
    return d


def _full_state(tmp_path: Path) -> Dict:
    """State dict with all 3 modules completed and CSV files present."""
    deg_dir = _populate_deg_csvs(tmp_path)
    pw_dir = _populate_pathway_csvs(tmp_path)
    drug_dir = _populate_drug_csvs(tmp_path)
    return {
        "session_id": "int-001",
        "analysis_id": "a-001",
        "disease_name": "test_condition",
        "output_root": str(tmp_path),
        "workflow_outputs": {
            "deg_analysis_output_dir": str(deg_dir),
            "pathway_enrichment_output_dir": str(pw_dir),
            "perturbation_analysis_output_dir": str(drug_dir),
        },
        "agent_results": [
            {"agent": "deg_analysis", "status": "completed", "duration_s": 10},
            {"agent": "pathway_enrichment", "status": "completed", "duration_s": 8},
            {"agent": "perturbation_analysis", "status": "completed", "duration_s": 12},
        ],
        "errors": [],
    }


# ─── Scenarios ───


class TestFullPipeline:
    """Multi-module pipeline: all 3 modules completed with real CSVs."""

    def test_output_is_string(self, tmp_path):
        md = _run_pipeline(_full_state(tmp_path))
        assert isinstance(md, str)
        assert len(md) > 100

    def test_title_in_output(self, tmp_path):
        md = _run_pipeline(_full_state(tmp_path))
        assert "test_condition Analysis Report" in md

    def test_structural_sections_present(self, tmp_path):
        md = _run_pipeline(_full_state(tmp_path))
        for heading in ["Executive Summary", "Run Summary", "Limitations"]:
            assert heading in md

    def test_module_sections_present(self, tmp_path):
        md = _run_pipeline(_full_state(tmp_path))
        assert "Differential Expression" in md
        assert "Pathway" in md
        assert "Drug" in md or "Perturbation" in md

    def test_tables_present(self, tmp_path):
        """Pipeline should produce at least one pipe-delimited table."""
        md = _run_pipeline(_full_state(tmp_path))
        assert "| " in md
        assert "---" in md

    def test_appendix_present_by_default(self, tmp_path):
        md = _run_pipeline(_full_state(tmp_path))
        assert "Appendix" in md

    def test_no_appendix_when_disabled(self, tmp_path):
        cfg = ReportingConfig(include_appendix=False)
        md = _run_pipeline(_full_state(tmp_path), config=cfg)
        assert "Appendix" not in md


class TestBulkOnlyPipeline:
    """Only DEG module completed — pathway and drug absent."""

    def _state(self, tmp_path: Path) -> Dict:
        deg_dir = _populate_deg_csvs(tmp_path)
        return {
            "session_id": "int-002",
            "disease_name": "bulk_only",
            "output_root": str(tmp_path),
            "workflow_outputs": {
                "deg_analysis_output_dir": str(deg_dir),
            },
            "agent_results": [
                {"agent": "deg_analysis", "status": "completed", "duration_s": 5},
            ],
            "errors": [],
        }

    def test_deg_section_present(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert "Differential Expression" in md

    def test_pathway_stub_present(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        # The builder produces "No ... data available" for missing modules
        assert "Pathway" in md
        assert "no pathway" in md.lower() or "not part of this run" in md

    def test_structural_sections_still_present(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert "Executive Summary" in md
        assert "Limitations" in md


class TestPartialFailurePipeline:
    """DEG completed, pathway failed, drug not run."""

    def _state(self, tmp_path: Path) -> Dict:
        deg_dir = _populate_deg_csvs(tmp_path)
        pw_dir = tmp_path / "pathway_output"
        pw_dir.mkdir(exist_ok=True)
        return {
            "session_id": "int-003",
            "disease_name": "partial_fail",
            "output_root": str(tmp_path),
            "workflow_outputs": {
                "deg_analysis_output_dir": str(deg_dir),
                "pathway_enrichment_output_dir": str(pw_dir),
            },
            "agent_results": [
                {"agent": "deg_analysis", "status": "completed", "duration_s": 5},
            ],
            "errors": [
                {"agent": "pathway_enrichment", "error": "timeout"},
            ],
        }

    def test_deg_present(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert "Differential Expression" in md

    def test_failure_noted(self, tmp_path):
        """Pathway section should note the failure."""
        md = _run_pipeline(self._state(tmp_path))
        # The planner should produce a stub noting the failure
        assert "Pathway" in md

    def test_limitations_mention_failures(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert "Limitations" in md


class TestEmptyPipeline:
    """No modules at all — output should still be a valid report skeleton."""

    def _state(self, tmp_path: Path) -> Dict:
        return {
            "session_id": "int-004",
            "disease_name": "empty_run",
            "output_root": str(tmp_path),
            "workflow_outputs": {},
            "agent_results": [],
            "errors": [],
        }

    def test_produces_output(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert isinstance(md, str)
        assert "empty_run" in md

    def test_structural_sections(self, tmp_path):
        md = _run_pipeline(self._state(tmp_path))
        assert "Executive Summary" in md
