"""Tests for manifest_builder and registry_builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..data_layer.manifest_builder import manifest_from_state
from ..data_layer.registry_builder import build_artifact_index
from ..data_layer.schemas.manifest import ModuleStatus
from .conftest import write_csv


class TestManifestFromState:
    def test_builds_from_full_state(self, mock_supervisor_state):
        manifest = manifest_from_state(mock_supervisor_state)
        assert manifest.session_id == "sess-mock"
        assert manifest.disease_name == "mock_disease"
        assert len(manifest.modules) == 3

    def test_all_modules_completed(self, mock_supervisor_state):
        manifest = manifest_from_state(mock_supervisor_state)
        for m in manifest.modules:
            assert m.status == ModuleStatus.COMPLETED

    def test_failed_module_captured(self, mock_supervisor_state):
        mock_supervisor_state["errors"] = [
            {"agent": "deg_analysis", "error": "out of memory"}
        ]
        # Remove from results so it's only in errors
        mock_supervisor_state["agent_results"] = [
            r for r in mock_supervisor_state["agent_results"]
            if r["agent"] != "deg_analysis"
        ]
        manifest = manifest_from_state(mock_supervisor_state)
        deg = manifest.get_module("deg_analysis")
        assert deg is not None
        assert deg.status == ModuleStatus.FAILED

    def test_empty_state(self):
        manifest = manifest_from_state({})
        assert manifest.session_id == ""
        assert manifest.modules == []

    def test_output_files_collected(self, mock_supervisor_state):
        mock_supervisor_state["workflow_outputs"]["deg_analysis_volcano_path"] = "/fake/volcano.png"
        manifest = manifest_from_state(mock_supervisor_state)
        deg = manifest.get_module("deg_analysis")
        assert deg is not None
        assert "deg_analysis_output_dir" in deg.output_files

    def test_base_dir_suffix_recognised(self, tmp_path):
        d = tmp_path / "multi_out"
        d.mkdir()
        state = {
            "session_id": "s1",
            "workflow_outputs": {"multiomics_integration_base_dir": str(d)},
            "agent_results": [],
            "errors": [],
        }
        manifest = manifest_from_state(state)
        assert manifest.get_module("multiomics_integration") is not None


class TestBuildArtifactIndex:
    def test_discovers_artefacts_from_manifest(self, full_manifest, tmp_path):
        # Put a matching CSV in the DEG output dir
        deg_dir = Path(full_manifest.modules[0].output_dir)
        write_csv(deg_dir / "test_DEGs.csv", ["gene", "padj", "log2FoldChange"],
                  [["A", "0.01", "2.0"]])
        index = build_artifact_index(full_manifest)
        assert len(index.artifacts) >= 1
        assert index.get("deg_table") is not None

    def test_empty_manifest_returns_empty_index(self, empty_manifest):
        index = build_artifact_index(empty_manifest)
        assert index.artifacts == []

    def test_missing_adapter_skipped_gracefully(self):
        """Modules with no registered adapter produce no artifacts but don't crash."""
        from ..data_layer.schemas.manifest import ModuleRun, RunManifest
        manifest = RunManifest(
            session_id="x",
            modules=[ModuleRun(
                module_name="nonexistent_module",
                status=ModuleStatus.COMPLETED,
                output_dir="/tmp",
            )],
        )
        index = build_artifact_index(manifest)
        assert index.artifacts == []

    def test_failed_module_skipped(self, partial_manifest, tmp_path):
        """Failed modules are not passed to adapters."""
        deg_dir = Path(partial_manifest.modules[0].output_dir)
        write_csv(deg_dir / "test_DEGs.csv", ["gene", "padj"], [["A", "0.01"]])
        index = build_artifact_index(partial_manifest)
        # Only DEG module should produce artifacts (pathway failed, drug not run)
        modules_in_index = {a.module for a in index.artifacts}
        assert "pathway_enrichment" not in modules_in_index
        assert "perturbation_analysis" not in modules_in_index
