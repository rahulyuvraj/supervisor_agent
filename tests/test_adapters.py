"""Tests for data_layer adapters — DEG, Pathway, Drug."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..data_layer.schemas.manifest import ModuleRun, ModuleStatus
from ..data_layer.adapters.base import ADAPTER_REGISTRY
from ..data_layer.adapters.deg import DEGAdapter
from ..data_layer.adapters.pathway import PathwayAdapter
from ..data_layer.adapters.drug import DrugDiscoveryAdapter
from .conftest import write_csv


# ─── Registry ───


class TestAdapterRegistry:
    def test_all_three_registered(self):
        assert "deg_analysis" in ADAPTER_REGISTRY
        assert "pathway_enrichment" in ADAPTER_REGISTRY
        assert "perturbation_analysis" in ADAPTER_REGISTRY

    def test_registry_maps_to_correct_classes(self):
        assert ADAPTER_REGISTRY["deg_analysis"] is DEGAdapter
        assert ADAPTER_REGISTRY["pathway_enrichment"] is PathwayAdapter
        assert ADAPTER_REGISTRY["perturbation_analysis"] is DrugDiscoveryAdapter


# ─── DEGAdapter ───


class TestDEGAdapter:
    def _make_adapter(self, tmp_path: Path, status=ModuleStatus.COMPLETED) -> DEGAdapter:
        d = tmp_path / "deg_out"
        d.mkdir(exist_ok=True)
        run = ModuleRun(module_name="deg_analysis", status=status, output_dir=str(d))
        return DEGAdapter(run)

    def test_discover_finds_deg_table(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "deg_out" / "lupus_DEGs.csv",
            ["gene", "log2FoldChange", "pvalue", "padj"],
            [["A", "1.5", "0.01", "0.05"]],
        )
        entries = adapter.discover()
        assert len(entries) >= 1
        labels = {e.label for e in entries}
        assert "deg_table" in labels

    def test_discover_finds_gene_priorities(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "deg_out" / "Final_Gene_Priorities.csv",
            ["Gene", "Score"], [["X", "0.9"]],
        )
        entries = adapter.discover()
        labels = {e.label for e in entries}
        assert "gene_priorities" in labels

    def test_qc_flags_missing_padj(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "deg_out" / "weird_DEGs.csv",
            ["gene", "log2FoldChange"],  # no padj/pvalue
            [["A", "1.5"]],
        )
        entries = adapter.discover()
        deg_entry = next((e for e in entries if e.label == "deg_table"), None)
        assert deg_entry is not None
        assert "missing_significance" in deg_entry.qc_flags

    def test_qc_flags_missing_log2fc(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "deg_out" / "minimal_DEGs.csv",
            ["gene", "padj"],
            [["A", "0.01"]],
        )
        entries = adapter.discover()
        deg_entry = next((e for e in entries if e.label == "deg_table"), None)
        assert deg_entry is not None
        assert "missing_log2fc" in deg_entry.qc_flags

    def test_unavailable_returns_empty(self, tmp_path):
        run = ModuleRun(module_name="deg_analysis", status=ModuleStatus.FAILED)
        adapter = DEGAdapter(run)
        assert adapter.discover() == []

    def test_no_csvs_returns_empty(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        assert adapter.discover() == []

    def test_unmatched_csv_ignored(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "deg_out" / "random_file.csv",
            ["col1", "col2"], [["a", "b"]],
        )
        assert adapter.discover() == []


# ─── PathwayAdapter ───


class TestPathwayAdapter:
    def _make_adapter(self, tmp_path: Path) -> PathwayAdapter:
        d = tmp_path / "pw_out"
        d.mkdir(exist_ok=True)
        run = ModuleRun(module_name="pathway_enrichment",
                        status=ModuleStatus.COMPLETED, output_dir=str(d))
        return PathwayAdapter(run)

    def test_discover_finds_consolidation(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "pw_out" / "Pathways_Consolidated.csv",
            ["pathway", "p_value", "combined_score"], [["PW1", "0.01", "10"]],
        )
        entries = adapter.discover()
        labels = {e.label for e in entries}
        assert "pathway_consolidation" in labels

    def test_qc_missing_pathway_name(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "pw_out" / "pathway_consolidation_v2.csv",
            ["score", "genes"],  # no pathway column
            [["10", "A;B"]],
        )
        entries = adapter.discover()
        pw = next((e for e in entries if e.label == "pathway_consolidation"), None)
        assert pw is not None
        assert "missing_pathway_name" in pw.qc_flags

    def test_qc_missing_pvalue(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "pw_out" / "pathway_consolidation_v2.csv",
            ["pathway", "genes"],  # no p_value
            [["PW1", "A;B"]],
        )
        entries = adapter.discover()
        pw = next((e for e in entries if e.label == "pathway_consolidation"), None)
        assert pw is not None
        assert "missing_p_value" in pw.qc_flags


# ─── DrugDiscoveryAdapter ───


class TestDrugDiscoveryAdapter:
    def _make_adapter(self, tmp_path: Path) -> DrugDiscoveryAdapter:
        d = tmp_path / "drug_out"
        d.mkdir(exist_ok=True)
        run = ModuleRun(module_name="perturbation_analysis",
                        status=ModuleStatus.COMPLETED, output_dir=str(d))
        return DrugDiscoveryAdapter(run)

    def test_discover_finds_drug_gene_pairs(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "drug_out" / "Final_GeneDrug_Pairs.csv",
            ["drug", "gene", "score"], [["D1", "G1", "0.9"]],
        )
        entries = adapter.discover()
        labels = {e.label for e in entries}
        assert "drug_gene_pairs" in labels

    def test_known_qc_issues_appended(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "drug_out" / "drug_scores_v2.csv",
            ["drug", "score", "source"], [["D1", "0.8", "ChEMBL"]],
        )
        entries = adapter.discover()
        scores = next((e for e in entries if e.label == "drug_scores"), None)
        assert scores is not None
        # Known QC issues should be appended automatically
        assert "ot_zero_scores_possible" in scores.qc_flags

    def test_qc_missing_drug_identifier(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "drug_out" / "Final_GeneDrug_Pairs.csv",
            ["target", "score"],  # no drug column
            [["G1", "0.5"]],
        )
        entries = adapter.discover()
        pair = next((e for e in entries if e.label == "drug_gene_pairs"), None)
        assert pair is not None
        assert "missing_drug_identifier" in pair.qc_flags

    def test_mechanism_summary_known_qc(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        write_csv(
            tmp_path / "drug_out" / "mechanism_of_action.csv",
            ["drug", "mechanism"], [["D1", "inhibitor"]],
        )
        entries = adapter.discover()
        mech = next((e for e in entries if e.label == "mechanism_summary"), None)
        assert mech is not None
        assert "action_type_may_be_unknown" in mech.qc_flags
