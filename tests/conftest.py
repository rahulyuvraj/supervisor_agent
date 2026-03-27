"""Shared fixtures for reporting engine tests.

All mock data is domain-free: generic module names, structural labels, and
synthetic column names. Zero hardcoded genes, pathways, diseases, or file names.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

from ..data_layer.schemas.manifest import ModuleRun, ModuleStatus, RunManifest
from ..data_layer.schemas.registry import ArtifactEntry, ArtifactIndex
from ..data_layer.schemas.evidence import Confidence, EvidenceCard, ConflictRecord
from ..data_layer.schemas.sections import ReportingConfig, SectionBlock, SectionMeta, TableBlock


# ─── RunManifest factories ───


def _make_module(
    name: str,
    status: ModuleStatus = ModuleStatus.COMPLETED,
    output_dir: str | None = None,
    **kwargs,
) -> ModuleRun:
    return ModuleRun(module_name=name, status=status, output_dir=output_dir, **kwargs)


@pytest.fixture()
def full_manifest(tmp_path: Path) -> RunManifest:
    """Manifest with DEG, Pathway, and Drug modules all COMPLETED."""
    deg_dir = tmp_path / "deg_output"
    pw_dir = tmp_path / "pathway_output"
    drug_dir = tmp_path / "drug_output"
    for d in (deg_dir, pw_dir, drug_dir):
        d.mkdir()
    return RunManifest(
        session_id="sess-001",
        analysis_id="ana-001",
        disease_name="test_condition",
        output_root=str(tmp_path),
        modules=[
            _make_module("deg_analysis", output_dir=str(deg_dir)),
            _make_module("pathway_enrichment", output_dir=str(pw_dir)),
            _make_module("perturbation_analysis", output_dir=str(drug_dir)),
        ],
    )


@pytest.fixture()
def partial_manifest(tmp_path: Path) -> RunManifest:
    """Manifest with DEG completed, Pathway failed, Drug not run."""
    deg_dir = tmp_path / "deg_output"
    deg_dir.mkdir()
    return RunManifest(
        session_id="sess-002",
        disease_name="partial_condition",
        modules=[
            _make_module("deg_analysis", output_dir=str(deg_dir)),
            _make_module("pathway_enrichment", status=ModuleStatus.FAILED,
                         error_message="timeout during enrichment"),
            _make_module("perturbation_analysis", status=ModuleStatus.NOT_RUN),
        ],
    )


@pytest.fixture()
def empty_manifest() -> RunManifest:
    """Manifest with no modules."""
    return RunManifest(session_id="sess-empty")


# ─── CSV file factories ───


def write_csv(path: Path, headers: List[str], rows: List[List[str]]) -> Path:
    """Write a CSV and return the path."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


@pytest.fixture()
def deg_csv(tmp_path: Path) -> Path:
    """A minimal DEG table CSV with expected columns."""
    return write_csv(
        tmp_path / "deg_output" / "sample_DEGs.csv",
        ["gene", "log2FoldChange", "pvalue", "padj", "baseMean"],
        [
            ["GeneA", "2.5", "0.001", "0.01", "500"],
            ["GeneB", "-1.8", "0.0001", "0.002", "1200"],
            ["GeneC", "0.3", "0.4", "0.8", "300"],
        ],
    )


@pytest.fixture()
def gene_priorities_csv(tmp_path: Path) -> Path:
    """A gene priorities CSV."""
    return write_csv(
        tmp_path / "deg_output" / "Final_Gene_Priorities.csv",
        ["Gene", "Score", "Rank"],
        [["GeneA", "0.95", "1"], ["GeneB", "0.87", "2"]],
    )


@pytest.fixture()
def pathway_csv(tmp_path: Path) -> Path:
    """A pathway consolidation CSV."""
    d = tmp_path / "pathway_output"
    d.mkdir(exist_ok=True)
    return write_csv(
        d / "Pathways_Consolidated.csv",
        ["pathway", "p_value", "combined_score", "genes"],
        [
            ["PathwayX", "0.001", "15.3", "GeneA;GeneB"],
            ["PathwayY", "0.05", "8.1", "GeneC"],
            ["PathwayZ", "0.12", "3.2", "GeneD"],
        ],
    )


@pytest.fixture()
def drug_csv(tmp_path: Path) -> Path:
    """A drug gene pairs CSV."""
    d = tmp_path / "drug_output"
    d.mkdir(exist_ok=True)
    return write_csv(
        d / "Final_GeneDrug_Pairs.csv",
        ["drug", "gene", "score", "mechanism"],
        [
            ["DrugAlpha", "GeneA", "0.92", "inhibitor"],
            ["DrugBeta", "GeneB", "0.0", "unknown"],
        ],
    )


@pytest.fixture()
def drug_scores_csv(tmp_path: Path) -> Path:
    d = tmp_path / "drug_output"
    d.mkdir(exist_ok=True)
    return write_csv(
        d / "drug_scores_ranked.csv",
        ["drug_name", "overall_score", "source"],
        [["DrugAlpha", "0.85", "ChEMBL"], ["DrugBeta", "0.0", "OT"]],
    )


# ─── ArtifactIndex factories ───


@pytest.fixture()
def sample_artifact_index() -> ArtifactIndex:
    """Index with entries from all three adapter modules."""
    return ArtifactIndex(artifacts=[
        ArtifactEntry(
            path="/fake/deg_results.csv", module="deg_analysis",
            label="deg_table", columns=["gene", "padj", "log2FoldChange"],
            row_count=100,
        ),
        ArtifactEntry(
            path="/fake/priorities.csv", module="deg_analysis",
            label="gene_priorities", columns=["Gene", "Score"], row_count=20,
        ),
        ArtifactEntry(
            path="/fake/pathways.csv", module="pathway_enrichment",
            label="pathway_consolidation",
            columns=["pathway", "p_value", "combined_score"], row_count=50,
        ),
        ArtifactEntry(
            path="/fake/drug_pairs.csv", module="perturbation_analysis",
            label="drug_gene_pairs",
            columns=["drug", "gene", "score"], row_count=30,
            qc_flags=["ot_zero_scores_possible"],
        ),
    ])


@pytest.fixture()
def empty_artifact_index() -> ArtifactIndex:
    return ArtifactIndex(artifacts=[])


# ─── EvidenceCard factories ───


@pytest.fixture()
def sample_evidence_cards() -> List[EvidenceCard]:
    return [
        EvidenceCard(
            finding="100 DEGs identified", module="deg_analysis",
            artifact_label="deg_table", metric_name="deg_count",
            metric_value=100.0, confidence=Confidence.MEDIUM, section="deg_findings",
        ),
        EvidenceCard(
            finding="15 significant pathways", module="pathway_enrichment",
            artifact_label="pathway_consolidation", metric_name="pathway_count",
            metric_value=15.0, confidence=Confidence.MEDIUM, section="pathway_findings",
        ),
        EvidenceCard(
            finding="30 drug-gene pairs", module="perturbation_analysis",
            artifact_label="drug_gene_pairs", metric_name="pair_count",
            metric_value=30.0, confidence=Confidence.FLAGGED, section="drug_findings",
        ),
    ]


# ─── SectionBlock factories ───


@pytest.fixture()
def sample_sections() -> List[SectionBlock]:
    return [
        SectionBlock(id="executive_summary", title="Executive Summary", order=0),
        SectionBlock(id="deg_findings", title="DEG Analysis", order=3,
                     meta=SectionMeta(module="deg_analysis",
                                      artifact_labels=["deg_table", "gene_priorities"])),
        SectionBlock(id="pathway_findings", title="Pathway Analysis", order=4,
                     meta=SectionMeta(module="pathway_enrichment",
                                      artifact_labels=["pathway_consolidation"])),
    ]


# ─── Config ───


@pytest.fixture()
def default_config() -> ReportingConfig:
    return ReportingConfig()


# ─── Mock supervisor state ───


@pytest.fixture()
def mock_supervisor_state(tmp_path: Path) -> Dict:
    """State dict shaped like SupervisorGraphState after a full pipeline run."""
    deg_dir = tmp_path / "deg_out"
    pw_dir = tmp_path / "pathway_out"
    drug_dir = tmp_path / "drug_out"
    for d in (deg_dir, pw_dir, drug_dir):
        d.mkdir()
    return {
        "session_id": "sess-mock",
        "analysis_id": "ana-mock",
        "disease_name": "mock_disease",
        "output_root": str(tmp_path),
        "user_query": "generate a structured report",
        "workflow_outputs": {
            "deg_analysis_output_dir": str(deg_dir),
            "pathway_enrichment_output_dir": str(pw_dir),
            "perturbation_analysis_output_dir": str(drug_dir),
        },
        "agent_results": [
            {"agent": "deg_analysis", "status": "completed", "duration_s": 12.3},
            {"agent": "pathway_enrichment", "status": "completed", "duration_s": 8.1},
            {"agent": "perturbation_analysis", "status": "completed", "duration_s": 15.0},
        ],
        "errors": [],
    }
