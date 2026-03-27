"""
core_agent.py — internal causality pipeline engine.

Adapted from the standalone demo with targeted fixes:
  1. EligibilityChecker gate dedup: worst-severity-wins (not first-wins).
  2. Plan rejection: ``PlanRejectedError`` replaces ``sys.exit(0)``.
  3. Gate filtering: ``GATE_WORKFLOW_MAP`` applied after intent classification.
  4. Non-interactive: no ``input()`` calls — safe for headless execution.
  5. Two additional DAG column signatures in FileInspector.

Sections:
    1. Enums (IntentID, GateStatus, FilePhase)
    2. Exception (PlanRejectedError)
    3. Dataclasses
    4. PLATFORM_TOOLS list
    5. CAUSAL_MODULE_CHAINS dict
    6. FileInspector class
    7. EligibilityChecker class
    8. WorkflowRouter class
    9. CausalitySupervisorAgent class
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .constants import (
    CAUSALITY_MODEL,
    EVIDENCE_WEIGHTS,
    GATE_EDGE_CONFIDENCE,
    GATE_MIN_MR_INSTRUMENTS,
    GATE_MIN_SAMPLES,
    GATE_WARN_SAMPLES,
    LIT_CACHE_TTL_DAYS,
)

log = logging.getLogger("causal_platform")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Enums
# ═════════════════════════════════════════════════════════════════════════════

class IntentID(str, Enum):
    I_01 = "I_01"   # Causal Drivers Discovery
    I_02 = "I_02"   # Directed Causality X->Y
    I_03 = "I_03"   # Intervention / Actionability
    I_04 = "I_04"   # Comparative Causality
    I_05 = "I_05"   # Counterfactual / What-If
    I_06 = "I_06"   # Evidence Inspection / Explain
    I_07 = "I_07"   # Standard Association Analysis


class GateStatus(str, Enum):
    PASS  = "pass"
    WARN  = "warn"
    BLOCK = "block"
    SKIP  = "skip"


class FilePhase(str, Enum):
    BIO          = "bio_prep"
    INTERVENTION = "intervention_prep"
    DAG          = "dag_build"
    CAUSAL       = "causal_core"
    ACQUISITION  = "acquisition"
    UNKNOWN      = "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Exception
# ═════════════════════════════════════════════════════════════════════════════

class PlanRejectedError(RuntimeError):
    """Raised when the execution plan is rejected (replaces demo sys.exit)."""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    gate_id: str
    name:    str
    status:  GateStatus
    message: str
    value:   Any = None


@dataclass
class FileAuditResult:
    file_path:     str
    file_name:     str
    type_id:       str
    type_label:    str
    phase:         FilePhase
    platform_tool: str
    errors:        list = field(default_factory=list)
    warnings:      list = field(default_factory=list)
    infos:         list = field(default_factory=list)
    gates:         list = field(default_factory=list)
    n_rows:        int  = 0
    n_cols:        int  = 0
    columns:       list = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass
class LiteraturePaper:
    pmid:            Optional[str] = None
    doi:             Optional[str] = None
    title:           str = ""
    abstract:        str = ""
    authors:         str = ""
    year:            Optional[int] = None
    journal:         str = ""
    source:          str = ""
    relevance_score: float = 0.0


@dataclass
class LitClaim:
    entity_x:      str
    relation:      str
    entity_y:      str
    direction:     str
    evidence_type: str
    strength:      str
    pmid:          Optional[str] = None
    quote:         str = ""
    confidence:    float = 0.0


@dataclass
class LitBrief:
    inferred_context:       str  = ""
    key_entities:           list = field(default_factory=list)
    search_queries_used:    list = field(default_factory=list)
    papers_found:           int  = 0
    papers_processed:       int  = 0
    claims:                 list = field(default_factory=list)
    high_confidence_edges:  list = field(default_factory=list)
    conflicts:              list = field(default_factory=list)
    causal_vs_associative:  str  = "mixed"
    prior_evidence_summary: str  = ""
    recommended_modules:    list = field(default_factory=list)
    data_gaps:              list = field(default_factory=list)
    supervisor_brief:       str  = ""
    conflict_rate:          float = 0.0
    error:                  Optional[str] = None


@dataclass
class ParsedIntent:
    intent_id:              IntentID
    intent_name:            str
    confidence:             float
    needs_clarification:    bool
    clarifying_question:    Optional[str]
    context:                dict = field(default_factory=dict)
    entities:               dict = field(default_factory=dict)
    requires:               dict = field(default_factory=dict)
    needs_literature_first: bool = True
    requires_existing_dag:  bool = False
    routing_summary:        str  = ""
    module_chain:           list = field(default_factory=list)
    parallel_blocks:        list = field(default_factory=list)
    fallback:               str  = ""


@dataclass
class FinalResult:
    headline:            str  = ""
    analyzed_context:    str  = ""
    top_findings:        list = field(default_factory=list)
    tier1_candidates:    list = field(default_factory=list)
    tier2_candidates:    list = field(default_factory=list)
    actionable_targets:  list = field(default_factory=list)
    evidence_quality:    dict = field(default_factory=dict)
    caveats:             list = field(default_factory=list)
    next_experiments:    list = field(default_factory=list)
    missing_data_impact: list = field(default_factory=list)
    modules_run:         list = field(default_factory=list)
    artifacts_produced:  list = field(default_factory=list)


@dataclass
class ClarificationResult:
    """Defined here (not in intelligence.py) to avoid circular imports."""
    questions:        list = field(default_factory=list)
    answers:          list = field(default_factory=list)
    enriched_query:   str  = ""
    enriched_context: dict = field(default_factory=dict)
    can_proceed:      bool = True
    skipped:          bool = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLATFORM_TOOLS
# ═════════════════════════════════════════════════════════════════════════════

PLATFORM_TOOLS: list[dict] = [
    {"id": "T_00", "label": "Cohort Data Retrieval", "phase": "acquisition",
     "tool": "cohort_data_retrieval()", "outputs": ["raw_counts.csv", "metadata.csv"],
     "desc": "Retrieves a matching cohort from GEO/SRA using disease name from query.",
     "auto_fetch": True, "auto_fetch_source": "GEO / SRA by disease name",
     "condition": lambda p: p["no_data"]},

    {"id": "T_04b", "label": "Single-Cell RNA-seq Pipeline", "phase": "bio_prep",
     "tool": "sc_pipeline()", "outputs": ["sc_norm.h5ad", "cell_annotations.csv", "pseudobulk.csv"],
     "desc": "QC, normalisation, clustering, and cell-type annotation for scRNA-seq data.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_sc"]},

    {"id": "T_01", "label": "Expression Normalization", "phase": "bio_prep",
     "tool": "normalize_expression()", "outputs": ["expr_norm.parquet", "qc_report.json"],
     "desc": "Log2-CPM normalisation, QC filtering, and variance-based gene selection.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_expression"] and not p["has_deg_output"]},

    {"id": "T_02", "label": "Differential Expression (DESeq2)", "phase": "bio_prep",
     "tool": "run_deseq2()", "outputs": ["DEGs_prioritized.csv"],
     "desc": "Case vs control DEG analysis using phenotype labels from metadata.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_metadata"] and not p["has_deg_output"]},

    {"id": "T_03", "label": "Pathway Enrichment", "phase": "bio_prep",
     "tool": "pathway_enrichment()", "outputs": ["Pathways_Enrichment.csv"],
     "desc": "ORA + GSEA across GO / KEGG / Reactome / MSigDB using DEG results.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: not p["has_pathway_output"]},

    {"id": "T_04", "label": "Cell-Type Deconvolution", "phase": "bio_prep",
     "tool": "deconvolution()", "outputs": ["cell_fractions.csv", "deconv_confidence.json"],
     "desc": "BisQue (SC reference), CIBERSORT (uploaded matrix), or LM22 (built-in) deconvolution.",
     "auto_fetch": True, "auto_fetch_source": "built-in LM22 immune markers (or BisQue if SC uploaded)",
     "condition": lambda p: True},

    {"id": "T_05", "label": "Temporal / Pseudotime Pipeline", "phase": "bio_prep",
     "tool": "temporal_pipeline()", "outputs": ["temporal_gene_fits.tsv", "granger_edges_raw.csv"],
     "desc": "Impulse model + Granger causality; infers pseudotime if no temporal files uploaded.",
     "auto_fetch": True, "auto_fetch_source": "pseudotime inference from expression",
     "condition": lambda p: not p["has_temporal"]},

    {"id": "T_06", "label": "CRISPR / Perturbation Pipeline", "phase": "intervention_prep",
     "tool": "perturbation_pipeline()", "outputs": ["CausalDrivers_Ranked.csv", "GeneEssentiality_ByMedian.csv"],
     "desc": "ACE scoring; uses uploaded CRISPR files or auto-fetches DepMap Avana + LINCS L1000.",
     "auto_fetch": True, "auto_fetch_source": "DepMap Avana CRISPR screen + LINCS L1000 by disease name",
     "condition": lambda p: True},

    {"id": "T_07", "label": "Prior Knowledge Network (SIGNOR)", "phase": "intervention_prep",
     "tool": "prior_knowledge_pipeline()",
     "outputs": ["SIGNOR_Subnetwork_Edges.tsv", "kg_annotations.json"],
     "desc": "Uses uploaded SIGNOR edges; else auto-fetches STRING v12 + KEGG + SIGNOR API.",
     "auto_fetch": True, "auto_fetch_source": "STRING v12 + KEGG + SIGNOR by disease gene set",
     "condition": lambda p: True},

    {"id": "T_08", "label": "GWAS / eQTL / MR Preprocessing", "phase": "intervention_prep",
     "tool": "gwas_eqtl_mr_pipeline()",
     "outputs": ["genetic-evidence.xlsx", "GeneLevel_GeneticEvidence.tsv", "MR_MAIN_RESULTS_ALL_GENES.csv"],
     "desc": "Uses uploaded GWAS/eQTL/MR files; else auto-fetches GWAS Catalog + eQTL Catalogue.",
     "auto_fetch": True, "auto_fetch_source": "GWAS Catalog + eQTL Catalogue + OpenGWAS by disease name",
     "condition": lambda p: True},
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CAUSAL_MODULE_CHAINS
# ═════════════════════════════════════════════════════════════════════════════

CAUSAL_MODULE_CHAINS: dict = {
    "I_01": [
        {"id": "M12",  "label": "DAGBuilder",           "phase": "dag_build",
         "algo": "PC/FCI/GES consensus + multi-modal evidence fusion + MR edge constraints + 1000x bootstrap",
         "outputs": ["consensus_causal_dag.json"]},
        {"id": "M13",  "label": "DAGValidator",          "phase": "causal_core",
         "algo": "Edge stability + PPI + pathway coherence validation",
         "outputs": ["validated_dag.json", "edge_stability.csv"]},
        {"id": "M14",  "label": "CentralityCalculator",  "phase": "causal_core",
         "algo": "kME + betweenness + PageRank -> causal_importance_score -> Tier 1/2/3",
         "outputs": ["causal_importance_scores.csv", "ranked_targets.csv", "tier_assignments.csv"]},
        {"id": "M15",  "label": "EvidenceAggregator",    "phase": "causal_core",
         "algo": "6-stream weighted fusion: Gen(0.30)+Pert(0.25)+Temp(0.20)+Net(0.15)+Expr(0.05)+Imm(0.05)",
         "outputs": ["evidence_matrix.parquet", "conflict_flags.csv"]},
        {"id": "M_DC", "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion + do-calculus: directionality test + confounder removal",
         "outputs": ["do_calculus_results.json", "backdoor_sets.csv", "causal_effects.csv"]},
    ],
    "I_02": [
        {"id": "M12",  "label": "DAGBuilder (targeted)", "phase": "dag_build",
         "algo": "Targeted feature set for X->Y test + MR constraints + 1000x bootstrap",
         "outputs": ["consensus_causal_dag.json"]},
        {"id": "M_DC", "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion: tests X->Y directionality and removes confounding",
         "outputs": ["do_calculus_results.json", "causal_effects.csv"]},
    ],
    "I_03": [
        {"id": "M_IS", "label": "InSilicoSimulator",        "phase": "causal_core",
         "algo": "do(X=0) graph propagation: dose-response + compensation + resistance discovery",
         "outputs": ["predicted_changes.csv", "compensation_pathways.json", "resistance_mechanisms.csv"]},
        {"id": "M_PI", "label": "PharmaInterventionEngine", "phase": "causal_core",
         "algo": "DrugBank + ChEMBL + DepMap + LINCS: therapeutic efficacy vs systemic safety -> TPP",
         "outputs": ["target_product_profile.json", "prioritized_targets.csv", "drug_candidates.csv"]},
    ],
    "I_04": [
        {"id": "M12",   "label": "DAGBuilder (group A)", "phase": "dag_build",
         "algo": "Group A DAG: PC/FCI + multi-modal + MR constraints + bootstrap",
         "outputs": ["consensus_dag_a.json"]},
        {"id": "M12b",  "label": "DAGBuilder (group B)", "phase": "dag_build",
         "algo": "Group B DAG: PC/FCI + multi-modal + MR constraints + bootstrap (parallel)",
         "outputs": ["consensus_dag_b.json"]},
        {"id": "M13",   "label": "DAGValidator",          "phase": "causal_core",
         "algo": "Bootstrap + PPI + pathway coherence per group",
         "outputs": ["validated_dags.json", "edge_stability_per_group.csv"]},
        {"id": "M14",   "label": "CentralityCalculator",  "phase": "causal_core",
         "algo": "kME + PageRank per group -> causal_importance_score per group",
         "outputs": ["causal_importance_per_group.csv", "ranked_targets_per_group.csv"]},
        {"id": "M15",   "label": "EvidenceAggregator",    "phase": "causal_core",
         "algo": "6-stream weighted evidence integration per group",
         "outputs": ["evidence_matrix_per_group.parquet"]},
        {"id": "M_DC",  "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion per group + delta directionality comparison",
         "outputs": ["do_calculus_per_group.json", "direction_delta.json"]},
        {"id": "DELTA", "label": "Delta Analysis",        "phase": "causal_core",
         "algo": "Shared/group-specific/lost edges + delta causal_importance_scores",
         "outputs": ["delta_graph.csv", "conserved_edges.csv", "group_specific_drivers.csv"]},
    ],
    "I_05": [
        {"id": "M_IS", "label": "InSilicoSimulator",        "phase": "causal_core",
         "algo": "do(X=0) counterfactual propagation: reveal resistance + compensation pathways",
         "outputs": ["predicted_changes.csv", "compensation_pathways.json", "resistance_mechanisms.csv"]},
        {"id": "M_DC", "label": "DoCalculusEngine",          "phase": "causal_core",
         "algo": "Validates intervention target is causal node + residual confounding check",
         "outputs": ["do_calculus_results.json"], "optional": True},
        {"id": "M_PI", "label": "PharmaInterventionEngine",  "phase": "causal_core",
         "algo": "Drug prioritisation for simulated intervention",
         "outputs": ["target_product_profile.json"], "optional": True},
    ],
    "I_06": [
        {"id": "M15", "label": "EvidenceAggregator (read-only)", "phase": "causal_core",
         "algo": "Retrieve evidence matrix and conflict flags from prior run",
         "outputs": ["evidence_breakdown.json"]},
        {"id": "M14", "label": "CentralityCalculator (read-only)", "phase": "causal_core",
         "algo": "Retrieve causal_importance_scores and Tier assignments from prior run",
         "outputs": ["gap_analysis.json", "citations.csv"]},
    ],
    "I_07": [],
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FileInspector
# ═════════════════════════════════════════════════════════════════════════════

class FileInspector:
    """Inspects uploaded data files; returns a :class:`FileAuditResult`."""

    BINARY_EXTS = {".h5ad", ".pkl", ".rds", ".h5", ".hdf5", ".bam", ".bai", ".loom"}

    _FILENAME_RULES: list[tuple] = [
        ("expression",      "Expression / count matrix",
         FilePhase.BIO, "T_01",
         r"_raw_count\.tsv$|_counts\.csv$|rnaseq"),
        ("metadata",        "Sample metadata with outcome labels",
         FilePhase.ACQUISITION, "T_01",
         r"prep_meta\.csv$|-METADATA\.xlsx$|_meta\.csv$|metadata"),
        ("deg_output",      "DEG output (pre-computed)",
         FilePhase.BIO, "T_02",
         r"_DEGs_prioritized\.csv$|DEG"),
        ("pathway_output",  "Pathway enrichment output (pre-computed)",
         FilePhase.BIO, "T_03",
         r"_Pathways_Enrichment\.csv$|pathway.*enrichment"),
        ("signature_matrix","Cell-type signature / deconvolution reference matrix",
         FilePhase.BIO, "T_04",
         r"signature_matrix\.tsv$|signature_matrix"),
        ("gwas_mr",         "GWAS / MR results",
         FilePhase.INTERVENTION, "T_08",
         r"__genetic-evidence\.xlsx$|genetic.*evidence"),
        ("eqtl_gene",       "eQTL gene-level genetic evidence",
         FilePhase.INTERVENTION, "T_08",
         r"_GeneLevel_GeneticEvidence\.tsv$"),
        ("eqtl_variant",    "eQTL variant-level genetic evidence",
         FilePhase.INTERVENTION, "T_08",
         r"_VariantLevel_GeneticEvidence\.tsv$"),
        ("mr_results",      "Mendelian Randomization results",
         FilePhase.INTERVENTION, "T_08",
         r"MR_MAIN_RESULTS_ALL_GENES\.csv$|MR_MAIN"),
        ("crispr_guide",    "CRISPR guide-level screen",
         FilePhase.INTERVENTION, "T_06",
         r"CRISPR_GuideLevel_Avana|CRISPR.*GuideLevel"),
        ("crispr_essentiality", "Gene essentiality scores",
         FilePhase.INTERVENTION, "T_06",
         r"GeneEssentiality_ByMedian\.csv$|GeneEssentiality"),
        ("crispr_ranked",   "Causal drivers ranked",
         FilePhase.INTERVENTION, "T_06",
         r"CausalDrivers_Ranked\.csv$|CausalDrivers"),
        ("temporal_fits",   "Temporal / impulse model fits",
         FilePhase.BIO, "T_05",
         r"temporal_gene_fits\.tsv$|temporal.*gene.*fits"),
        ("granger_edges",   "Granger causality edges",
         FilePhase.BIO, "T_05",
         r"granger_edges_raw\.csv$|granger_edges"),
        ("prior_network",   "Prior knowledge network (SIGNOR / STRING / KEGG)",
         FilePhase.INTERVENTION, "T_07",
         r"SIGNOR_Subnetwork_Edges\.tsv$|SIGNOR"),
    ]

    def __init__(self):
        self._col_signatures = None  # lazy — built on first inspect()

    @property
    def col_signatures(self):
        if self._col_signatures is None:
            self._col_signatures = self._build_col_signatures()
        return self._col_signatures

    # ── column signature helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_col_signatures() -> list[tuple]:
        import pandas as pd

        def _cols(df: pd.DataFrame, *names: str) -> bool:
            c = set(df.columns.str.lower())
            return all(n in c for n in names)

        def _has_sample_col(df: pd.DataFrame) -> bool:
            return any("sample" in c.lower() for c in df.columns)

        def _has_outcome_col(df: pd.DataFrame) -> bool:
            kw = ["disease_status", "outcome", "phenotype", "label",
                  "group", "response", "status", "trait", "condition",
                  "diagnosis", "class", "category"]
            return any(any(k in c.lower() for k in kw) for c in df.columns)

        def _is_expression(df: pd.DataFrame) -> bool:
            gene_kw = {"gene", "gene_id", "geneid", "ens-id", "ensembl",
                       "symbol", "feature_id", "transcript_id", "description"}
            col_lower = [c.lower() for c in df.columns]
            has_gene = any(any(g in c for g in gene_kw) for c in col_lower)
            if not has_gene:
                return False
            non_gene = [c for c in df.columns
                        if not any(g in c.lower() for g in gene_kw)]
            if not non_gene:
                return False
            sample_like = sum(
                1 for c in non_gene
                if re.match(r"(GSM|SRR|ERR|DRR|SAMN|SRS|S\d+|sample[_\s]?\d+|\d+)", c, re.I)
            )
            return sample_like / len(non_gene) >= 0.5

        return [
            ("temporal_fits",   "Temporal / impulse model fits",
             FilePhase.BIO, "T_05",
             lambda df: _cols(df, "gene_id", "pattern", "r2_impulse")),
            ("granger_edges",   "Granger causality edges",
             FilePhase.BIO, "T_05",
             lambda df: _cols(df, "source", "target", "effect_f", "q_value")),
            ("crispr_ranked",   "CRISPR / ACE perturbation drivers",
             FilePhase.INTERVENTION, "T_06",
             lambda df: _cols(df, "gene", "ace", "rank")),
            ("gwas_mr",         "GWAS / MR results",
             FilePhase.INTERVENTION, "T_08",
             lambda df: _cols(df, "gene", "or", "b") and
                        any(c in df.columns.str.lower() for c in ["pval", "p_value", "pvalue"])),
            ("eqtl_gene",       "eQTL data",
             FilePhase.INTERVENTION, "T_08",
             lambda df: _cols(df, "gene", "snp", "slope")),
            # ── two DAG signatures added in integration ──
            ("dag_edges",       "DAG edge list",
             FilePhase.DAG, "M12",
             lambda df: _cols(df, "source", "target", "weight", "edge_type")),
            ("dag_nodes",       "DAG node annotations",
             FilePhase.DAG, "M14",
             lambda df: _cols(df, "node_id", "causal_tier")),
            # ── original continued ──
            ("prior_network",   "Prior knowledge network (SIGNOR / STRING / KEGG)",
             FilePhase.INTERVENTION, "T_07",
             lambda df: _cols(df, "source", "target", "mechanism")),
            ("gwas_mr",         "GWAS summary statistics",
             FilePhase.INTERVENTION, "T_08",
             lambda df: any(c in df.columns.str.lower()
                            for c in ["snp_id", "rsid", "snp", "variant_id"])),
            ("expression",      "Expression / count matrix",
             FilePhase.BIO, "T_01",
             lambda df: _is_expression(df)),
            ("metadata",        "Sample metadata with outcome labels",
             FilePhase.ACQUISITION, "T_01",
             lambda df: _has_sample_col(df) and _has_outcome_col(df)),
            ("metadata_partial","Sample metadata (no outcome column)",
             FilePhase.ACQUISITION, "T_01",
             lambda df: _has_sample_col(df)),
            ("signature_matrix","Cell-type signature / deconvolution reference matrix",
             FilePhase.BIO, "T_04",
             lambda df: any(c.lower() in ("gene", "gene_id", "geneid") for c in df.columns)
                        and len(df.columns) >= 4
                        and not any(c in df.columns.str.lower()
                                    for c in ["ace", "pattern", "mechanism", "source"])),
            # Fallback
            ("expression",      "Expression / count matrix",
             FilePhase.BIO, "T_01",
             lambda df: True),
        ]

    # ── inspect ───────────────────────────────────────────────────────────────

    def inspect(self, file_path) -> FileAuditResult:
        path  = Path(file_path)
        fname = path.name

        if path.suffix.lower() in self.BINARY_EXTS:
            return FileAuditResult(
                file_path=str(path), file_name=fname,
                type_id="sc_data", type_label="Single-cell data (.h5ad / binary)",
                phase=FilePhase.BIO, platform_tool="T_04b",
                infos=["Single-cell binary -> T_04b runs before T_04."],
            )

        nl = fname.lower()
        for tid, tlabel, tphase, ttool, pattern in self._FILENAME_RULES:
            if re.search(pattern, nl, re.I):
                try:
                    df = self._load_tabular(path, fname)
                except Exception as exc:
                    return FileAuditResult(
                        file_path=str(path), file_name=fname,
                        type_id=tid, type_label=tlabel,
                        phase=tphase, platform_tool=ttool,
                        errors=[f"Parse error: {exc}"],
                    )
                audit = FileAuditResult(
                    file_path=str(path), file_name=fname,
                    type_id=tid, type_label=tlabel,
                    phase=tphase, platform_tool=ttool,
                    n_rows=len(df), n_cols=len(df.columns),
                    columns=list(df.columns),
                )
                self._audit(audit, df)
                return audit

        try:
            df = self._load_tabular(path, fname)
        except Exception as exc:
            return FileAuditResult(
                file_path=str(path), file_name=fname,
                type_id="unknown", type_label="Could not parse",
                phase=FilePhase.UNKNOWN, platform_tool="?",
                errors=[f"Parse error: {exc}"],
            )

        type_id, label, phase, tool = "unknown", "Unknown", FilePhase.UNKNOWN, "?"
        for tid, tlabel, tphase, ttool, test in self.col_signatures:
            try:
                if test(df):
                    type_id, label, phase, tool = tid, tlabel, tphase, ttool
                    break
            except Exception:
                continue

        audit = FileAuditResult(
            file_path=str(path), file_name=fname,
            type_id=type_id, type_label=label,
            phase=phase, platform_tool=tool,
            n_rows=len(df), n_cols=len(df.columns),
            columns=list(df.columns),
        )
        self._audit(audit, df)
        return audit

    @staticmethod
    def _load_tabular(path: Path, fname: str):
        import pandas as pd
        if fname.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, nrows=300)
        sep = "\t" if fname.endswith((".tsv", ".txt")) else ","
        return pd.read_csv(path, sep=sep, nrows=300)

    # ── type-specific audits ──────────────────────────────────────────────────

    def _audit(self, audit: FileAuditResult, df: pd.DataFrame):
        import pandas as pd
        tid = audit.type_id
        n   = len(df)
        if tid == "metadata":
            self._audit_metadata(audit, df, n)
        elif tid == "metadata_partial":
            audit.errors.append(
                "G_01 BLOCK: No outcome/disease_status column. "
                "Add 'outcome', 'disease_status', 'phenotype', or 'group'."
            )
            audit.gates.append(GateResult("G_01", "Phenotype label completeness",
                                          GateStatus.BLOCK, "Missing"))
        elif tid == "expression":
            self._audit_expression(audit, df, n)
        elif tid == "deg_output":
            sig_col = next((c for c in df.columns if "padj" in c.lower() or "fdr" in c.lower()), None)
            sig = 0
            if sig_col:
                sig = int((pd.to_numeric(df[sig_col], errors="coerce") < 0.05).sum())
            audit.infos.append(f"{n} genes in DEG output; {sig} significant (padj<0.05).")
        elif tid == "pathway_output":
            audit.infos.append(f"{n} pathways in enrichment output.")
        elif tid in ("gwas_mr", "gwas_raw", "mr_results", "eqtl_gene", "eqtl_variant"):
            self._audit_gwas(audit, df, n)
        elif tid in ("crispr_guide", "crispr_essentiality", "crispr_ranked"):
            self._audit_crispr(audit, df, n)
        elif tid == "temporal_fits":
            self._audit_temporal_fits(audit, df, n)
        elif tid == "granger_edges":
            self._audit_granger_edges(audit, df, n)
        elif tid == "prior_network":
            self._audit_prior_network(audit, df, n)
        elif tid == "signature_matrix":
            cell_types = [c for c in df.columns if c.lower() not in ("gene", "gene_id", "geneid")]
            audit.infos.append(f"{n:,} genes x {len(cell_types)} reference cell types")
            if len(cell_types) < 3:
                audit.warnings.append("Fewer than 3 cell types — deconvolution resolution limited.")
            if n < 1000:
                audit.warnings.append("Fewer than 1,000 reference genes — consider a broader matrix.")

    def _audit_metadata(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        outcome_cols = [c for c in df.columns
                        if re.search(r"outcome|phenotype|status|label|group|response|trait|disease", c, re.I)]
        audit.infos.append(f"{n} samples; outcome columns: {', '.join(outcome_cols[:4]) or 'none'}")
        audit.gates.append(GateResult(
            "G_01", "Phenotype label completeness",
            GateStatus.PASS if outcome_cols else GateStatus.BLOCK,
            "Outcome column(s) present" if outcome_cols
            else "No outcome/phenotype column — causal analysis requires outcome labels.",
        ))
        if n < GATE_MIN_SAMPLES:
            audit.errors.append(f"G_02 BLOCK: {n} samples < {GATE_MIN_SAMPLES} min.")
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.BLOCK, f"{n} samples"))
        elif n < GATE_WARN_SAMPLES:
            audit.warnings.append(f"G_02 WARN: {n} samples — potentially underpowered.")
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.WARN, f"{n} samples"))
        else:
            audit.infos.append(f"G_02 pass: {n} samples >= {GATE_MIN_SAMPLES}.")
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.PASS, f"{n} samples"))

    @staticmethod
    def _audit_expression(audit: FileAuditResult, df: pd.DataFrame, n: int):
        n_samples = len(df.columns) - 1
        if n < 100:
            audit.warnings.append(f"Only {n} feature rows — may be pre-filtered.")
        else:
            audit.infos.append(f"{n:,} features x {n_samples} samples.")
        if n_samples < GATE_MIN_SAMPLES:
            audit.errors.append(f"G_02 BLOCK: {n_samples} samples in expression (<{GATE_MIN_SAMPLES}).")

    @staticmethod
    def _audit_temporal_fits(audit: FileAuditResult, df: pd.DataFrame, n: int):
        import pandas as pd
        good_r2 = int(pd.to_numeric(df.get("r2_impulse", pd.Series(dtype=float)), errors="coerce").ge(0.5).sum())
        sig = int(pd.to_numeric(df.get("p_adj", pd.Series(dtype=float)), errors="coerce").lt(0.05).sum())
        audit.infos.append(f"{n:,} genes; {sig} significant; {good_r2} well-fitted (R2>=0.5)")
        if n > 0 and good_r2 / n < 0.15:
            audit.warnings.append(f"Only {round(good_r2 / n * 100)}% genes have R2>=0.5.")

    @staticmethod
    def _audit_granger_edges(audit: FileAuditResult, df: pd.DataFrame, n: int):
        import pandas as pd
        q_col = next((c for c in df.columns if c.lower() in ("q_value", "qvalue", "q")), None)
        sig_q = int(pd.to_numeric(df[q_col], errors="coerce").lt(0.05).sum()) if q_col else 0
        audit.infos.append(f"{n:,} Granger edges; {sig_q} q<0.05")

    @staticmethod
    def _audit_gwas(audit: FileAuditResult, df: pd.DataFrame, n: int):
        audit.infos.append(f"{n} rows in genetic evidence file.")
        audit.gates.append(GateResult(
            "G_06", "MR instrument sufficiency",
            GateStatus.PASS if n >= GATE_MIN_MR_INSTRUMENTS else GateStatus.WARN,
            f"{n} rows (need >={GATE_MIN_MR_INSTRUMENTS})",
        ))

    @staticmethod
    def _audit_crispr(audit: FileAuditResult, df: pd.DataFrame, n: int):
        import pandas as pd
        ace_col = next((c for c in df.columns if c.lower() == "ace"), None)
        if ace_col:
            ace = pd.to_numeric(df[ace_col], errors="coerce")
            drivers = int((ace > 0.1).sum())
            audit.infos.append(f"{n} genes; {drivers} causal drivers (ACE>0.1)")
            audit.gates.append(GateResult(
                "G_07", "IV strength / ACE coverage",
                GateStatus.PASS if drivers > 0 else GateStatus.WARN,
                f"{drivers} driver genes with ACE>0.1",
            ))
        else:
            audit.infos.append(f"{n} rows in CRISPR file.")

    @staticmethod
    def _audit_prior_network(audit: FileAuditResult, df: pd.DataFrame, n: int):
        mech_col = next((c for c in df.columns if "mechanism" in c.lower()), None)
        mechs = list(df[mech_col].dropna().unique()[:5]) if mech_col else []
        audit.infos.append(f"{n} prior edges; mechanisms: {', '.join(str(m) for m in mechs)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EligibilityChecker  (Bug 1 fixed: worst-severity dedup)
# ═════════════════════════════════════════════════════════════════════════════

_SEVERITY = {GateStatus.BLOCK: 3, GateStatus.WARN: 2, GateStatus.PASS: 1, GateStatus.SKIP: 0}


class EligibilityChecker:
    """Aggregates per-file gate results and fills cross-file gates G_03–G_09."""

    def evaluate(self, audits: list[FileAuditResult]) -> list[GateResult]:
        # --- Bug-1 fix: keep worst severity per gate_id, not first occurrence ---
        seen: dict[str, tuple[int, GateResult]] = {}  # gate_id → (severity, GateResult)
        for audit in audits:
            for g in audit.gates:
                sev = _SEVERITY.get(g.status, 0)
                prev = seen.get(g.gate_id)
                if prev is None or sev > prev[0]:
                    seen[g.gate_id] = (sev, g)
        gates: list[GateResult] = [v[1] for v in seen.values()]
        seen_ids = set(seen)

        type_map: dict[str, FileAuditResult] = {a.type_id: a for a in audits}

        if "G_03" not in seen_ids:
            has_expr = any(a.type_id in ("expression", "sc_data") for a in audits)
            gates.append(GateResult(
                "G_03", "Expression matrix availability",
                GateStatus.PASS if has_expr else GateStatus.WARN,
                "Expression matrix detected" if has_expr
                else "No expression matrix — normalization needs raw counts or pre-normalized data",
            ))
            seen_ids.add("G_03")

        meta = type_map.get("metadata")
        if meta and "G_04" not in seen_ids:
            batch_cols = [c for c in meta.columns if re.search(r"batch|run|plate|lane|center|site", c, re.I)]
            gates.append(GateResult(
                "G_04", "Batch confounding risk",
                GateStatus.WARN if batch_cols else GateStatus.PASS,
                f"Batch column(s): {', '.join(batch_cols)}" if batch_cols else "No batch columns",
            ))
            seen_ids.add("G_04")

        if "G_05" not in seen_ids:
            gates.append(GateResult("G_05", "Graph stability (convergence)",
                                    GateStatus.SKIP, "Evaluated during M12/M13"))
            seen_ids.add("G_05")

        if "G_08" not in seen_ids:
            has_intervention = any(
                a.type_id in ("crispr_guide", "crispr_essentiality", "crispr_ranked")
                for a in audits
            )
            has_meta = any(a.type_id in ("metadata", "metadata_partial") for a in audits)
            needs_warn = has_intervention and not has_meta
            gates.append(GateResult(
                "G_08", "Intervention context mismatch",
                GateStatus.WARN if needs_warn else GateStatus.PASS,
                "Intervention data without matched metadata" if needs_warn else "Context match OK",
            ))
            seen_ids.add("G_08")

        if "G_09" not in seen_ids:
            gates.append(GateResult(
                "G_09", f"Validated edge confidence >={GATE_EDGE_CONFIDENCE}",
                GateStatus.SKIP,
                f"Edges below {GATE_EDGE_CONFIDENCE} excluded; kept as exploratory",
            ))
            seen_ids.add("G_09")

        return sorted(gates, key=lambda g: g.gate_id)

    @staticmethod
    def is_blocked(gates: list[GateResult]) -> bool:
        return any(g.status == GateStatus.BLOCK for g in gates)

    @staticmethod
    def filter_by_intent(gates: list[GateResult], intent_id: str) -> list[GateResult]:
        """Return only gates applicable to the given intent per GATE_WORKFLOW_MAP."""
        from .routing_rules import GATE_WORKFLOW_MAP
        allowed = GATE_WORKFLOW_MAP.get(intent_id)
        if allowed is None or not allowed:
            return []  # empty list means no gates apply
        return [g for g in gates if g.gate_id in allowed]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WorkflowRouter
# ═════════════════════════════════════════════════════════════════════════════

_UPLOAD_FLAGS: dict[str, list] = {
    "T_04": ["has_sig_matrix", "has_sc"],
    "T_05": ["has_temporal"],
    "T_06": ["has_crispr"],
    "T_07": ["has_prior_network"],
    "T_08": ["has_gwas", "has_eqtl", "has_mr"],
}

_HARD_SKIP: dict[str, str] = {
    "T_00":  "Files uploaded — cohort retrieval not needed",
    "T_04b": "No .h5ad / single-cell file uploaded",
    "T_01":  "DEG output already uploaded — normalization skipped",
    "T_02":  "No metadata or DEG output already uploaded — DESeq2 skipped",
    "T_03":  "Pathway enrichment output already uploaded",
    "T_05":  "Temporal fits / Granger edges already uploaded",
}


class WorkflowRouter:
    """Builds an ordered execution plan as a list of step dicts."""

    def build_steps(self, intent: ParsedIntent, audits: list,
                    clarification_context: dict | None = None) -> list:
        clarification_context = clarification_context or {}
        tid_set = {a.type_id for a in audits}
        p = self._presence_flags(tid_set, audits, clarification_context)

        iid = intent.intent_id.value if hasattr(intent.intent_id, "value") else str(intent.intent_id)

        platform = self._select_platform_tools(iid)
        steps: list[dict] = []

        for tool in platform:
            condition = tool.get("condition")
            runs = condition(p) if callable(condition) else True

            if not runs:
                skip = _HARD_SKIP.get(tool["id"], "Condition not met")
                data_source = None
            else:
                skip = None
                data_source = self._data_source(tool, p)

            steps.append({
                "id":                tool["id"],
                "label":             tool["label"],
                "phase":             tool["phase"],
                "tool":              tool["tool"],
                "outputs":           tool["outputs"],
                "desc":              tool["desc"],
                "skip":              skip,
                "required":          not tool.get("optional", False),
                "data_source":       data_source,
                "auto_fetch":        tool.get("auto_fetch", False) and not self._has_uploaded(tool["id"], p),
                "auto_fetch_source": tool.get("auto_fetch_source", ""),
            })

        for mod in CAUSAL_MODULE_CHAINS.get(iid, []):
            is_optional = mod.get("optional", False)
            skip = "Optional module — skipped" if is_optional else None
            steps.append({
                "id":                mod["id"],
                "label":             mod["label"],
                "phase":             mod["phase"],
                "tool":              mod["algo"],
                "outputs":           mod["outputs"],
                "desc":              mod.get("algo", ""),
                "skip":              skip,
                "required":          not is_optional,
                "data_source":       "computed from pipeline outputs",
                "auto_fetch":        False,
                "auto_fetch_source": "",
            })

        return steps

    @staticmethod
    def _data_source(tool: dict, p: dict) -> str:
        tid = tool["id"]
        if tid == "T_00":
            return f"auto-fetch: {tool.get('auto_fetch_source', 'GEO/SRA')}"
        if tid in ("T_01", "T_02", "T_03"):
            return "uploaded file"
        if tid == "T_04b":
            return "uploaded .h5ad file"
        if tool.get("auto_fetch"):
            if WorkflowRouter._has_uploaded(tid, p):
                return "uploaded file"
            return f"auto-fetch: {tool.get('auto_fetch_source', 'public database')}"
        return "uploaded file"

    @staticmethod
    def _has_uploaded(tool_id: str, p: dict) -> bool:
        return any(p.get(f) for f in _UPLOAD_FLAGS.get(tool_id, []))

    @staticmethod
    def _presence_flags(tid_set: set, audits: list, ctx: dict) -> dict:
        p: dict[str, bool] = {
            "no_data":            len(audits) == 0,
            "has_sc":             "sc_data" in tid_set,
            "has_expression":     "expression" in tid_set,
            "has_metadata":       any(t in tid_set for t in ("metadata", "phenotype", "metadata_partial")),
            "has_deg_output":     "deg_output" in tid_set,
            "has_pathway_output": "pathway_output" in tid_set,
            "has_sig_matrix":     "signature_matrix" in tid_set,
            "has_temporal":       any(t in tid_set for t in ("temporal_fits", "granger_edges")),
            "has_gwas":           any(t in tid_set for t in ("gwas_mr", "gwas_raw")),
            "has_eqtl":           any(t in tid_set for t in ("eqtl_gene", "eqtl_variant")),
            "has_mr":             "mr_results" in tid_set,
            "has_crispr":         any(t in tid_set
                                     for t in ("crispr_guide", "crispr_essentiality", "crispr_ranked")),
            "has_prior_network":  "prior_network" in tid_set,
        }
        p["has_genetic"]          = p["has_gwas"] or p["has_eqtl"] or p["has_mr"]
        p["has_expression_or_sc"] = p["has_expression"] or p["has_sc"]

        data_ans = str(ctx.get("data_availability", "")).lower()
        if "crispr" in data_ans or "perturbation" in data_ans:
            p["has_crispr"] = True
        if "temporal" in data_ans or "longitudinal" in data_ans:
            p["has_temporal"] = True
        if "gwas" in data_ans or "genetic" in data_ans:
            p["has_gwas"] = True
        if "eqtl" in data_ans:
            p["has_eqtl"] = True
        return p

    @staticmethod
    def _select_platform_tools(iid: str) -> list:
        if iid == "I_06":
            return []
        if iid == "I_07":
            return [t for t in PLATFORM_TOOLS if t["id"] in ("T_01", "T_02", "T_03")]
        if iid in ("I_03", "I_05"):
            return []
        return PLATFORM_TOOLS


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CausalitySupervisorAgent
# ═════════════════════════════════════════════════════════════════════════════

_PHASE_ORDER  = ["acquisition", "bio_prep", "intervention_prep", "dag_build", "causal_core"]
_PHASE_LABELS = {
    "acquisition":       "PHASE 1 — Data Acquisition",
    "bio_prep":          "PHASE 2 — Biological Preparation",
    "intervention_prep": "PHASE 3 — Intervention Preparation",
    "dag_build":         "PHASE 4 — DAG Construction (M12)",
    "causal_core":       "PHASE 5 — Causal Core Analysis",
}


class CausalitySupervisorAgent:
    """
    Orchestrates the full causality pipeline from file inspection to synthesis.

    Unlike the demo ``SupervisorAgent`` this class:
    - takes an ``llm_bridge`` with a ``complete()`` interface (no Anthropic SDK);
    - is non-interactive (no ``input()`` calls);
    - raises :class:`PlanRejectedError` instead of ``sys.exit()``;
    - filters gates via ``GATE_WORKFLOW_MAP`` per intent.
    """

    LIT_CACHE_TTL_DAYS = LIT_CACHE_TTL_DAYS

    def __init__(self, llm_bridge):
        """
        Parameters
        ----------
        llm_bridge : CausalityLLMBridge
            Object with ``complete(user_msg, system_prompt, max_tokens, as_json)``
            matching the signature of the demo's ``ClaudeClient``.
        """
        from .intelligence import (
            ClarificationEngine, IntentClassifier,
            LiteraturePipeline, ResultSynthesiser, StepNarrator,
        )
        self._llm           = llm_bridge
        self._inspector     = FileInspector()
        self._eligibility   = EligibilityChecker()
        self._router        = WorkflowRouter()
        self._clarification = ClarificationEngine(self._llm)
        self._intent_clf    = IntentClassifier(self._llm)
        self._lit           = LiteraturePipeline(self._llm)
        self._narrator      = StepNarrator(self._llm)
        self._synthesiser   = ResultSynthesiser(self._llm)

    # ── Pre-run clarification ─────────────────────────────────────────────────

    def pre_clarify(self, query: str, file_paths: list | None = None) -> list:
        file_paths = file_paths or []
        try:
            intent = self._intent_clf.classify(query)
            intent_name = intent.intent_name
        except Exception:
            intent_name = ""
        file_types: list[str] = []
        for fp in file_paths:
            try:
                audit = self._inspector.inspect(fp)
                file_types.append(audit.type_id)
            except Exception:
                pass
        return self._clarification.generate_questions(query, file_types, intent_name)

    def build_clarification(self, query: str, questions: list,
                            answers: list, skipped: bool = False) -> ClarificationResult:
        return self._clarification.build_result(query, questions, answers, skipped)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        query:                str,
        file_paths:           list | None = None,
        clarification_result: ClarificationResult | None = None,
        skip_literature:      bool  = False,
        verbose:              bool  = False,
        output_dir:           str   = "pipeline_outputs",
    ) -> dict:
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_output_dir = f"{output_dir}/{run_id}"

        clarif_ctx: dict = {}
        if clarification_result and not clarification_result.skipped:
            query      = clarification_result.enriched_query
            clarif_ctx = clarification_result.enriched_context

        log.info("[%s] Causality pipeline start  query=%s", run_id, query[:80])
        file_paths = file_paths or []

        # ── 1. File inspection ────────────────────────────────────────────────
        log.info("[1/7] Inspecting %d file(s)...", len(file_paths))
        audits: list[FileAuditResult] = []
        for fp in file_paths:
            audit = self._inspector.inspect(fp)
            audits.append(audit)
            log.info("  %s -> %s (%s)", audit.file_name, audit.type_label, audit.platform_tool)

        # ── 2. Eligibility gates (all) ────────────────────────────────────────
        log.info("[2/7] Running eligibility gates...")
        all_gates = self._eligibility.evaluate(audits)

        # ── 3. Intent classification ──────────────────────────────────────────
        log.info("[3/7] Classifying intent...")
        intent = self._intent_clf.classify(query)
        iid = intent.intent_id.value if hasattr(intent.intent_id, "value") else str(intent.intent_id)
        log.info("  %s: %s (%.0f%% confidence)", iid, intent.intent_name, intent.confidence * 100)

        already_clarified = bool(clarification_result)
        if intent.needs_clarification and not already_clarified:
            return {
                "run_id":              run_id,
                "status":              "needs_clarification",
                "clarifying_question": intent.clarifying_question,
                "intent_partial":      asdict(intent),
                "file_audits":         [asdict(a) for a in audits],
                "gate_results":        [asdict(g) for g in all_gates],
            }

        # ── 3b. Filter gates by intent (GATE_WORKFLOW_MAP) ───────────────────
        active_gates = self._eligibility.filter_by_intent(all_gates, iid)
        if self._eligibility.is_blocked(active_gates):
            blocking = [g for g in active_gates if g.status == GateStatus.BLOCK]
            msg = "Pipeline blocked:\n" + "\n".join(
                f"  BLOCK {g.gate_id}: {g.message}" for g in blocking
            )
            log.warning(msg)
            return {
                "run_id":           run_id,
                "status":           "blocked",
                "gate_results":     [asdict(g) for g in all_gates],
                "file_audits":      [asdict(a) for a in audits],
                "blocking_message": msg,
            }

        # ── 4. Literature pipeline ────────────────────────────────────────────
        lit_brief: LitBrief | None = None
        should_run_lit = (
            not skip_literature
            and intent.needs_literature_first
            and iid not in ("I_06", "I_07")
        )
        if should_run_lit:
            cache_hit, lit_brief = self._load_lit_cache(query, output_dir)
            if cache_hit:
                log.info("[4/7] Literature pipeline — loaded from cache")
            else:
                log.info("[4/7] Running literature pipeline (LIT_01–LIT_08)...")
                try:
                    lit_brief = self._lit.run(
                        query=query,
                        intent_name=intent.intent_name,
                        entities=intent.entities or {},
                    )
                    self._save_lit_cache(query, lit_brief, output_dir)
                    log.info("  papers=%d claims=%d conflicts=%d",
                             lit_brief.papers_processed, len(lit_brief.claims), len(lit_brief.conflicts))
                except Exception as exc:
                    log.warning("Literature pipeline failed: %s", exc)
                    lit_brief = LitBrief(error=str(exc))
        else:
            log.info("[4/7] Literature pipeline — skipped")

        # ── 5. Build execution plan ───────────────────────────────────────────
        log.info("[5/7] Building execution plan...")
        steps = self._router.build_steps(intent, audits, clarif_ctx)
        active = [s for s in steps if not s.get("skip")]
        log.info("  %d active, %d skipped", len(active), len(steps) - len(active))

        # ── 6. Execute steps phase by phase ───────────────────────────────────
        log.info("[6/7] Executing pipeline steps...")
        executed_steps: list[dict] = []
        artifact_store: dict = {}

        phase_groups: dict[str, list] = {ph: [] for ph in _PHASE_ORDER}
        for step in steps:
            ph = step.get("phase", "causal_core")
            phase_groups.setdefault(ph, []).append(step)

        from .tool_registry import TOOL_REGISTRY

        def _run_one_step(step: dict) -> dict:
            if step.get("skip"):
                log.debug("  SKIP %s: %s", step["id"], step["skip"])
                return {**step, "status": "skipped"}
            step_id = step["id"]
            fn = TOOL_REGISTRY.get(step_id)
            exec_status = "pending"
            new_arts: dict = {}
            if fn is None:
                log.debug("  [%s] not implemented (None in registry)", step_id)
            else:
                os.makedirs(run_output_dir, exist_ok=True)
                try:
                    result = fn(artifact_store, audits, intent, run_output_dir)
                    new_arts = result or {}
                    exec_status = "done"
                    log.info("  [%s] done — artifacts: %s", step_id, list(new_arts.keys()))
                except NotImplementedError:
                    exec_status = "pending"
                except RuntimeError as exc:
                    log.warning("  [%s] pipeline error: %s", step_id, exc)
                    exec_status = "error"
                except Exception as exc:
                    log.warning("  [%s] unexpected error: %s", step_id, exc, exc_info=True)
                    exec_status = "error"
            artifact_store.update(new_arts)
            narration = self._narrator.narrate(step, intent)
            return {**step, "status": exec_status, "narration": narration,
                    "artifacts_out": list(new_arts.keys())}

        pipeline_aborted = False
        for phase_name in _PHASE_ORDER:
            p_steps = phase_groups.get(phase_name, [])
            if not p_steps:
                continue
            active_in_phase = [s for s in p_steps if not s.get("skip")]
            if not active_in_phase:
                for s in p_steps:
                    executed_steps.append({**s, "status": "skipped"})
                continue

            log.info("  %s", _PHASE_LABELS.get(phase_name, phase_name.upper()))
            phase_results: list[dict] = []
            for step in p_steps:
                sr = _run_one_step(step)
                phase_results.append(sr)
            executed_steps.extend(phase_results)

            if phase_name != "causal_core":
                done = sum(1 for s in phase_results if s.get("status") == "done")
                pend = sum(1 for s in phase_results if s.get("status") == "pending")
                errs = sum(1 for s in phase_results if s.get("status") == "error")
                log.info("  Phase checkpoint: %d done, %d pending, %d errors", done, pend, errs)

        # ── 7. Synthesise result ──────────────────────────────────────────────
        log.info("[7/7] Synthesising results...")
        result = self._synthesiser.synthesise(
            query=query, intent=intent, steps_run=executed_steps,
            lit_brief=lit_brief, audits=audits, artifact_store=artifact_store,
        )
        log.info("Result: %s", result.headline)

        pipeline_result = {
            "run_id":       run_id,
            "status":       "complete" if not pipeline_aborted else "aborted",
            "query":        query,
            "intent":       asdict(intent),
            "context":      intent.context,
            "gate_results": [asdict(g) for g in all_gates],
            "file_audits":  [asdict(a) for a in audits],
            "lit_brief":    asdict(lit_brief) if lit_brief else None,
            "steps":        executed_steps,
            "result":       asdict(result),
        }
        self._write_next_steps(pipeline_result, output_dir)
        return pipeline_result

    # ── Literature cache ──────────────────────────────────────────────────────

    @staticmethod
    def _lit_cache_path(query: str, output_dir: str) -> Path:
        key = hashlib.md5(query.strip().lower().encode()).hexdigest()[:12]
        return Path(output_dir) / "lit_cache" / f"{key}.json"

    def _load_lit_cache(self, query: str, output_dir: str) -> tuple[bool, LitBrief | None]:
        path = self._lit_cache_path(query, output_dir)
        if not path.exists():
            return False, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if (datetime.now() - cached_at).days > self.LIT_CACHE_TTL_DAYS:
                return False, None
            bd = data["lit_brief"]
            brief = LitBrief(
                inferred_context=bd.get("inferred_context", ""),
                key_entities=bd.get("key_entities", []),
                search_queries_used=bd.get("search_queries_used", []),
                papers_found=bd.get("papers_found", 0),
                papers_processed=bd.get("papers_processed", 0),
                claims=[LitClaim(**c) for c in bd.get("claims", [])],
                high_confidence_edges=bd.get("high_confidence_edges", []),
                conflicts=bd.get("conflicts", []),
                causal_vs_associative=bd.get("causal_vs_associative", "mixed"),
                prior_evidence_summary=bd.get("prior_evidence_summary", ""),
                recommended_modules=bd.get("recommended_modules", []),
                data_gaps=bd.get("data_gaps", []),
                supervisor_brief=bd.get("supervisor_brief", ""),
                conflict_rate=bd.get("conflict_rate", 0.0),
            )
            return True, brief
        except Exception as exc:
            log.debug("Lit cache load failed: %s", exc)
            return False, None

    def _save_lit_cache(self, query: str, brief: LitBrief, output_dir: str) -> None:
        path = self._lit_cache_path(query, output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps({"cached_at": datetime.now().isoformat(), "lit_brief": asdict(brief)},
                           indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Lit cache save failed: %s", exc)

    # ── Next-steps artefact ───────────────────────────────────────────────────

    def _write_next_steps(self, result: dict, output_dir: str) -> None:
        intent_d = result.get("intent", {})
        steps    = result.get("steps", [])
        gates    = result.get("gate_results", [])

        pending_impl = [
            {"module_id": s["id"], "label": s["label"], "phase": s.get("phase", "")}
            for s in steps
            if not s.get("skip") and s.get("status") == "pending"
        ]
        warnings = [
            {"gate": g["gate_id"], "name": g["name"], "message": g["message"]}
            for g in gates if g["status"] in ("warn", "block")
        ]

        doc = {
            "run_id":                         result["run_id"],
            "query":                          result.get("query", ""),
            "intent":                         f"{intent_d.get('intent_id')} -- {intent_d.get('intent_name')}",
            "modules_pending_implementation": pending_impl,
            "eligibility_warnings":           warnings,
        }

        out_path = Path(output_dir) / "next_steps.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        log.debug("Next steps written to %s", out_path)
