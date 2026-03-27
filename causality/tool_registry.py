"""
Tool dispatch table for the causality pipeline.

Every slot starts as ``None``.  ``tool_wiring.populate()`` fills them
at runtime **after** ``adapter.check_preconditions()`` has verified that
the required workflow_state keys exist.

Tool signature::

    fn(artifact_store: dict,
       audits: list,
       intent: "ParsedIntent",
       output_dir: str) -> dict

Returns a dict of new artifact key → path/value, merged into
``artifact_store`` for downstream steps.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# fmt: off

TOOL_REGISTRY: Dict[str, Optional[Callable[..., Dict[str, Any]]]] = {

    # ── Acquisition ───────────────────────────────────────────────────────────
    "T_00":  None,   # Cohort retrieval (GEO/SRA)

    # ── Bio Prep ──────────────────────────────────────────────────────────────
    "T_04b": None,   # Single-cell pipeline
    "T_01":  None,   # Expression normalisation
    "T_02":  None,   # DEG (DESeq2)
    "T_03":  None,   # Pathway enrichment
    "T_04":  None,   # Cell-type deconvolution
    "T_05":  None,   # Temporal / pseudotime
    "T_06":  None,   # CRISPR / perturbation

    # ── Intervention Prep ─────────────────────────────────────────────────────
    "T_07":  None,   # Prior knowledge network (SIGNOR)
    "T_08":  None,   # GWAS / eQTL / MR

    # ── DAG Construction ──────────────────────────────────────────────────────
    "M12":   None,   # DAGBuilder
    "M12b":  None,   # DAGBuilder group-B
    "M13":   None,   # DAGValidator
    "M14":   None,   # CentralityCalculator
    "M15":   None,   # EvidenceAggregator

    # ── Causal Analysis ───────────────────────────────────────────────────────
    "M_DC":  None,   # Do-Calculus engine
    "M_IS":  None,   # In-Silico simulator
    "M_PI":  None,   # Pharma intervention engine

    # ── Comparative ───────────────────────────────────────────────────────────
    "DELTA": None,   # Cross-group DAG comparison
}

# fmt: on
