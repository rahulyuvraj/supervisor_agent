"""
Routing fallbacks — per-intent required data and precedence rules.

Each intent declares *groups* of required workflow_state keys.  The adapter
runs ``check_preconditions()`` **before** tool_wiring touches the registry,
so missing groups are caught early.

Lookup order (handled by adapter):
    1.  workflow_state  (output of a prior executor)
    2.  uploaded_files  via ``get_available_inputs()``
    3.  Degrade gracefully — skip the affected module chain and log a warning.
"""
from __future__ import annotations

from typing import Dict, List

# fmt: off

# Maps each intent to the data groups it depends on.
# Keys inside each group are *actual* workflow_state / get_available_inputs()
# key names — verified against pipeline_executors.py and state.py.
ROUTING_FALLBACKS: Dict[str, Dict[str, List[str]]] = {

    # ── I_01  Full Causal Discovery ───────────────────────────────────────────
    "I_01": {
        "expression":      ["cohort_output_dir", "harmonized_counts_file"],
        "metadata":        ["metadata_file"],
        "deg_results":     ["deg_base_dir", "prioritized_genes_path"],
        "pathway_results": ["pathway_output_dir", "pathway_consolidation_path"],
    },

    # ── I_02  Single-Gene Deep Dive ───────────────────────────────────────────
    "I_02": {
        "expression":  ["cohort_output_dir", "harmonized_counts_file"],
        "deg_results": ["deg_base_dir", "prioritized_genes_path"],
    },

    # ── I_03  Pathway-Centric Causal Subgraph ─────────────────────────────────
    "I_03": {
        "pathway_results": [
            "pathway_output_dir",
            "pathway_enrichment_path",
            "pathway_consolidation_path",
        ],
    },

    # ── I_04  Drug-Target Causal Chain ────────────────────────────────────────
    "I_04": {
        "expression":      ["cohort_output_dir", "harmonized_counts_file"],
        "deg_results":     ["deg_base_dir", "prioritized_genes_path"],
        "pathway_results": ["pathway_output_dir", "pathway_consolidation_path"],
    },

    # ── I_05  What-If Perturbation ────────────────────────────────────────────
    "I_05": {
        "prior_dag": ["causality_output_dir"],
    },

    # ── I_06  Explain Existing DAG ────────────────────────────────────────────
    "I_06": {
        "prior_dag": ["causality_output_dir"],
    },

    # ── I_07  Standard Association (escalation) ──────────────────────────────
    "I_07": {},
}

# fmt: on
