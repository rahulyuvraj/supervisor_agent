"""
Intent-level routing configuration — thresholds and gate applicability.

Values sourced from the Excel specification sheets (Supervisor_Routing,
Eligibility_Gates "Applies To" column).
"""
from __future__ import annotations

# Minimum LLM confidence required before routing to each intent.
# Below this threshold the adapter yields a clarification request instead.
CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "I_01": 0.70,   # Full Causal Discovery
    "I_02": 0.65,   # Single-Gene Deep Dive
    "I_03": 0.65,   # Pathway-Centric Causal Subgraph
    "I_04": 0.70,   # Drug-Target Causal Chain
    "I_05": 0.75,   # What-If Perturbation
    "I_06": 0.60,   # Explain Existing DAG
    "I_07": 0.60,   # Standard Association Analysis (escalation)
}

# Which eligibility gates apply to each intent workflow.
# EligibilityChecker.evaluate() filters its output to only gates in this list.
GATE_WORKFLOW_MAP: dict[str, list[str]] = {
    "I_01": ["G_01", "G_02", "G_03", "G_04", "G_05", "G_06", "G_07", "G_08", "G_09"],
    "I_02": ["G_01", "G_02", "G_03", "G_06", "G_09"],
    "I_03": ["G_07", "G_08"],
    "I_04": ["G_01", "G_02", "G_03", "G_04", "G_05", "G_06", "G_07", "G_08", "G_09"],
    "I_05": [],
    "I_06": [],
    "I_07": [],
}
