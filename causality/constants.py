"""
Consolidated constants for the causality pipeline.

Single source of truth — agent.py L44-73 and intelligence.py L35-51
both had identical copies.  Now everything lives here.
"""
from __future__ import annotations

# ── LLM defaults (overridden at runtime by llm_bridge) ───────────────────────
CAUSALITY_MODEL = "claude-sonnet-4-20250514"  # demo default; Bedrock model used in prod

MAX_TOKENS_INTENT  = 800
MAX_TOKENS_LIT     = 1_000
MAX_TOKENS_NARRATE = 400
MAX_TOKENS_RESULT  = 1_200
MAX_TOKENS_CLAIMS  = 1_000
MAX_TOKENS_BRIEF   = 1_000

# ── Literature endpoints ──────────────────────────────────────────────────────
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_BASE   = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S2_BASE     = "https://api.semanticscholar.org/graph/v1"

LIT_MAX_PAPERS     = 30
LIT_TOP_K          = 15
LIT_TIMEOUT        = 12.0
LIT_CACHE_TTL_DAYS = 7

# ── Eligibility gate thresholds ───────────────────────────────────────────────
GATE_MIN_SAMPLES        = 30
GATE_WARN_SAMPLES       = 60
GATE_MIN_MR_INSTRUMENTS = 3
GATE_EDGE_CONFIDENCE    = 0.70

# ── Evidence-stream weights (6-stream fusion) ─────────────────────────────────
EVIDENCE_WEIGHTS: dict[str, float] = {
    "genetic":      0.30,
    "perturbation": 0.25,
    "temporal":     0.20,
    "network":      0.15,
    "expression":   0.05,
    "immuno":       0.05,
}
