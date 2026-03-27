"""LangGraph state definitions for the Supervisor Agent graph.

SupervisorGraphState is the shared state flowing through all graph nodes.
SupervisorResult is the structured output for API consumers.
"""

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, TypedDict


def _merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dicts (shallow). Handles None from MemorySaver restore."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    merged = {**left, **right}
    return {k: v for k, v in merged.items()
            if not isinstance(v, str) or len(v) < 10_000}


_MAX_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))


def _capped_add(left: list, right: list) -> list:
    if left is None:
        left = []
    if right is None:
        right = []
    return (left + right)[-_MAX_HISTORY:]


class SupervisorGraphState(TypedDict, total=False):
    """Shared state for the supervisor LangGraph workflow."""

    # ── Intent layer ──
    user_query: str
    disease_name: str
    uploaded_files: Annotated[Dict[str, str], _merge_dicts]       # filename → local filepath
    detected_file_types: Annotated[Dict[str, str], _merge_dicts]  # filename → detected type key
    routing_decision: Dict[str, Any]         # serialized RoutingDecision
    is_general_query: bool
    general_response: str
    needs_response: bool                     # whether to run LLM synthesis after execution
    response_format: str                     # "chat" | "pdf" | "docx" | "none"
    report_theme: str                        # CSS theme preset: "default" | "clinical" | "minimal"
    style_instructions: str                  # freeform styling text extracted from query
    molecular_report_format: str             # "pdf" | "docx" — for molecular report output
    requested_top_n: int                     # CSV rows to read (default 10, from "top N" in query)
    has_existing_results: bool               # True when workflow_outputs has prior run data

    # ── Planning layer ──
    execution_plan: List[str]                # AgentType.value list in run order
    agents_skipped: List[str]
    current_agent_index: int

    # ── Execution layer ──
    workflow_outputs: Annotated[Dict[str, Any], _merge_dicts]  # merges across nodes
    agent_results: Annotated[List[Dict[str, Any]], _capped_add]
    errors: Annotated[List[Dict[str, Any]], _capped_add]
    retry_counts: Annotated[Dict[str, int], _merge_dicts]      # agent_type.value → attempts used

    # ── Conversation layer — accumulates across invocations via MemorySaver ──
    conversation_history: Annotated[List[Dict[str, str]], _capped_add]

    # ── Response layer ──
    final_response: str

    # ── Telemetry ──
    llm_call_log: Annotated[List[Dict[str, Any]], _capped_add]

    # ── Infrastructure ──
    session_id: str
    analysis_id: str
    user_id: str
    output_root: str
    status: str                              # routed | planned | executing | completed | failed


@dataclass
class SupervisorResult:
    """Structured output returned by run_supervisor() for API consumers."""

    status: str
    final_response: str = ""
    output_dir: str = ""
    key_files: Dict[str, str] = field(default_factory=dict)
    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: int = 0
