"""Conditional edge functions for the supervisor LangGraph workflow."""

import re

from ..agent_registry import AGENT_REGISTRY, AgentType
from .state import SupervisorGraphState

# Matches explicit structured/analysis report requests — NOT molecular report
# (which routes via MOLECULAR_REPORT_RE in intent_node).
# Allow optional format word (pdf/docx/word) between qualifier and "report".
_FMT_WORD = r'(?:(?:pdf|docx?|word)\s+)?'
_STRUCTURED_REPORT_RE = re.compile(
    rf'\b(structured\s+{_FMT_WORD}report|analysis\s+{_FMT_WORD}report'
    rf'|generate\s+(?:a\s+)?{_FMT_WORD}report\b'
    rf'|full\s+{_FMT_WORD}report|pipeline\s+{_FMT_WORD}report'
    rf'|evidence\s+{_FMT_WORD}report)',
    re.IGNORECASE,
)


def check_general_query(state: SupervisorGraphState) -> str:
    """After intent_node: general → END, follow_up → response,
    structured_report → structured_report, else → plan."""
    is_general = state.get("is_general_query")
    has_results = state.get("has_existing_results")
    wants_response = state.get("needs_response")
    query = state.get("user_query", "")

    # Structured report intent: user has results and explicitly asks for a report
    if has_results and _STRUCTURED_REPORT_RE.search(query):
        return "structured_report"

    if is_general:
        # If user explicitly requested a document format (pdf/docx), always
        # route through response_node so it can synthesize from conversation
        # history and render the file — even without prior agent outputs.
        if state.get("response_format") in ("pdf", "docx"):
            return "follow_up"
        # Route knowledge / follow-up queries through response_node for
        # full 2-pass LLM synthesis instead of showing only the router's
        # brief suggested_response.
        return "follow_up" if wants_response else "general"

    # Agent-specific query — but if the target agent already produced its
    # outputs, the user is asking about existing results, not requesting
    # a new run.
    if has_results and wants_response:
        agent_type = state.get("routing_decision", {}).get("agent_type")
        if agent_type:
            try:
                info = AGENT_REGISTRY[AgentType(agent_type)]
                wo = state.get("workflow_outputs", {})
                if info.produces and any(k in wo for k in info.produces):
                    return "follow_up"
            except (ValueError, KeyError):
                pass

    return "needs_execution"


def route_next(state: SupervisorGraphState) -> str:
    """After each executor pass: loop, structured_report, report, or exit."""
    plan = state.get("execution_plan", [])
    index = state.get("current_agent_index", 0)
    if index < len(plan):
        return "execute_agent"

    # After all agents complete, check if structured report was requested
    query = state.get("user_query", "")
    if _STRUCTURED_REPORT_RE.search(query):
        return "structured_report"

    if state.get("needs_response"):
        return "report"
    # All agents failed → route to response for actionable error summary
    if state.get("errors") and not state.get("agent_results"):
        return "report"
    return "done"
