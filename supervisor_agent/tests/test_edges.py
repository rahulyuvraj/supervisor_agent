"""Tests for supervisor_agent.langgraph.edges — routing logic."""

import re
import pytest

from supervisor_agent.langgraph.edges import (
    _STRUCTURED_REPORT_RE,
    check_general_query,
)
from supervisor_agent.langgraph.nodes import _detect_needs_response


# ── Regex tests ──────────────────────────────────────────────────────────────


class TestStructuredReportRegex:
    """_STRUCTURED_REPORT_RE must match report requests with/without format words."""

    @pytest.mark.parametrize("text", [
        "generate a report",
        "generate report",
        "Generate a Report for ERBB2",
        "generate a pdf report",
        "generate a docx report",
        "generate a word report",
        "generate a PDF report with detailed summary and overview",
        "generate pdf report",
        "Can you generate a pdf report with summary",
        "structured report",
        "structured pdf report",
        "analysis report",
        "analysis pdf report",
        "full report",
        "full pdf report",
        "pipeline report",
        "evidence report",
        "evidence docx report",
    ])
    def test_matches(self, text):
        assert _STRUCTURED_REPORT_RE.search(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        "what is a report card",
        "tell me about ERBB2",
        "run DEG analysis",
        "what genes are upregulated",
        "molecular report",       # handled by _MOLECULAR_REPORT_RE
        "patient report",         # handled by _MOLECULAR_REPORT_RE
    ])
    def test_no_match(self, text):
        assert not _STRUCTURED_REPORT_RE.search(text), f"Should NOT match: {text!r}"


# ── check_general_query tests ────────────────────────────────────────────────


def _state(**overrides):
    """Build a minimal SupervisorGraphState dict for edge tests."""
    base = {
        "user_query": "",
        "is_general_query": False,
        "has_existing_results": False,
        "needs_response": False,
        "response_format": "none",
        "routing_decision": {},
    }
    base.update(overrides)
    return base


class TestCheckGeneralQuery:
    """Routing decisions in check_general_query."""

    def test_general_chat_goes_to_end(self):
        """Plain chat with no results → 'general' (END)."""
        s = _state(user_query="hello", is_general_query=True)
        assert check_general_query(s) == "general"

    def test_general_with_results_goes_to_followup(self):
        """General query with prior results → 'follow_up' (response_node)."""
        s = _state(
            user_query="summarize findings",
            is_general_query=True,
            has_existing_results=True,
            needs_response=True,
        )
        assert check_general_query(s) == "follow_up"

    def test_pdf_format_forces_followup_even_without_results(self):
        """User asks for PDF report without prior agent results →
        should still go to response_node (not END)."""
        s = _state(
            user_query="generate a pdf report with detailed summary",
            is_general_query=True,
            has_existing_results=False,
            needs_response=True,
            response_format="pdf",
        )
        assert check_general_query(s) == "follow_up"

    def test_docx_format_forces_followup_even_without_results(self):
        s = _state(
            user_query="create a docx report",
            is_general_query=True,
            has_existing_results=False,
            needs_response=True,
            response_format="docx",
        )
        assert check_general_query(s) == "follow_up"

    def test_pdf_with_results_routes_structured_report(self):
        """If user has results AND asks to generate a report → structured_report."""
        s = _state(
            user_query="generate a pdf report",
            is_general_query=True,
            has_existing_results=True,
            needs_response=True,
            response_format="pdf",
        )
        assert check_general_query(s) == "structured_report"

    def test_structured_report_with_results(self):
        s = _state(
            user_query="generate a structured report",
            is_general_query=True,
            has_existing_results=True,
            needs_response=True,
        )
        assert check_general_query(s) == "structured_report"

    def test_no_general_flag_routes_to_execution(self):
        """Non-general query without results → needs_execution."""
        s = _state(
            user_query="run DEG analysis",
            is_general_query=False,
            has_existing_results=False,
        )
        assert check_general_query(s) == "needs_execution"

    def test_chat_format_general_no_results_routes_to_followup(self):
        """Chat format with needs_response=True → 'follow_up' (response_node)
        even without prior results, so the full 2-pass LLM synthesis runs."""
        s = _state(
            user_query="tell me about ERBB2",
            is_general_query=True,
            has_existing_results=False,
            needs_response=True,
            response_format="chat",
        )
        assert check_general_query(s) == "follow_up"

    def test_general_no_results_no_response_goes_to_end(self):
        """General query with needs_response=False → 'general' (END)."""
        s = _state(
            user_query="hello",
            is_general_query=True,
            has_existing_results=False,
            needs_response=False,
            response_format="chat",
        )
        assert check_general_query(s) == "general"


# ── _detect_needs_response tests ─────────────────────────────────────────────


class TestDetectNeedsResponse:
    """Follow-up / elaboration patterns should trigger needs_response=True."""

    @pytest.mark.parametrize("query", [
        "can you elaborate more?",
        "tell me more about that",
        "what are your sources?",
        "go deeper into the mechanism",
        "expand on the discussion",
        "in more detail please",
        "more about the kinase domain",
        "can you explain the binding affinity?",
        "how do you know that?",
        "what is that based on?",
        "continue",
        "keep going",
        "further analysis please",
    ])
    def test_followup_patterns_detected(self, query):
        assert _detect_needs_response(query), f"Should detect: {query!r}"

    @pytest.mark.parametrize("query", [
        # Reporting signals (already covered)
        "what are the top genes?",
        "summarize findings",
        "explain the results",
        "show me the pathways",
    ])
    def test_reporting_signals_still_detected(self, query):
        assert _detect_needs_response(query), f"Should detect: {query!r}"

    @pytest.mark.parametrize("query", [
        "run DEG analysis",
        "execute the pipeline",
        "perform gene prioritization",
    ])
    def test_execution_queries_not_detected(self, query):
        assert not _detect_needs_response(query), f"Should NOT detect: {query!r}"

    @pytest.mark.parametrize("query", [
        "hello",
        "ok",
        "thanks",
    ])
    def test_plain_chat_not_detected(self, query):
        assert not _detect_needs_response(query), f"Should NOT detect: {query!r}"
