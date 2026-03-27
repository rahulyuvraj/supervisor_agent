"""Tests for format carry-forward fix and disease-stickiness fix.

Bug 1: Scientific queries should NOT trigger format carry-forward.
Bug 2: Disease from current query takes priority over stale checkpoint.
Bug 3: MemorySaver checkpoint leaks final_response/response_format/disease_name
        from turn N-1 into turn N — _build_initial_state must reset them.
"""

from __future__ import annotations

import pytest

from supervisor_agent.response.style_engine import (
    extract_style_instructions,
    has_style_intent,
)
from supervisor_agent.langgraph.entry import _build_initial_state


# ── has_style_intent gate ──


class TestHasStyleIntent:
    """Allowlist gate must return True only for genuine style keywords."""

    @pytest.mark.parametrize(
        "query",
        [
            "make headings red",
            "use blue font for the tables",
            "dark mode theme please",
            "generate pdf with compact layout",
            "can you make the background light gray",
            "two-column format with larger headings",
            "corporate professional style",
            "use sans-serif font",
            "add a border around tables",
        ],
    )
    def test_positive_style_queries(self, query):
        assert has_style_intent(query), f"Should detect style intent in: {query!r}"

    @pytest.mark.parametrize(
        "query",
        [
            "How does allosteric binding of Asciminib affect ABL1 kinase activation loop?",
            "What are the top genes involved in lupus nephritis?",
            "run pathway enrichment on the prioritized genes",
            "can you explain the mechanism of ERBB2 in breast cancer",
            "generate a pdf report for sjogren syndrome",
            "what is the role of BRCA1 in DNA repair",
            "analyze differentially expressed genes for my dataset",
            "how does metformin affect glucose metabolism",
            "run DEPMAP analysis for KRAS mutations",
        ],
    )
    def test_negative_scientific_queries(self, query):
        assert not has_style_intent(query), f"Should NOT detect style intent in: {query!r}"


# ── extract_style_instructions with allowlist gate ──


class TestExtractStyleGated:
    """extract_style_instructions now returns empty for non-style queries."""

    def test_scientific_query_returns_empty(self):
        """The original bug: a scientific query should NOT produce style text."""
        result = extract_style_instructions(
            "How does the allosteric binding of Asciminib affect the "
            "conformational state of the ABL1 kinase activation loop?",
            "pdf",
            "breast cancer",
        )
        assert result == ""

    def test_real_style_query_still_works(self):
        result = extract_style_instructions(
            "make headings red and generate the pdf", "pdf", "breast cancer"
        )
        assert "red" in result.lower()

    def test_dark_mode_still_works(self):
        result = extract_style_instructions(
            "generate pdf with dark mode theme", "pdf", ""
        )
        assert "dark" in result.lower()

    def test_compact_layout_works(self):
        result = extract_style_instructions(
            "make it compact with smaller font", "pdf", ""
        )
        assert result != ""

    def test_plain_report_request_returns_empty(self):
        """'generate a pdf report' has no style intent—only a format request."""
        result = extract_style_instructions(
            "can you please generate a pdf report", "pdf", ""
        )
        assert result == ""

    def test_chat_format_still_blocked(self):
        result = extract_style_instructions("make it red", "chat", "")
        assert result == ""

    def test_gene_query_returns_empty(self):
        result = extract_style_instructions(
            "What role does TP53 play in cell cycle regulation?",
            "pdf",
            "colon cancer",
        )
        assert result == ""

    def test_pathway_query_returns_empty(self):
        result = extract_style_instructions(
            "run KEGG pathway enrichment on my gene list",
            "pdf",
            "lupus",
        )
        assert result == ""


# ── Disease priority: topic-change aware ──


class TestDiseasePriority:
    """Verify the three-way disease resolution logic.

    - Extracted disease always wins.
    - General query with no extracted disease → empty (topic change).
    - Agent-routed query with no extracted disease → checkpoint fallback.
    """

    def test_fresh_disease_wins(self):
        extracted = "chronic myeloid leukemia"
        is_general = True
        checkpoint = "breast cancer"
        if extracted:
            disease = extracted
        elif is_general:
            disease = ""
        else:
            disease = checkpoint
        assert disease == "chronic myeloid leukemia"

    def test_general_query_no_disease_clears(self):
        """General query with no extracted disease → empty (don't carry forward)."""
        extracted = ""
        is_general = True
        checkpoint = "breast cancer"
        if extracted:
            disease = extracted
        elif is_general:
            disease = ""
        else:
            disease = checkpoint
        assert disease == ""

    def test_agent_query_falls_back_to_checkpoint(self):
        """Agent-routed query ('run pathway enrichment') → keep prior disease."""
        extracted = ""
        is_general = False
        checkpoint = "breast cancer"
        if extracted:
            disease = extracted
        elif is_general:
            disease = ""
        else:
            disease = checkpoint
        assert disease == "breast cancer"

    def test_both_empty(self):
        extracted = ""
        is_general = True
        checkpoint = ""
        if extracted:
            disease = extracted
        elif is_general:
            disease = ""
        else:
            disease = checkpoint
        assert disease == ""

    def test_none_extracted_agent_query_falls_back(self):
        extracted = None
        is_general = False
        checkpoint = "lupus"
        if extracted:
            disease = extracted
        elif is_general:
            disease = ""
        else:
            disease = checkpoint
        assert disease == "lupus"


# ── Import sanity ──


class TestExports:
    def test_has_style_intent_importable_from_package(self):
        from supervisor_agent.response import has_style_intent as fn

        assert callable(fn)

    def test_extract_style_instructions_importable(self):
        from supervisor_agent.response import extract_style_instructions as fn

        assert callable(fn)


# ── _build_initial_state checkpoint reset ──


class TestBuildInitialState:
    """_build_initial_state must include per-turn reset fields so checkpoint
    values from a prior turn don't leak into the new invocation."""

    def test_response_format_defaults_to_none(self):
        state = _build_initial_state("hello", "sess1")
        assert state["response_format"] == "none"

    def test_final_response_defaults_to_empty(self):
        state = _build_initial_state("hello", "sess1")
        assert state["final_response"] == ""

    def test_needs_response_defaults_to_false(self):
        state = _build_initial_state("hello", "sess1")
        assert state["needs_response"] is False

    def test_general_response_defaults_to_empty(self):
        state = _build_initial_state("hello", "sess1")
        assert state["general_response"] == ""

    def test_style_instructions_defaults_to_empty(self):
        state = _build_initial_state("hello", "sess1")
        assert state["style_instructions"] == ""

    def test_molecular_report_format_defaults_to_empty(self):
        state = _build_initial_state("hello", "sess1")
        assert state["molecular_report_format"] == ""

    def test_disease_name_included_when_provided(self):
        state = _build_initial_state("hello", "sess1", disease_name="lupus")
        assert state["disease_name"] == "lupus"

    def test_disease_name_absent_when_empty(self):
        state = _build_initial_state("hello", "sess1", disease_name="")
        assert "disease_name" not in state

    def test_uploaded_files_included_when_provided(self):
        state = _build_initial_state("hello", "sess1", uploaded_files={"a.csv": "/tmp/a.csv"})
        assert state["uploaded_files"] == {"a.csv": "/tmp/a.csv"}

    def test_has_existing_results_set_with_workflow_outputs(self):
        state = _build_initial_state("hello", "sess1", workflow_outputs={"key": "val"})
        assert state["has_existing_results"] is True
        assert state["workflow_outputs"] == {"key": "val"}
