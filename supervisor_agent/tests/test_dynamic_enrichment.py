"""Tests for upgraded synthesizer prompts, builders, and document renderer.

Verifies:
1. Enhanced prompt content (key sections present)
2. New prompt builder signatures and output structure
3. Adaptive budget parameter in collect_output_summaries
4. Report scope detection
5. Enhanced _build_html() semantics
6. Enhanced BASE_CSS structural properties
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from supervisor_agent.response.synthesizer import (
    DOCUMENT_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    STRUCTURED_REPORT_PROMPT,
    build_document_user_prompt,
    build_response_user_prompt,
)
from supervisor_agent.response.style_engine import BASE_CSS, build_css
from supervisor_agent.response.document_renderer import _build_html


# ── Prompt content tests ──


class TestEnhancedPrompts:
    """Verify enhanced prompts contain key sections from enhanced_prompts.py."""

    def test_response_prompt_has_calibration(self):
        assert "RESPONSE CALIBRATION" in RESPONSE_SYSTEM_PROMPT

    def test_response_prompt_has_data_grounding(self):
        assert "DATA GROUNDING RULES" in RESPONSE_SYSTEM_PROMPT

    def test_response_prompt_has_cross_module(self):
        assert "CROSS-MODULE SYNTHESIS" in RESPONSE_SYSTEM_PROMPT

    def test_response_prompt_has_analytical_enrichment(self):
        assert "ANALYTICAL ENRICHMENT" in RESPONSE_SYSTEM_PROMPT

    def test_response_prompt_no_generic_disclaimers(self):
        assert "consult a physician" in RESPONSE_SYSTEM_PROMPT  # Warned against

    def test_document_prompt_has_structure(self):
        assert "DOCUMENT STRUCTURE" in DOCUMENT_SYSTEM_PROMPT

    def test_document_prompt_has_rules(self):
        assert "DOCUMENT RULES" in DOCUMENT_SYSTEM_PROMPT

    def test_document_prompt_has_data_grounding(self):
        assert "DATA GROUNDING" in DOCUMENT_SYSTEM_PROMPT

    def test_document_scales_to_data(self):
        assert "Single module" in DOCUMENT_SYSTEM_PROMPT
        assert "8-12 pages" in DOCUMENT_SYSTEM_PROMPT

    def test_structured_prompt_has_evidence_tracing(self):
        assert "evidence card" in STRUCTURED_REPORT_PROMPT.lower()

    def test_structured_prompt_has_section_guidance(self):
        assert "SECTION-SPECIFIC GUIDANCE" in STRUCTURED_REPORT_PROMPT

    def test_structured_prompt_has_critical_constraint(self):
        assert "CRITICAL CONSTRAINT" in STRUCTURED_REPORT_PROMPT


# ── Prompt builder tests (new signatures) ──


class TestBuildResponseUserPrompt:
    """Verify new build_response_user_prompt signature and output."""

    def test_basic_call(self):
        result = build_response_user_prompt(
            query="What are the top genes?",
            disease_name="lupus",
            summaries={"deg": "STAT1,TNF,IL6"},
        )
        assert "What are the top genes?" in result
        assert "lupus" in result
        assert "STAT1" in result

    def test_with_enrichment_data(self):
        result = build_response_user_prompt(
            query="top genes",
            disease_name="lupus",
            summaries={"deg": "gene data"},
            enrichment_data={"Gene Annotations (Ensembl)": "STAT1 | protein_coding"},
        )
        assert "EXTERNAL ANNOTATIONS" in result
        assert "ENSEMBL" in result.upper()

    def test_with_modules(self):
        result = build_response_user_prompt(
            query="summarize",
            disease_name="lupus",
            summaries={"deg": "data"},
            available_modules=["deg", "pathway"],
            failed_modules=["drug"],
        )
        assert "PIPELINE STATUS" in result
        assert "deg" in result
        assert "drug" in result

    def test_with_conversation_history(self):
        result = build_response_user_prompt(
            query="follow up",
            disease_name="",
            summaries={},
            conversation_history=[
                {"role": "user", "content": "What about TNF?"},
                {"role": "assistant", "content": "TNF is a key inflammatory gene."},
            ],
        )
        assert "RECENT CONVERSATION" in result
        assert "TNF" in result

    def test_empty_enrichment_excluded(self):
        result = build_response_user_prompt(
            query="test",
            disease_name="",
            summaries={"data": "stuff"},
            enrichment_data={},
        )
        assert "EXTERNAL ANNOTATIONS" not in result

    def test_none_enrichment_excluded(self):
        result = build_response_user_prompt(
            query="test",
            disease_name="",
            summaries={"data": "stuff"},
            enrichment_data=None,
        )
        assert "EXTERNAL ANNOTATIONS" not in result


class TestBuildDocumentUserPrompt:
    """Verify new build_document_user_prompt signature and output."""

    def test_basic_call(self):
        result = build_document_user_prompt(
            query="Generate a PDF report",
            disease_name="breast cancer",
            summaries={"deg": "gene data"},
        )
        assert "Generate a PDF report" in result
        assert "breast cancer" in result
        assert "REPORT SCOPE: standard" in result

    def test_brief_scope(self):
        result = build_document_user_prompt(
            query="brief",
            disease_name="lupus",
            summaries={},
            report_scope="brief",
        )
        assert "REPORT SCOPE: brief" in result
        assert "SCOPE NOTE" in result
        assert "concise" in result.lower()

    def test_comprehensive_scope(self):
        result = build_document_user_prompt(
            query="comprehensive report",
            disease_name="lupus",
            summaries={},
            report_scope="comprehensive",
        )
        assert "REPORT SCOPE: comprehensive" in result
        assert "8-12 pages" in result

    def test_with_enrichment(self):
        result = build_document_user_prompt(
            query="pdf",
            disease_name="lupus",
            summaries={"deg": "data"},
            enrichment_data={"ChEMBL": "drug info"},
        )
        assert "EXTERNAL ANNOTATIONS" in result

    def test_with_style_instructions(self):
        result = build_document_user_prompt(
            query="pdf",
            disease_name="lupus",
            summaries={},
            style_instructions="dark mode, serif font",
        )
        assert "STYLING INSTRUCTIONS" in result
        assert "dark mode" in result

    def test_module_count_in_output(self):
        result = build_document_user_prompt(
            query="pdf",
            disease_name="lupus",
            summaries={"deg": "data1", "pathway": "data2", "empty": ""},
        )
        # Should count only non-empty modules
        assert "2 modules" in result


# ── Enhanced BASE_CSS tests ──


class TestEnhancedCSS:
    """Verify structural CSS improvements."""

    def test_page_break_inside_avoid_on_tables(self):
        assert "page-break-inside: avoid" in BASE_CSS

    def test_orphans_widows(self):
        assert "orphans: 2" in BASE_CSS
        assert "widows: 2" in BASE_CSS

    def test_report_body_class(self):
        assert ".report-body" in BASE_CSS

    def test_report_footer_class(self):
        assert ".report-footer" in BASE_CSS

    def test_section_divider_class(self):
        assert ".section-divider" in BASE_CSS

    def test_blockquote_styled(self):
        assert "blockquote" in BASE_CSS

    def test_print_media_query(self):
        assert "@media print" in BASE_CSS

    def test_heading_page_break_after_avoid(self):
        assert "page-break-after: avoid" in BASE_CSS

    def test_build_css_still_works(self):
        css = build_css("default", "A4", "h1 { color: red !important; }")
        assert "color: red" in css
        assert "--color-primary" in css


# ── Enhanced _build_html() tests ──


class TestEnhancedBuildHtml:
    """Verify semantic HTML structure improvements."""

    def test_has_lang_attribute(self):
        html = _build_html("Test", "", "# Hello")
        assert "lang='en'" in html or 'lang="en"' in html

    def test_has_report_body_main(self):
        html = _build_html("Test", "", "content")
        assert '<main class="report-body">' in html

    def test_has_report_footer(self):
        html = _build_html("Test", "", "content")
        assert '<div class="report-footer">' in html

    def test_still_renders_markdown(self):
        html = _build_html("Test", "", "## Section\n\nSome text")
        assert "<h2" in html

    def test_disease_context(self):
        html = _build_html("Test", "lupus", "content")
        assert "lupus" in html

    def test_auto_landscape_wide_tables(self):
        wide_table = "| a | b | c | d | e | f | g | h | i |\n" + "| - " * 9 + "|\n"
        html = _build_html("Test", "", wide_table)
        assert "landscape" in html


# ── Report scope detection tests ──


class TestReportScopeDetection:
    """Test _detect_report_scope from nodes.py."""

    def test_import(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope is not None

    def test_brief(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope("give me a brief summary") == "brief"

    def test_comprehensive(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope("generate a comprehensive analysis") == "comprehensive"

    def test_detailed(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope("I want a detailed report") == "comprehensive"

    def test_default_standard(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope("generate a pdf report for lupus") == "standard"

    def test_full(self):
        from supervisor_agent.langgraph.nodes import _detect_report_scope
        assert _detect_report_scope("give me the full analysis") == "comprehensive"


# ── Scope budgets test ──


class TestScopeBudgets:
    def test_budgets_defined(self):
        from supervisor_agent.langgraph.nodes import SCOPE_BUDGETS
        assert SCOPE_BUDGETS["chat"] == 8000
        assert SCOPE_BUDGETS["brief"] == 10000
        assert SCOPE_BUDGETS["standard"] == 15000
        assert SCOPE_BUDGETS["comprehensive"] == 20000


# ── CSS brace-balance tests ──


class TestCssBraceBalance:
    """Regression: truncated LLM CSS must not produce unclosed rules."""

    def test_balanced_css_unchanged(self):
        from supervisor_agent.response.style_engine import _balance_braces
        css = "h1 { color: red; } h2 { font-size: 14pt; }"
        assert _balance_braces(css) == css

    def test_truncated_rule_stripped(self):
        from supervisor_agent.response.style_engine import _balance_braces
        css = "h1 { color: red; } h2 { font-size: 14pt;"
        result = _balance_braces(css)
        assert result == "h1 { color: red; }"
        assert result.count("{") == result.count("}")

    def test_empty_string(self):
        from supervisor_agent.response.style_engine import _balance_braces
        assert _balance_braces("") == ""

    def test_fully_broken_no_closing(self):
        from supervisor_agent.response.style_engine import _balance_braces
        # No balanced closing brace at all — returned as-is (nothing to truncate to)
        css = "h1 { color: red;"
        result = _balance_braces(css)
        assert result == css  # no last_balanced found

    def test_sanitize_css_fixes_truncation(self):
        from supervisor_agent.response.style_engine import sanitize_css
        css = "h1 { color: red !important; } h2 { font-size:"
        result = sanitize_css(css)
        assert result.count("{") == result.count("}")


# ── Empty-data PDF downgrade tests ──


class TestEmptyDataPdfDisclaimer:
    """When no pipeline data exists, PDF still renders but with a disclaimer."""

    def test_has_pipeline_data_guard(self):
        import inspect
        from supervisor_agent.langgraph.nodes import _response_node_inner
        src = inspect.getsource(_response_node_inner)
        assert "has_pipeline_data" in src

    def test_disclaimer_prepended_not_downgraded(self):
        """Verify the guard adds a disclaimer instead of blocking PDF generation."""
        import inspect
        from supervisor_agent.langgraph.nodes import _response_node_inner
        src = inspect.getsource(_response_node_inner)
        # Should NOT downgrade fmt to chat
        assert 'fmt = "chat"' not in src or 'fmt = "chat"' in src.split("has_pipeline_data")[0]
        # Should prepend a disclaimer note
        assert "Note:" in src
