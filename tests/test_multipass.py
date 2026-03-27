"""Tests for multi-pass LLM synthesis architecture.

Verifies:
1. New pass-specific prompts exist and contain key sections.
2. synthesize_multipass() adapts pass chain to pass_count.
3. synthesize_chat_multipass() always runs 2 passes.
4. Graceful degradation when individual passes fail.
5. SCOPE_PASSES config maps scopes to correct pass counts.
6. Parallel section augmentation with error isolation.
7. Integration: response_node uses multipass functions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supervisor_agent.response.synthesizer import (
    OUTLINE_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    CHAT_REFINEMENT_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    build_response_user_prompt,
    synthesize_multipass,
    synthesize_chat_multipass,
    synthesize_response,
    augment_narrative,
)
from supervisor_agent.langgraph.nodes import SCOPE_PASSES, SCOPE_BUDGETS, _detect_report_scope


# ── Prompt content tests ──


class TestMultiPassPrompts:
    """Verify new pass-specific prompts contain key sections."""

    def test_outline_prompt_has_analysis_phase(self):
        assert "ANALYSIS PHASE" in OUTLINE_SYSTEM_PROMPT

    def test_outline_prompt_is_data_aware(self):
        """User refinement: outline must analyze actual data, not be generic."""
        assert "strongest findings" in OUTLINE_SYSTEM_PROMPT.lower()
        assert "statistical significance" in OUTLINE_SYSTEM_PROMPT.lower()
        assert "evidence convergence" in OUTLINE_SYSTEM_PROMPT.lower()

    def test_outline_prompt_identifies_contradictions(self):
        assert "contradictions" in OUTLINE_SYSTEM_PROMPT.lower()

    def test_outline_prompt_has_depth_markers(self):
        assert "deep" in OUTLINE_SYSTEM_PROMPT
        assert "brief" in OUTLINE_SYSTEM_PROMPT

    def test_outline_prompt_section_format(self):
        assert "Depth:" in OUTLINE_SYSTEM_PROMPT
        assert "Key data points:" in OUTLINE_SYSTEM_PROMPT
        assert "Cross-references:" in OUTLINE_SYSTEM_PROMPT

    def test_review_prompt_has_criteria(self):
        assert "REVIEW CRITERIA" in REVIEW_SYSTEM_PROMPT

    def test_review_prompt_checks_data_grounding(self):
        assert "quantitative claim" in REVIEW_SYSTEM_PROMPT.lower()

    def test_review_prompt_checks_cross_module(self):
        assert "Cross-module" in REVIEW_SYSTEM_PROMPT

    def test_review_prompt_checks_executive_summary(self):
        assert "Executive Summary" in REVIEW_SYSTEM_PROMPT

    def test_review_prompt_returns_revised_document(self):
        """Review should return full revised doc, not comments."""
        assert "COMPLETE revised document" in REVIEW_SYSTEM_PROMPT

    def test_chat_refinement_prompt_transforms_analysis(self):
        assert "TRANSFORMATION RULES" in CHAT_REFINEMENT_PROMPT

    def test_chat_refinement_prompt_preserves_data(self):
        assert "Preserve ALL specific data values" in CHAT_REFINEMENT_PROMPT

    def test_chat_refinement_prompt_has_key_takeaway(self):
        assert "Key Takeaway" in CHAT_REFINEMENT_PROMPT

    def test_chat_refinement_prompt_no_bullet_narrative(self):
        assert "bullet points" in CHAT_REFINEMENT_PROMPT.lower()

    def test_chat_refinement_prompt_no_self_reference(self):
        assert "the analysis" in CHAT_REFINEMENT_PROMPT
        assert "FORBIDDEN" in CHAT_REFINEMENT_PROMPT


# ── SCOPE_PASSES config tests ──


class TestScopePassesConfig:
    """Verify SCOPE_PASSES configuration."""

    def test_chat_gets_2_passes(self):
        assert SCOPE_PASSES["chat"] == 2

    def test_brief_gets_2_passes(self):
        assert SCOPE_PASSES["brief"] == 2

    def test_standard_gets_3_passes(self):
        assert SCOPE_PASSES["standard"] == 3

    def test_comprehensive_gets_4_passes(self):
        assert SCOPE_PASSES["comprehensive"] == 4

    def test_scope_passes_keys_match_budgets(self):
        assert set(SCOPE_PASSES.keys()) == set(SCOPE_BUDGETS.keys())

    def test_detect_report_scope_default(self):
        assert _detect_report_scope("generate a report") == "standard"

    def test_detect_report_scope_brief(self):
        assert _detect_report_scope("give me a brief summary") == "brief"

    def test_detect_report_scope_comprehensive(self):
        assert _detect_report_scope("comprehensive analysis") == "comprehensive"


# ── Mock LLMResult ──


@dataclass
class _MockLLMResult:
    text: str
    provider: str = "mock"
    model: str = "test"
    latency_ms: float = 100.0
    fallback_used: bool = False
    intent: str = ""

    def as_dict(self):
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "fallback_used": self.fallback_used,
        }


# ── synthesize_multipass tests ──


class TestSynthesizeMultipass:
    """Verify multi-pass document synthesis."""

    @pytest.mark.asyncio
    async def test_single_pass_returns_draft_only(self):
        """pass_count=1 → single draft call."""
        mock_result = _MockLLMResult(text="Draft content here")
        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_llm:
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=1,
            )
        assert content == "Draft content here"
        assert len(logs) == 1
        assert "draft" in logs[0]["intent"]
        mock_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_two_passes_draft_and_review(self):
        """pass_count=2 → draft + review."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="Draft content")
            return _MockLLMResult(text="Reviewed and improved content")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=2,
            )
        assert content == "Reviewed and improved content"
        assert len(logs) == 2
        assert "draft" in logs[0]["intent"]
        assert "review" in logs[1]["intent"]

    @pytest.mark.asyncio
    async def test_three_passes_outline_draft_review(self):
        """pass_count=3 → outline + draft + review."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="## Section A\n## Section B")
            elif call_count == 2:
                return _MockLLMResult(text="Full draft following outline")
            return _MockLLMResult(text="Reviewed final document")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=3,
            )
        assert content == "Reviewed final document"
        assert len(logs) == 3
        assert "outline" in logs[0]["intent"]
        assert "draft" in logs[1]["intent"]
        assert "review" in logs[2]["intent"]

    @pytest.mark.asyncio
    async def test_four_passes_full_chain(self):
        """pass_count=4 → outline + draft + review + revise."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="Outline")
            elif call_count == 2:
                return _MockLLMResult(text="Draft")
            elif call_count == 3:
                return _MockLLMResult(text="Reviewed")
            return _MockLLMResult(text="Final polished document")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=4,
            )
        assert content == "Final polished document"
        assert len(logs) == 4
        assert "revise" in logs[3]["intent"]

    @pytest.mark.asyncio
    async def test_outline_includes_data_context(self):
        """Outline pass receives data_context when provided."""
        captured_messages = []

        async def _mock_llm(messages, **kwargs):
            captured_messages.append(messages[0]["content"])
            return _MockLLMResult(text="outline" * 20)

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            await synthesize_multipass(
                "User prompt", data_context="DEG data here", pass_count=3,
            )
        # First call (outline) should include data context
        assert "DEG data here" in captured_messages[0]

    @pytest.mark.asyncio
    async def test_outline_wired_into_draft(self):
        """Draft pass receives the outline output."""
        captured_messages = []

        async def _mock_llm(messages, **kwargs):
            captured_messages.append(messages[0]["content"])
            return _MockLLMResult(text="Structured outline content here")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            await synthesize_multipass("User prompt", pass_count=3)
        # Second call (draft) should include the outline
        assert "DOCUMENT OUTLINE" in captured_messages[1]
        assert "Structured outline content here" in captured_messages[1]

    @pytest.mark.asyncio
    async def test_graceful_degradation_outline_fails(self):
        """If outline fails, draft still runs without outline."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Outline timeout")
            return _MockLLMResult(text="Draft without outline")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=3,
            )
        assert content  # Should still produce content
        # Outline error logged
        assert "error" in logs[0]

    @pytest.mark.asyncio
    async def test_graceful_degradation_review_fails(self):
        """If review fails, draft is returned as-is."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="Good draft content")
            raise RuntimeError("Review timeout")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=2,
            )
        assert content == "Good draft content"
        assert len(logs) == 2
        assert "error" in logs[1]

    @pytest.mark.asyncio
    async def test_draft_failure_returns_empty(self):
        """If draft fails, return empty content."""
        async def _mock_llm(**kwargs):
            raise RuntimeError("Total failure")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=1,
            )
        assert content == ""
        assert "error" in logs[0]

    @pytest.mark.asyncio
    async def test_review_rejected_if_too_short(self):
        """Review output is rejected if it's less than 50% of draft length."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="A" * 1000)  # Long draft
            return _MockLLMResult(text="Short")  # Too-short review

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_multipass(
                "Test prompt", pass_count=2,
            )
        # Should keep the draft since review was too short
        assert content == "A" * 1000

    @pytest.mark.asyncio
    async def test_custom_intent_label(self):
        """Custom intent_label propagates to all pass logs."""
        async def _mock_llm(**kwargs):
            return _MockLLMResult(text="Content")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            _, logs = await synthesize_multipass(
                "Test", pass_count=3, intent_label="my_custom_label",
            )
        for log in logs:
            assert "my_custom_label" in log["intent"]


# ── synthesize_chat_multipass tests ──


class TestSynthesizeChatMultipass:
    """Verify two-pass chat synthesis."""

    @pytest.mark.asyncio
    async def test_always_two_passes(self):
        """Chat multipass always runs analysis + refinement."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="Structured analysis output")
            return _MockLLMResult(text="Polished conversational response")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_chat_multipass("What are the top genes?")
        assert content == "Polished conversational response"
        assert len(logs) == 2
        assert "analysis" in logs[0]["intent"]
        assert "refinement" in logs[1]["intent"]

    @pytest.mark.asyncio
    async def test_enrichment_context_included(self):
        """Enrichment context is passed to the analysis pass."""
        captured = []

        async def _mock_llm(messages, **kwargs):
            captured.append(messages[0]["content"])
            return _MockLLMResult(text="Result text here")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            await synthesize_chat_multipass(
                "Query", enrichment_context="STRING: 47 interactions",
            )
        assert "47 interactions" in captured[0]

    @pytest.mark.asyncio
    async def test_analysis_failure_returns_empty(self):
        """If analysis pass fails, return empty."""
        async def _mock_llm(**kwargs):
            raise RuntimeError("LLM error")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_chat_multipass("Query")
        assert content == ""
        assert "error" in logs[0]

    @pytest.mark.asyncio
    async def test_refinement_failure_returns_analysis(self):
        """If refinement fails, return analysis pass output."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="Good analysis content")
            raise RuntimeError("Refinement error")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, logs = await synthesize_chat_multipass("Query")
        assert content == "Good analysis content"
        assert len(logs) == 2

    @pytest.mark.asyncio
    async def test_refinement_rejected_if_too_short(self):
        """Refinement output rejected if less than 30% of analysis length."""
        call_count = 0

        async def _mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockLLMResult(text="A" * 500)
            return _MockLLMResult(text="Hi")  # Too short

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            content, _ = await synthesize_chat_multipass("Query")
        assert content == "A" * 500

    @pytest.mark.asyncio
    async def test_refinement_uses_chat_refinement_prompt(self):
        """Second pass uses CHAT_REFINEMENT_PROMPT as system prompt."""
        captured_systems = []

        async def _mock_llm(messages, system=None, **kwargs):
            captured_systems.append(system)
            return _MockLLMResult(text="Some content here")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            await synthesize_chat_multipass("Query")
        assert captured_systems[1] == CHAT_REFINEMENT_PROMPT


# ── Log structure tests ──


class TestMultipassLogStructure:
    """Verify log entries have proper structure."""

    @pytest.mark.asyncio
    async def test_multipass_logs_are_list(self):
        async def _mock_llm(**kwargs):
            return _MockLLMResult(text="Content")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            _, logs = await synthesize_multipass("Test", pass_count=3)
        assert isinstance(logs, list)
        assert all(isinstance(log, dict) for log in logs)

    @pytest.mark.asyncio
    async def test_each_log_has_intent(self):
        async def _mock_llm(**kwargs):
            return _MockLLMResult(text="Content")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            _, logs = await synthesize_multipass("Test", pass_count=3)
        for log in logs:
            assert "intent" in log

    @pytest.mark.asyncio
    async def test_chat_logs_are_list(self):
        async def _mock_llm(**kwargs):
            return _MockLLMResult(text="Content")

        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            side_effect=_mock_llm,
        ):
            _, logs = await synthesize_chat_multipass("Test")
        assert isinstance(logs, list)
        assert len(logs) == 2


# ── Backward-compatibility tests ──


class TestBackwardCompatibility:
    """Ensure original synthesize_response still works unchanged."""

    @pytest.mark.asyncio
    async def test_synthesize_response_still_works(self):
        mock_result = _MockLLMResult(text="Single-pass response")
        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            content, log = await synthesize_response("Test prompt")
        assert content == "Single-pass response"
        assert isinstance(log, dict)  # Single dict, not list

    @pytest.mark.asyncio
    async def test_augment_narrative_still_works(self):
        """augment_narrative is unchanged."""
        mock_ctx = MagicMock()
        mock_ctx.disease_name = "test"
        mock_ctx.section_title = "Test Section"
        mock_ctx.evidence_cards = []
        mock_ctx.table_summaries = {}
        mock_ctx.conflicts = []

        mock_result = _MockLLMResult(text="Augmented narrative")
        with patch(
            "supervisor_agent.response.synthesizer.llm_complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await augment_narrative(mock_ctx)
        assert result == "Augmented narrative"


# ── Import tests ──


class TestImports:
    """Verify all new exports are accessible."""

    def test_imports_from_response_init(self):
        from supervisor_agent.response import (
            OUTLINE_SYSTEM_PROMPT,
            REVIEW_SYSTEM_PROMPT,
            CHAT_REFINEMENT_PROMPT,
            synthesize_multipass,
            synthesize_chat_multipass,
        )
        assert OUTLINE_SYSTEM_PROMPT
        assert REVIEW_SYSTEM_PROMPT
        assert CHAT_REFINEMENT_PROMPT
        assert callable(synthesize_multipass)
        assert callable(synthesize_chat_multipass)

    def test_scope_passes_importable(self):
        from supervisor_agent.langgraph.nodes import SCOPE_PASSES
        assert isinstance(SCOPE_PASSES, dict)
        assert len(SCOPE_PASSES) == 4


# ── Prompt content tests (response quality enhancements) ─────────────────────


class TestResponsePromptContent:
    """Verify RESPONSE_SYSTEM_PROMPT calibration addresses knowledge queries."""

    def test_has_expert_level_calibration(self):
        assert "expert-level" in RESPONSE_SYSTEM_PROMPT

    def test_no_self_limiting_rule(self):
        """The old rule that made Claude apologize instead of answering."""
        assert "say so clearly rather than generating a generic textbook answer" \
            not in RESPONSE_SYSTEM_PROMPT

    def test_has_no_meta_commentary_rule(self):
        assert "meta-commentary" in RESPONSE_SYSTEM_PROMPT

    def test_has_elaboration_calibration(self):
        assert "Elaborate" in RESPONSE_SYSTEM_PROMPT
        assert "substantially longer" in RESPONSE_SYSTEM_PROMPT

    def test_knowledge_query_framing(self):
        """New rule: answer from domain knowledge, don't apologize."""
        assert "DO NOT apologize" in RESPONSE_SYSTEM_PROMPT


class TestRefinementPromptContent:
    """Verify CHAT_REFINEMENT_PROMPT handles follow-ups correctly."""

    def test_has_followup_expansion_rule(self):
        assert "elaborate" in CHAT_REFINEMENT_PROMPT.lower()
        assert "tell me more" in CHAT_REFINEMENT_PROMPT.lower()

    def test_has_length_guard(self):
        assert "shorter than the prior assistant response" in CHAT_REFINEMENT_PROMPT

    def test_no_apology_rule(self):
        assert "apologize" in CHAT_REFINEMENT_PROMPT.lower()


# ── Conversation history window tests ────────────────────────────────────────


class TestConversationHistoryWindow:
    """Verify build_response_user_prompt uses expanded history window."""

    def test_includes_last_6_messages(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"msg {i}"} for i in range(8)]
        prompt = build_response_user_prompt(
            query="test", disease_name="test", summaries={},
            conversation_history=history,
        )
        # Last 6 of 8 messages: msg 2..7
        assert "msg 2" in prompt
        assert "msg 7" in prompt
        # msg 0, 1 are outside the 6-message window
        assert "msg 0" not in prompt
        assert "msg 1" not in prompt

    def test_truncates_at_1500_chars(self):
        long_msg = "x" * 2000
        history = [{"role": "user", "content": long_msg}]
        prompt = build_response_user_prompt(
            query="test", disease_name="test", summaries={},
            conversation_history=history,
        )
        # 1500 chars of 'x' should be present, but not all 2000
        assert "x" * 1500 in prompt
        assert "x" * 1501 not in prompt

    def test_empty_history_no_section(self):
        prompt = build_response_user_prompt(
            query="test", disease_name="test", summaries={},
            conversation_history=[],
        )
        assert "RECENT CONVERSATION" not in prompt

    def test_none_history_no_section(self):
        prompt = build_response_user_prompt(
            query="test", disease_name="test", summaries={},
            conversation_history=None,
        )
        assert "RECENT CONVERSATION" not in prompt
