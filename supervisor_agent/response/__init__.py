"""Response module — synthesis, query functions, data discovery, document rendering.

Business logic for the supervisor's response_node. LangGraph nodes import from
this package instead of implementing inline.
"""

from .data_discovery import (
    collect_output_summaries,
    classify_csv_type,
    discover_relevant_csvs,
)
from .document_renderer import render_docx, render_pdf
from .query_functions import df_to_table, regex_fast_path
from .style_engine import extract_style_instructions, has_style_intent, generate_style_css
from .synthesizer import (
    DOCUMENT_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    STRUCTURED_REPORT_PROMPT,
    OUTLINE_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    CHAT_REFINEMENT_PROMPT,
    build_document_user_prompt,
    build_response_user_prompt,
    synthesize_response,
    synthesize_multipass,
    synthesize_chat_multipass,
    augment_narrative,
)
from .enrichment_dispatcher import enrich_for_response

__all__ = [
    "collect_output_summaries",
    "classify_csv_type",
    "discover_relevant_csvs",
    "render_docx",
    "render_pdf",
    "df_to_table",
    "regex_fast_path",
    "extract_style_instructions",
    "has_style_intent",
    "generate_style_css",
    "RESPONSE_SYSTEM_PROMPT",
    "DOCUMENT_SYSTEM_PROMPT",
    "STRUCTURED_REPORT_PROMPT",
    "OUTLINE_SYSTEM_PROMPT",
    "REVIEW_SYSTEM_PROMPT",
    "CHAT_REFINEMENT_PROMPT",
    "build_response_user_prompt",
    "build_document_user_prompt",
    "synthesize_response",
    "synthesize_multipass",
    "synthesize_chat_multipass",
    "augment_narrative",
    "enrich_for_response",
]
