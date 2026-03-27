"""Node functions for the supervisor LangGraph workflow.

Each function receives (state, config) and returns a partial state dict.
Nodes delegate to existing modules — no business logic is duplicated here.
"""

import asyncio
import functools
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from langchain_core.runnables import RunnableConfig

from ..agent_registry import (
    AGENT_REGISTRY, AgentType, FILE_TYPE_TO_INPUT_KEY,
)
from ..llm_provider import llm_complete
from ..response import (
    collect_output_summaries,
    classify_csv_type,
    discover_relevant_csvs,
    regex_fast_path,
    build_response_user_prompt,
    build_document_user_prompt,
    synthesize_response,
    synthesize_multipass,
    synthesize_chat_multipass,
    augment_narrative,
    render_pdf,
    render_docx,
    extract_style_instructions,
    has_style_intent,
    generate_style_css,
    enrich_for_response,
    DOCUMENT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)
from ..executors import (
    StatusType, StatusUpdate,
    execute_cohort_retrieval,
    execute_deg_analysis,
    execute_gene_prioritization,
    execute_pathway_enrichment,
    execute_deconvolution,
    execute_temporal_analysis,
    execute_harmonization,
    execute_mdp_analysis,
    execute_perturbation_analysis,
    execute_multiomics_integration,
    execute_fastq_processing,
    execute_molecular_report,
    execute_crispr_perturb_seq,
    execute_crispr_screening,
    execute_crispr_targeted,
    execute_causality,
)
from ..data_layer.manifest_builder import manifest_from_state
from ..data_layer.registry_builder import build_artifact_index
from ..data_layer.schemas.sections import NarrativeMode, ReportingConfig
from ..data_layer.schemas.evidence import NarrativeContext
from ..reporting_engine.planner import ReportPlanner
from ..reporting_engine.builders.base import BUILDER_REGISTRY
from ..reporting_engine.evidence import score_findings, detect_conflicts
from ..reporting_engine.validation import ValidationGuard, ValidationError
from ..reporting_engine.enrichment import ReportEnricher
from ..reporting_engine.renderers.markdown_renderer import render_markdown
from ..router import IntentRouter
from ..state import ConversationState, UploadedFile
from ..supervisor import (
    _detect_file_type,
    _build_execution_chain,
    _extract_contextual_inputs,
    _validate_required_input_paths,
)
from .state import SupervisorGraphState

logger = logging.getLogger(__name__)

EXECUTOR_MAP = {
    AgentType.COHORT_RETRIEVAL:       execute_cohort_retrieval,
    AgentType.DEG_ANALYSIS:           execute_deg_analysis,
    AgentType.GENE_PRIORITIZATION:    execute_gene_prioritization,
    AgentType.PATHWAY_ENRICHMENT:     execute_pathway_enrichment,
    AgentType.DECONVOLUTION:          execute_deconvolution,
    AgentType.TEMPORAL_ANALYSIS:      execute_temporal_analysis,
    AgentType.HARMONIZATION:          execute_harmonization,
    AgentType.MDP_ANALYSIS:           execute_mdp_analysis,
    AgentType.PERTURBATION_ANALYSIS:  execute_perturbation_analysis,
    AgentType.MULTIOMICS_INTEGRATION: execute_multiomics_integration,
    AgentType.FASTQ_PROCESSING:       execute_fastq_processing,
    AgentType.MOLECULAR_REPORT:       execute_molecular_report,
    AgentType.CRISPR_PERTURB_SEQ:     execute_crispr_perturb_seq,
    AgentType.CRISPR_SCREENING:       execute_crispr_screening,
    AgentType.CRISPR_TARGETED:        execute_crispr_targeted,
    AgentType.CAUSALITY:                execute_causality,
}

# Keys that get seeded into workflow_outputs from uploads alone (no agent ran).
# Used to avoid false positives in has_existing_results.
_UPLOAD_DERIVED_KEYS = frozenset(
    k for keys in FILE_TYPE_TO_INPUT_KEY.values() for k in keys
)


# ── Cached IntentRouter (builds OpenAI client + capabilities text once) ──


@functools.lru_cache(maxsize=1)
def get_intent_router() -> IntentRouter:
    return IntentRouter()


def detect_file_type(filename: str, filepath: str) -> str:
    """Standalone wrapper around the supervisor's file-type detection."""
    return _detect_file_type(filepath)


# Strict patterns for metadata files — avoids false positives on
# "metabolomics", "metastasis", "meta_analysis", etc.
_METADATA_RE = re.compile(r'(^|[_\-])metadata([_\-.]|$)', re.IGNORECASE)

# ── Reporting intent detection ──

_REPORTING_SIGNALS = re.compile(
    r'\b(what are|what is|what\'s|show me|summarize|summary|findings|'
    r'results for|results of|tell me|explain|describe|list the|'
    r'which genes|which pathways|which drugs|how many|top\s+\d+|'
    r'report|overview|compare|between|'
    r'generate|create|make|produce)\b', re.IGNORECASE,
)

# Follow-up / elaboration signals — routes through response_node for
# richer LLM synthesis instead of short-circuiting to END.
_FOLLOWUP_SIGNALS = re.compile(
    r'\b(elaborate|more detail|more about|tell me more|expand on|'
    r'can you explain|go deeper|in more detail|further|'
    r'sources?|how do you know|what.*based on|'
    r'continue|keep going)\b', re.IGNORECASE,
)

# Molecular report triggers — routes to the ReportingPipelineAgent, NOT the
# dynamic style-engine report renderer.
# Allow optional format word (pdf/docx/word) between qualifier and "report"
_FMT_WORD = r'(?:(?:pdf|docx?|word)\s+)?'
_MOLECULAR_REPORT_RE = re.compile(
    rf'\b(molecular\s+(?:analysis\s+)?{_FMT_WORD}report|'
    rf'patient\s+{_FMT_WORD}report|'
    rf'comprehensive\s+{_FMT_WORD}report|'
    rf'full\s+{_FMT_WORD}report)\b',
    re.IGNORECASE,
)
_EXEC_ONLY_START = re.compile(
    r'^(run|execute|perform|start|do|analyze|process)\b', re.IGNORECASE,
)
_TOP_N_RE = re.compile(r'\btop\s+(\d+)\b', re.IGNORECASE)
_ALL_LIST_RE = re.compile(
    r'\b(all|complete|full|entire)\s+(list|genes|pathways|drugs|results)\b',
    re.IGNORECASE,
)
_FORMAT_PDF = re.compile(r'\bpdf\b', re.IGNORECASE)
_FORMAT_DOCX = re.compile(r'\b(docx?|word\s+doc(ument)?)\b', re.IGNORECASE)
_IMPLICIT_DOC_RE = re.compile(
    r'\b(generate|create|make|produce|build|write)\b.{0,30}\breport\b',
    re.IGNORECASE,
)

# Short affirmative replies that confirm the previous turn's pending action
_CONFIRMATION_RE = re.compile(
    r'^(ok|okay|yes|sure|go\s*ahead|confirm(ed)?|proceed|do\s*it|'
    r'yeah|yep|yea|y|please|please\s+do|sounds?\s+good|lgtm)\b',
    re.IGNORECASE,
)

_THEME_RE = {
    "clinical": re.compile(r'\b(clinical|medical|formal|pharma)\b', re.IGNORECASE),
    "minimal": re.compile(r'\b(minimal|simple|plain|clean|basic)\b', re.IGNORECASE),
}


def _detect_report_theme(query: str) -> str:
    for theme, pattern in _THEME_RE.items():
        if pattern.search(query):
            return theme
    return "default"


def _detect_needs_response(query: str) -> bool:
    if _REPORTING_SIGNALS.search(query):
        return True
    if _FOLLOWUP_SIGNALS.search(query):
        return True
    if _EXEC_ONLY_START.search(query):
        return False
    return False


def _detect_response_format(query: str, needs_response: bool) -> str:
    if _FORMAT_PDF.search(query):
        return "pdf"
    if _FORMAT_DOCX.search(query):
        return "docx"
    if _IMPLICIT_DOC_RE.search(query):
        return "pdf"
    if not needs_response:
        return "none"
    return "chat"


def _extract_top_n(query: str) -> int:
    m = _TOP_N_RE.search(query)
    if m:
        return min(int(m.group(1)), 100)
    if _ALL_LIST_RE.search(query):
        return 100
    return 10


def _build_available_inputs(state: SupervisorGraphState) -> Dict[str, Any]:
    """Merge workflow outputs + uploaded files + params into executor inputs."""
    inputs = dict(state.get("workflow_outputs", {}))

    for filename, filepath in state.get("uploaded_files", {}).items():
        ftype = state.get("detected_file_types", {}).get(filename)
        for key in FILE_TYPE_TO_INPUT_KEY.get(ftype, []):
            inputs.setdefault(key, filepath)

        # Filename-heuristic fallbacks for keys FILE_TYPE_TO_INPUT_KEY can't cover
        name_lower = filename.lower()
        if _METADATA_RE.search(name_lower):
            inputs.setdefault("metadata_file", filepath)
        if name_lower.endswith(".h5ad"):
            inputs.setdefault("h5ad_file", filepath)
        if ftype in ("deg_results", "prioritized_genes"):
            inputs.setdefault("deg_base_dir", str(Path(filepath).parent))

        inputs[filename] = filepath

    if state.get("disease_name"):
        inputs["disease_name"] = state["disease_name"]
    inputs.update(state.get("routing_decision", {}).get("extracted_params", {}))

    # Pass through context needed by molecular report executor
    inputs.setdefault("user_query", state.get("user_query", ""))
    mol_fmt = state.get("molecular_report_format")
    if mol_fmt:
        inputs["report_output_format"] = mol_fmt

    return inputs


def _get_progress_callback(config: RunnableConfig) -> Optional[Callable]:
    return config.get("configurable", {}).get("progress_callback")


# ── Graph nodes ──


async def intent_node(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Layer 1 — Route user query, detect files, determine reporting intent."""
    query = state["user_query"]
    try:
        return await _intent_node_inner(state, config)
    except Exception as exc:
        logger.exception("intent_node crashed — degrading to general chat: %s", exc)
        return {
            "routing_decision": {"agent_type": None, "is_general_query": True,
                                 "reasoning": f"intent error: {exc}"},
            "detected_file_types": {},
            "disease_name": state.get("disease_name", ""),
            "is_general_query": True,
            "general_response": "",
            "final_response": "",
            "needs_response": True,
            "response_format": "chat",
            "requested_top_n": 10,
            "report_theme": "default",
            "style_instructions": "",
            "has_existing_results": bool(state.get("workflow_outputs")),
            "conversation_history": [{"role": "user", "content": query}],
            "status": "routed",
        }


async def _intent_node_inner(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Inner implementation of intent_node — may raise."""
    query = state["user_query"]
    uploaded = state.get("uploaded_files", {})
    detected = {name: detect_file_type(name, path) for name, path in uploaded.items()}

    # ── Confirmation fast-path: replay previous turn's pending action ──
    # MemorySaver checkpoint retains prior turn's response_format/top_n/needs_response
    # before this node overwrites them. Short confirmations replay that intent.
    query_stripped = query.strip()
    prev_fmt = state.get("response_format", "none")
    if (len(query_stripped) < 30
            and _CONFIRMATION_RE.match(query_stripped)
            and prev_fmt in ("pdf", "docx")):
        logger.info(f"intent_node: confirmation detected, replaying fmt={prev_fmt}")
        callback = _get_progress_callback(config)
        if callback:
            callback(StatusUpdate(
                status_type=StatusType.ROUTING,
                title="🔀 Routing decision",
                message=f"Confirmed → generating {prev_fmt.upper()} report",
                agent_name="intent",
            ))
        return {
            "routing_decision": {"agent_type": None, "is_general_query": True,
                                 "reasoning": "confirmation → response"},
            "detected_file_types": detected,
            "disease_name": state.get("disease_name", ""),
            "is_general_query": True,
            "general_response": "",
            "final_response": "",
            "needs_response": True,
            "response_format": prev_fmt,
            "requested_top_n": state.get("requested_top_n", 10),
            "report_theme": state.get("report_theme", "default"),
            "style_instructions": extract_style_instructions(
                query, prev_fmt, state.get("disease_name", ""),
            ) or state.get("style_instructions", ""),
            "has_existing_results": True,
            "conversation_history": [{"role": "user", "content": query}],
            "status": "routed",
        }

    # ConversationState shim for IntentRouter.route() compatibility
    conv_state = ConversationState(session_id=state.get("session_id", ""))
    conv_state.current_disease = state.get("disease_name")
    conv_state.workflow_state = dict(state.get("workflow_outputs", {}))
    for name, path in uploaded.items():
        try:
            fsize = Path(path).stat().st_size
        except OSError:
            fsize = 0
        conv_state.uploaded_files[name] = UploadedFile(
            filename=name, filepath=path,
            file_type=detected.get(name, "unknown"),
            upload_time=time.time(),
            size_bytes=fsize,
        )

    # Replay accumulated conversation history so router sees prior turns
    for turn in state.get("conversation_history", []):
        role = turn.get("role", "user")
        text = turn.get("content", "")
        if role == "user":
            conv_state.add_user_message(text)
        else:
            conv_state.add_assistant_message(text)

    decision = await get_intent_router().route(query, conv_state)

    relevant_input_names = set()
    if decision.is_multi_agent and decision.agent_pipeline:
        for intent in decision.agent_pipeline:
            agent_info = AGENT_REGISTRY.get(intent.agent_type)
            if not agent_info:
                continue
            relevant_input_names.update(inp.name for inp in agent_info.required_inputs)
            relevant_input_names.update(inp.name for inp in agent_info.optional_inputs)
    elif decision.agent_type and decision.agent_type in AGENT_REGISTRY:
        agent_info = AGENT_REGISTRY[decision.agent_type]
        relevant_input_names.update(inp.name for inp in agent_info.required_inputs)
        relevant_input_names.update(inp.name for inp in agent_info.optional_inputs)

    decision.extracted_params = {
        **_extract_contextual_inputs(query, relevant_input_names),
        **decision.extracted_params,
    }

    # Prefer freshly-extracted disease from the current query.
    # For general queries with no extracted disease, DON'T inherit
    # a stale disease from the checkpoint — the user changed topic.
    extracted_disease = decision.extracted_params.get("disease_name") or ""
    _wants_doc = (_FORMAT_PDF.search(query) or _FORMAT_DOCX.search(query)
                  or _IMPLICIT_DOC_RE.search(query))
    if extracted_disease:
        disease = extracted_disease
    elif decision.is_general_query and _wants_doc:
        # Report-generation follow-up ("generate a pdf for same") → keep prior disease
        disease = state.get("disease_name", "")
    elif decision.is_general_query:
        # General query with no disease mention → don't carry forward stale context
        disease = ""
    else:
        # Agent-routed query (e.g. "run pathway enrichment") → keep prior disease
        disease = state.get("disease_name", "")

    # Emit routing status so Streamlit can display the decision
    callback = _get_progress_callback(config)
    if callback:
        agents_str = (
            ", ".join(
                a.agent_name if hasattr(a, "agent_name") else a.get("agent_name", "?")
                for a in (decision.agent_pipeline or [])
            )
            if decision.is_multi_agent
            else (decision.agent_name or "general")
        )
        callback(StatusUpdate(
            status_type=StatusType.ROUTING,
            title="🔀 Routing decision",
            message=f"Route → {agents_str}: {decision.reasoning}" if decision.reasoning else f"Route → {agents_str}",
            agent_name="intent",
        ))

    needs_resp = _detect_needs_response(query)
    resp_fmt = _detect_response_format(query, needs_resp)
    if resp_fmt in ("pdf", "docx"):
        needs_resp = True

    # Style carry-forward: re-use prior document format when user requests
    # styling changes without explicitly mentioning "pdf"/"docx" again.
    # Gate: only trigger if the query contains genuine style keywords
    # (colors, fonts, layout, themes) — not arbitrary scientific text.
    prev_fmt = state.get("response_format", "none")
    if resp_fmt not in ("pdf", "docx") and prev_fmt in ("pdf", "docx"):
        if has_style_intent(query):
            style_probe = extract_style_instructions(query, prev_fmt, disease)
            if style_probe:
                logger.info("intent_node: style carry-forward %s → %s (style=%r)",
                            resp_fmt, prev_fmt, style_probe)
                resp_fmt = prev_fmt
                needs_resp = True

    # Molecular report: detect format, default to PDF
    is_molecular = _MOLECULAR_REPORT_RE.search(query)
    mol_fmt = ""
    if is_molecular:
        if _FORMAT_DOCX.search(query):
            mol_fmt = "docx"
        else:
            mol_fmt = "pdf"
        resp_fmt = "none"
        needs_resp = True

    # Populate workflow_outputs from uploaded files so response_node can find them
    wo_from_uploads: Dict[str, Any] = {}
    for fname, fpath in uploaded.items():
        ftype = detected.get(fname)
        for key in FILE_TYPE_TO_INPUT_KEY.get(ftype, []):
            wo_from_uploads.setdefault(key, fpath)

    return {
        "routing_decision": decision.to_dict(),
        "detected_file_types": detected,
        "disease_name": disease,
        "is_general_query": decision.is_general_query,
        "general_response": decision.suggested_response or "",
        "final_response": "",  # clear stale turn-N-1 response from checkpoint
        "needs_response": needs_resp,
        "response_format": resp_fmt,
        "molecular_report_format": mol_fmt,
        "report_theme": _detect_report_theme(query),
        "style_instructions": extract_style_instructions(query, resp_fmt, disease),
        "requested_top_n": _extract_top_n(query),
        "has_existing_results": bool(
            {k for k in state.get("workflow_outputs", {}) if k not in _UPLOAD_DERIVED_KEYS}
        ),
        "workflow_outputs": wo_from_uploads,
        "conversation_history": (
            [{"role": "user", "content": query},
             {"role": "assistant", "content": decision.suggested_response}]
            if decision.is_general_query and decision.suggested_response
            else [{"role": "user", "content": query}]
        ),
        "status": "routed",
    }


async def plan_node(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Layer 2 — Build execution plan from routing decision."""
    rd = state.get("routing_decision", {})

    # Guard: molecular_report requires explicit intent (regex match in intent_node).
    # If the LLM router over-routed a generic "I need a PDF report" here, fall
    # through to the dynamic renderer instead.
    if not state.get("molecular_report_format") and rd.get("agent_type") == AgentType.MOLECULAR_REPORT.value:
        return {"execution_plan": [], "status": "completed"}

    # Multi-agent pipeline already ordered by router
    if rd.get("is_multi_agent") and rd.get("agent_pipeline"):
        agents = [a["agent_type"] for a in rd["agent_pipeline"]]
        if not state.get("molecular_report_format"):
            agents = [a for a in agents if a != AgentType.MOLECULAR_REPORT.value]

        # Skip agents whose outputs are already available from uploads
        available_keys = set(state.get("workflow_outputs", {}).keys())
        for ftype in state.get("detected_file_types", {}).values():
            for key in FILE_TYPE_TO_INPUT_KEY.get(ftype, []):
                available_keys.add(key)
        filtered = []
        for a in agents:
            info = AGENT_REGISTRY.get(AgentType(a))
            if info and info.produces and all(k in available_keys for k in info.produces):
                continue
            filtered.append(a)
            if info:
                available_keys |= set(info.produces)

        return {
            "execution_plan": filtered,
            "agents_skipped": [],
            "current_agent_index": 0,
            "status": "planned",
        }

    # Single agent — derive full upstream chain
    agent_type_val = rd.get("agent_type")
    if not agent_type_val:
        return {"execution_plan": [], "status": "completed"}

    target = AgentType(agent_type_val)

    # Merge workflow outputs + file-type-implied keys into available set
    available_keys = set(state.get("workflow_outputs", {}).keys())
    detected = state.get("detected_file_types", {})
    for ftype in detected.values():
        for key in FILE_TYPE_TO_INPUT_KEY.get(ftype, []):
            available_keys.add(key)

    primary_type = next(iter(detected.values()), None) if detected else None
    agents_to_run, agents_skipped, _ = _build_execution_chain(
        target, available_keys, primary_type,
    )

    return {
        "execution_plan": [a.value for a in agents_to_run],
        "agents_skipped": [a.value for a in agents_skipped],
        "current_agent_index": 0,
        "status": "planned",
    }


# ── Error classification & self-healing ──

_TRANSIENT_API_RE = re.compile(
    r'429|rate.?limit|timeout|timed?.?out|50[23]|connection.*(refused|reset|error)',
    re.IGNORECASE,
)
_TRANSIENT_RUNTIME_RE = re.compile(
    r'cannot schedule|shutdown|broken pipe|reset by peer|event loop|errno'
    r'|non.?zero exit status|exit status \d+|CalledProcessError',
    re.IGNORECASE,
)
_PERMANENT_INPUT_RE = re.compile(
    r'not found|no .* files|directory.*missing|does not exist',
    re.IGNORECASE,
)
_PERMANENT_AUTH_RE = re.compile(r'40[13]|unauthorized|forbidden', re.IGNORECASE)
_PERMANENT_DATA_RE = re.compile(
    r'column|shape|parse|decode|format|incompatible|mismatch',
    re.IGNORECASE,
)


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Classify an exception into (category, is_retryable)."""
    if isinstance(exc, FileNotFoundError):
        return ("permanent_input", False)
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return ("permanent_data", False)
    msg = str(exc)
    if _PERMANENT_AUTH_RE.search(msg):
        return ("permanent_auth", False)
    if _TRANSIENT_API_RE.search(msg):
        return ("transient_api", True)
    if _TRANSIENT_RUNTIME_RE.search(msg):
        return ("transient_runtime", True)
    if _PERMANENT_DATA_RE.search(msg):
        return ("permanent_data", False)
    if _PERMANENT_INPUT_RE.search(msg):
        return ("permanent_input", False)
    return ("unknown", False)


def _preflight_check(agent_info, available_inputs: Dict[str, Any]) -> Optional[str]:
    """Validate file inputs exist on disk before running the executor."""
    for inp in agent_info.required_inputs:
        if not (inp.is_file and inp.required):
            continue
        path = available_inputs.get(inp.name)
        if path and isinstance(path, str) and not os.path.exists(path):
            return f"Required input '{inp.name}' not found at: {path}"
    path_err = _validate_required_input_paths(agent_info, available_inputs)
    if path_err:
        return path_err
    return None


def _cleanup_partial_output(agent_name: str, session_id: str) -> None:
    """Remove partial output directory before retry."""
    output_dir = Path(__file__).parent.parent / "outputs" / agent_name / session_id
    if output_dir.is_dir():
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info(f"Cleaned partial output: {output_dir}")


# ── Execution nodes ──


async def execution_router(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Pass-through node — routing logic lives in the route_next edge function."""
    return {}


def _build_conv_state_shim(state: SupervisorGraphState) -> ConversationState:
    """Create a fresh ConversationState shim from graph state."""
    cs = ConversationState(session_id=state.get("session_id", ""))
    cs.workflow_state = dict(state.get("workflow_outputs", {}))
    cs.current_disease = state.get("disease_name")
    for name, path in state.get("uploaded_files", {}).items():
        try:
            fsize = Path(path).stat().st_size
        except OSError:
            fsize = 0
        cs.uploaded_files[name] = UploadedFile(
            filename=name, filepath=path,
            file_type=state.get("detected_file_types", {}).get(name, "unknown"),
            upload_time=time.time(),
            size_bytes=fsize,
        )
    cs.add_user_message(state.get("user_query", ""))
    return cs


def _compute_skip_to(plan: List[str], index: int, available_keys: set) -> int:
    """Find the next runnable agent index after a failure, skipping broken deps."""
    skip_to = index + 1
    for future_idx in range(index + 1, len(plan)):
        future_agent = AGENT_REGISTRY[AgentType(plan[future_idx])]
        if future_agent.requires_one_of and not any(
            k in available_keys for k in future_agent.requires_one_of
        ):
            skip_to = future_idx + 1
        else:
            break
    return skip_to


async def agent_executor(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Execute the current agent with pre-flight validation and self-healing retry."""
    plan = state.get("execution_plan", [])
    index = state.get("current_agent_index", 0)
    total = len(plan)
    agent_type = AgentType(plan[index])
    agent_info = AGENT_REGISTRY[agent_type]
    executor_fn = EXECUTOR_MAP[agent_type]
    callback = _get_progress_callback(config)
    available_inputs = _build_available_inputs(state)
    session_id = state.get("session_id", "")

    # Pre-flight: catch missing-file errors before wasting executor time
    preflight_err = _preflight_check(agent_info, available_inputs)
    if preflight_err:
        logger.warning(f"Pre-flight failed for {agent_info.name}: {preflight_err}")
        if callback:
            callback(StatusUpdate(
                status_type=StatusType.ERROR,
                title=f"❌ {agent_info.display_name} — Missing Input",
                message=preflight_err,
                agent_name=agent_info.name,
            ))
        skip_to = _compute_skip_to(plan, index, set(state.get("workflow_outputs", {}).keys()))
        return {
            "current_agent_index": skip_to,
            "errors": [{
                "agent": agent_info.name, "agent_type": agent_type.value,
                "error": preflight_err, "category": "permanent_input",
                "retried": 0, "duration_s": 0,
                "skipped_downstream": plan[index + 1:skip_to],
            }],
            "status": "executing",
        }

    max_retries = int(os.environ.get("SUPERVISOR_MAX_RETRIES", "2"))
    retry_counts = dict(state.get("retry_counts", {}))
    attempt = 0

    while True:
        conv_state = _build_conv_state_shim(state)
        conv_state.start_agent_execution(agent_info.name, agent_info.display_name, available_inputs)
        start_t = time.time()

        if callback:
            progress_base = index / total if total > 1 else 0.0
            title = (f"⏳ Retrying {agent_info.display_name} (attempt {attempt + 1}/{max_retries + 1})"
                     if attempt > 0
                     else f"🚀 Running {agent_info.display_name} ({index + 1}/{total})")
            callback(StatusUpdate(
                status_type=StatusType.EXECUTING,
                title=title,
                message="Resetting progress..." if attempt > 0 else "Starting analysis...",
                agent_name=agent_info.name,
                progress=progress_base,
            ))

        try:
            async for update in executor_fn(agent_info, available_inputs, conv_state):
                if update.progress is not None and total > 1:
                    update.progress = (index + update.progress) / total
                if callback:
                    callback(update)

            conv_state.complete_agent_execution(conv_state.workflow_state)
            duration = round(time.time() - start_t, 1)
            logger.info(f"Agent {agent_info.display_name} completed in {duration}s"
                        + (f" (after {attempt} retries)" if attempt else ""))

            if callback and total > 1:
                callback(StatusUpdate(
                    status_type=StatusType.PROGRESS,
                    title=f"✅ {agent_info.display_name} Complete ({index + 1}/{total})",
                    message="Moving to next agent..." if index + 1 < total else "All agents complete",
                    agent_name=agent_info.name,
                    progress=(index + 1) / total,
                ))

            return {
                "workflow_outputs": conv_state.workflow_state,
                "current_agent_index": index + 1,
                "retry_counts": retry_counts,
                "agent_results": [{
                    "agent": agent_info.name,
                    "agent_type": agent_type.value,
                    "status": "completed",
                    "duration_s": duration,
                    "retried": attempt,
                }],
                "status": "executing",
            }

        except Exception as exc:
            duration = round(time.time() - start_t, 1)
            category, retryable = _classify_error(exc)
            attempt += 1

            if retryable and attempt <= max_retries:
                retry_counts[agent_type.value] = attempt
                # Differentiated backoff: API errors recover fast, runtime needs more time
                backoff = (2 ** attempt) if category == "transient_api" else (5 ** attempt)
                logger.warning(
                    f"Retrying {agent_info.name} (attempt {attempt + 1}/{max_retries + 1}) "
                    f"after {category} error: {exc}  — backoff {backoff}s"
                )
                _cleanup_partial_output(agent_info.name, session_id)
                conv_state.fail_agent_execution(str(exc))
                await asyncio.sleep(backoff)
                continue

            # Non-retryable or retries exhausted
            logger.exception(f"Agent {agent_info.name} failed ({category}): {exc}")
            conv_state.fail_agent_execution(str(exc))
            if callback:
                callback(StatusUpdate(
                    status_type=StatusType.ERROR,
                    title=f"❌ {agent_info.display_name} Failed",
                    message=str(exc),
                    agent_name=agent_info.name,
                ))

            available_keys = set(conv_state.workflow_state.keys())
            skip_to = _compute_skip_to(plan, index, available_keys)

            return {
                "workflow_outputs": conv_state.workflow_state,
                "current_agent_index": skip_to,
                "retry_counts": retry_counts,
                "errors": [{
                    "agent": agent_info.name,
                    "agent_type": agent_type.value,
                    "error": str(exc),
                    "category": category,
                    "retried": attempt,
                    "duration_s": duration,
                    "skipped_downstream": plan[index + 1:skip_to],
                }],
                "status": "executing",
            }


# ── Response node ──

SCOPE_BUDGETS = {"chat": 8000, "brief": 10000, "standard": 15000, "comprehensive": 20000}
SCOPE_PASSES = {"chat": 2, "brief": 2, "standard": 3, "comprehensive": 4}

_SCOPE_KEYWORDS = {
    "brief": {"brief", "summary", "quick", "short", "concise"},
    "comprehensive": {"comprehensive", "detailed", "everything", "full", "in-depth", "deep"},
}


def _detect_report_scope(query: str) -> str:
    """Simple keyword match — no LLM call."""
    ql = query.lower()
    for scope, keywords in _SCOPE_KEYWORDS.items():
        if any(kw in ql for kw in keywords):
            return scope
    return "standard"


async def response_node(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Synthesize pipeline results into a natural-language response.

    Thin orchestrator — delegates to supervisor_agent.response module:
      regex fast-path → LLM synthesis → optional PDF/DOCX rendering.
    """
    query = state["user_query"]
    try:
        return await _response_node_inner(state, config)
    except Exception as exc:
        logger.exception("response_node crashed — returning graceful fallback: %s", exc)
        # Last-resort: single LLM call with just the user query
        fallback = ""
        try:
            from ..response.synthesizer import synthesize_response
            fallback, _ = await synthesize_response(
                f"Answer this biomedical question concisely:\n\n{query}",
                intent_label="response_fallback",
            )
        except Exception:
            pass
        if not fallback:
            fallback = (
                "I encountered an issue generating the full response. "
                "Please try rephrasing your question or asking again."
            )
        return {
            "final_response": fallback,
            "conversation_history": [{"role": "assistant", "content": fallback}],
            "llm_call_log": [],
            "status": "completed",
        }


async def _response_node_inner(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Inner implementation of response_node — may raise."""
    fmt = state.get("response_format", "chat")
    query = state["user_query"]
    top_n = state.get("requested_top_n", 10)
    wo = state.get("workflow_outputs", {})
    uploaded = state.get("uploaded_files", {})
    callback = _get_progress_callback(config)

    logger.info(f"response_node: fmt={fmt}, top_n={top_n}")

    # ── 0. Passthrough for molecular report pipeline output ──
    # The reporting pipeline already generated its own DOCX/PDF —
    # skip the dynamic renderer entirely and surface the files.
    report_path = wo.get("report_pdf_path") or wo.get("report_docx_path")
    if report_path:
        ext = Path(report_path).suffix.lstrip(".")
        fname = Path(report_path).name
        summary = wo.get("report_summary", {})
        detail_parts = [f"📄 Your **{ext.upper()}** molecular report is ready for download.\n"]
        detail_parts.append(f"**File:** `{fname}`\n")
        if summary:
            detail_parts.append("**Report Highlights:**")
            for k, v in summary.items():
                detail_parts.append(f"- {k.replace('_', ' ').title()}: {v}")
        chat_msg = "\n".join(detail_parts)
        return {
            "final_response": chat_msg,
            "conversation_history": [{"role": "assistant", "content": chat_msg}],
            "workflow_outputs": {"response_report_path": report_path},
            "status": "completed",
        }

    if callback:
        callback(StatusUpdate(
            status_type=StatusType.THINKING,
            title="🧠 Synthesizing results...",
            message="Reading analysis outputs and preparing response",
            agent_name="response",
        ))

    # ── 1. Discover CSVs, load DataFrames, classify by type ──
    csv_list = discover_relevant_csvs(wo, query, uploaded)
    csvs_by_type: Dict[str, pd.DataFrame] = {}
    best_df: Optional[pd.DataFrame] = None
    for csv_path, _score in csv_list:
        ctype = classify_csv_type(csv_path)
        if ctype not in csvs_by_type:
            try:
                csvs_by_type[ctype] = pd.read_csv(
                    csv_path, nrows=50_000,
                    encoding="utf-8", encoding_errors="replace",
                )
            except Exception:
                pass
        if best_df is None:
            best_df = csvs_by_type.get(ctype)

    # ── 2. Try regex fast-path (deterministic, no LLM) ──
    fast_data = regex_fast_path(query, csvs_by_type, top_n, best_df)

    # ── 3. Build prompt and synthesize via LLM ──
    is_doc = fmt in ("pdf", "docx")
    doc_prompt = DOCUMENT_SYSTEM_PROMPT if is_doc else ""

    # Resolve dynamic CSS: always regenerate when user provides new instructions
    style_instr = state.get("style_instructions", "")
    cached_css = wo.get("custom_css", "")
    need_css = is_doc and bool(style_instr)

    # Distill pipeline status for new prompt builder signatures
    disease_name = state.get("disease_name", "") or ""
    available_modules = [r["agent"] for r in (state.get("agent_results") or [])]
    failed_modules = [e["agent"] for e in (state.get("errors") or [])]
    report_scope = _detect_report_scope(query) if is_doc else "chat"
    char_budget = SCOPE_BUDGETS.get(report_scope, 8000)
    conversation_history = state.get("conversation_history", [])

    # Run API enrichment in parallel with CSS generation (if needed)
    enrichment_data: dict = {}
    async def _do_enrichment():
        nonlocal enrichment_data
        try:
            enrichment_data = await enrich_for_response(
                query, csvs_by_type, disease_name, timeout=15.0,
            )
        except Exception as exc:
            logger.debug("Enrichment failed gracefully: %s", exc)

    if fast_data and fast_data.strip():
        # Fast-path: still enrich + maybe CSS in parallel
        coros = [_do_enrichment()]
        if need_css:
            coros.append(generate_style_css(style_instr, state.get("report_theme", "default")))
        results = await asyncio.gather(*coros, return_exceptions=True)
        custom_css = results[1] if need_css and not isinstance(results[1], Exception) else cached_css

        summaries = {"query_result": fast_data}
        if is_doc:
            user_prompt = build_document_user_prompt(
                query=query, disease_name=disease_name, summaries=summaries,
                enrichment_data=enrichment_data or None,
                style_instructions=style_instr or None,
                report_scope=report_scope,
                conversation_history=conversation_history or None,
            )
            pass_count = SCOPE_PASSES.get(report_scope, 3)
            content, llm_logs = await synthesize_multipass(
                user_prompt, pass_count=pass_count,
                system_prompt=doc_prompt,
                intent_label="response_synthesis_fast_doc",
            )
        else:
            user_prompt = build_response_user_prompt(
                query=query, disease_name=disease_name, summaries=summaries,
                enrichment_data=enrichment_data or None,
                conversation_history=conversation_history or None,
                available_modules=available_modules or None,
                failed_modules=failed_modules or None,
            )
            content, llm_logs = await synthesize_chat_multipass(
                user_prompt,
                intent_label="response_synthesis_fast_chat",
            )
        if not content:
            content = f"## Query Results\n\n```\n{fast_data}\n```"
    else:
        # Full path: collect summaries + enrich + maybe CSS in parallel
        summaries = collect_output_summaries(wo, query, top_n, uploaded, budget=char_budget)
        coros = [_do_enrichment()]
        if need_css:
            coros.append(generate_style_css(style_instr, state.get("report_theme", "default")))
        results = await asyncio.gather(*coros, return_exceptions=True)
        custom_css = results[1] if need_css and not isinstance(results[1], Exception) else cached_css

        if is_doc:
            user_prompt = build_document_user_prompt(
                query=query, disease_name=disease_name, summaries=summaries,
                enrichment_data=enrichment_data or None,
                style_instructions=style_instr or None,
                report_scope=report_scope,
                conversation_history=conversation_history or None,
            )
            pass_count = SCOPE_PASSES.get(report_scope, 3)
            content, llm_logs = await synthesize_multipass(
                user_prompt, pass_count=pass_count,
                system_prompt=doc_prompt,
                intent_label="response_synthesis_full_doc",
            )
        else:
            user_prompt = build_response_user_prompt(
                query=query, disease_name=disease_name, summaries=summaries,
                enrichment_data=enrichment_data or None,
                conversation_history=conversation_history or None,
                available_modules=available_modules or None,
                failed_modules=failed_modules or None,
            )
            content, llm_logs = await synthesize_chat_multipass(
                user_prompt,
                intent_label="response_synthesis_full_chat",
            )
        if not content:
            # Fallback: summarise agent_results + errors as plain text
            lines = ["## Analysis Summary\n"]
            for r in state.get("agent_results", []):
                lines.append(f"✅ **{r['agent']}** completed in {r.get('duration_s', '?')}s")
            for e in state.get("errors", []):
                lines.append(f"❌ **{e['agent']}** failed: {e.get('error', 'unknown')}")
            dir_items = [f"- {k}: `{v}`" for k, v in wo.items()
                         if isinstance(v, str) and k.endswith(("_output_dir", "_base_dir", "_path"))]
            if dir_items:
                lines.append("\n**Output Locations:**")
                lines.extend(dir_items)
            content = "\n".join(lines)

    # ── 4. Document rendering (PDF / DOCX) ──
    has_pipeline_data = bool(csv_list) or bool(state.get("agent_results"))
    if fmt in ("pdf", "docx") and not has_pipeline_data:
        content = (
            "> **Note:** This report is based on general biomedical knowledge "
            "and external database annotations, not on uploaded pipeline "
            "analysis results. For a data-driven report, upload your data and "
            "run the analysis pipeline first.\n\n"
            + content
        )

    if fmt in ("pdf", "docx"):
        disease = state.get("disease_name", "") or ""
        # Guard: if disease_name looks like raw query text, fall back
        _BAD_TITLE_RE = re.compile(
            r'\b(generate|create|make|produce|build|write|report|pdf|docx)\b',
            re.IGNORECASE,
        )
        if disease and (len(disease) > 50 or _BAD_TITLE_RE.search(disease)):
            # Try to find a real disease from conversation history
            disease = ""
            for msg in reversed(conversation_history):
                prior_disease = state.get("disease_name", "")
                if prior_disease and len(prior_disease) <= 50 and not _BAD_TITLE_RE.search(prior_disease):
                    disease = prior_disease
                    break
        title = f"{disease.title()} Analysis Report" if disease else "Biomedical Analysis Report"
        ts = int(time.time())
        slug = re.sub(r"[^a-z0-9]+", "_", disease.lower())[:40] if disease else "report"
        filename = f"{slug}_{ts}.{fmt}"

        report_dir = Path("agentic_ai_wf/shared/reports/dynamic_reports") / str(ts)
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            report_dir = Path(tempfile.gettempdir()) / f"supervisor_report_{ts}"
            report_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(report_dir / filename)

        # Ensure fast_data tables are in the document even if LLM omitted them
        doc_content = content
        if fast_data and fast_data.strip() not in content:
            doc_content = f"## Query Results\n\n{fast_data}\n\n{content}"

        theme = state.get("report_theme", "default")
        loop = asyncio.get_running_loop()
        renderer = render_pdf if fmt == "pdf" else render_docx
        try:
            await loop.run_in_executor(
                None, renderer, title, disease, doc_content, output_path, theme, custom_css,
            )
            chat_msg = (
                f"📄 Your **{fmt.upper()}** report has been generated and is ready for download.\n\n"
                f"**File:** `{filename}`\n\n"
                f"Here's a summary of the content:\n\n{content[:500]}"
            )
            if len(content) > 500:
                chat_msg += "…"
            if fmt == "docx" and custom_css:
                chat_msg += ("\n\n> **Note:** Custom styling is fully supported in PDF. "
                             "DOCX formatting applies best-effort styling with some limitations.")
            wo_update = {"response_report_path": output_path}
            if custom_css:
                wo_update["custom_css"] = custom_css
        except Exception as exc:
            logger.exception(f"Failed to render {fmt.upper()} report")
            chat_msg = (
                f"⚠️ {fmt.upper()} rendering failed ({exc}). "
                f"Here are the results as text:\n\n{content}"
            )
            wo_update = {}

        return {
            "final_response": chat_msg,
            "conversation_history": [{"role": "assistant", "content": chat_msg}],
            "llm_call_log": llm_logs,
            "workflow_outputs": wo_update,
            "status": "completed",
        }

    # ── Chat path — return synthesized content directly ──
    return {
        "final_response": content,
        "conversation_history": [{"role": "assistant", "content": content}],
        "llm_call_log": llm_logs,
        "status": "completed",
    }


# ── Structured Report Generation ──


async def report_generation_node(state: SupervisorGraphState, config: RunnableConfig) -> dict:
    """Generate a structured, evidence-traced Markdown report from pipeline outputs.

    Pipeline: ManifestBuilder → Adapters → Planner → Builders → Evidence →
              Validation → Markdown Renderer → optional PDF/DOCX.
    """
    callback = _get_progress_callback(config)
    wo = state.get("workflow_outputs", {}) or {}
    query = state.get("user_query", "")
    disease = state.get("disease_name", "") or ""
    theme = state.get("report_theme", "default")
    fmt = state.get("response_format", "chat")

    if callback:
        callback(StatusUpdate(
            status_type=StatusType.THINKING,
            title="📊 Generating structured report...",
            message="Building evidence-traced analysis report",
            agent_name="report_generation",
        ))

    loop = asyncio.get_running_loop()

    # ── 1. Build manifest + artifact index (sync, in executor) ──
    def _build_data_layer():
        manifest = manifest_from_state(state)
        index = build_artifact_index(manifest)
        return manifest, index

    manifest, index = await loop.run_in_executor(None, _build_data_layer)

    if not manifest.completed_modules():
        msg = "⚠️ No completed analysis modules found. Cannot generate a structured report."
        return {
            "final_response": msg,
            "conversation_history": [{"role": "assistant", "content": msg}],
            "status": "completed",
        }

    # ── 2. Plan → Build → Evidence (sync) ──
    def _build_sections():
        config_obj = ReportingConfig(theme=theme)

        # Plan sections
        planner = ReportPlanner(manifest, index, config_obj)
        sections = planner.plan()

        # Build each section
        all_cards = []
        for section in sections:
            builder_cls = BUILDER_REGISTRY.get(section.id)
            if builder_cls:
                builder = builder_cls(manifest, index, config_obj)
                section = builder.build(section)
                all_cards.extend(builder.collect_evidence(section))

        return sections, all_cards, config_obj

    sections, all_cards, config_obj = await loop.run_in_executor(
        None, _build_sections,
    )

    # ── 2b. Enrich sections with live API data (async) ──
    try:
        enricher = ReportEnricher()
        sections, extra_cards = await enricher.enrich(
            sections, all_cards, disease=disease,
        )
        all_cards.extend(extra_cards)
    except Exception as exc:
        logger.warning("API enrichment failed (non-fatal): %s", exc)

    # ── 2c. Score → Validate → Render (sync) ──
    def _finalise_report(sections, all_cards):
        # Evidence scoring + conflict detection
        scored_cards = score_findings(all_cards)
        conflicts = detect_conflicts(scored_cards)

        # Validation
        guard = ValidationGuard(index)
        try:
            warnings = guard.validate(sections, scored_cards)
            if warnings:
                logger.info("Report validation warnings: %s", warnings)
        except ValidationError as exc:
            logger.warning("Report validation errors: %s", exc.errors)

        # Render Markdown
        title = f"{disease.title()} Analysis Report" if disease else "Analysis Report"
        md_content = render_markdown(
            sections, scored_cards, conflicts, title=title, disease=disease,
        )
        return md_content, title

    md_content, title = await loop.run_in_executor(
        None, _finalise_report, sections, all_cards,
    )

    # ── 2d. Optional LLM narrative augmentation (gated by NarrativeMode) ──
    if config_obj.narrative_mode == NarrativeMode.LLM_AUGMENTED:
        # Build augmentation tasks for all sections in parallel
        async def _augment_section(section):
            if not section.body:
                return section, None
            section_cards = [c for c in all_cards if c.section == section.id]
            ctx = NarrativeContext(
                disease_name=disease,
                section_title=section.title,
                evidence_cards=section_cards,
                table_summaries={t.caption: t.source_artifact for t in section.tables if t.caption},
            )
            augmented = await augment_narrative(ctx)
            return section, augmented

        tasks = [_augment_section(s) for s in sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Section augmentation failed (non-fatal): %s", res)
                continue
            section, augmented = res
            if augmented:
                section.body = augmented

        # Re-render markdown with augmented section bodies
        md_content = await loop.run_in_executor(
            None, lambda: render_markdown(
                sections, all_cards, detect_conflicts(all_cards),
                title=title, disease=disease,
            ),
        )

        # ── 2e. Review pass on the full rendered markdown ──
        review_input = (
            f"═══ ORIGINAL DATA CONTEXT ═══\n\n"
            f"Disease: {disease}\nQuery: {query}\n"
            f"Modules: {', '.join(manifest.completed_modules())}\n\n"
            f"═══ DRAFT DOCUMENT TO REVIEW ═══\n\n{md_content}"
        )
        try:
            from ..response.synthesizer import REVIEW_SYSTEM_PROMPT as _REVIEW_PROMPT
            review_result = await llm_complete(
                messages=[{"role": "user", "content": review_input}],
                system=_REVIEW_PROMPT,
                temperature=0.1,
            )
            if review_result.text and len(review_result.text) > len(md_content) * 0.5:
                md_content = review_result.text
        except Exception as exc:
            logger.warning("Review pass on structured report failed (non-fatal): %s", exc)

    # ── 3. Write output files ──
    ts = int(time.time())
    slug = re.sub(r"[^a-z0-9]+", "_", disease.lower())[:40] if disease else "report"
    report_dir = Path("agentic_ai_wf/shared/reports/structured_reports") / str(ts)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        report_dir = Path(tempfile.gettempdir()) / f"structured_report_{ts}"
        report_dir.mkdir(parents=True, exist_ok=True)

    md_path = str(report_dir / f"{slug}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    wo_update = {"structured_report_md_path": md_path}

    # ── 4. Optional PDF/DOCX rendering ──
    if fmt in ("pdf", "docx"):
        from ..response import render_pdf, render_docx, generate_style_css
        output_path = str(report_dir / f"{slug}_{ts}.{fmt}")
        renderer = render_pdf if fmt == "pdf" else render_docx
        # Honour user style instructions for structured reports too
        sr_style = state.get("style_instructions", "")
        if sr_style:
            try:
                custom_css = await generate_style_css(
                    sr_style, state.get("report_theme", "default"),
                )
            except Exception:
                custom_css = wo.get("custom_css", "")
        else:
            custom_css = wo.get("custom_css", "")
        try:
            await loop.run_in_executor(
                None, renderer, title, disease, md_content, output_path, theme, custom_css,
            )
            wo_update["structured_report_path"] = output_path
            chat_msg = (
                f"📊 Your structured analysis report has been generated.\n\n"
                f"**Format:** {fmt.upper()}\n"
                f"**File:** `{Path(output_path).name}`\n\n"
                f"The report includes evidence traceability, cross-module findings, "
                f"and quality assessments for all completed analyses."
            )
        except Exception as exc:
            logger.exception("Structured report %s rendering failed", fmt.upper())
            chat_msg = (
                f"📊 Structured Markdown report generated successfully.\n"
                f"⚠️ {fmt.upper()} rendering failed ({exc}).\n\n"
                f"Markdown report saved to: `{Path(md_path).name}`"
            )
    else:
        chat_msg = (
            f"📊 Structured analysis report generated.\n\n"
            f"**File:** `{Path(md_path).name}`\n\n"
            f"{md_content[:500]}"
        )
        if len(md_content) > 500:
            chat_msg += "…"

    return {
        "final_response": chat_msg,
        "conversation_history": [{"role": "assistant", "content": chat_msg}],
        "workflow_outputs": wo_update,
        "status": "completed",
    }
