"""
adapter.py — Async executor bridging CausalitySupervisorAgent to the supervisor framework.

Follows the established ``execute_*`` pattern in pipeline_executors.py:
async generator that yields StatusUpdate objects and mutates state.workflow_state.

Ordering contract:
    1. Discover files via file_mapper.map_files(workflow_state)
    2. Wire tool slots via tool_wiring.populate(workflow_state)
    3. Run CausalitySupervisorAgent.run() in a thread (sync → async bridge)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List

from ..executors.base import StatusType, StatusUpdate

if TYPE_CHECKING:
    from ..agent_registry import AgentInfo
    from ..state import ConversationState

log = logging.getLogger("causal_platform")


# ── helpers (mirrors helpers at top of pipeline_executors.py) ─────────────

def _get_output_dir(agent_name: str, session_id: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "outputs" / agent_name / session_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_user_query(inputs: Dict[str, Any], state: "ConversationState") -> str:
    query = inputs.get("query") or inputs.get("disease_name", "")
    if not query:
        from ..state import MessageRole
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                query = msg.content
                break
    return query or ""


def _sync_llm_fn(user_msg: str, system_prompt: str, max_tokens: int) -> str:
    """Sync LLM wrapper matching CausalityLLMBridge's call_llm_fn protocol.

    Calls _bedrock_invoke directly (already sync) with proper message format.
    Falls back to the LocalClient inside llm_bridge if Bedrock is unavailable.
    """
    from ..llm_provider import _bedrock_invoke
    messages = [{"role": "user", "content": user_msg}]
    return _bedrock_invoke(
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        system=system_prompt,
        response_format=None,
    )


def _collect_generated_files(output_dir: Path) -> List[Dict[str, Any]]:
    """Walk output_dir and return file metadata dicts for StatusUpdate."""
    files = []
    if not output_dir.is_dir():
        return files
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
            })
    return files


# ── main executor ─────────────────────────────────────────────────────────

async def execute_causality(
    agent_info: "AgentInfo",
    inputs: Dict[str, Any],
    state: "ConversationState",
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the causality analysis pipeline.

    This is the supervisor executor entry-point registered in EXECUTOR_MAP.
    It bridges the sync CausalitySupervisorAgent into the async generator
    protocol expected by agent_executor() in langgraph/nodes.py.
    """
    from . import tool_wiring, file_mapper
    from .llm_bridge import CausalityLLMBridge
    from .core_agent import CausalitySupervisorAgent

    try:
        query = _get_user_query(inputs, state)
        if not query:
            raise ValueError("No query provided for causality analysis.")

        disease_name = (
            inputs.get("disease_name")
            or state.workflow_state.get("disease_name", "")
        )

        output_dir = _get_output_dir("causality", state.session_id)

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🔬 Causality Analysis Starting",
            message=f"Running causal inference pipeline"
                    + (f" for **{disease_name}**" if disease_name else "")
                    + "...",
            details=(
                f"Query: {query[:120]}\n"
                f"Output: {output_dir}"
            ),
            agent_name=agent_info.name,
            progress=0.1,
        )

        # 1. Discover input files from upstream agents' outputs
        file_paths = file_mapper.map_files(state.workflow_state)
        log.info("causality adapter: %d input files discovered", len(file_paths))

        # 2. Wire tool slots from workflow_state
        wired = tool_wiring.populate(state.workflow_state)
        log.info("causality adapter: %d tool slots wired", wired)

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Inputs Prepared",
            message=f"Discovered {len(file_paths)} files, wired {wired} tool slots",
            agent_name=agent_info.name,
            progress=0.2,
        )

        # 3. Build LLM bridge (sync wrapper around Bedrock)
        bridge = CausalityLLMBridge(call_llm_fn=_sync_llm_fn)

        # 4. Instantiate and run the agent in a thread
        agent = CausalitySupervisorAgent(bridge)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run(
                query=query,
                file_paths=file_paths,
                output_dir=str(output_dir),
            ),
        )

        # 5. Handle clarification / blocked outcomes
        status = result.get("status", "error")

        if status == "needs_clarification":
            state.workflow_state["causality_needs_clarification"] = True
            state.workflow_state["causality_clarifying_question"] = result.get("clarifying_question", "")
            yield StatusUpdate(
                status_type=StatusType.WAITING_INPUT,
                title="❓ Clarification Needed",
                message=result.get("clarifying_question", "Please clarify your query."),
                agent_name=agent_info.name,
                progress=0.3,
            )
            return

        if status == "blocked":
            state.workflow_state["causality_blocked"] = True
            state.workflow_state["causality_blocking_message"] = result.get("blocking_message", "")
            yield StatusUpdate(
                status_type=StatusType.ERROR,
                title="🚫 Causality Pipeline Blocked",
                message=result.get("blocking_message", "Pipeline blocked by eligibility gates."),
                agent_name=agent_info.name,
            )
            return

        # 6. Populate workflow_state with outputs
        state.workflow_state["causality_output_dir"] = str(output_dir)
        state.workflow_state["causality_result"] = result.get("result", {})
        state.workflow_state["causality_intent"] = result.get("intent", {})
        state.workflow_state["causality_gates"] = result.get("gate_results", [])
        state.workflow_state["causality_file_audits"] = result.get("file_audits", [])
        state.workflow_state["causality_lit_brief"] = result.get("lit_brief")
        state.workflow_state["causality_steps"] = result.get("steps", [])
        state.workflow_state["causality_run_id"] = result.get("run_id", "")
        state.workflow_state["causality_query"] = result.get("query", query)
        state.workflow_state["causality_context"] = result.get("context", "")

        generated = _collect_generated_files(output_dir)

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Causality Analysis Complete",
            message=f"Causal inference pipeline finished"
                    + (f" for {disease_name}" if disease_name else ""),
            details=(
                f"Run ID: {result.get('run_id', 'N/A')}\n"
                f"Intent: {result.get('intent', {}).get('intent_name', 'N/A')}\n"
                f"Files generated: {len(generated)}\n"
                f"Results saved to: {output_dir}"
            ),
            agent_name=agent_info.name,
            progress=1.0,
            generated_files=generated,
            output_dir=str(output_dir),
        )

    except Exception as e:
        log.exception("Causality analysis failed: %s", e)
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Causality Analysis Failed",
            message=f"Error: {e}",
            agent_name=agent_info.name,
        )
        raise
