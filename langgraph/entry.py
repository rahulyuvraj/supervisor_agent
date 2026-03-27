"""Public entry points for the Supervisor LangGraph workflow.

Two invocation modes:
  run_supervisor_stream()  — AsyncGenerator[StatusUpdate] for real-time UI
  run_supervisor()         — returns SupervisorResult for programmatic callers
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from langgraph.types import RunnableConfig

from ..executors.base import StatusType, StatusUpdate
from .graph import get_supervisor_graph
from .state import SupervisorGraphState, SupervisorResult

logger = logging.getLogger(__name__)


def _build_initial_state(
    user_query: str,
    session_id: str,
    analysis_id: str = "",
    output_root: str = "",
    disease_name: str = "",
    uploaded_files: Optional[Dict[str, str]] = None,
    workflow_outputs: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> SupervisorGraphState:
    """Normalize caller inputs into a graph-ready state dict.

    Fields that must NOT leak from a prior turn's checkpoint are
    explicitly reset here so the new invocation starts clean.
    """
    state: SupervisorGraphState = {
        "user_query": user_query,
        "session_id": session_id,
        # Reset per-turn fields so MemorySaver checkpoint doesn't leak them
        "response_format": "none",
        "final_response": "",
        "needs_response": False,
        "general_response": "",
        "style_instructions": "",
        "molecular_report_format": "",
    }
    if analysis_id:
        state["analysis_id"] = analysis_id
    if output_root:
        state["output_root"] = output_root
    if user_id:
        state["user_id"] = user_id
    if disease_name:
        state["disease_name"] = disease_name
    if uploaded_files:
        state["uploaded_files"] = uploaded_files
    if workflow_outputs:
        state["workflow_outputs"] = workflow_outputs
        state["has_existing_results"] = True
    return state


async def run_supervisor_stream(
    user_query: str,
    session_id: str,
    analysis_id: str = "",
    output_root: str = "",
    disease_name: str = "",
    uploaded_files: Optional[Dict[str, str]] = None,
    workflow_outputs: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> AsyncGenerator[StatusUpdate, None]:
    """Stream StatusUpdates from the LangGraph supervisor.

    Bridges the callback-based graph nodes to an AsyncGenerator via an
    asyncio.Queue with a None sentinel so the drain loop terminates
    reliably regardless of success or failure inside the graph.
    """
    graph = get_supervisor_graph()
    initial_state = _build_initial_state(
        user_query, session_id, analysis_id, output_root,
        disease_name, uploaded_files, workflow_outputs, user_id,
    )
    queue: asyncio.Queue[Optional[StatusUpdate]] = asyncio.Queue()

    from ..logging_utils import analysis_id_var
    analysis_id_var.set(analysis_id or session_id[:8])

    def _callback(update: StatusUpdate) -> None:
        queue.put_nowait(update)

    config = RunnableConfig(configurable={
        "thread_id": session_id,
        "progress_callback": _callback,
    })

    async def _run():
        try:
            return await graph.ainvoke(initial_state, config)
        finally:
            queue.put_nowait(None)

    yield StatusUpdate(StatusType.THINKING, "Analyzing request…", user_query)

    task = asyncio.create_task(_run())

    # Drain callback queue until sentinel
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    except (GeneratorExit, asyncio.CancelledError):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return

    # Retrieve final state from the completed graph
    try:
        result_state = task.result()
    except Exception as exc:
        logger.exception("Supervisor graph raised an exception")
        yield StatusUpdate(StatusType.ERROR, "Workflow failed", str(exc))
        return

    # Build graph-state payload for sync-back by callers
    graph_state = {
        "workflow_outputs": result_state.get("workflow_outputs", {}),
        "disease_name": result_state.get("disease_name", ""),
        "agent_results": result_state.get("agent_results", []),
    }

    # Determine the single terminal StatusUpdate
    final = result_state.get("final_response", "")
    general = result_state.get("general_response", "")
    errors = result_state.get("errors", [])
    has_results = bool(result_state.get("agent_results"))

    if result_state.get("is_general_query") and not final:
        terminal = StatusUpdate(StatusType.COMPLETED, "Response ready", general)
    elif final:
        terminal = StatusUpdate(StatusType.COMPLETED, "Analysis complete", final)
    elif errors and not has_results:
        summary = "; ".join(e.get("error", "unknown") for e in errors)
        terminal = StatusUpdate(StatusType.ERROR, "All agents failed", summary)
    else:
        terminal = StatusUpdate(StatusType.COMPLETED, "Pipeline complete",
                                "All requested agents completed successfully.")

    terminal._graph_state = graph_state

    # Attach generated report file only if THIS invocation produced a document.
    # Without the format guard, a stale response_report_path from a prior turn
    # would be re-served on every subsequent general query.
    fmt = result_state.get("response_format", "none")
    if fmt in ("pdf", "docx"):
        wo = result_state.get("workflow_outputs", {})
        report_path = wo.get("response_report_path") or wo.get("report_pdf_path") or wo.get("report_docx_path")
        if report_path:
            terminal.generated_files = [report_path]
            terminal.agent_name = "response"

    yield terminal


async def run_supervisor(
    user_query: str,
    session_id: str,
    analysis_id: str = "",
    output_root: str = "",
    disease_name: str = "",
    uploaded_files: Optional[Dict[str, str]] = None,
    workflow_outputs: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> SupervisorResult:
    """Non-streaming invocation — returns a SupervisorResult directly."""
    graph = get_supervisor_graph()
    initial_state = _build_initial_state(
        user_query, session_id, analysis_id, output_root,
        disease_name, uploaded_files, workflow_outputs, user_id,
    )
    config = RunnableConfig(configurable={"thread_id": session_id})

    t0 = time.monotonic()
    try:
        state = await graph.ainvoke(initial_state, config)
    except Exception as exc:
        logger.exception("Supervisor graph raised an exception")
        return SupervisorResult(status="failed", final_response=str(exc))

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    response = (
        state.get("final_response")
        or state.get("general_response")
        or ""
    )

    return SupervisorResult(
        status=state.get("status", "completed"),
        final_response=response,
        output_dir=state.get("output_root", ""),
        agent_results=state.get("agent_results", []),
        errors=state.get("errors", []),
        execution_time_ms=elapsed_ms,
    )
