# Supervisor Agent Integration Guide

> **Audience:** Senior software engineer integrating into Django + FastAPI + Celery on EC2.
> **Frontend:** js/React. **Storage:** EC2 local disk (S3 migration planned). **Auth:** External — leave clean hooks.

---

## 1. Architecture Overview

The Supervisor Agent is an LLM-powered orchestrator for multi-agent bioinformatics pipelines. It receives a user query, classifies intent via an LLM router, builds a dependency-aware execution plan across 16 specialized agents (DEG analysis, pathway enrichment, drug discovery, CRISPR analysis, etc.), executes them sequentially with real-time status streaming, and synthesizes pipeline outputs into natural-language responses or styled PDF/DOCX documents through a multi-pass LLM synthesis engine.

### Mermaid Diagram

```mermaid
graph TD
    A[User Query] --> B[IntentRouter<br/>LLM-based routing]
    B --> C{check_general_query}
    C -->|general| D[END - return suggested_response]
    C -->|follow_up| E[response_node<br/>Multi-pass LLM synthesis]
    C -->|structured_report| F[structured_report_node<br/>Evidence-traced report]
    C -->|needs_execution| G[plan_node<br/>Build execution plan]
    G --> H[execution_router]
    H -->|execute_agent| I[agent_executor<br/>Run pipeline agent]
    I --> H
    H -->|structured_report| F
    H -->|report| E
    H -->|done| D
    E --> J[END - return final_response]
    F --> K[END - return structured report]

    subgraph "Response Engine"
        E --> E1[Data Discovery]
        E1 --> E2[API Enrichment]
        E2 --> E3[Multi-pass Synthesis]
        E3 --> E4[Optional PDF/DOCX Render]
    end
```

### File Map: `supervisor_agent/`

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports: `SupervisorAgent`, `SessionManager`, `ConversationState` |
| `supervisor.py` | Legacy `SupervisorAgent` class, file-type detection, execution chain builder, CRISPR input hydration |
| `router.py` | `IntentRouter` — LLM-based query routing with multi-agent pipeline detection |
| `agent_registry.py` | `AgentType` enum, `AgentInfo` dataclass, `AGENT_REGISTRY` dict (16 agents), `PIPELINE_ORDER`, `FILE_TYPE_TO_INPUT_KEY` |
| `state.py` | `ConversationState`, `SessionManager`, `Message`, `UploadedFile`, `AgentExecution` dataclasses |
| `llm_provider.py` | `llm_complete()` — unified async LLM call (Bedrock primary, OpenAI fallback) |
| `logging_utils.py` | `SupervisorLogger`, `CorrelationFilter`, `analysis_id_var` context var |
| `utils.py` | Shared utility functions |
| `test_supervisor.py` | Unit tests for supervisor |

### File Map: `supervisor_agent/langgraph/`

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `graph.py` | `build_supervisor_graph()` — StateGraph construction with nodes and edges; `get_supervisor_graph()` cached singleton |
| `state.py` | `SupervisorGraphState` TypedDict (shared graph state), `SupervisorResult` dataclass (API output) |
| `nodes.py` | `intent_node()`, `plan_node()`, `execution_router()`, `agent_executor()`, `response_node()`, `report_generation_node()` |
| `edges.py` | `check_general_query()`, `route_next()` — conditional edge functions |
| `entry.py` | `run_supervisor()`, `run_supervisor_stream()` — public API entry points |
| `checkpointer.py` | `get_checkpointer()` — MemorySaver (dev) or PostgresSaver (prod) factory |
| `test_langgraph_e2e.py` | End-to-end integration tests |

### File Map: `supervisor_agent/response/`

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports all public symbols from sub-modules |
| `synthesizer.py` | `synthesize_multipass()` (4-pass doc pipeline), `synthesize_chat_multipass()` (2-pass chat), system prompts, prompt builders |
| `document_renderer.py` | `render_pdf()` (WeasyPrint), `render_docx()` (htmldocx) — markdown→HTML→file |
| `style_engine.py` | CSS theme presets, `extract_style_instructions()`, `generate_style_css()` via LLM, `build_css()` |
| `data_discovery.py` | `discover_relevant_csvs()`, `classify_csv_type()`, `collect_output_summaries()`, `query_csv_data()` |
| `query_functions.py` | 5 deterministic query functions: `get_top_genes`, `get_top_pathways`, `get_top_drugs`, `get_gene_detail`, `get_full_summary`; `regex_fast_path()` dispatcher |
| `enrichment_dispatcher.py` | `enrich_for_response()` — entity extraction + parallel API adapter calls (Ensembl, STRING, DGIdb, ChEMBL, Reactome, etc.) |

### File Map: `supervisor_agent/executors/`

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports all executor functions and `StatusType`/`StatusUpdate` |
| `base.py` | `StatusType` enum (9 values), `StatusUpdate` dataclass, `OutputValidation`, `validate_pipeline_output()` |
| `pipeline_executors.py` | 15 executor functions — one per agent type; manages output dirs, calls agent pipelines, yields `StatusUpdate` |

---

## 2. Entry Points (Production API Surface)

### `supervisor_agent/langgraph/entry.py`

#### `run_supervisor()` — Non-streaming, returns result directly

```python
async def run_supervisor(
    user_query: str,           # The user's natural-language query
    session_id: str,           # Unique session identifier (used for checkpointing + output dirs)
    analysis_id: str = "",     # Optional analysis run identifier (used in logging)
    output_root: str = "",     # Base directory for agent outputs (overrides default)
    disease_name: str = "",    # Pre-extracted disease name (optional — router can extract it)
    uploaded_files: Optional[Dict[str, str]] = None,   # {filename: local_filepath}
    workflow_outputs: Optional[Dict[str, Any]] = None,  # Prior run results to resume from
    user_id: str = "",         # External user ID for auth hooks
) -> SupervisorResult
```

**`SupervisorResult` dataclass fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | `"completed"`, `"failed"` |
| `final_response` | `str` | Markdown-formatted response text |
| `output_dir` | `str` | Root output directory path |
| `key_files` | `Dict[str, str]` | Named key files (default empty) |
| `agent_results` | `List[Dict[str, Any]]` | Per-agent completion records: `{agent, agent_type, status, duration_s, retried}` |
| `errors` | `List[Dict[str, Any]]` | Per-agent error records: `{agent, agent_type, error, category, retried, duration_s, skipped_downstream}` |
| `execution_time_ms` | `int` | Total wall-clock time in milliseconds |

**Example call:**

```python
import asyncio
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor

result = asyncio.run(run_supervisor(
    user_query="What are the top 10 differentially expressed genes in lupus?",
    session_id="abc-123-def",
    disease_name="lupus",
    uploaded_files={"lupus_counts.csv": "/data/uploads/lupus_counts.csv"},
))
print(result.status)           # "completed"
print(result.final_response)   # Markdown text
print(result.agent_results)    # [{agent: "deg_analysis", status: "completed", ...}]
```

#### `run_supervisor_stream()` — Async generator of StatusUpdate objects

```python
async def run_supervisor_stream(
    user_query: str,
    session_id: str,
    analysis_id: str = "",
    output_root: str = "",
    disease_name: str = "",
    uploaded_files: Optional[Dict[str, str]] = None,
    workflow_outputs: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> AsyncGenerator[StatusUpdate, None]
```

**How to consume:**

```python
async for update in run_supervisor_stream(
    user_query="Run full pipeline for breast cancer",
    session_id="session-456",
    disease_name="breast cancer",
):
    print(f"[{update.status_type}] {update.title}: {update.message}")
    if update.status_type == StatusType.COMPLETED:
        # Terminal update — check update.generated_files for report paths
        # Access graph state via update._graph_state
        graph_state = getattr(update, "_graph_state", {})
```

The stream:
1. Yields an initial `THINKING` update immediately
2. Yields `StatusUpdate` objects from the callback queue as the graph progresses (ROUTING, EXECUTING, PROGRESS, etc.)
3. Yields a terminal `COMPLETED` or `ERROR` update as the final item
4. The terminal update has `_graph_state` dict attached with `workflow_outputs`, `disease_name`, `agent_results`
5. If `response_format` was `"pdf"` or `"docx"`, the terminal update has `generated_files` list with the report path

#### `run_supervisor_sync()` — Synchronous wrapper for Celery

There is no explicit `run_supervisor_sync()` function in the codebase. For Celery integration, wrap `run_supervisor()` with `asyncio.run()`:

```python
import asyncio
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor

def run_supervisor_sync(**kwargs):
    return asyncio.run(run_supervisor(**kwargs))
```

### `supervisor_agent/supervisor.py` — `SupervisorAgent` class

```python
class SupervisorAgent:
    def __init__(self, api_key: Optional[str] = None):
        # Initializes IntentRouter, SessionManager, SupervisorLogger
        # api_key: Optional OpenAI key (reads from env if not provided)

    async def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        uploaded_files: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> AsyncGenerator[StatusUpdate, None]:
        # Yields StatusUpdate objects during processing
```

**`USE_LANGGRAPH_SUPERVISOR` flag:** Read from `os.environ.get("USE_LANGGRAPH_SUPERVISOR", "false")`. When `"true"`, `process_message()` delegates to the LangGraph graph via `run_supervisor_stream()`. When `"false"`, runs the legacy inline execution path in `supervisor.py`. **For production, set `USE_LANGGRAPH_SUPERVISOR=true`.**

### `StatusUpdate` dataclass (`supervisor_agent/executors/base.py`)

```python
@dataclass
class StatusUpdate:
    status_type: StatusType    # Enum value (see below)
    title: str                 # Short display title (e.g., "🚀 Running DEG Analysis")
    message: str               # Main message (supports markdown)
    details: Optional[str]     # Collapsible detail text
    progress: Optional[float]  # 0.0 to 1.0
    agent_name: Optional[str]  # Which agent emitted this
    timestamp: float           # time.time() when created
    generated_files: Optional[List[Dict[str, Any]]]  # Files produced (on completion)
    output_dir: Optional[str]  # Output directory (on completion)
```

**`StatusType` enum values:**

| Value | When Emitted |
|-------|-------------|
| `THINKING` | Start of processing, LLM analysis beginning |
| `ROUTING` | Intent classification result available |
| `VALIDATING` | Input validation in progress |
| `EXECUTING` | Agent execution started |
| `PROGRESS` | Mid-execution progress update (0.0–1.0) |
| `COMPLETED` | Agent or full workflow completed successfully |
| `ERROR` | Agent or workflow failed |
| `WAITING_INPUT` | Missing required inputs — needs user response |
| `INFO` | Informational message |

### FastAPI Endpoint Example

```python
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio
import uuid

from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor, run_supervisor_stream
from agentic_ai_wf.supervisor_agent.executors.base import StatusType

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    disease_name: Optional[str] = ""
    uploaded_files: Optional[Dict[str, str]] = None
    workflow_outputs: Optional[Dict[str, dict]] = None
    user_id: Optional[str] = ""

class QueryResponse(BaseModel):
    status: str
    response: str
    output_dir: str
    agent_results: list
    errors: list
    execution_time_ms: int

@app.post("/api/supervisor/query")
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Synchronous query — returns when complete. Use for simple queries."""
    session_id = request.session_id or str(uuid.uuid4())
    result = await run_supervisor(
        user_query=request.query,
        session_id=session_id,
        disease_name=request.disease_name or "",
        uploaded_files=request.uploaded_files,
        workflow_outputs=request.workflow_outputs,
        user_id=request.user_id or "",
    )
    return QueryResponse(
        status=result.status,
        response=result.final_response,
        output_dir=result.output_dir,
        agent_results=result.agent_results,
        errors=result.errors,
        execution_time_ms=result.execution_time_ms,
    )

@app.websocket("/ws/supervisor/stream")
async def stream_query(websocket: WebSocket):
    """WebSocket endpoint for real-time StatusUpdate streaming."""
    await websocket.accept()
    data = await websocket.receive_json()
    session_id = data.get("session_id") or str(uuid.uuid4())

    async for update in run_supervisor_stream(
        user_query=data["query"],
        session_id=session_id,
        disease_name=data.get("disease_name", ""),
        uploaded_files=data.get("uploaded_files"),
        workflow_outputs=data.get("workflow_outputs"),
        user_id=data.get("user_id", ""),
    ):
        await websocket.send_json({
            "status_type": update.status_type.value,
            "title": update.title,
            "message": update.message,
            "details": update.details,
            "progress": update.progress,
            "agent_name": update.agent_name,
            "generated_files": update.generated_files,
            "output_dir": update.output_dir,
        })
        if update.status_type in (StatusType.COMPLETED, StatusType.ERROR):
            break
    await websocket.close()
```

### Celery Task Example

```python
import asyncio
import uuid
from celery import Celery

from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor

celery_app = Celery("supervisor_tasks", broker="redis://localhost:6379/0")

@celery_app.task(bind=True, max_retries=1, time_limit=3600)
def run_analysis_task(self, query, files, disease_name, analysis_id=None):
    """
    Celery task wrapping the supervisor agent.

    Args:
        query: User's natural-language query
        files: Dict of {filename: filepath} for uploaded files
        disease_name: Disease context string
        analysis_id: Optional analysis run ID
    """
    session_id = analysis_id or str(uuid.uuid4())

    result = asyncio.run(run_supervisor(
        user_query=query,
        session_id=session_id,
        analysis_id=analysis_id or session_id,
        disease_name=disease_name,
        uploaded_files=files,
    ))

    return {
        "status": result.status,
        "response": result.final_response,
        "output_dir": result.output_dir,
        "agent_results": result.agent_results,
        "errors": result.errors,
        "execution_time_ms": result.execution_time_ms,
    }
```

For real-time progress from Celery, use `run_supervisor_stream()` with a Redis pub/sub bridge:

```python
@celery_app.task(bind=True)
def run_analysis_task_with_progress(self, query, files, disease_name, analysis_id):
    import redis
    r = redis.Redis()

    async def _run():
        async for update in run_supervisor_stream(
            user_query=query,
            session_id=analysis_id,
            disease_name=disease_name,
            uploaded_files=files,
        ):
            r.publish(f"supervisor:{analysis_id}", json.dumps({
                "status_type": update.status_type.value,
                "title": update.title,
                "message": update.message,
                "progress": update.progress,
            }))

    asyncio.run(_run())
```

---

## 3. LangGraph Graph Structure

### `supervisor_agent/langgraph/graph.py`

**Nodes:**

| Node Name | Function | Purpose |
|-----------|----------|---------|
| `intent` | `intent_node()` | Route query, detect files, determine response format |
| `plan` | `plan_node()` | Build ordered execution plan from routing decision |
| `router` | `execution_router()` | Pass-through — routing logic in `route_next` edge |
| `executor` | `agent_executor()` | Execute current agent with retry and error handling |
| `response` | `response_node()` | Synthesize results into natural-language + optional PDF/DOCX |
| `structured_report` | `report_generation_node()` | Evidence-traced structured report from pipeline outputs |

**Edges:**

| From | To | Type | Condition |
|------|----|------|-----------|
| `START` | `intent` | Direct | Always |
| `intent` | `END` | Conditional | `check_general_query` returns `"general"` |
| `intent` | `response` | Conditional | `check_general_query` returns `"follow_up"` |
| `intent` | `structured_report` | Conditional | `check_general_query` returns `"structured_report"` |
| `intent` | `plan` | Conditional | `check_general_query` returns `"needs_execution"` |
| `plan` | `router` | Direct | Always |
| `router` | `executor` | Conditional | `route_next` returns `"execute_agent"` |
| `router` | `structured_report` | Conditional | `route_next` returns `"structured_report"` |
| `router` | `response` | Conditional | `route_next` returns `"report"` |
| `router` | `END` | Conditional | `route_next` returns `"done"` |
| `executor` | `router` | Direct | Always (loops back for next agent) |
| `response` | `END` | Direct | Always |
| `structured_report` | `END` | Direct | Always |

**Graph compiled with:** `graph.compile(checkpointer=get_checkpointer())`

### Complete Flow Sequences

**Path A — General query (e.g., "What is ERBB2?"):**
1. `START` → `intent_node`: Routes query, sets `is_general_query=True`, `needs_response=True`
2. `check_general_query` → returns `"follow_up"` (because `needs_response=True`)
3. `response_node`: Discovers CSVs (if any), runs 2-pass chat synthesis, returns response
4. → `END`

**Path B — Pipeline execution (e.g., "Run DEG analysis for lupus"):**
1. `START` → `intent_node`: Routes query, identifies `deg_analysis` agent, sets `is_general_query=False`
2. `check_general_query` → returns `"needs_execution"`
3. `plan_node`: Builds `execution_plan=["deg_analysis"]` (or chain with upstream agents)
4. `execution_router` → `route_next` → `"execute_agent"` (index < plan length)
5. `agent_executor`: Runs DEG analysis, updates `workflow_outputs` and `current_agent_index`
6. Back to `execution_router` → `route_next` → `"report"` or `"done"` (plan complete)
7. If `"report"` → `response_node`: Synthesizes results → `END`
8. If `"done"` → `END`

**Path C — Multi-agent pipeline (e.g., "Full pipeline for breast cancer"):**
1–3. Same as Path B, but `plan_node` builds `execution_plan=["cohort_retrieval", "deg_analysis", "gene_prioritization", "pathway_enrichment"]`
4–6. Loop: `execution_router` → `agent_executor` → `execution_router` for each agent
7. After all agents: `route_next` → `"report"` → `response_node` → `END`

### `SupervisorGraphState` TypedDict

| Key | Type | Purpose | Written By | Read By |
|-----|------|---------|-----------|---------|
| `user_query` | `str` | Original user query | entry | intent, response, edges |
| `disease_name` | `str` | Extracted disease context | intent | plan, executor, response |
| `uploaded_files` | `Dict[str, str]` (merge) | `{filename: filepath}` | entry | intent, executor, response |
| `detected_file_types` | `Dict[str, str]` (merge) | `{filename: type_key}` | intent | plan, executor |
| `routing_decision` | `Dict[str, Any]` | Serialized RoutingDecision | intent | plan, edges |
| `is_general_query` | `bool` | Whether query is general (not agent-specific) | intent | edges |
| `general_response` | `str` | Router's suggested response for general queries | intent | entry |
| `needs_response` | `bool` | Whether to run LLM synthesis after execution | intent | edges, response |
| `response_format` | `str` | `"chat"` / `"pdf"` / `"docx"` / `"none"` | intent | response, entry |
| `report_theme` | `str` | CSS theme: `"default"` / `"clinical"` / `"minimal"` | intent | response |
| `style_instructions` | `str` | Freeform styling text extracted from query | intent | response |
| `molecular_report_format` | `str` | `"pdf"` / `"docx"` for molecular report | intent | plan, executor |
| `requested_top_n` | `int` | CSV rows to read (default 10, from "top N") | intent | response |
| `has_existing_results` | `bool` | True when `workflow_outputs` has prior run data | intent, entry | edges |
| `execution_plan` | `List[str]` | `AgentType.value` list in run order | plan | executor, router edge |
| `agents_skipped` | `List[str]` | Agents skipped (already satisfied) | plan | — |
| `current_agent_index` | `int` | Current position in execution_plan | plan, executor | executor, router edge |
| `workflow_outputs` | `Dict[str, Any]` (merge) | Accumulates agent output keys/paths | executor, intent | executor, response, entry |
| `agent_results` | `List[Dict]` (append) | Per-agent completion records | executor | response, entry |
| `errors` | `List[Dict]` (append) | Per-agent error records | executor | response, entry, edges |
| `retry_counts` | `Dict[str, int]` (merge) | `agent_type.value → attempts` | executor | executor |
| `conversation_history` | `List[Dict[str, str]]` (append) | Accumulates `{role, content}` across turns | intent, response | intent (replay), response |
| `final_response` | `str` | Terminal markdown response | response | entry |
| `llm_call_log` | `List[Dict]` (append) | LLM call telemetry | response | — |
| `session_id` | `str` | Session identifier | entry | executor, checkpointer |
| `analysis_id` | `str` | Analysis run identifier | entry | logging |
| `user_id` | `str` | External user ID | entry | — |
| `output_root` | `str` | Base output directory override | entry | executor |
| `status` | `str` | `"routed"` / `"planned"` / `"executing"` / `"completed"` / `"failed"` | all nodes | entry |

**Annotated merge reducers:**
- `_merge_dicts`: Shallow dict merge, drops values > 10,000 chars. Used for `workflow_outputs`, `uploaded_files`, `detected_file_types`, `retry_counts`.
- `_capped_add`: List append with cap at `MAX_CONVERSATION_HISTORY` (default 50). Used for `agent_results`, `errors`, `conversation_history`, `llm_call_log`.

### `ConversationState` class (`supervisor_agent/state.py`)

| Field | Type | Purpose |
|-------|------|---------|
| `session_id` | `str` | UUID for session (auto-generated) |
| `user_id` | `Optional[str]` | External user ID |
| `created_at` | `float` | Session creation timestamp |
| `last_activity` | `float` | Last interaction timestamp |
| `messages` | `List[Message]` | Full conversation history |
| `uploaded_files` | `Dict[str, UploadedFile]` | Registered uploaded files |
| `agent_executions` | `List[AgentExecution]` | Agent run history |
| `workflow_state` | `Dict[str, Any]` | Output keys from completed agents |
| `current_disease` | `Optional[str]` | Active disease context |
| `current_agent` | `Optional[str]` | Currently running agent name |
| `waiting_for_input` | `bool` | Whether awaiting user input |
| `required_inputs` | `List[str]` | What inputs are needed |
| `pending_agent_type` | `Optional[str]` | Agent type waiting for inputs |

### `SessionManager` (`supervisor_agent/state.py`)

In-memory dict-based session store. `MAX_SESSIONS` env var caps active sessions (default 1000). For production, replace with Redis or database-backed store.

Key methods:
- `get_or_create_session(session_id: Optional[str]) -> ConversationState`
- `get_session(session_id: str) -> Optional[ConversationState]`

### `supervisor_agent/langgraph/nodes.py` — Node Details

#### `intent_node(state, config) -> dict`

**Reads:** `user_query`, `uploaded_files`, `workflow_outputs`, `disease_name`, `conversation_history`, `response_format` (from checkpoint for confirmation fast-path)

**Writes:** `routing_decision`, `detected_file_types`, `disease_name`, `is_general_query`, `general_response`, `final_response`, `needs_response`, `response_format`, `molecular_report_format`, `report_theme`, `style_instructions`, `requested_top_n`, `has_existing_results`, `workflow_outputs`, `conversation_history`, `status`

**Logic:**
1. **Confirmation fast-path**: Short replies like "ok", "yes", "go ahead" replay the previous turn's pending document format
2. **File type detection**: Iterates `uploaded_files`, calls `_detect_file_type()` for each
3. **LLM routing**: Creates `ConversationState` shim, calls `IntentRouter.route(query, conv_state)` for LLM-based intent classification
4. **Format detection** (all regex, zero LLM):
   - `_detect_needs_response(query)`: Matches reporting signals ("what are", "show me", "summarize") and follow-up signals ("elaborate", "more detail")
   - `_detect_response_format(query, needs_resp)`: Detects "pdf"/"docx"/"word" keywords, or implicit doc request ("generate a report")
   - `_detect_report_scope(query)`: "brief"/"comprehensive"/"standard" from keywords
   - `_extract_top_n(query)`: Extracts N from "top N" patterns (default 10, max 100)
   - `_detect_report_theme(query)`: "clinical"/"minimal"/"default" from keywords
5. **Style carry-forward**: If prior turn was PDF/DOCX and current query has style keywords (colors, fonts), inherits prior format
6. **Molecular report detection**: Regex `_MOLECULAR_REPORT_RE` matches "molecular report", "patient report", "comprehensive report", "full report"

#### `response_node(state, config) -> dict`

**Full flow:**
1. **Passthrough check**: If `workflow_outputs` has `report_pdf_path` or `report_docx_path` (from molecular report pipeline), surfaces those files directly
2. **CSV discovery**: `discover_relevant_csvs()` finds and scores CSVs across all agent output directories
3. **Regex fast-path**: `regex_fast_path()` tries deterministic answers for "top N genes/pathways/drugs" and "tell me about GENE" queries
4. **API enrichment** (parallel): `enrich_for_response()` calls external APIs (Ensembl, STRING, DGIdb, ChEMBL, Reactome, KEGG, ClinicalTrials.gov) for entity context
5. **CSS generation** (parallel, if PDF/DOCX with style instructions): `generate_style_css()` via LLM
6. **LLM synthesis**:
   - Chat format → `synthesize_chat_multipass()` (2 passes)
   - Document format → `synthesize_multipass()` (2-4 passes based on scope)
7. **PDF/DOCX rendering** (if requested): `render_pdf()` or `render_docx()`, saves to `agentic_ai_wf/shared/reports/dynamic_reports/{timestamp}/`

#### `agent_executor(state, config) -> dict`

1. Reads `execution_plan[current_agent_index]` to get current agent type
2. Looks up `EXECUTOR_MAP[agent_type]` for the executor function
3. Builds `available_inputs` from `workflow_outputs` + `uploaded_files` + `extracted_params`
4. **Pre-flight validation**: Checks required file paths exist on disk
5. **Execution**: Calls executor as async generator, forwards `StatusUpdate`s to progress callback
6. **Retry logic**: Classifies errors as transient (retryable: API rate limits, timeouts, runtime errors) or permanent (auth, data format, missing files). Max retries = `SUPERVISOR_MAX_RETRIES` env var (default 2). Exponential backoff.
7. **Skip logic**: On failure, `_compute_skip_to()` skips downstream agents that depend on the failed agent's outputs
8. Updates `workflow_outputs`, `current_agent_index`, `agent_results` or `errors`

#### `reporting_node()` — Structured Evidence Report

Invoked when user requests "structured report", "evidence report", or "analysis report" after pipeline runs. Uses the `reporting_engine` sub-package: builds a manifest from state, plans sections, builds evidence-scored sections, detects cross-module conflicts, renders validated markdown.

### `supervisor_agent/langgraph/edges.py`

#### `check_general_query(state) -> str`

Decision tree:
1. If `has_existing_results=True` AND query matches `_STRUCTURED_REPORT_RE` → `"structured_report"`
2. If `is_general_query=True`:
   - If `response_format` is `"pdf"` or `"docx"` → `"follow_up"` (to render document)
   - If `needs_response=True` → `"follow_up"` (for LLM synthesis)
   - Else → `"general"` (short-circuit to END)
3. If `has_existing_results=True` AND `needs_response=True` AND target agent's outputs already exist → `"follow_up"`
4. Else → `"needs_execution"`

#### `route_next(state) -> str`

Decision tree:
1. If `current_agent_index < len(execution_plan)` → `"execute_agent"`
2. If all agents done AND query matches `_STRUCTURED_REPORT_RE` → `"structured_report"`
3. If `needs_response=True` → `"report"`
4. If errors exist but no agent_results → `"report"` (error summary)
5. Else → `"done"`

---

## 4. Intent Router

### `supervisor_agent/router.py`

#### `IntentRouter` class

```python
class IntentRouter:
    def __init__(self, api_key: Optional[str] = None):
        # Pre-builds agent_capabilities text from AGENT_REGISTRY
        # No api_key needed — uses llm_complete() which reads from env

    async def route(
        self,
        user_query: str,                    # The user's message
        conversation_state: ConversationState  # Current session state
    ) -> RoutingDecision
```

**`ROUTING_SYSTEM_PROMPT` contains:**
- Full agent capabilities text (auto-generated from `AGENT_REGISTRY`)
- Agent dependency ordering rules (13 numbered agents with strict order)
- Multi-agent query examples
- MDP multi-disease examples
- Topic-change detection instructions
- Expected JSON response format

**LLM JSON schema the router expects:**

```json
{
    "is_multi_agent": true,
    "agents": [
        {
            "agent_name": "deg_analysis",
            "confidence": 0.95,
            "reasoning": "User wants differential expression",
            "order": 1
        }
    ],
    "extracted_params": {
        "disease_name": "lupus",
        "disease_names": ["lupus"],
        "tissue_filter": null,
        "experiment_filter": null,
        "technique": null
    },
    "is_general_query": false,
    "suggested_response": null,
    "reasoning": "User wants DEG analysis on uploaded data"
}
```

**`RoutingDecision` dataclass fields:**

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | `Optional[AgentType]` | Primary agent (first in pipeline) |
| `agent_name` | `Optional[str]` | Agent name string |
| `confidence` | `float` | 0.0–1.0 confidence score |
| `reasoning` | `str` | Human-readable explanation |
| `extracted_params` | `Dict[str, Any]` | Parameters extracted from query |
| `missing_inputs` | `List[str]` | Required inputs not available |
| `is_general_query` | `bool` | True if no specific agent needed |
| `suggested_response` | `Optional[str]` | LLM response for general queries |
| `is_multi_agent` | `bool` | True if multiple agents detected |
| `agent_pipeline` | `List[AgentIntent]` | Ordered list of agents to execute |

**How `is_general` vs agent routing works:**
- LLM returns `is_general_query: true` for knowledge questions, greetings, follow-ups
- LLM returns `is_general_query: false` with specific agent(s) for pipeline requests
- Multi-agent plans: LLM returns `is_multi_agent: true` with ordered `agents` list
- Pipeline is **re-sorted** by `PIPELINE_ORDER` to enforce correct dependency order (e.g., gene_prioritization always before pathway_enrichment)

**`BEDROCK_LIGHT_MODEL_ID` optimization:** The router uses the same `llm_complete()` function as all other LLM calls. There is no separate light model for routing — all calls go through the primary Bedrock model (or OpenAI fallback). The `BEDROCK_LIGHT_MODEL_ID` environment variable is referenced in the `ROUTING_SYSTEM_PROMPT` comments but routing uses the standard `BEDROCK_MODEL_ID`.

---

## 5. Agent Registry & Pipeline Executors

### `supervisor_agent/agent_registry.py`

#### `AgentInfo` dataclass fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | `AgentType` | Enum value |
| `name` | `str` | Unique name string |
| `display_name` | `str` | Human-readable name with emoji |
| `description` | `str` | One-line description |
| `detailed_description` | `str` | Full multi-paragraph description |
| `required_inputs` | `List[InputRequirement]` | What this agent needs |
| `optional_inputs` | `List[InputRequirement]` | Optional inputs |
| `outputs` | `List[OutputSpec]` | What this agent produces |
| `keywords` | `List[str]` | Intent matching keywords |
| `example_queries` | `List[str]` | Example user queries |
| `estimated_time` | `str` | Runtime estimate string |
| `depends_on` | `List[AgentType]` | Direct dependencies |
| `produces` | `List[str]` | State keys this agent writes |
| `requires_one_of` | `List[str]` | At least one must be available |

#### Complete Agent Registry

| Agent | Name | Required Inputs | Produces (State Keys) | Depends On | Est. Time |
|-------|------|----------------|----------------------|------------|-----------|
| Cohort Retrieval | `cohort_retrieval` | `disease_name` | `cohort_output_dir`, `cohort_summary_text` | — | 2-10 min |
| DEG Analysis | `deg_analysis` | `counts_file` OR `cohort_output_dir`, `disease_name` | `deg_base_dir`, `deg_input_file` | — | 5-15 min |
| Gene Prioritization | `gene_prioritization` | `deg_input_file` OR `deg_base_dir`, `disease_name` | `prioritized_genes_path` | DEG Analysis | 3-8 min |
| Pathway Enrichment | `pathway_enrichment` | `prioritized_genes_path`, `disease_name` | `pathway_consolidation_path` | Gene Prioritization | 5-15 min |
| Deconvolution | `deconvolution` | `bulk_file`, `disease_name` | `deconvolution_output_dir` | — | 5-20 min |
| Temporal Analysis | `temporal_analysis` | `counts_file`, `metadata_file`, `disease_name` | `temporal_output_dir` | — | 10-30 min |
| Harmonization | `harmonization` | `counts_file`, `metadata_file`, `disease_name` | `harmonization_output_dir` | — | 5-15 min |
| MDP Analysis | `mdp_analysis` | (none required) | `mdp_output_dir` | — | 5-20 min |
| Perturbation Analysis | `perturbation_analysis` | `prioritized_genes_path`, `pathway_consolidation_path`, `disease_name` | `perturbation_output_dir` | Pathway Enrichment | 15-45 min |
| Multi-Omics Integration | `multiomics_integration` | `disease_name`, `multiomics_layers` | `multiomics_output_dir` | — | 20-60 min |
| FASTQ Processing | `fastq_processing` | `fastq_input_dir`, `disease_name` | `fastq_output_dir` | — | 10-60 min |
| Molecular Report | `molecular_report` | `prioritized_genes_path`, `pathway_consolidation_path`, `disease_name` | `report_output_dir`, `report_docx_path`, `report_pdf_path` | Pathway Enrichment | 2-10 min |
| CRISPR Perturb-seq | `crispr_perturb_seq` | `crispr_10x_input_dir` | `crispr_perturb_seq_output_dir` | — | 30-120 min |
| CRISPR Screening | `crispr_screening` | `crispr_screening_input_dir` | `crispr_screening_output_dir` | — | 15-60 min |
| CRISPR Targeted | `crispr_targeted` | `protospacer` | `crispr_targeted_output_dir` | — | 15-60 min |
| Causality | `causality` | (none required) | `causality_output_dir` | — | 5-15 min |

#### `get_agent_capabilities_text()` and `get_agent_capabilities_slim()`

- `get_agent_capabilities_text()` → Returns formatted text listing all agents with name, description, keywords (first 10), example queries (first 3), and required inputs. Called by `IntentRouter.__init__()` to build the routing system prompt.
- `get_agent_capabilities_slim()` is not defined in the codebase. Only `get_agent_capabilities_text()` exists.

### Executor Function Table

| Agent | Executor Function | File Path | Input State Keys | Output State Keys | Output Directory Pattern | Runtime |
|-------|------------------|-----------|------------------|-------------------|--------------------------|---------|
| cohort_retrieval | `execute_cohort_retrieval` | `executors/pipeline_executors.py` | `disease_name` | `cohort_output_dir`, `cohort_summary_text` | `supervisor_agent/outputs/cohort_retrieval/{session_id}/{disease_safe}/` | 2-10 min |
| deg_analysis | `execute_deg_analysis` | `executors/pipeline_executors.py` | `counts_file` or `cohort_output_dir`, `disease_name`, `metadata_file` | `deg_base_dir`, `deg_input_file` | `supervisor_agent/outputs/deg_analysis/{session_id}/` | 5-15 min |
| gene_prioritization | `execute_gene_prioritization` | `executors/pipeline_executors.py` | `deg_base_dir` or `deg_input_file`, `disease_name` | `prioritized_genes_path`, `gene_prioritization_output_dir` | `supervisor_agent/outputs/gene_prioritization/{session_id}/` | 3-8 min |
| pathway_enrichment | `execute_pathway_enrichment` | `executors/pipeline_executors.py` | `prioritized_genes_path`, `disease_name` | `pathway_consolidation_path`, `pathway_enrichment_path`, `pathway_output_dir` | `supervisor_agent/outputs/pathway_enrichment/{session_id}/enrichment/` | 5-15 min |
| deconvolution | `execute_deconvolution` | `executors/pipeline_executors.py` | `bulk_file`, `disease_name`, `technique`, `h5ad_file`, `metadata_file` | `deconvolution_output_dir` | `supervisor_agent/outputs/deconvolution/{session_id}/` | 5-20 min |
| temporal_analysis | `execute_temporal_analysis` | `executors/pipeline_executors.py` | `counts_file`, `metadata_file`, `disease_name` | `temporal_output_dir` | `supervisor_agent/outputs/temporal_analysis/{session_id}/` | 10-30 min |
| harmonization | `execute_harmonization` | `executors/pipeline_executors.py` | `counts_file`, `metadata_file`, `disease_name` | `harmonization_output_dir` | `supervisor_agent/outputs/harmonization/{session_id}/` | 5-15 min |
| mdp_analysis | `execute_mdp_analysis` | `executors/pipeline_executors.py` | `disease_name`, `disease_names`, `counts_file` | `mdp_output_dir` | `supervisor_agent/outputs/mdp_analysis/{session_id}/` | 5-20 min |
| perturbation_analysis | `execute_perturbation_analysis` | `executors/pipeline_executors.py` | `prioritized_genes_path`, `pathway_consolidation_path`, `disease_name` | `perturbation_output_dir` | `supervisor_agent/outputs/perturbation_analysis/{session_id}/` | 15-45 min |
| multiomics_integration | `execute_multiomics_integration` | `executors/pipeline_executors.py` | `disease_name`, `multiomics_layers` dict | `multiomics_output_dir` | `supervisor_agent/outputs/multiomics_integration/{session_id}/` | 20-60 min |
| fastq_processing | `execute_fastq_processing` | `executors/pipeline_executors.py` | `fastq_input_dir`, `disease_name` | `fastq_output_dir` | `supervisor_agent/outputs/fastq_processing/{session_id}/` | 10-60 min |
| molecular_report | `execute_molecular_report` | `executors/pipeline_executors.py` | `prioritized_genes_path`, `pathway_consolidation_path`, `disease_name`, `xcell_path`, `patient_info_path` | `report_output_dir`, `report_docx_path`, `report_pdf_path` | `supervisor_agent/outputs/molecular_report/{session_id}/` | 2-10 min |
| crispr_perturb_seq | `execute_crispr_perturb_seq` | `executors/pipeline_executors.py` | `crispr_10x_input_dir` | `crispr_perturb_seq_output_dir` | `supervisor_agent/outputs/crispr_perturb_seq/{session_id}/` | 30-120 min |
| crispr_screening | `execute_crispr_screening` | `executors/pipeline_executors.py` | `crispr_screening_input_dir`, `modes` | `crispr_screening_output_dir` | `supervisor_agent/outputs/crispr_screening/{session_id}/` | 15-60 min |
| crispr_targeted | `execute_crispr_targeted` | `executors/pipeline_executors.py` | `crispr_targeted_input_dir`, `protospacer`, `target_gene` | `crispr_targeted_output_dir` | `supervisor_agent/outputs/crispr_targeted/{session_id}/` | 15-60 min |
| causality | `execute_causality` | `causality/adapter.py` | `disease_name`, `workflow_outputs` (reads upstream dirs) | `causality_output_dir` | `supervisor_agent/outputs/causality/{session_id}/` | 5-15 min |

**External dependencies by executor:**
- **DEG Analysis**: R + DESeq2, edgeR (via rpy2)
- **Gene Prioritization**: GeneCards API, PPI network databases
- **Pathway Enrichment**: KEGG, GO, Reactome, WikiPathways APIs
- **Deconvolution**: R + xCell/CIBERSORTx/BisQue packages (via rpy2)
- **Temporal Analysis**: R + impulse model packages
- **Harmonization**: R + sva (ComBat)
- **FASTQ Processing**: FastQC, Cutadapt, Trimmomatic, Salmon (system binaries)
- **CRISPR Screening/Targeted**: Nextflow, Singularity/Docker, nf-core/crisprseq, samtools, Java
- **Molecular Report**: WeasyPrint (PDF), python-docx + htmldocx (DOCX)

---

## 6. LLM Provider

### `supervisor_agent/llm_provider.py`

#### `llm_complete()` — Async unified LLM call

```python
async def llm_complete(
    messages: List[Dict[str, str]],       # Chat messages [{role, content}]
    temperature: float = 0.1,              # LLM temperature
    max_tokens: int = 4096,                # Max output tokens
    system: Optional[str] = None,          # System prompt
    response_format: Optional[Dict] = None # {"type": "json_object"} for JSON mode
) -> LLMResult
```

**`LLMResult` dataclass fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | LLM response text |
| `provider` | `str` | `"bedrock"` or `"openai"` |
| `model` | `str` | Model ID used |
| `latency_ms` | `int` | Response latency in ms |
| `fallback_used` | `bool` | True if OpenAI fallback was used |
| `intent` | `str` | Caller-populated telemetry label |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |

#### Bedrock Primary + OpenAI Fallback Chain

1. Check `USE_BEDROCK` env var (must be `"true"`) AND `AWS_ACCESS_KEY_ID` exists
2. If Bedrock available → call `_bedrock_invoke()` via `loop.run_in_executor()` (thread pool, max workers = `LLM_MAX_CONCURRENCY` env, default 10)
3. If Bedrock call raises any exception → log warning, fall through to OpenAI
4. OpenAI fallback → `AsyncOpenAI(api_key=OPENAI_API_KEY)` → `gpt-4o` model
5. If OpenAI also fails → raises `LLMProviderError`

#### Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `USE_BEDROCK` | Enable Bedrock as primary LLM | `"false"` | Yes (set to `"true"` for production) |
| `BEDROCK_MODEL_ID` | Bedrock model identifier | `us.anthropic.claude-opus-4-5-20251101-v1:0` | Yes (if Bedrock) |
| `AWS_ACCESS_KEY_ID` | AWS access key | — | Yes (if Bedrock) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — | Yes (if Bedrock) |
| `AWS_REGION` | AWS region | `us-east-1` | No |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | — | Yes (if no Bedrock) |
| `LLM_MAX_CONCURRENCY` | Thread pool size for Bedrock calls | `10` | No |

#### Token Logging Format

```
LLM [bedrock] 1523 in / 847 out tokens, 2341ms
LLM [openai] 2100 in / 1200 out tokens, 3500ms
```

---

## 7. Response Engine (Dynamic Response & Document Generation)

### `supervisor_agent/response/synthesizer.py`

#### `synthesize_multipass()` — 4-pass document pipeline

```python
async def synthesize_multipass(
    user_prompt: str,
    data_context: str = "",
    pass_count: int = 3,
    system_prompt: str = "",
    intent_label: str = "multipass_synthesis",
) -> tuple[str, list[Dict[str, Any]]]
```

**Pass chain adapts to `pass_count`:**

| pass_count | Passes | Pipeline |
|-----------|--------|----------|
| 1 | Single draft | draft only |
| 2 | Draft → Review | draft → REVIEW_SYSTEM_PROMPT revision |
| 3 | Outline → Draft → Review | OUTLINE_SYSTEM_PROMPT → DOCUMENT_SYSTEM_PROMPT → REVIEW_SYSTEM_PROMPT |
| 4 | Outline → Draft → Review → Revise | + final polish pass at temperature=0.05 |

**Scope passes and budgets (from `SCOPE_PASSES` / `SCOPE_BUDGETS`):**

| Scope | Passes | Char Budget | Triggered By |
|-------|--------|-------------|-------------|
| `chat` | 2 | 8,000 | Default for non-document queries |
| `brief` | 2 | 10,000 | "brief", "summary", "quick", "short", "concise" |
| `standard` | 3 | 15,000 | Default for document queries |
| `comprehensive` | 4 | 20,000 | "comprehensive", "detailed", "everything", "full", "in-depth" |

#### `synthesize_chat_multipass()` — 2-pass chat pipeline

```python
async def synthesize_chat_multipass(
    user_prompt: str,
    enrichment_context: str = "",
    system_prompt: str = "",
    intent_label: str = "chat_multipass",
) -> tuple[str, list[Dict[str, Any]]]
```

- **Pass 1 (Analysis):** Uses `RESPONSE_SYSTEM_PROMPT` to extract findings, cross-module connections, and data references
- **Pass 2 (Refinement):** Uses `CHAT_REFINEMENT_PROMPT` to transform structured analysis into polished conversational response

### `supervisor_agent/response/document_renderer.py`

#### `render_pdf()`

```python
def render_pdf(
    title: str,          # Report title
    disease: str,        # Disease context (shown in header)
    content: str,        # Markdown content to render
    output_path: str,    # Absolute file path for output PDF
    theme: str = "default",   # "default" | "clinical" | "minimal"
    custom_css: str = ""      # LLM-generated CSS overrides
) -> None
```

**Dependencies:** `weasyprint`, `markdown` (with extensions: tables, fenced_code, toc)

#### `render_docx()`

```python
def render_docx(
    title: str,
    disease: str,
    content: str,
    output_path: str,
    theme: str = "default",
    custom_css: str = ""
) -> None
```

**Dependencies:** `python-docx`, `htmldocx`

**Post-processing:** Proportional column widths, repeat header rows on page breaks, base font (Calibri 11pt), heading color extraction from CSS.

### `supervisor_agent/response/style_engine.py`

**Theme presets:**

| Theme | Primary Color | Font | Header BG |
|-------|-------------|------|-----------|
| `default` | `#028090` | Segoe UI, system-ui | `#028090` |
| `clinical` | `#1a365d` | Georgia, Times New Roman | `#1a365d` |
| `minimal` | `#333333` | Helvetica Neue, Arial | `#f0f0f0` |

**Style extraction:** `extract_style_instructions(query, fmt, disease)` — regex-based, zero LLM. Only returns non-empty when query contains genuine style keywords (colors, fonts, layout, theming). Strips functional words, disease name, and standalone numbers.

**CSS generation:** `generate_style_css(style_instructions, base_theme)` — LLM call with `CSS_GENERATION_PROMPT` system prompt. Output is sanitized: `@import` blocked, `url()` blocked, `expression()` blocked, `javascript:` blocked, unbalanced braces truncated.

### `supervisor_agent/response/data_discovery.py`

**`discover_relevant_csvs(workflow_outputs, user_query, uploaded_files)`:**
- Scans all `_output_dir` and `_base_dir` paths in `workflow_outputs` for `**/*.csv`
- Scores by: header column match to query words, high-value filename patterns (+3 for Final_Gene_Priorities, pathway_consolidation, etc.), penalizes large files (>10MB) and deep nesting
- Returns top 8 scored CSVs

**`collect_output_summaries(workflow_outputs, user_query, top_n, uploaded_files, budget=8000)`:**
- Reads CSV head data (up to `top_n` rows) from discovered CSVs
- Respects char budget, returns `Dict[str, str]` of `{filename: pipe_delimited_table}`

### `supervisor_agent/response/query_functions.py`

**5 deterministic query functions:**

| Function | Trigger Pattern | CSV Type | Returns |
|----------|----------------|----------|---------|
| `get_top_genes(df, n, query)` | "top N genes/markers/candidates" | genes | Sorted by composite_score, gene_score, etc. |
| `get_top_pathways(df, n, query)` | "top N pathways/signaling" | pathways | Sorted by p_value, combined_score |
| `get_top_drugs(df, n, query)` | "top N drugs/compounds" | drugs | Sorted by score, confidence |
| `get_gene_detail(df, gene_name)` | "tell me about BRCA1" | genes | All columns for matched gene |
| `get_full_summary(df, n)` | "summary/overview/findings" | any | Head N rows of best DataFrame |

**`regex_fast_path(query, csvs_by_type, top_n, best_df)`:**
- Intent matching via compiled regex patterns
- Returns pipe-delimited text or `None` if no regex matches
- If matched, the fast-path data is still sent through LLM synthesis for natural language formatting

---

## 8. File I/O & Output Directory Map

### Output Directory Patterns

All agent outputs default to: `supervisor_agent/outputs/{agent_name}/{session_id}/`

Generated by `_get_output_dir(agent_name, session_id)` in `executors/pipeline_executors.py`. Both `agent_name` and `session_id` are validated against `^[a-zA-Z0-9_-]+$`.

| Agent | Output Directory Pattern | Key Files Produced | File Formats |
|-------|--------------------------|---------------------|-------------|
| cohort_retrieval | `outputs/cohort_retrieval/{session_id}/{disease_safe}/` | `*_counts_data.csv`, `*_metadata.csv`, `*_cohort_details.json` | CSV, JSON |
| deg_analysis | `outputs/deg_analysis/{session_id}/` | `*_DEGs.csv`, `*_MA_plot.png`, `*_volcano_plot.png` | CSV, PNG |
| gene_prioritization | `outputs/gene_prioritization/{session_id}/` | `*_Final_Gene_Priorities.csv`, `*_DEGs_prioritized.csv` | CSV |
| pathway_enrichment | `outputs/pathway_enrichment/{session_id}/enrichment/` | `*_Pathways_Consolidated.csv`, `*_enrichment.csv` | CSV |
| deconvolution | `outputs/deconvolution/{session_id}/` | `CIBERSORT_results.csv`, `*_cell_composition.png` | CSV, PNG |
| temporal_analysis | `outputs/temporal_analysis/{session_id}/` | `*_temporal_genes.csv`, `*_trajectory.png` | CSV, PNG |
| harmonization | `outputs/harmonization/{session_id}/` | `*_harmonized_counts.csv`, `*_QC_plots.png` | CSV, PNG |
| mdp_analysis | `outputs/mdp_analysis/{session_id}/` | `*_pathway_comparison.csv`, `*_shared_pathways.csv` | CSV |
| perturbation_analysis | `outputs/perturbation_analysis/{session_id}/` | `*_DEPMAP_results.csv`, `*_L1000_results.csv`, `*_integration.csv` | CSV |
| multiomics_integration | `outputs/multiomics_integration/{session_id}/` | `*_integration_results.csv`, `*_biomarkers.csv` | CSV, PNG |
| fastq_processing | `outputs/fastq_processing/{session_id}/` | `*_multiqc_report.html`, `*_quant.sf`, `*_gene_counts.csv` | HTML, CSV |
| molecular_report | `outputs/molecular_report/{session_id}/` | `*.docx`, `*.pdf` | DOCX, PDF |
| crispr_perturb_seq | `outputs/crispr_perturb_seq/{session_id}/` | Stage 1-13 outputs, `*_report.html` | CSV, PNG, HTML |
| crispr_screening | `outputs/crispr_screening/{session_id}/` | MAGeCK/BAGEL2 results | CSV, TSV |
| crispr_targeted | `outputs/crispr_targeted/{session_id}/` | `crisprseq_targeted/`, indel quantification | CSV, HTML |
| causality | `outputs/causality/{session_id}/` | Evidence matrices, causal narratives | CSV, JSON, MD |
| dynamic reports | `agentic_ai_wf/shared/reports/dynamic_reports/{timestamp}/` | `{disease_slug}_{timestamp}.pdf` or `.docx` | PDF, DOCX |

### File Upload Handling

Uploaded files arrive as `{filename: local_filepath}` dict in `uploaded_files` parameter. The `intent_node`:
1. Calls `_detect_file_type(filepath)` for each file (reads first few rows, inspects columns)
2. Maps detected types to input keys via `FILE_TYPE_TO_INPUT_KEY`:

| Detected Type | Maps To State Keys |
|--------------|-------------------|
| `raw_counts` | `counts_file`, `bulk_file` |
| `deg_results` | `deg_input_file` |
| `prioritized_genes` | `prioritized_genes_path` |
| `pathway_results` | `pathway_consolidation_path` |
| `multiomics_layer` | `multiomics_layers` |
| `patient_info` | `patient_info_path` |
| `deconvolution_results` | `xcell_path` |
| `crispr_10x_data` | `crispr_10x_input_dir` |
| `crispr_count_table` | `crispr_screening_input_dir` |
| `crispr_fastq_data` | `crispr_targeted_input_dir` |

3. Additional heuristics: metadata filenames (`_METADATA_RE`), `.h5ad` → `h5ad_file`, DEG/prioritized files → `deg_base_dir` (parent directory)

### Hardcoded vs Configurable Paths

- **Agent output dirs**: Relative to `supervisor_agent/outputs/`. Hardcoded base path in `_get_output_dir()` = `Path(__file__).parent.parent / "outputs"`. Override via `output_root` parameter in `run_supervisor()`.
- **Dynamic report dir**: Hardcoded to `agentic_ai_wf/shared/reports/dynamic_reports/{timestamp}/`. Falls back to `tempfile.gettempdir()` on permission error.
- **CRISPR runtime dirs**: `{project_root}/temp/supervisor_crispr/{session_id}/{name}/`

---

## 9. State Flow & Data Dependencies

### Main Pipeline Chain

```
cohort_retrieval → deg_analysis → gene_prioritization → pathway_enrichment → perturbation_analysis → molecular_report
```

This is defined in `PIPELINE_ORDER`:
```python
PIPELINE_ORDER = [
    AgentType.COHORT_RETRIEVAL,
    AgentType.DEG_ANALYSIS,
    AgentType.GENE_PRIORITIZATION,
    AgentType.PATHWAY_ENRICHMENT,
    AgentType.PERTURBATION_ANALYSIS,
    AgentType.MOLECULAR_REPORT,
]
```

### Independent Modules (can run in parallel / standalone)

- `deconvolution` — requires `bulk_file`
- `temporal_analysis` — requires `counts_file` + `metadata_file`
- `harmonization` — requires `counts_file` + `metadata_file`
- `multiomics_integration` — requires `multiomics_layers`
- `mdp_analysis` — no required inputs (uses counts/DEGs if available, or Neo4j knowledge graph)
- `fastq_processing` — requires `fastq_input_dir`
- `crispr_perturb_seq` — requires `crispr_10x_input_dir`
- `crispr_screening` — requires `crispr_screening_input_dir`
- `crispr_targeted` — requires `protospacer`
- `causality` — no required inputs (reads upstream outputs if available)

### Dependent Modules

- `perturbation_analysis` — requires `prioritized_genes_path` + `pathway_consolidation_path` (must run after pathway_enrichment)
- `molecular_report` — requires `prioritized_genes_path` + `pathway_consolidation_path` (must run after pathway_enrichment)

### State Key Flow

| State Key | Produced By | Consumed By |
|-----------|------------|-------------|
| `cohort_output_dir` | `cohort_retrieval` | `deg_analysis` (as `geo_dir`) |
| `cohort_summary_text` | `cohort_retrieval` | `response_node` (data discovery) |
| `deg_base_dir` | `deg_analysis` | `gene_prioritization` |
| `deg_input_file` | `deg_analysis` | `gene_prioritization` |
| `prioritized_genes_path` | `gene_prioritization` | `pathway_enrichment`, `perturbation_analysis`, `molecular_report` |
| `gene_prioritization_output_dir` | `gene_prioritization` | `response_node` (data discovery) |
| `pathway_consolidation_path` | `pathway_enrichment` | `perturbation_analysis`, `molecular_report` |
| `pathway_enrichment_path` | `pathway_enrichment` | `response_node` (data discovery) |
| `pathway_output_dir` | `pathway_enrichment` | `response_node` (data discovery) |
| `deconvolution_output_dir` | `deconvolution` | `molecular_report` (optional: `xcell_path`), `response_node` |
| `temporal_output_dir` | `temporal_analysis` | `response_node` |
| `harmonization_output_dir` | `harmonization` | `response_node` |
| `mdp_output_dir` | `mdp_analysis` | `response_node` |
| `perturbation_output_dir` | `perturbation_analysis` | `response_node` |
| `multiomics_output_dir` | `multiomics_integration` | `response_node` |
| `fastq_output_dir` | `fastq_processing` | `response_node` |
| `report_output_dir` | `molecular_report` | `response_node` |
| `report_docx_path` | `molecular_report` | `response_node` (passthrough) |
| `report_pdf_path` | `molecular_report` | `response_node` (passthrough) |
| `crispr_perturb_seq_output_dir` | `crispr_perturb_seq` | `response_node` |
| `crispr_screening_output_dir` | `crispr_screening` | `response_node` |
| `crispr_targeted_output_dir` | `crispr_targeted` | `response_node` |
| `causality_output_dir` | `causality` | `response_node` |
| `custom_css` | `response_node` | `response_node` (carry-forward across turns) |
| `response_report_path` | `response_node` | `entry.py` (terminal update) |

---

## 10. Configuration & Environment Variables

### Complete Environment Variable Reference

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `USE_BEDROCK` | Enable Bedrock as primary LLM provider | `"false"` | Yes |
| `BEDROCK_MODEL_ID` | Bedrock Claude model ID | `us.anthropic.claude-opus-4-5-20251101-v1:0` | If Bedrock |
| `AWS_ACCESS_KEY_ID` | AWS access key for Bedrock | — | If Bedrock |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for Bedrock | — | If Bedrock |
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` | No |
| `OPENAI_API_KEY` | OpenAI API key (fallback or primary) | — | If no Bedrock |
| `USE_LANGGRAPH_SUPERVISOR` | Use LangGraph graph instead of legacy path | `"false"` | Yes (set `"true"`) |
| `CHECKPOINTER_BACKEND` | LangGraph checkpointer: `"memory"` or `"postgres"` | `"memory"` | No |
| `CHECKPOINTER_DB_URL` | PostgreSQL connection string for checkpointer | — | If postgres |
| `MAX_CONVERSATION_HISTORY` | Max conversation turns retained in state | `50` | No |
| `MAX_SESSIONS` | Max active sessions in SessionManager | `1000` | No |
| `MAX_CSV_BYTES` | Max CSV file size for data discovery (bytes) | `104857600` (100MB) | No |
| `SUPERVISOR_MAX_RETRIES` | Max retry attempts per agent on transient failure | `2` | No |
| `LLM_MAX_CONCURRENCY` | Thread pool size for Bedrock executor | `10` | No |

### R/Bioconductor Dependencies

Required for DEG analysis, deconvolution, temporal analysis, harmonization:

| R Package | Used By |
|-----------|---------|
| `DESeq2` | DEG Analysis |
| `edgeR` | DEG Analysis (alternative) |
| `sva` (ComBat) | Harmonization |
| `xCell` | Deconvolution |
| `BisqueRNA` | Deconvolution (BisQue method) |
| `Seurat` | Deconvolution, Perturb-seq |

### Python Key Dependencies

| Package | Purpose | Used By |
|---------|---------|---------|
| `langgraph` | Graph orchestration framework | Graph compilation, state management |
| `langchain-core` | RunnableConfig type | Node config passing |
| `weasyprint` | PDF rendering from HTML | `render_pdf()` |
| `python-docx` + `htmldocx` | DOCX rendering from HTML | `render_docx()` |
| `markdown` | Markdown → HTML conversion | Document renderer |
| `boto3` + `botocore` | AWS Bedrock client | LLM provider |
| `openai` | OpenAI client (fallback) | LLM provider |
| `pandas` | DataFrame operations | Data discovery, query functions |
| `rpy2` | R language bridge | DEG, deconvolution, harmonization |
| `streamlit` | Current demo UI (not needed for production) | `streamlit_app/app.py` |
| `python-dotenv` | `.env` file loading | Multiple modules |

---

## 11. Error Handling & Recovery

### Error Propagation Flow

```
Executor raises exception
    → _classify_error(exc) → (category, is_retryable)
    → If retryable AND attempt <= max_retries:
        → _cleanup_partial_output() (removes partial files)
        → Backoff sleep (2^attempt for API, 5^attempt for runtime)
        → Retry executor
    → Else:
        → Error added to state.errors[]
        → _compute_skip_to() determines how many downstream agents to skip
        → current_agent_index jumps to next runnable agent
        → StatusUpdate(ERROR) sent to progress callback
```

### Error Categories

| Category | Retryable | Pattern Match |
|----------|-----------|--------------|
| `transient_api` | Yes | 429, rate limit, timeout, 502, 503, connection errors |
| `transient_runtime` | Yes | Cannot schedule, broken pipe, non-zero exit, CalledProcessError |
| `permanent_input` | No | Not found, no files, directory missing, does not exist |
| `permanent_auth` | No | 401, 403, unauthorized, forbidden |
| `permanent_data` | No | Column, shape, parse, decode, format, incompatible, mismatch |
| `unknown` | No | Everything else |

### LangGraph Checkpointing

**MemorySaver (default):** In-memory checkpointing for development. State is lost on process restart. Conversation state (including `conversation_history`) persists across turns within the same process.

**PostgresSaver (production):** Set `CHECKPOINTER_BACKEND=postgres` and `CHECKPOINTER_DB_URL=postgresql://user:pass@host:5432/db`. Enables crash recovery — if process dies mid-execution, restart and re-invoke with same `session_id` to resume from last checkpoint.

### Retry Configuration

- **`SUPERVISOR_MAX_RETRIES`** (default 2): Total retry attempts per agent
- **Backoff**: `2^attempt` seconds for `transient_api`, `5^attempt` seconds for `transient_runtime`
- **Partial cleanup**: `_cleanup_partial_output()` removes `outputs/{agent_name}/{session_id}/` before retry

### Monitoring: Key Log Lines

```
# Routing decision
📍 Routing decision: agent=deg_analysis, confidence=0.95, general=False

# Agent execution
🚀 Running DEG Analysis (1/3)
Agent DEG Analysis completed in 45.2s
Agent DEG Analysis completed in 45.2s (after 1 retries)

# LLM calls
LLM [bedrock] 1523 in / 847 out tokens, 2341ms
LLM [openai] 2100 in / 1200 out tokens, 3500ms

# Errors
Pre-flight failed for deg_analysis: Required input 'counts_file' not found at: /data/missing.csv
Retrying deg_analysis (attempt 2/3) after transient_api error: 429 rate limit  — backoff 4s
Agent deg_analysis failed (permanent_data): Column 'condition' not found in metadata

# Style engine
style_engine: generated 342 chars CSS for 'red headings, dark mode'

# Response
response_node: fmt=pdf, top_n=10
```

---

## 12. Integration Checklist

### Step-by-step for production deployment

- [ ] **1. Install Python dependencies**
  ```bash
  pip install -r requirements.txt
  # Key packages: langgraph, langchain-core, weasyprint, python-docx, htmldocx,
  # boto3, openai, pandas, rpy2, markdown, python-dotenv
  ```

- [ ] **2. Install R dependencies** (for DEG, deconvolution, harmonization, temporal analysis)
  ```R
  install.packages("BiocManager")
  BiocManager::install(c("DESeq2", "edgeR", "sva", "xCell", "BisqueRNA"))
  # Optional: install.packages("Seurat") for Perturb-seq
  ```

- [ ] **3. Install system dependencies** (for FASTQ processing, CRISPR)
  ```bash
  # FastQC, Cutadapt, Trimmomatic, Salmon — for FASTQ processing
  # Nextflow, Singularity — for CRISPR screening/targeted
  # samtools, Java — for CRISPR targeted
  # WeasyPrint system dependencies:
  apt-get install -y libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
  ```

- [ ] **4. Set environment variables** (`.env` file or system env)
  ```bash
  USE_BEDROCK=true
  BEDROCK_MODEL_ID=us.anthropic.claude-opus-4-5-20251101-v1:0
  AWS_ACCESS_KEY_ID=your_key
  AWS_SECRET_ACCESS_KEY=your_secret
  AWS_REGION=us-east-1
  OPENAI_API_KEY=sk-fallback-key     # Fallback
  USE_LANGGRAPH_SUPERVISOR=true
  CHECKPOINTER_BACKEND=postgres       # For production
  CHECKPOINTER_DB_URL=postgresql://user:pass@localhost:5432/supervisor
  SUPERVISOR_MAX_RETRIES=2
  ```

- [ ] **5. Create FastAPI endpoint** that calls `run_supervisor()` (see Section 2 for code)

- [ ] **6. Create Celery task** that calls `run_supervisor()` via `asyncio.run()` (see Section 2 for code)

- [ ] **7. Set up WebSocket bridge** for `StatusUpdate` streaming during pipeline execution (see Section 2 for WebSocket example)

- [ ] **8. Configure output directory** — ensure the process has write access to `supervisor_agent/outputs/` and `agentic_ai_wf/shared/reports/dynamic_reports/`

- [ ] **9. Wire file upload** — map uploaded files to `{filename: local_filepath}` dict and pass as `uploaded_files` parameter. Files should land on EC2 local disk. The path in the dict must be the absolute local path.

- [ ] **10. Auth hooks** — wrap `run_supervisor()` and `run_supervisor_stream()` calls with your external auth middleware. The `user_id` parameter flows through state but is not enforced. Add auth at the FastAPI/Django layer.

- [ ] **11. Verify with general query:**
  ```python
  result = await run_supervisor(
      user_query="what is erbb2 in breast cancer",
      session_id="test-general-001",
  )
  assert result.status == "completed"
  assert len(result.final_response) > 100
  ```

- [ ] **12. Verify pipeline execution:**
  ```python
  result = await run_supervisor(
      user_query="run DEG analysis for lupus",
      session_id="test-pipeline-001",
      disease_name="lupus",
      uploaded_files={"lupus_counts.csv": "/data/test/lupus_counts.csv"},
  )
  assert result.status == "completed"
  assert any(r["agent"] == "deg_analysis" for r in result.agent_results)
  ```

---

## 13. Async/Sync Boundaries (for Celery Integration)

### Fast Operations (< 5 seconds)

| Operation | Where | Sync/Async |
|-----------|-------|-----------|
| Intent detection + routing | `intent_node` | Async (single LLM call ~1-3s) |
| File type detection | `_detect_file_type()` | Sync (reads first rows of CSV) |
| Format/scope detection | `_detect_response_format()`, `_detect_report_scope()` | Sync (regex only) |
| Execution plan building | `plan_node` | Sync (in-memory logic) |
| File classification | `classify_csv_type()` | Sync (filename + column check) |

### Slow Operations (minutes to hours)

| Operation | Where | Typical Duration |
|-----------|-------|-----------------|
| Pipeline execution (DEG) | `agent_executor` → `execute_deg_analysis` | 5-15 min |
| Pipeline execution (pathway) | `agent_executor` → `execute_pathway_enrichment` | 5-15 min |
| Pipeline execution (perturbation) | `agent_executor` → `execute_perturbation_analysis` | 15-45 min |
| Pipeline execution (CRISPR) | `agent_executor` → `execute_crispr_*` | 15-120 min |
| FASTQ processing | `agent_executor` → `execute_fastq_processing` | 10-60 min |
| Multi-pass document synthesis | `response_node` → `synthesize_multipass` | 30-90s (3-4 LLM calls) |
| PDF/DOCX rendering | `response_node` → `render_pdf/render_docx` | 5-30s |

### Recommended Split

```
Frontend (Next.js) → FastAPI → Intent detection (fast, sync response)
                            ↓
                    If needs_execution:
                        → Push to Celery queue
                        → Return task_id to frontend
                        → Celery worker runs run_supervisor()
                        → Redis pub/sub for StatusUpdate streaming
                        → WebSocket pushes updates to frontend
                            ↓
                    If general query (is_general=True, no pipeline):
                        → Run run_supervisor() in FastAPI async handler
                        → Return response directly (2-5s)
```

### Function Boundary

- **`run_supervisor()`** — Full async invocation, returns `SupervisorResult`. Use for Celery tasks via `asyncio.run()`.
- **`run_supervisor_stream()`** — Full async generator, yields `StatusUpdate` objects. Use for real-time WebSocket streaming.

### Progress Callback for Celery WebSocket Updates

`run_supervisor_stream()` uses an internal `asyncio.Queue` with callback-based bridge. For Celery, consume the async generator and publish each `StatusUpdate` to Redis:

```python
# In Celery task
async def _run_with_progress(query, session_id, disease_name, files, redis_client):
    async for update in run_supervisor_stream(
        user_query=query,
        session_id=session_id,
        disease_name=disease_name,
        uploaded_files=files,
    ):
        redis_client.publish(
            f"supervisor:progress:{session_id}",
            json.dumps({
                "status_type": update.status_type.value,
                "title": update.title,
                "message": update.message,
                "progress": update.progress,
                "agent_name": update.agent_name,
                "generated_files": update.generated_files,
            })
        )

# WebSocket handler subscribes to supervisor:progress:{session_id}
```

Alternatively, use `run_supervisor()` (non-streaming) for Celery and implement coarse-grained progress via Celery task state updates. The streaming approach gives finer-grained feedback but requires the Redis pub/sub bridge.
