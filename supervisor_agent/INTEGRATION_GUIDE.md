# Supervisor Agent — Integration Guide for Software Lead

**Purpose:** This document provides everything needed to replace the Streamlit test UI with your production frontend. It covers the exact API surface, data contracts, streaming protocol, state management, file handling, and deployment considerations.

**Date:** 2026-03-25
**Codebase Version:** `supervisor_agent_V0.1`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Two Integration Paths](#2-two-integration-paths)
3. [Preferred Path: LangGraph Entry Points](#3-preferred-path-langgraph-entry-points)
4. [Legacy Path: SupervisorAgent Class](#4-legacy-path-supervisoragent-class)
5. [Streaming Protocol & StatusUpdate Contract](#5-streaming-protocol--statusupdate-contract)
6. [Session & State Management](#6-session--state-management)
7. [File Upload Handling](#7-file-upload-handling)
8. [File Download & Report Retrieval](#8-file-download--report-retrieval)
9. [Agent Registry (Capabilities Discovery)](#9-agent-registry-capabilities-discovery)
10. [Error Handling Contract](#10-error-handling-contract)
11. [Concurrency & Multi-User Considerations](#11-concurrency--multi-user-considerations)
12. [Environment & Configuration](#12-environment--configuration)
13. [Known Limitations & Gotchas](#13-known-limitations--gotchas)
14. [Suggested Backend Architecture](#14-suggested-backend-architecture)
15. [Quick-Start: Minimal Integration Example](#15-quick-start-minimal-integration-example)

---

## 1. Architecture Overview

```
┌──────────────┐         ┌──────────────────────────────────────────┐
│  Your UI     │         │  supervisor_agent/                       │
│  (replaces   │  HTTP/  │                                          │
│  Streamlit)  │  WS     │  ┌──────────────────────────────────┐   │
│              │ ◄──────►│  │ langgraph/entry.py               │   │
│  - Chat UI   │         │  │  run_supervisor_stream()  ← USE  │   │
│  - Uploads   │         │  │  run_supervisor()         ← USE  │   │
│  - Downloads │         │  └──────────┬───────────────────────┘   │
│  - Progress  │         │             │                            │
│              │         │  ┌──────────▼───────────────────────┐   │
│              │         │  │ LangGraph StateGraph              │   │
│              │         │  │ intent → plan → execute → respond │   │
│              │         │  └──────────────────────────────────┘   │
└──────────────┘         └──────────────────────────────────────────┘
```

The Streamlit app (`streamlit_app/app.py`) is a **test harness only**. It wraps the supervisor agent with a chat UI. Your production frontend replaces Streamlit entirely and talks directly to the supervisor's Python API.

**What Streamlit does that you'll need to replicate:**
1. Accepts user text input and file uploads
2. Calls the supervisor's async streaming API
3. Consumes `StatusUpdate` objects for real-time progress
4. Displays the final response (markdown text)
5. Offers file downloads (CSVs, PDFs, DOCXs, PNGs)
6. Maintains session continuity across conversation turns

**What Streamlit does that you can ignore:**
- Custom CSS styling (lines 61-270 of app.py)
- `enhanced_prompts.py` — duplicate of prompts already in `response/synthesizer.py` (not used at runtime)
- The `pages/1_structured_report.py` viewer — just reads a markdown file and renders it

---

## 2. Two Integration Paths

| Path | Entry Point | Best For | Status |
|------|-------------|----------|--------|
| **A. LangGraph (Preferred)** | `langgraph.entry.run_supervisor_stream()` | New integrations, clean API | Production-ready |
| **B. SupervisorAgent (Legacy)** | `supervisor.SupervisorAgent.process_message()` | Backward compat with old Streamlit | Works but wraps Path A |

**Recommendation:** Use **Path A** directly. The `SupervisorAgent.process_message()` method (Path B) just delegates to `run_supervisor_stream()` internally (see `supervisor.py:1860-1895`) and adds a thin ConversationState sync layer. If you manage your own session state, Path A is simpler and more direct.

---

## 3. Preferred Path: LangGraph Entry Points

### 3.1 Streaming Mode — `run_supervisor_stream()`

**Import:**
```python
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream
from agentic_ai_wf.supervisor_agent.executors.base import StatusType, StatusUpdate
```

**Signature:**
```python
async def run_supervisor_stream(
    user_query: str,                              # Required: the user's message
    session_id: str,                              # Required: unique session identifier
    analysis_id: str = "",                        # Optional: unique ID for this analysis run
    output_root: str = "",                        # Optional: base dir for outputs
    disease_name: str = "",                       # Optional: carry-forward disease context
    uploaded_files: Optional[Dict[str, str]] = None,  # Optional: {filename: local_filepath}
    workflow_outputs: Optional[Dict[str, Any]] = None, # Optional: prior turn's outputs
) -> AsyncGenerator[StatusUpdate, None]:
```

**Usage:**
```python
import asyncio

async def handle_user_message(query: str, session_id: str, files: dict):
    async for update in run_supervisor_stream(
        user_query=query,
        session_id=session_id,
        uploaded_files=files,          # {"counts.csv": "/tmp/uploads/counts.csv"}
        workflow_outputs=prior_outputs, # from previous turn's _graph_state
    ):
        # Send update to frontend via WebSocket/SSE
        send_to_frontend({
            "type": update.status_type.value,   # "thinking", "routing", "executing", etc.
            "title": update.title,               # Short display title
            "message": update.message,           # Main content (markdown)
            "details": update.details,           # Collapsible extra info (optional)
            "progress": update.progress,         # 0.0 to 1.0 (optional)
            "agent_name": update.agent_name,     # Which agent is running (optional)
            "generated_files": update.generated_files,  # Files created (on COMPLETED)
            "timestamp": update.timestamp,
        })

        # The LAST StatusUpdate (COMPLETED or ERROR) contains the final answer
        if update.status_type in (StatusType.COMPLETED, StatusType.ERROR):
            final_response = update.message

            # CRITICAL: Extract _graph_state for session continuity
            graph_state = getattr(update, '_graph_state', None)
            if graph_state:
                # Store these for the NEXT turn's workflow_outputs parameter
                save_session_state(session_id, {
                    "workflow_outputs": graph_state.get("workflow_outputs", {}),
                    "disease_name": graph_state.get("disease_name", ""),
                })

            # Check for generated report files
            if hasattr(update, 'generated_files') and update.generated_files:
                report_paths = update.generated_files  # List of file paths
```

### 3.2 Non-Streaming Mode — `run_supervisor()`

For batch/API callers that don't need real-time progress:

```python
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor
from agentic_ai_wf.supervisor_agent.langgraph.state import SupervisorResult

result: SupervisorResult = await run_supervisor(
    user_query="What are the top 10 differentially expressed genes?",
    session_id="session-123",
    uploaded_files={"counts.csv": "/data/counts.csv"},
)

# SupervisorResult fields:
result.status              # "completed" | "failed"
result.final_response      # Markdown string — the answer
result.output_dir          # Path to output directory
result.agent_results       # List of per-agent result dicts
result.errors              # List of error dicts
result.execution_time_ms   # Total execution time
```

---

## 4. Legacy Path: SupervisorAgent Class

If you prefer the legacy interface (used by Streamlit):

```python
from agentic_ai_wf.supervisor_agent import SupervisorAgent, SessionManager

session_manager = SessionManager()
supervisor = SupervisorAgent(
    session_manager=session_manager,
    upload_dir="/tmp/uploads"
)

# Get or create a session
session = session_manager.get_or_create_session(session_id="user-123")

# Process message (async generator)
async for update in supervisor.process_message(
    user_message="Run DEG analysis on my uploaded data",
    session_id=session.session_id,
    uploaded_files={"counts.csv": "/tmp/uploads/counts.csv"},
):
    # Same StatusUpdate objects as Path A
    handle_status_update(update)
```

**Note:** This path manages `ConversationState` internally and syncs `_graph_state` back automatically. You don't need to manually carry `workflow_outputs` between turns.

---

## 5. Streaming Protocol & StatusUpdate Contract

Every `StatusUpdate` yielded by the stream has this shape:

```python
@dataclass
class StatusUpdate:
    status_type: StatusType    # Enum — see below
    title: str                 # Short heading, e.g., "🧠 Analyzing request..."
    message: str               # Body text, supports Markdown
    details: Optional[str]     # Extra info for expandable UI sections
    progress: Optional[float]  # 0.0 to 1.0 for progress bars
    agent_name: Optional[str]  # Which agent is active (None for system events)
    timestamp: float           # Unix timestamp
    generated_files: Optional[List]  # File paths on COMPLETED events
    output_dir: Optional[str]  # Output directory path on COMPLETED events
```

### StatusType Enum Values

| Value | When Emitted | UI Treatment |
|-------|-------------|--------------|
| `"thinking"` | Query analysis, LLM synthesis | Show spinner/pulse animation |
| `"routing"` | Intent classification done | Show routing decision text |
| `"validating"` | Input validation | Show checklist |
| `"executing"` | Agent pipeline started | Show agent name + spinner |
| `"progress"` | Mid-execution update | Update progress bar |
| `"info"` | Informational (non-blocking) | Show as info banner |
| `"completed"` | Agent or full pipeline done | **Show final response + downloads** |
| `"error"` | Failure occurred | Show error message in red |
| `"waiting_input"` | Needs user input | Show input prompt |

### Typical Event Sequence

For a query like "Run pathway enrichment on my data":

```
1. StatusUpdate(type=THINKING,   title="Analyzing request…",       agent=None)
2. StatusUpdate(type=ROUTING,    title="🔀 Routing decision",      agent="intent", message="Route → pathway_enrichment: ...")
3. StatusUpdate(type=EXECUTING,  title="🚀 Running Pathway...",    agent="pathway_enrichment", progress=0.1)
4. StatusUpdate(type=PROGRESS,   title="Enriching pathways...",    agent="pathway_enrichment", progress=0.5)
5. StatusUpdate(type=COMPLETED,  title="✅ Pathway Complete",      agent="pathway_enrichment", progress=1.0, generated_files=[...])
6. StatusUpdate(type=THINKING,   title="🧠 Synthesizing...",       agent="response")
7. StatusUpdate(type=COMPLETED,  title="Analysis complete",        agent=None, message="## Pathway Enrichment Results\n\n...")
```

**The LAST `COMPLETED` update** contains:
- `message`: The full markdown response to display to the user
- `generated_files`: List of downloadable file paths (if report was generated)
- `_graph_state` (private attr): Session state for next turn continuity

### Multi-Agent Pipeline Sequence

For "Run full analysis pipeline":
```
THINKING → ROUTING →
  EXECUTING(deg) → PROGRESS(deg) → COMPLETED(deg) →
  EXECUTING(prioritization) → PROGRESS(prioritization) → COMPLETED(prioritization) →
  EXECUTING(pathway) → PROGRESS(pathway) → COMPLETED(pathway) →
THINKING(response synthesis) →
COMPLETED(final response with markdown)
```

---

## 6. Session & State Management

### Session ID
- Must be a **unique string per user conversation** (UUID recommended)
- The LangGraph checkpoint stores state keyed by `session_id` (mapped to LangGraph's `thread_id`)
- Same `session_id` across turns = conversational memory (LLM sees prior exchanges)
- Different `session_id` = fresh conversation

### Cross-Turn State Continuity

The supervisor maintains state between turns via LangGraph checkpointing. Two things accumulate:

1. **`conversation_history`** — list of `{role, content}` dicts (grows unboundedly — see audit)
2. **`workflow_outputs`** — dict of all agent output paths and data (merged across turns)

**How Streamlit carries state (you must replicate):**

```python
# After each turn's COMPLETED update:
graph_state = getattr(final_update, '_graph_state', None)

# On next turn, pass back:
run_supervisor_stream(
    ...,
    workflow_outputs=graph_state["workflow_outputs"],  # Agent output paths
    disease_name=graph_state["disease_name"],          # Disease context
)
```

If using the Legacy `SupervisorAgent` path, this sync happens automatically inside `process_message()` via `ConversationState`.

### What's Stored in `workflow_outputs`

| Key | Type | Set By | Description |
|-----|------|--------|-------------|
| `cohort_output_dir` | `str` | Cohort agent | Path to downloaded datasets |
| `deg_base_dir` | `str` | DEG agent | Path to DEG results directory |
| `deg_input_file` | `str` | DEG agent | Path to DEG CSV file |
| `prioritized_genes_path` | `str` | Prioritization agent | Path to prioritized genes CSV |
| `pathway_consolidation_path` | `str` | Pathway agent | Path to pathway results CSV |
| `deconvolution_output_dir` | `str` | Deconvolution agent | Path to deconv results |
| `response_report_path` | `str` | Response node | Path to generated PDF/DOCX |
| `structured_report_md_path` | `str` | Report node | Path to structured Markdown report |
| `custom_css` | `str` | Style engine | Cached CSS for report styling |
| `disease_name` | `str` | Intent node | Extracted disease context |
| *(+ more per agent)* | | | |

---

## 7. File Upload Handling

### What Streamlit Does (you replicate this)

1. User selects files in the UI
2. Streamlit saves them to `./streamlit_uploads/{filename}` (see `app.py:481-491`)
3. A dict `{filename: filepath}` is passed to the supervisor

### Your Implementation

```python
# 1. Save uploaded file to a temporary/persistent location on the server
file_path = f"/data/uploads/{session_id}/{secure_filename(file.name)}"
save_file(file, file_path)

# 2. Build the files dict
uploaded_files = {
    "pancreatic_cancer_counts.csv": file_path,
    "pancreatic_cancer_metadata.csv": metadata_path,
}

# 3. Pass to supervisor
run_supervisor_stream(
    user_query="Run DEG analysis for pancreatic cancer",
    session_id=session_id,
    uploaded_files=uploaded_files,
)
```

### Accepted File Types
(from `app.py:661`):
`csv`, `tsv`, `xlsx`, `h5ad`, `txt`, `json`, `fastq`, `fq`, `gz`, `mtx`

### File Type Detection
The supervisor auto-detects file types based on column headers:
- `raw_counts` — numeric matrix with gene column
- `deg_results` — has log2FC + pvalue columns
- `prioritized_genes` — has priority/score + gene columns
- `metadata` — has sample/condition columns
- `gene_list` — plain text, one gene per line
- `json_data` — JSON with pathway/gene keys

**No file type annotation needed from the frontend** — the supervisor handles detection.

---

## 8. File Download & Report Retrieval

### Generated Files
On `COMPLETED` events, the `StatusUpdate` may contain:
- `generated_files`: List of absolute file paths (CSVs, PNGs, PDFs)
- `output_dir`: Path to the output directory

### Report Files
When the user requests a PDF/DOCX report, the path is in:
- `workflow_outputs["response_report_path"]` — dynamic style-engine report
- `workflow_outputs["structured_report_md_path"]` — evidence-traced Markdown report
- `workflow_outputs["structured_report_path"]` — evidence-traced PDF/DOCX

### Download Strategy for Your Backend

```python
# Option A: Serve files directly from disk (simple)
@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    # Validate path is under allowed output directory
    file_path = get_validated_path(session_id, filename)
    return FileResponse(file_path)

# Option B: Upload to object storage (S3/GCS) on generation, return URL
# Better for distributed deployments
```

### Downloadable Output Keys (from Streamlit sidebar, `app.py:763-773`)

| workflow_outputs Key | Display Name | Type |
|---------------------|--------------|------|
| `prioritized_genes_path` | Prioritized Genes | CSV file |
| `pathway_consolidation_path` | Pathway Analysis | CSV file |
| `deg_base_dir` | DEG Results | Directory (ZIP) |
| `deconvolution_output_dir` | Deconvolution Results | Directory (ZIP) |
| `cohort_output_dir` | Cohort Data | Directory (ZIP) |
| `mdp_output_dir` | MDP Analysis Results | Directory (ZIP) |
| `temporal_output_dir` | Temporal Analysis Results | Directory (ZIP) |
| `harmonization_output_dir` | Harmonization Results | Directory (ZIP) |

---

## 9. Agent Registry (Capabilities Discovery)

The frontend can enumerate available agents for UI display:

```python
from agentic_ai_wf.supervisor_agent.agent_registry import AGENT_REGISTRY

for agent_type, agent_info in AGENT_REGISTRY.items():
    print(agent_info.display_name)       # "Differential Expression Analysis"
    print(agent_info.description)        # Human-readable description
    print(agent_info.estimated_time)     # "2-5 minutes"
    print(agent_info.example_queries)    # ["Run DEG analysis...", ...]
    print(agent_info.required_inputs)    # [AgentInput(name="counts_file", ...)]
    print(agent_info.optional_inputs)    # [AgentInput(name="metadata_file", ...)]
```

This is what Streamlit uses to populate the sidebar agent cards (`app.py:640-652`). Your UI can use this to show available capabilities dynamically.

---

## 10. Error Handling Contract

### Error Events
Errors are communicated as `StatusUpdate(status_type=StatusType.ERROR, ...)`:

```python
StatusUpdate(
    status_type=StatusType.ERROR,
    title="❌ DEG Analysis Failed",
    message="Required input 'counts_file' not found at: /tmp/missing.csv",
    agent_name="deg_analysis",
)
```

### Partial Failures
In multi-agent pipelines, one agent can fail while others succeed. The final `COMPLETED` message will mention which agents failed:

```
"✅ DEG analysis completed. ❌ Pathway enrichment failed: missing input. ✅ Response synthesized."
```

### Fatal Failures
If the entire graph crashes, `run_supervisor()` returns `SupervisorResult(status="failed", final_response="error message")`. For streaming, the last yielded update will be `ERROR`.

### User Input Required
When the supervisor needs more information:
```python
StatusUpdate(
    status_type=StatusType.WAITING_INPUT,
    title="📎 Additional Input Needed",
    message="Please upload a counts CSV file to proceed with DEG analysis.",
)
```
Your UI should prompt the user to provide the missing input.

---

## 11. Concurrency & Multi-User Considerations

### Current State (Streamlit Test Mode)
- Streamlit runs **one async loop per user session** via `asyncio.run()` (`app.py:1074`)
- SessionManager is **in-memory only** — not shared across processes
- MemorySaver (LangGraph checkpointing) is **in-memory only**

### Production Requirements

| Concern | Current | Required for Production |
|---------|---------|----------------------|
| Session storage | In-memory dict | Redis or PostgreSQL |
| Checkpoint storage | MemorySaver (in-memory) | PostgresSaver (LangGraph built-in) |
| Concurrent requests | Single-process Streamlit | ASGI server (uvicorn + FastAPI) with async handlers |
| File storage | Local filesystem | Shared filesystem or S3 |
| Process model | Single process | Multiple workers behind load balancer |

### Thread Safety Warning
The following global singletons are **not thread-safe** (see audit C5):
- `_graph` in `langgraph/graph.py`
- `_bedrock_client` / `_openai_client` in `llm_provider.py`
- `_intent_router` in `langgraph/nodes.py`

**Fix:** Initialize these at application startup, not lazily. Or add threading locks.

### Session Isolation
Each `session_id` gets its own LangGraph checkpoint. Two different sessions running simultaneously are fully isolated. However, two concurrent requests for the **same** `session_id` will race on the checkpoint — ensure your API serializes requests per session.

---

## 12. Environment & Configuration

### Required Environment Variables

```bash
# LLM Provider (at least one required)
USE_BEDROCK=true                    # "true" to use AWS Bedrock as primary
AWS_ACCESS_KEY_ID=...               # Bedrock auth
AWS_SECRET_ACCESS_KEY=...           # Bedrock auth
AWS_REGION=us-east-1                # Bedrock region
BEDROCK_MODEL_ID=us.anthropic.claude-opus-4-5-20251101-v1:0

OPENAI_API_KEY=...                  # Fallback LLM (always set this)

# Optional API Keys (adapters degrade gracefully without them)
OPENFDA_API_KEY=...                 # Higher rate limits for FDA
NCBI_API_KEY=...                    # Higher rate limits for NCBI/Ensembl

# Feature Flags
KEGG_ENABLED=false                  # KEGG API (academic use only)
USE_LANGGRAPH_SUPERVISOR=true       # Enable LangGraph mode (vs legacy)
SUPERVISOR_MAX_RETRIES=2            # Max retries per pipeline agent

# Paths
DECONV_SC_BASE_DIR=/path/to/sc/refs # Single-cell reference files
```

### Python Path Setup
The Streamlit app hacks `sys.path` (`app.py:21-22`). In production, install the package properly:
```bash
pip install -e /path/to/agentic_ai_wf
```

Or set `PYTHONPATH` in your deployment config.

---

## 13. Known Limitations & Gotchas

### Must-Know for Integration

1. **`_graph_state` is a private attribute** — set dynamically on the last `StatusUpdate` via `terminal._graph_state = graph_state` (`entry.py:148`). Access it with `getattr(update, '_graph_state', None)`. This is the **only way** to get cross-turn state in streaming mode.

2. **`conversation_history` grows unboundedly** — long sessions will eventually cause token-limit errors in LLM calls. Implement a trim/eviction policy in your backend (max ~50 turns recommended).

3. **Reports are written to relative paths** — `agentic_ai_wf/shared/reports/dynamic_reports/` and `agentic_ai_wf/shared/reports/structured_reports/`. In production, set `output_root` parameter to control where files go.

4. **`generated_files` on StatusUpdate can be `List[str]` or `List[Dict]`** — the type annotation says `List[Dict[str, Any]]` but in practice it's often `List[str]` (just file paths). Handle both:
   ```python
   for item in update.generated_files or []:
       path = item if isinstance(item, str) else item.get("path", "")
   ```

5. **The supervisor is async-only** — both `run_supervisor_stream()` and `run_supervisor()` are `async`. You need an async server (FastAPI/Starlette) or wrap with `asyncio.run()`.

6. **File paths in workflow_outputs are absolute local paths** — they won't work if the backend and frontend are on different machines. Either serve files via an endpoint or copy to shared storage.

7. **No built-in request cancellation** — the Streamlit app has a "Stop" button (`app.py:876-878`) that sets a flag, but this only works because the generator loop checks it. For production, use Python's `asyncio.Task.cancel()` on the stream consumer task.

8. **PDF rendering (WeasyPrint) requires system dependencies** — `libpango`, `libcairo`, `libffi`, etc. Ensure your production container has these installed.

---

## 14. Suggested Backend Architecture

### Recommended Stack

```
┌─────────────────────┐
│  Your Frontend      │
│  (React/Vue/etc.)   │
└──────────┬──────────┘
           │ WebSocket / SSE
┌──────────▼──────────┐
│  FastAPI Backend     │
│                      │
│  POST /api/chat      │  ← Accept query + files
│  WS   /api/stream    │  ← Stream StatusUpdates
│  GET  /api/files/:id │  ← Serve generated files
│  GET  /api/agents    │  ← List available agents
│  GET  /api/health    │  ← Health check
└──────────┬──────────┘
           │ Python async
┌──────────▼──────────┐
│  supervisor_agent    │
│  run_supervisor_     │
│  stream()            │
└─────────────────────┘
```

### FastAPI Example Skeleton

```python
from fastapi import FastAPI, WebSocket, UploadFile, File
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream
from agentic_ai_wf.supervisor_agent.executors.base import StatusType
import json, uuid

app = FastAPI()

# In-memory session store (replace with Redis in production)
sessions = {}

@app.websocket("/api/stream/{session_id}")
async def stream_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    while True:
        # Receive user message
        data = json.loads(await websocket.receive_text())
        query = data["query"]
        files = sessions.get(session_id, {}).get("files", {})
        prior_outputs = sessions.get(session_id, {}).get("workflow_outputs", {})
        disease = sessions.get(session_id, {}).get("disease_name", "")

        # Stream supervisor responses
        async for update in run_supervisor_stream(
            user_query=query,
            session_id=session_id,
            uploaded_files=files,
            workflow_outputs=prior_outputs or None,
            disease_name=disease,
        ):
            await websocket.send_json({
                "type": update.status_type.value,
                "title": update.title,
                "message": update.message,
                "details": update.details,
                "progress": update.progress,
                "agent_name": update.agent_name,
                "generated_files": update.generated_files,
                "timestamp": update.timestamp,
            })

            # Save state for next turn
            if update.status_type == StatusType.COMPLETED:
                gs = getattr(update, '_graph_state', None)
                if gs:
                    sessions.setdefault(session_id, {}).update({
                        "workflow_outputs": gs.get("workflow_outputs", {}),
                        "disease_name": gs.get("disease_name", ""),
                    })

@app.post("/api/upload/{session_id}")
async def upload_files(session_id: str, files: list[UploadFile] = File(...)):
    upload_dir = f"/data/uploads/{session_id}"
    os.makedirs(upload_dir, exist_ok=True)
    saved = {}
    for f in files:
        path = f"{upload_dir}/{f.filename}"
        with open(path, "wb") as out:
            out.write(await f.read())
        saved[f.filename] = path
    sessions.setdefault(session_id, {}).setdefault("files", {}).update(saved)
    return {"uploaded": list(saved.keys())}

@app.get("/api/agents")
async def list_agents():
    from agentic_ai_wf.supervisor_agent.agent_registry import AGENT_REGISTRY
    return [
        {
            "id": at.value,
            "name": ai.display_name,
            "description": ai.description,
            "estimated_time": ai.estimated_time,
            "example_queries": ai.example_queries,
            "required_inputs": [{"name": i.name, "is_file": i.is_file} for i in ai.required_inputs],
        }
        for at, ai in AGENT_REGISTRY.items()
    ]
```

---

## 15. Quick-Start: Minimal Integration Example

The absolute simplest integration — a CLI that replicates what Streamlit does:

```python
#!/usr/bin/env python3
"""Minimal integration example — replaces Streamlit with a CLI."""

import asyncio
import uuid
from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream
from agentic_ai_wf.supervisor_agent.executors.base import StatusType

async def main():
    session_id = str(uuid.uuid4())
    workflow_outputs = {}
    disease_name = ""

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break

        print("\n--- Processing ---")
        async for update in run_supervisor_stream(
            user_query=query,
            session_id=session_id,
            workflow_outputs=workflow_outputs or None,
            disease_name=disease_name,
        ):
            # Show progress
            status = update.status_type.value
            if status in ("thinking", "routing", "executing", "progress"):
                prog = f" [{int(update.progress*100)}%]" if update.progress else ""
                print(f"  [{status}] {update.title}{prog}")

            # Final response
            if status == "completed":
                print(f"\nAssistant:\n{update.message}\n")

                # Save state for next turn
                gs = getattr(update, '_graph_state', None)
                if gs:
                    workflow_outputs = gs.get("workflow_outputs", {})
                    disease_name = gs.get("disease_name", "")

                # Report downloads
                if update.generated_files:
                    print(f"  📁 Generated files: {update.generated_files}")

            elif status == "error":
                print(f"\n❌ Error: {update.message}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary: What to Tell Your Software Lead

> "The Streamlit app is a test harness. To integrate, you need to:
>
> 1. **Call `run_supervisor_stream()`** — it's an async generator that yields `StatusUpdate` objects with progress, then a final response
> 2. **Stream StatusUpdates to the frontend** via WebSocket or SSE — each has `type`, `title`, `message`, `progress`, `agent_name`
> 3. **Save `_graph_state`** from the last COMPLETED update and pass `workflow_outputs` back on the next turn for session continuity
> 4. **Handle file uploads** by saving to disk and passing `{filename: path}` dict
> 5. **Serve generated files** — report PDFs/DOCXs and analysis CSVs are written to local disk; expose via an API endpoint
>
> The entire data contract is: text in → StatusUpdate stream out → text + files at the end. No WebSocket protocol, no custom serialization — just Python async generators."
