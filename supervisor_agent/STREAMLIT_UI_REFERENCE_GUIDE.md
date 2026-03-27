# Streamlit UI Reference Guide — BiRAGAS

> **Purpose:** Enable a frontend developer to rebuild the entire Streamlit UI in Next.js/React without reading Streamlit code.
>
> **Source files analyzed:**
> - `streamlit_app/app.py` (main chat UI)
> - `streamlit_app/pages/1_structured_report.py` (report viewer page)
> - `streamlit_app/.streamlit/config.toml` (server config)
> - `supervisor_agent/supervisor.py` (StatusUpdate, StatusType, SupervisorAgent)
> - `supervisor_agent/state.py` (ConversationState, SessionManager, Message, UploadedFile)
> - `supervisor_agent/agent_registry.py` (AgentType, AgentInfo, all 16 agent definitions)
> - `supervisor_agent/router.py` (RoutingDecision, AgentIntent)

---

## 1. Application Layout & Navigation

### Page Configuration

```
Title:   "🧬 BiRAGAS"
Icon:    🧬
Layout:  wide (full-width)
Sidebar: expanded by default
```

### Overall Layout

```
┌──────────────────┬────────────────────────────────────────────────┐
│                  │                                                │
│    SIDEBAR       │              MAIN AREA                        │
│    (~300px)      │              (flex-1)                          │
│                  │                                                │
│  ┌────────────┐  │  ┌──────────────────────────────────────────┐  │
│  │ Title:     │  │  │ # 🧬 BiRAGAS                             │  │
│  │ 🧬 Bio-   │  │  │ Your intelligent guide to transcriptome  │  │
│  │ informatics│  │  │ analysis                                 │  │
│  │ Assistant  │  │  │ ────────────────────────────────────────  │  │
│  ├────────────┤  │  │                                          │  │
│  │ AGENTS     │  │  │  [Welcome Cards - 6 capability tiles]    │  │
│  │ (expanders)│  │  │  OR                                      │  │
│  ├────────────┤  │  │  [Chat Message History]                  │  │
│  │ FILE       │  │  │  [Activity Log Accordion]                │  │
│  │ UPLOAD     │  │  │  [Live Status Placeholder]               │  │
│  ├────────────┤  │  │  [Stop Button (when processing)]         │  │
│  │ SESSION    │  │  │                                          │  │
│  │ INFO       │  │  ├──────────────────────────────────────────┤  │
│  ├────────────┤  │  │ [Chat Input Bar]                         │  │
│  │ DOWNLOAD   │  │  ├──────────────────────────────────────────┤  │
│  │ RESULTS    │  │  │ Footer: "🧬 Bioinformatics AI Assistant  │  │
│  ├────────────┤  │  │ | Powered by LangGraph & OpenAI"         │  │
│  │ STRUCTURED │  │  └──────────────────────────────────────────┘  │
│  │ REPORT LINK│  │                                                │
│  ├────────────┤  │                                                │
│  │ [🗑 Clear] │  │                                                │
│  └────────────┘  │                                                │
└──────────────────┴────────────────────────────────────────────────┘
```

### UI States

| State | Main Area Content | Sidebar State |
|-------|-------------------|---------------|
| **Initial (empty)** | Welcome cards showing 6 capabilities (Cohort Retrieval, DEG Analysis, Gene Prioritization, Pathway Analysis, Deconvolution, Structured Reports) with icons and descriptions | Agents list, empty upload, session info, no downloads |
| **Query submitted** | User message bubble appears, live processing container activates with pipeline plan and agent step log | Chat input disabled, Stop button appears |
| **Analysis running** | Real-time structured log: agent headers (✅/⚡/🔹), step-by-step updates with icons, inline plots, progress bars | Same, files accumulate in Downloads |
| **Results displayed** | Full chat history with user/assistant/status messages, activity log accordion at bottom | Download Results section populated with categorized files |
| **Error** | Error message in red styling, pipeline stops with partial completion list | Error state reflected in activity log |
| **Waiting for input** | Purple-styled message requesting specific missing inputs | Sidebar unchanged, awaiting user response |

### Initialization Sequence (on first load)

1. `st.set_page_config()` — page title, icon, layout
2. Inject custom CSS (270 lines of inline `<style>`)
3. `init_session_state()`:
   - Create `SessionManager` (in-memory session store)
   - Create `SupervisorAgent` (with upload dir `./streamlit_uploads`)
   - Create `ConversationState` (new UUID session)
   - Initialize empty `messages`, `activity_log`, `generated_files`
4. Render sidebar (agents, upload, session info)
5. Render main area (chat container with history, or welcome cards if empty)
6. Place chat input bar at bottom

### Multi-Page Navigation

- **Main page** (`app.py`): Chat UI
- **Structured Report Viewer** (`pages/1_structured_report.py`): Dedicated report page with TOC sidebar, section search, interactive/raw tabs, evidence badges, conflict panels, and download buttons (MD/PDF/DOCX)
- Navigation: Sidebar link "📊 Open Report Viewer" appears after a structured report is generated

---

## 2. Session State Map

| Key | Type | Initial Value | Set By | Read By | Purpose |
|-----|------|---------------|--------|---------|---------|
| `session_manager` | `SessionManager` | `SessionManager()` | `init_session_state()` | `supervisor`, `app.py` | In-memory session store (maps session_id → ConversationState) |
| `supervisor` | `SupervisorAgent` | new instance | `init_session_state()` | `process_message()` | Main orchestrator — routes queries, executes agents |
| `conversation` | `ConversationState` | new session | `init_session_state()`, Clear button | sidebar, `process_message()` | Backend state: messages, uploads, workflow_state, disease |
| `messages` | `list[dict]` | `[]` | `process_message()`, Clear button | chat_container rendering | UI message history (separate from backend ConversationState.messages) |
| `current_status` | `StatusUpdate \| None` | `None` | (unused legacy) | (unused) | Reserved for current status display |
| `uploaded_files_cache` | `dict[str, str]` | `{}` | file upload handler | `process_message()` | Maps filename → saved filepath on disk |
| `processing` | `bool` | `False` | `process_message()` start/end | chat_input disabled check, Stop button visibility | Whether an async processing task is running |
| `cancel_requested` | `bool` | `False` | Stop button click | `process_message()` loop | Signal to abort the current processing loop |
| `current_task` | `None` | `None` | (reserved) | (reserved) | Reserved for async task handle |
| `activity_log` | `list[dict]` | `[]` | `add_activity_log()` | `render_activity_log()` | Timestamped log of all StatusUpdates (max 50 entries) |
| `generated_files` | `dict[str, dict]` | `{}` | `register_generated_files()` | sidebar Downloads section | Tracks files per agent: `{agent_name: {plots: [...], data: [...], reports: [...], other: [...]}}` |

### Persistence Behavior

- **All keys persist within a Streamlit session** (browser tab). Streamlit session state survives re-runs triggered by `st.rerun()` and widget interactions.
- **None persist across browser refresh** — session state is in-memory only. The `SessionManager` itself is also in-memory (production would use Redis/DB).
- **Clear Conversation resets:** `messages`, `conversation` (new session), `uploaded_files_cache`, `activity_log`, `generated_files`.
- **Backend-consumed keys:** `conversation` (used by supervisor internals), `uploaded_files_cache` (passed as `uploaded_files` dict to `process_message()`).

---

## 3. User Input Flow

### Chat Input

- **Component:** `st.chat_input("Ask me about bioinformatics analysis...")`
- **Disabled when:** `st.session_state.processing == True`
- **On submit:**
  1. Append `{"role": "user", "content": prompt}` to `st.session_state.messages`
  2. Display user message in chat container
  3. Set `processing = True`, `cancel_requested = False`
  4. Copy `uploaded_files_cache` dict
  5. Call `asyncio.run(process_message())` — this calls `supervisor.process_message()` as an async generator
  6. Iterate over `StatusUpdate` yields, updating UI in real-time via `status_placeholder`
  7. On completion: clear placeholder, append final assistant message, set `processing = False`
  8. Call `st.rerun()` to refresh the entire page with updated state

### File Upload

- **Component:** `st.file_uploader` in sidebar
- **Accepted types:** `csv`, `tsv`, `xlsx`, `h5ad`, `txt`, `json`, `fastq`, `fq`, `gz`, `mtx`
- **Multiple files:** Yes (`accept_multiple_files=True`)
- **Size limits:** 25 GB per file (configured in `.streamlit/config.toml`: `maxUploadSize = 25000`)
- **Help text:** "Upload tabular inputs, matrix files, or sequencing reads for supervisor-driven analysis"

**Upload Flow:**

1. User drops/selects files in sidebar uploader
2. For each file not already in `uploaded_files_cache`:
   - Save to `./streamlit_uploads/{filename}` via `save_uploaded_file()`
   - Store `filename → filepath` mapping in `uploaded_files_cache`
3. Display "✅ {filename}" for each uploaded file in sidebar
4. When user submits a chat message, `uploaded_files_cache` is passed to `supervisor.process_message(uploaded_files=...)`
5. Supervisor registers each file in `ConversationState.uploaded_files`, auto-detects file type (`raw_counts`, `deg_results`, `pathway_results`, `prioritized_genes`, `patient_info`, `multiomics_layer`, `crispr_*`, `fastq_*`, etc.), and maps to appropriate input keys in `workflow_state`

**File Type Detection Logic (in supervisor.py `_detect_file_type()`):**

| Detected Type | Trigger | Maps To Input Key(s) |
|---------------|---------|----------------------|
| `raw_counts` | Gene ID column + all numeric sample columns, no DEG columns | `counts_file`, `bulk_file` |
| `deg_results` | Has both fold-change AND p-value columns | `deg_input_file`, `deg_base_dir` |
| `prioritized_genes` | Has `composite_score`, `druggability_score`, etc. | `prioritized_genes_path` |
| `pathway_results` | Has `pathway_name`, `pathway_id`, `enrichment_score`, etc. | `pathway_consolidation_path` |
| `multiomics_layer` | Filename contains known layer name (proteomics, metabolomics, etc.) | Accumulated in `multiomics_layers` dict |
| `patient_info` | Filename contains "patient" or has clinical columns | `patient_info_file` |
| `deconvolution_results` | Filename contains "cibersort", "xcell", "deconv", etc. | `deconvolution_results` |
| `crispr_10x_data` | Contains barcode + feature/gene + matrix files | `crispr_10x_input_dir` |
| `crispr_count_table` | Has sgRNA/guide columns | `crispr_screening_input_dir` |
| `fastq_file` / `fastq_directory` | `.fastq`, `.fq`, `.fastq.gz` extensions | `fastq_input_dir` |
| `json_data` | `.json` extension with pathway/gene/enrichment data | `json_data` |
| `gene_list` | `.txt` with short alphanumeric lines | `gene_list` |
| `unknown` | Could not determine | No automatic mapping |

### Disease Name Detection

Disease name is extracted from multiple sources (priority order):
1. **Router LLM extraction** — the routing LLM parses the query and returns `disease_name` in `extracted_params`
2. **Deterministic regex** in `_extract_contextual_inputs()`:
   - `disease is "lupus"` / `disease: lupus`
   - `for lupus analysis` / `analyzing lupus`
   - `condition: lupus`
3. **Prior conversation context** — `state.current_disease` from previous messages
4. **Stored in:** `ConversationState.current_disease` and `workflow_state["disease_name"]`

### Clear Conversation

**Button:** "🗑️ Clear Conversation" (full-width in sidebar)

**Resets:**
- `messages = []`
- `conversation = session_manager.create_session()` (new UUID, fresh ConversationState)
- `uploaded_files_cache = {}`
- `activity_log = []`
- `generated_files = {}`

**Triggers:** `st.rerun()`

**Does NOT reset:** `session_manager` (retains other sessions), `supervisor` (singleton agent instance)

---

## 4. Message Rendering System

### Message Format (UI-level)

Messages stored in `st.session_state.messages` use this structure:

```python
# User message
{"role": "user", "content": "Find datasets for lupus disease"}

# Assistant message (final response)
{"role": "assistant", "content": "🎉 **Cohort Retrieval Agent** completed successfully!\n..."}

# Status update (intermediate)
{"role": "status", "update": <StatusUpdate object>}
```

### Backend Message Format (`ConversationState.messages`)

```python
@dataclass
class Message:
    role: MessageRole        # "user" | "assistant" | "system" | "agent_status"
    content: str
    message_type: MessageType  # "text" | "file_upload" | "agent_routing" | "agent_progress" | "agent_result" | "error" | "thinking"
    timestamp: float         # time.time()
    metadata: dict           # Arbitrary metadata
```

### User Message Rendering

```python
with st.chat_message("user"):
    st.markdown(msg["content"])
```

Renders as a native Streamlit chat bubble (user avatar on right). Also has custom CSS class `.user-message` with gradient background (`#667eea` → `#764ba2`), white text, rounded corners (18px 18px 4px 18px), max-width 80%, right-aligned.

### Assistant Message Rendering

```python
with st.chat_message("assistant"):
    st.markdown(msg["content"])
```

Renders with Streamlit's native markdown support including: headers, bold, italic, lists, code blocks, links, tables. Custom CSS class `.assistant-message` with `#f0f2f6` background, dark text, rounded corners (18px 18px 18px 4px), max-width 85%.

### Status Update Rendering

Status messages call `render_status_update(update)` which renders:

1. **Title** — bold with status icon, or H3 with chain icon for pipeline updates
2. **Message** — markdown body
3. **Details** — collapsible `st.expander("Details")` (only if non-empty, no HTML, no "How to provide inputs")
4. **Progress bar** — `st.progress(update.progress)` if progress is not None
5. **Generated files/plots** — inline image grid on COMPLETED status
6. **Download buttons** — for workflow_state files on COMPLETED
7. **Divider** — `st.markdown("---")`

### Inline Plots

Generated PNG/JPG files are displayed in a 3-column grid inside an expander:

```python
with st.expander(f"📊 Generated Visualizations ({len(plots)} plots)", expanded=True):
    cols = st.columns(min(3, len(plots)))
    for idx, plot_path in enumerate(plots[:6]):
        with cols[idx % 3]:
            st.image(str(p), caption=p.stem, use_container_width=True)
```

Max 6 plots displayed; overflow shows info message: "📁 {N} more plots available in output directory"

### File Download Links

Files are rendered as `st.download_button` components with:
- Label: icon + filename (e.g., "📄 lupus_DEGs_filtered.csv")
- MIME types: `text/csv`, `text/tab-separated-values`, `image/png`, `application/pdf`, `application/zip`, `application/octet-stream`
- Key: unique per agent+filename to avoid Streamlit widget ID collisions

---

## 5. Real-Time Progress & Status Updates

### StatusType Enum (complete)

```python
class StatusType(str, Enum):
    THINKING      = "thinking"       # Analyzing query
    ROUTING       = "routing"        # Matching to agent(s)
    VALIDATING    = "validating"     # Checking input requirements
    EXECUTING     = "executing"      # Agent execution started
    PROGRESS      = "progress"       # Mid-execution progress
    INFO          = "info"           # Informational message
    COMPLETED     = "completed"      # Agent/pipeline complete
    ERROR         = "error"          # Execution failure
    WAITING_INPUT = "waiting_input"  # Need user input
```

### StatusUpdate Dataclass (complete)

```python
@dataclass
class StatusUpdate:
    status_type: StatusType
    title: str                           # Bold header text
    message: str                         # Markdown body
    details: Optional[str] = None        # Collapsible detail text
    progress: Optional[float] = None     # 0.0 to 1.0
    agent_name: Optional[str] = None     # Which agent this is from
    timestamp: float = None              # auto-set to time.time()
    generated_files: Optional[List[str]] = None  # File paths (plots, CSVs, etc.)
    output_dir: Optional[str] = None     # Agent output directory
```

### Status Type → UI Mapping

| status_type | CSS Class | Color | Icon | Color Dot | UI Behavior |
|-------------|-----------|-------|------|-----------|-------------|
| `THINKING` | `status-thinking` | Yellow bg, amber left-border | 🧠 | 🟡 | "Understanding Your Request" |
| `ROUTING` | `status-routing` | Blue bg, blue left-border | 🔍 | 🔵 | "Analyzing Intent" / "Routing to X Agent" / "Building Execution Pipeline" |
| `VALIDATING` | `status-routing` | Blue bg, blue left-border | 📋 | 🔵 | "Checking Requirements" |
| `EXECUTING` | `status-executing` | Green bg, green left-border | ⚡ | 🟢 | "Starting X Agent" / "Running X Agent (1/3)" |
| `PROGRESS` | `status-executing` | Green bg, green left-border | 🔄 | 🟢 | "File Registered" / "All Inputs Received" / progress updates |
| `INFO` | `status-routing` | Blue bg, blue left-border | ℹ️ | 🔵 | "Smart Routing" (skip info) |
| `COMPLETED` | `status-executing` | Green bg, green left-border | ✅ | 🟢 | Final response with generated files, download buttons |
| `ERROR` | `status-error` | Red bg, red left-border | ❌ | 🔴 | Error message with details |
| `WAITING_INPUT` | `status-waiting` | Purple bg, purple left-border | 📎 | 🟣 | "Additional Input Needed" / "Clarification Needed" |

### Streaming Flow

```
User submits query
    │
    ▼
supervisor.process_message() yields StatusUpdate objects
    │
    ├─► THINKING: "Understanding Your Request"
    ├─► ROUTING: "Analyzing Intent" / "Routing to X Agent"
    ├─► [If multi-agent] ROUTING: "Building Execution Pipeline" with agent list
    ├─► VALIDATING: "Checking Requirements"
    ├─► [If missing inputs] WAITING_INPUT: "Additional Input Needed" → STOP
    ├─► EXECUTING: "Starting X Agent"
    ├─► PROGRESS: File registration, intermediate results (yielded by executor)
    ├─► [For each agent in pipeline]:
    │     ├─► EXECUTING: "Pipeline Progress: Agent N/M"
    │     ├─► PROGRESS: agent-specific progress (executor yields)
    │     ├─► PROGRESS: "X Agent Complete (N/M)"
    │     └─► (repeat for next agent)
    ├─► COMPLETED: Final response with generated_files, output_dir
    └─► [On error] ERROR: "X Agent Failed"

For each yielded StatusUpdate:
    1. Check cancel_requested → break if true
    2. Append to all_status_updates
    3. Add to activity_log (timestamped entry)
    4. Track agent changes (agent_logs structure)
    5. Update status_placeholder with structured display:
       - "### 🔄 Processing Your Request" header
       - Pipeline plan (if available)
       - System logs (routing/planning, before first agent)
       - Per-agent section: header with status indicator → step messages → progress bar
       - Current step details in expander
       - Inline generated plots
    6. Append non-COMPLETED updates to messages as {"role": "status", "update": update}
    7. On COMPLETED: extract final_message, register generated files
    8. On ERROR/WAITING_INPUT: extract final_message
```

### Live Status Placeholder Structure

While processing, the `status_placeholder` container shows:

```
### 🔄 Processing Your Request
────────────────────────────────

📋 **Pipeline Plan**
Your request requires 3 agents:
🔍 Cohort Retrieval → 📊 DEG Analysis → 🎯 Gene Prioritization

#### ✅ **Cohort Retrieval**             ← completed agents
    ✅ Datasets downloaded successfully

#### ⚡ **Deg Analysis** _(running...)_  ← current agent
    🔄 Processing GSE12345...
    🔄 Running DESeq2 analysis...
    ⚡ Filtering significant DEGs...
    [=========>          ] 45%           ← progress bar

#### 🔹 **Gene Prioritization**          ← pending agent

📋 Current Step Details ▸               ← collapsible

📊 Generated Visualizations:            ← inline plots if available
   [volcano.png] [ma_plot.png] [heatmap.png]
```

### Activity Log

Collapsible section ("📋 Activity Log") showing last 20 entries in terminal-style format:

```
[14:32:05] System      🔍 Analyzing Intent
[14:32:06] deg_analysis ⚡ Starting DEG Analysis
[14:32:15] deg_analysis 🔄 Processing counts matrix...
[14:32:45] deg_analysis ✅ DEG Analysis Complete
```

Rendered as HTML with dark background (`#0f172a`), green text (`#10b981`), monospace font, max-height 400px with scroll.

---

## 6. Agent Sidebar

### Agent List

All 16 agents from `AGENT_REGISTRY` are listed as collapsible expanders:

| Agent | Display Name | Icon |
|-------|-------------|------|
| Cohort Retrieval | 🔍 Cohort Retrieval Agent | 🔍 |
| DEG Analysis | 📊 DEG Analysis Agent | 📊 |
| Gene Prioritization | 🎯 Gene Prioritization Agent | 🎯 |
| Pathway Enrichment | 🛤️ Pathway Enrichment Agent | 🛤️ |
| Deconvolution | 🧬 Deconvolution Agent | 🧬 |
| Temporal Analysis | ⏱️ Temporal Analysis Agent | ⏱️ |
| Harmonization | 🔗 Harmonization Agent | 🔗 |
| MDP Analysis | 🌐 MDP Analysis Agent | 🌐 |
| Perturbation Analysis | 💊 Perturbation Analysis Agent | 💊 |
| Multi-Omics Integration | Multi-Omics Integration | 🧬 |
| FASTQ Processing | 🧬 FASTQ Processing Agent | 🧬 |
| Molecular Report | 📋 Molecular Report Agent | 📋 |
| CRISPR Perturb-seq | 🧬 CRISPR Perturb-seq Agent | 🧬 |
| CRISPR Screening | 🔬 CRISPR Screening Agent | 🔬 |
| CRISPR Targeted | 🎯 CRISPR Targeted Agent | 🎯 |
| Causality | 🔬 Causality Agent | 🔬 |

### Expander Content (when expanded)

```
**{description}**
⏱️ Estimated time: {estimated_time}

**Required inputs:**
- 📎 counts_file       ← is_file=True
- ✏️ disease_name      ← is_file=False

**Example queries:**
- _Find datasets for lupus disease_
- _Search for breast cancer RNA-seq studies_
```

### Agent Status

Agents are **display-only** — not clickable or interactive. Status is not tracked in the sidebar; it's shown in the main area via the real-time processing log. The sidebar agents list is a static reference.

---

## 7. Session Info Panel

Located in sidebar below the file upload section.

### Fields Displayed

| Field | Format | Source |
|-------|--------|--------|
| Session ID | `` `{first_8_chars}...` `` | `conversation.session_id[:8]` |
| Messages | Integer count | `len(st.session_state.messages)` |
| Current disease | Disease name (only shown if set) | `conversation.current_disease` |

### Refresh Behavior

Session info refreshes on every Streamlit re-run (which happens after each `st.rerun()` call — i.e., after every message processed or Clear button clicked).

---

## 8. File Display & Downloads

### Sidebar Download Results Section

Appears when `conversation.workflow_state` or `generated_files` is non-empty.

#### Primary: Generated Files Tracking

Organized by agent, then by file category:

```
### 📥 Download Results

📁 Deg Analysis (15 files) ▸
    📊 Data Files:
        [📄 lupus_DEGs_filtered.csv]  ← download button
        [📄 lupus_DEGs_all.csv]
    📈 Plots:
        [🖼️ volcano_plot.png]
        [🖼️ ma_plot.png]
    📑 Reports:
        [📄 analysis_report.pdf]

📁 Gene Prioritization (3 files) ▸
    ...
```

#### Fallback: Workflow State Downloads

If `generated_files` is empty but `workflow_state` has output paths:

| workflow_state Key | Display Name | Type |
|-------------------|-------------|------|
| `prioritized_genes_path` | Prioritized Genes | Single CSV download |
| `pathway_consolidation_path` | Pathway Analysis | Single CSV download |
| `deg_base_dir` | DEG Results | ZIP folder download + browse |
| `deconvolution_output_dir` | Deconvolution Results | ZIP folder download + browse |
| `cohort_output_dir` | Cohort Data | ZIP folder download + browse |
| `mdp_output_dir` | MDP Analysis Results | ZIP folder download + browse |
| `temporal_output_dir` | Temporal Analysis Results | ZIP folder download + browse |
| `harmonization_output_dir` | Harmonization Results | ZIP folder download + browse |
| `reporting_output_dir` | Report Results | ZIP folder download + browse |

For folders: a ZIP download button is generated on-the-fly using `create_zip_from_folder()`, plus a "browse files" expander listing individual CSV/PDF/HTML files with download buttons.

### Structured Report Link

When `workflow_state["structured_report_md_path"]` is set and the file exists:

```
### 📊 Structured Report
✅ Report generated: **{filename}**
[📊 Open Report Viewer]  ← page link to pages/1_structured_report.py
```

### File Category Detection

```python
categories = {
    "plots":   [".png", ".jpg", ".jpeg", ".svg"],
    "data":    [".csv", ".tsv", ".xlsx", ".json"],
    "reports": [".pdf", ".docx", ".html"],
    "other":   [everything else]
}
```

### Generated File Collection

`_collect_generated_files(output_dir)` scans recursively for: `.png`, `.jpg`, `.jpeg`, `.csv`, `.tsv`, `.xlsx`, `.json`, `.pdf`, `.html`

---

## 9. Error Handling & User Feedback

### Error Display

- **Agent execution failure:** `StatusType.ERROR` status update with red-styled card:
  - Title: "❌ {Agent Display Name} Failed"
  - Message: "An error occurred during execution: {error_string}"
  - Details: "Please check your inputs and try again."
  - Also adds assistant message: "I encountered an error: {error}\n\nPlease check your inputs and try again."

- **Pipeline failure:** Pipeline stops at the failing agent. Completion summary shows which agents completed and which failed:
  - Title: "❌ {Agent} Failed ({N}/{M})"
  - Details: "Pipeline stopped. Completed agents: {list}"

- **Missing dependencies in pipeline:**
  - Title: "⚠️ Missing Dependencies"
  - Message: Lists which inputs are missing

### Waiting for Input (not an error)

Purple-styled card (`status-waiting`) with:
- Title: "📎 Additional Input Needed"
- Message: Lists each missing input with icon (📎 for files, ✏️ for text), description, and example
- Tip: "💡 Upload files in the sidebar, then send me a message and I'll start automatically!"
- Sets `state.waiting_for_input = True` and `state.pending_agent_type`

### Cancellation

- **Stop button:** "🛑 Stop Processing" appears in right column when `processing == True`
- Sets `cancel_requested = True`
- Shows warning: "⏳ Cancellation requested..."
- On next StatusUpdate yield, loop breaks
- Adds assistant message: "🛑 Processing was cancelled."

### Partial Results

When a pipeline fails mid-execution:
- Completed agents' outputs remain in `workflow_state`
- Generated files from completed agents are still available for download
- Error message lists which agents succeeded vs failed

### No Special Network/API Timeout UI

The app does not have explicit timeout handling or reconnection UI — this is a Streamlit limitation (single-page app, synchronous re-runs). The Next.js replacement should add proper WebSocket timeout/reconnect logic.

---

## 10. API Contract for Frontend Replacement

### Request (Frontend → Backend)

```typescript
interface AnalysisRequest {
  query: string;                           // User's natural language message
  files: Record<string, string>;           // {filename: server_filepath} — files already uploaded via separate endpoint
  session_id: string;                      // UUID — from ConversationState.session_id
  user_id?: string;                        // Optional authenticated user ID
}
```

The current Streamlit app sends these as arguments to `supervisor.process_message()`:

```python
async for update in supervisor.process_message(
    user_message=prompt,
    session_id=st.session_state.conversation.session_id,
    uploaded_files=uploaded_files_dict,   # {filename: filepath}
    user_id=st.session_state.get("user_id"),
):
```

### Response Stream (Backend → Frontend via WebSocket/SSE)

Each yielded `StatusUpdate` serializes to:

```typescript
interface StatusUpdateEvent {
  status_type: "thinking" | "routing" | "validating" | "executing" | "progress" | "info" | "completed" | "error" | "waiting_input";
  title: string;                          // Bold header (e.g., "🚀 Starting DEG Analysis")
  message: string;                        // Markdown body
  details: string | null;                 // Collapsible detail text
  progress: number | null;                // 0.0 to 1.0 (null if indeterminate)
  agent_name: string | null;              // e.g., "deg_analysis" (null for system events)
  timestamp: number;                      // Unix timestamp (seconds)
  generated_files: string[];              // Absolute file paths on server
  output_dir: string | null;              // Directory containing all outputs
}
```

### Final Response (on COMPLETED)

The last `StatusUpdate` with `status_type === "completed"` contains the full response in `message`. Additional context is in `workflow_state` on the backend:

```typescript
interface CompletedResponse {
  status_type: "completed";
  title: string;                          // "✅ DEG Analysis Complete!" or "🎉 Pipeline Complete!"
  message: string;                        // Full markdown response text
  details: string;                        // Output summary
  progress: 1.0;
  agent_name: string | null;
  generated_files: string[];              // All files from all agents in pipeline
  output_dir: string | null;
}
```

### Workflow State Keys (backend, available for API enrichment)

```typescript
interface WorkflowState {
  // Primary output paths
  cohort_output_dir?: string;
  cohort_summary_text?: string;
  deg_base_dir?: string;
  deg_input_file?: string;
  prioritized_genes_path?: string;
  pathway_consolidation_path?: string;
  deconvolution_output_dir?: string;
  perturbation_output_dir?: string;
  multiomics_output_dir?: string;
  temporal_output_dir?: string;
  harmonization_output_dir?: string;
  mdp_output_dir?: string;
  reporting_output_dir?: string;
  structured_report_md_path?: string;

  // Input tracking
  counts_file?: string;
  bulk_file?: string;
  metadata_file?: string;
  disease_name?: string;

  // Internal tracking (prefixed with _)
  _detected_file_type?: string;
  _executed_agents_this_run?: string[];
}
```

### File Upload Endpoint (new for Next.js)

The Streamlit app saves files directly to disk. For Next.js, implement:

```typescript
// POST /api/upload
// Content-Type: multipart/form-data
// Body: file (binary)
// Response: { filename: string, filepath: string, size: number, detected_type: string }
```

### File Download Endpoint (new for Next.js)

```typescript
// GET /api/files/{filepath}
// Response: Binary file with appropriate Content-Type header
```

---

## 11. Theming & Styling

### Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| User message gradient start | Purple-blue | `#667eea` |
| User message gradient end | Deep purple | `#764ba2` |
| Assistant message background | Light gray | `#f0f2f6` |
| Assistant message text | Dark gray | `#1f2937` |
| Status: thinking bg | Light yellow | `#fef3c7` |
| Status: thinking border | Amber | `#f59e0b` |
| Status: routing bg | Light blue | `#dbeafe` |
| Status: routing border | Blue | `#3b82f6` |
| Status: executing bg | Light green | `#d1fae5` |
| Status: executing border | Green | `#10b981` |
| Status: error bg | Light red | `#fee2e2` |
| Status: error border | Red | `#ef4444` |
| Status: waiting bg | Light purple | `#ede9fe` |
| Status: waiting border | Purple | `#8b5cf6` |
| Pipeline container bg | Dark indigo gradient | `#1e1b4b` → `#312e81` |
| Pipeline step active | Green | `#10b981` |
| Pipeline step completed | Dark green | `#059669` |
| Pipeline arrow | Indigo | `#6366f1` |
| Terminal log bg | Dark slate | `#0f172a` |
| Terminal log text | Green | `#10b981` |
| Terminal timestamp | Slate gray | `#64748b` |
| Terminal agent name | Amber | `#f59e0b` |
| Terminal status text | Cyan | `#06b6d4` |
| Progress bar gradient | Green → Blue | `#10b981` → `#3b82f6` |
| Agent card border | Light gray | `#e5e7eb` |
| Agent card shadow | Light | `rgba(0,0,0,0.1)` |
| Sidebar section bg | Near-white | `#f8fafc` |
| Upload area border | Light slate | `#cbd5e1` |
| Upload area hover border | Blue | `#3b82f6` |
| Footer text | Gray | `#9ca3af` |

### Fonts

- **Terminal log:** `'Monaco', 'Consolas', monospace` at 13px
- **Everything else:** Streamlit default (system sans-serif stack)

### CSS Architecture

All custom CSS is injected as a single `<style>` block via `st.markdown(unsafe_allow_html=True)`. Key classes:

```css
.main-chat          /* max-width: 900px, centered */
.user-message        /* gradient bg, white text, rounded */
.assistant-message   /* gray bg, dark text, rounded */
.status-thinking     /* yellow bg, amber left-border */
.status-routing      /* blue bg, blue left-border */
.status-executing    /* green bg, green left-border */
.status-error        /* red bg, red left-border */
.status-waiting      /* purple bg, purple left-border */
.progress-container  /* gray track, 8px height */
.progress-bar        /* green-blue gradient fill */
.agent-card          /* white bg, border, shadow, 12px radius */
.sidebar-section     /* near-white bg, 12px padding, 8px radius */
.upload-area         /* dashed border, hover effect */
.typing-indicator    /* 3 dots with bounce animation */
.pipeline-container  /* dark indigo gradient bg */
.pipeline-step       /* pill with active/completed/pending states */
.pipeline-arrow      /* indigo arrow between steps */
.terminal-log        /* dark bg, green monospace text, 400px max-height scroll */
```

### Report Viewer CSS (pages/1_structured_report.py)

Additional classes for the structured report page:

```css
.badge-high      /* Green bg (#059669), white text */
.badge-medium    /* Amber bg (#d97706), white text */
.badge-low       /* Gray bg (#6b7280), white text */
.badge-flagged   /* Red bg (#dc2626), white text */
.badge-api       /* Purple bg (#8b5cf6), white text, small */
.conflict-box    /* Yellow bg, amber left-border */
.section-card    /* White bg, border, shadow */
.toc-active      /* Bold, blue (#2563eb) */
.report-table th /* Slate bg (#f1f5f9) */
```

---

## 12. Component-by-Component Mapping

| Streamlit Component | Location | Purpose | Suggested React Equivalent |
|--------------------|----------|---------|----------------------------|
| `st.set_page_config()` | `app.py:50` | Page title, icon, layout | Next.js `metadata` export + layout component |
| `st.markdown(css, unsafe_allow_html=True)` | `app.py:61-270` | Inject custom CSS | Global CSS module or Tailwind classes |
| `st.sidebar` | `app.py:633` | Side panel container | Fixed sidebar in layout (`aside` with CSS) |
| `st.expander(label)` | `app.py:641`, `610`, `703` | Collapsible section | `@radix-ui/react-accordion` or shadcn `Accordion` |
| `st.file_uploader()` | `app.py:659` | Multi-file upload with drag-drop | `react-dropzone` or shadcn `FileUpload` |
| `st.chat_input()` | `app.py:881` | Text input at bottom of page | Custom `ChatInput` with `onSubmit`, `disabled` prop |
| `st.chat_message("user"/"assistant")` | `app.py:857-861` | Message bubble with avatar | Custom `MessageBubble` with role-based styling |
| `st.markdown()` | Throughout | Rich text rendering | `react-markdown` with `remark-gfm`, `rehype-raw` |
| `st.image()` | `app.py:597` | Display PNG/JPG inline | Next.js `Image` or `<img>` with proper sizing |
| `st.download_button()` | `app.py:361`, `712-758` | File download trigger | `<a href={blobUrl} download={filename}>` or `FileSaver.js` |
| `st.button()` | `app.py:835`, `876` | Action buttons | shadcn `Button` component |
| `st.columns()` | `app.py:350`, `591`, `874` | Horizontal layout grid | CSS Grid or Flexbox |
| `st.container()` | `app.py:852` | Grouping container | `<div>` wrapper |
| `st.empty()` | `app.py:870` | Placeholder for live updates | React state + conditional rendering |
| `st.progress()` | `app.py:451`, `1013` | Progress bar (0.0-1.0) | shadcn `Progress` or HTML `<progress>` |
| `st.spinner()` | (not directly used) | Loading indicator | shadcn `Skeleton` or custom spinner |
| `st.info()` / `st.warning()` / `st.success()` / `st.error()` | Various | Colored alert banners | shadcn `Alert` with variant prop |
| `st.selectbox()` | Report page:328 | Dropdown selector | shadcn `Select` or native `<select>` |
| `st.text_input()` | Report page:386 | Text input field | shadcn `Input` |
| `st.metric()` | Report page:396-404 | Stat display (label + value) | Custom `MetricCard` component |
| `st.tabs()` | Report page:409 | Tab navigation | shadcn `Tabs` |
| `st.code()` | Report page:425 | Code/text display with syntax | `react-syntax-highlighter` or `<pre>` |
| `st.dataframe()` | Report page:278 | Table display | `@tanstack/react-table` or shadcn `Table` |
| `st.page_link()` | `app.py:828` | Link to another page | Next.js `Link` component |
| `st.rerun()` | `app.py:841`, `1079` | Force UI refresh | React state update (automatic re-render) |
| `st.stop()` | Report page:322 | Halt rendering | Early `return` in component |
| `st.caption()` | Report page:246 | Small muted text | `<p className="text-sm text-muted">` |

---

## 13. WebSocket Event Stream Design

### Recommended WebSocket Protocol

```typescript
// Connection
const ws = new WebSocket(`wss://${host}/api/ws/analysis`);

// Send analysis request
ws.send(JSON.stringify({
  type: "analysis_request",
  payload: {
    query: "Find datasets for lupus",
    session_id: "uuid-here",
    user_id: "optional-user-id",
    uploaded_file_ids: ["file-uuid-1", "file-uuid-2"]
  }
}));

// Receive events
ws.onmessage = (event) => {
  const msg: WebSocketMessage = JSON.parse(event.data);
  switch (msg.type) {
    case "status_update":    handleStatusUpdate(msg.payload);    break;
    case "heartbeat":        resetTimeout();                     break;
    case "error":            handleError(msg.payload);           break;
    case "stream_complete":  handleComplete();                   break;
  }
};
```

### Message Types

```typescript
type WebSocketMessage =
  | { type: "status_update"; payload: StatusUpdateEvent }
  | { type: "heartbeat"; payload: { timestamp: number } }
  | { type: "error"; payload: { code: string; message: string } }
  | { type: "stream_complete"; payload: { session_id: string } };
```

### Reconnection Strategy

```
On disconnect:
  1. Attempt reconnect with exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s max
  2. On reconnect, send: { type: "resume", session_id: "..." }
  3. Backend replays any missed StatusUpdates since last received timestamp
  4. After 5 failed attempts, show "Connection lost" banner with manual retry button
  5. If analysis was in progress, show last known state + "Reconnecting..." indicator
```

### Buffering Strategy

```
Fast events (multiple per second during pipeline execution):
  1. Buffer incoming StatusUpdates in a queue
  2. Render at 200ms intervals (5 FPS) — batch all queued updates into single React state update
  3. For PROGRESS events with same agent_name, keep only the latest (replace, don't accumulate)
  4. For EXECUTING/COMPLETED events, always render immediately (don't drop)
  5. Activity log: append all events regardless of rendering throttle
```

### Long-Running Pipeline Support

Pipelines (especially FASTQ processing) can run 30+ minutes:

```
1. Server sends heartbeat every 15 seconds during execution
2. Client timeout: 60 seconds without any message → show "Pipeline may still be running..."
3. Show elapsed time counter: "Running for 12m 34s"
4. Progress bar updates based on pipeline-level progress: (current_agent_index + agent_progress) / total_agents
5. Allow user to navigate away and return — reconnect via session_id
```

---

## 14. Migration Checklist

### Phase 1: Core Infrastructure

- [ ] **1.1** Set up Next.js project with App Router, Tailwind CSS, shadcn/ui
- [ ] **1.2** Create FastAPI backend wrapping `SupervisorAgent.process_message()` with WebSocket endpoint
- [ ] **1.3** Implement file upload API (`POST /api/upload`) saving to server disk, returning file metadata
- [ ] **1.4** Implement file download API (`GET /api/files/:path`) serving generated files
- [ ] **1.5** Set up session management (Redis-backed, replacing in-memory `SessionManager`)

### Phase 2: Chat Interface

- [ ] **2.1** Build chat message history component with role-based rendering (user/assistant/status)
- [ ] **2.2** Implement `ChatInput` component with disabled state during processing
- [ ] **2.3** Implement markdown rendering for assistant messages (`react-markdown` + `remark-gfm` + `rehype-raw`)
- [ ] **2.4** Implement custom message bubble styling matching the CSS palette
- [ ] **2.5** Build WebSocket client with connection management, reconnection, and heartbeat

### Phase 3: Real-Time Updates

- [ ] **3.1** Implement `StatusUpdate` event handler that maps `status_type` to UI components
- [ ] **3.2** Build pipeline progress display: agent headers (✅/⚡/🔹), step messages, progress bars
- [ ] **3.3** Implement event buffering (200ms render interval, latest-only for PROGRESS)
- [ ] **3.4** Build activity log accordion with terminal-style rendering
- [ ] **3.5** Implement cancellation: Stop button sends cancel signal, backend yields cancelled state

### Phase 4: File Management

- [ ] **4.1** Build drag-and-drop file upload component (`react-dropzone`) in sidebar
- [ ] **4.2** Implement file type display with icons and detection feedback
- [ ] **4.3** Build inline image display for generated plots (3-column grid in expander)
- [ ] **4.4** Build download buttons for individual files (CSV, PDF, DOCX, PNG)
- [ ] **4.5** Build ZIP download for output directories
- [ ] **4.6** Build "Download Results" sidebar section with per-agent categorized files

### Phase 5: Sidebar

- [ ] **5.1** Build agent list with expand/collapse showing description, estimated time, required inputs, example queries
- [ ] **5.2** Build session info panel (Session ID, Messages count, Current disease)
- [ ] **5.3** Build Clear Conversation button with state reset
- [ ] **5.4** Build structured report link (conditional, when report is generated)

### Phase 6: Structured Report Viewer

- [ ] **6.1** Build report page with TOC sidebar navigation
- [ ] **6.2** Implement Markdown section parser (heading-based)
- [ ] **6.3** Implement pipe-table extraction and rendering with `@tanstack/react-table`
- [ ] **6.4** Build evidence badges (HIGH/MEDIUM/LOW/FLAGGED) and API badges
- [ ] **6.5** Build cross-module conflict warning panel
- [ ] **6.6** Implement section search/filter
- [ ] **6.7** Build Interactive View / Raw Markdown tabs
- [ ] **6.8** Build download buttons for MD/PDF/DOCX companion files

### Phase 7: Welcome & Error States

- [ ] **7.1** Build welcome screen with 6 capability cards (matching current design)
- [ ] **7.2** Implement error display cards with appropriate color-coded styling
- [ ] **7.3** Implement "waiting for input" state with purple styling and input instructions
- [ ] **7.4** Build connection lost / reconnecting banner
- [ ] **7.5** Build elapsed time counter for long-running pipelines

### Phase 8: Testing

- [ ] **8.1** Test: General knowledge query → LLM response (no agent execution)
- [ ] **8.2** Test: Single agent — "Find datasets for lupus" → Cohort Retrieval
- [ ] **8.3** Test: File upload + single agent — upload counts CSV, "Run DEG analysis for lupus"
- [ ] **8.4** Test: Multi-agent pipeline — "Analyze pancreatic cancer" → Cohort → DEG → Prioritization → Pathway
- [ ] **8.5** Test: Missing inputs flow — request pathway analysis without gene list → WAITING_INPUT → provide file → resume
- [ ] **8.6** Test: Cancellation — start long pipeline, click Stop, verify partial results preserved
- [ ] **8.7** Test: Error handling — invalid file format, agent execution failure, pipeline partial failure
- [ ] **8.8** Test: Structured report generation and viewer page
- [ ] **8.9** Test: Large file upload (multi-GB FASTQ) → long-running FASTQ Processing agent
- [ ] **8.10** Test: MDP multi-disease clarification flow (upload 1 file for 3 diseases → disambiguation)

---

## Appendix A: Complete Agent Registry

| AgentType | name | display_name | estimated_time | depends_on | produces | requires_one_of |
|-----------|------|-------------|----------------|------------|----------|-----------------|
| `COHORT_RETRIEVAL` | `cohort_retrieval` | 🔍 Cohort Retrieval Agent | 2-10 min | (none) | `cohort_output_dir`, `cohort_summary_text` | `disease_name` |
| `DEG_ANALYSIS` | `deg_analysis` | 📊 DEG Analysis Agent | 5-15 min | (none) | `deg_base_dir`, `deg_input_file` | `cohort_output_dir`, `counts_file` |
| `GENE_PRIORITIZATION` | `gene_prioritization` | 🎯 Gene Prioritization Agent | 3-8 min | `DEG_ANALYSIS` | `prioritized_genes_path` | `deg_base_dir`, `deg_input_file` |
| `PATHWAY_ENRICHMENT` | `pathway_enrichment` | 🛤️ Pathway Enrichment Agent | 5-15 min | `GENE_PRIORITIZATION` | `pathway_consolidation_path` | `prioritized_genes_path` |
| `DECONVOLUTION` | `deconvolution` | 🧬 Deconvolution Agent | 5-20 min | (none) | `deconvolution_output_dir`, `cibersort_results` | `bulk_file` |
| `TEMPORAL_ANALYSIS` | `temporal_analysis` | ⏱️ Temporal Analysis Agent | 5-15 min | varies | `temporal_output_dir` | varies |
| `HARMONIZATION` | `harmonization` | 🔗 Harmonization Agent | 10-30 min | varies | `harmonization_output_dir` | varies |
| `MDP_ANALYSIS` | `mdp_analysis` | 🌐 MDP Analysis Agent | 10-30 min | varies | `mdp_output_dir` | varies |
| `PERTURBATION_ANALYSIS` | `perturbation_analysis` | 💊 Perturbation Analysis Agent | 5-15 min | varies | `perturbation_output_dir` | varies |
| `MULTIOMICS_INTEGRATION` | `multiomics_integration` | Multi-Omics Integration | 10-30 min | varies | `multiomics_output_dir` | varies |
| `FASTQ_PROCESSING` | `fastq_processing` | 🧬 FASTQ Processing Agent | 15-60 min | (none) | `counts_file` | `fastq_input_dir` |
| `MOLECULAR_REPORT` | `molecular_report` | 📋 Molecular Report Agent | 5-15 min | varies | `reporting_output_dir`, `structured_report_md_path` | varies |
| `CRISPR_PERTURB_SEQ` | `crispr_perturb_seq` | 🧬 CRISPR Perturb-seq Agent | 15-45 min | (none) | varies | `crispr_10x_input_dir` |
| `CRISPR_SCREENING` | `crispr_screening` | 🔬 CRISPR Screening Agent | 10-30 min | (none) | varies | `crispr_screening_input_dir` |
| `CRISPR_TARGETED` | `crispr_targeted` | 🎯 CRISPR Targeted Agent | 10-30 min | (none) | varies | `crispr_targeted_input_dir` or `project_id` |
| `CAUSALITY` | `causality` | 🔬 Causality Agent | 5-15 min | varies | varies | varies |

### Standard Pipeline Order

```
COHORT_RETRIEVAL → DEG_ANALYSIS → GENE_PRIORITIZATION → PATHWAY_ENRICHMENT
```

When the user requests a downstream agent (e.g., pathway enrichment) and upstream data is missing, the supervisor automatically builds the full chain via `_build_execution_chain()`.

---

## Appendix B: ConversationState Schema

```typescript
interface ConversationState {
  session_id: string;              // UUID
  user_id: string | null;
  created_at: number;              // Unix timestamp
  last_activity: number;           // Unix timestamp
  messages: Message[];
  uploaded_files: Record<string, UploadedFile>;
  agent_executions: AgentExecution[];
  workflow_state: Record<string, any>;
  current_disease: string | null;
  current_agent: string | null;
  waiting_for_input: boolean;
  required_inputs: string[];
  pending_agent_type: string | null;
}

interface Message {
  role: "user" | "assistant" | "system" | "agent_status";
  content: string;
  type: "text" | "file_upload" | "agent_routing" | "agent_progress" | "agent_result" | "error" | "thinking";
  timestamp: number;
  metadata: Record<string, any>;
}

interface UploadedFile {
  filename: string;
  filepath: string;
  file_type: string;
  upload_time: number;
  size_bytes: number;
  description: string | null;
}

interface AgentExecution {
  agent_name: string;
  agent_display_name: string;
  start_time: number;
  end_time: number | null;
  status: "running" | "completed" | "failed";
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  error: string | null;
  logs: string[];
  duration_seconds: number;     // computed property
}
```

---

## Appendix C: RoutingDecision Schema

```typescript
interface RoutingDecision {
  agent_type: AgentType | null;
  agent_name: string | null;
  confidence: number;              // 0.0 to 1.0
  reasoning: string;
  extracted_params: Record<string, any>;  // {disease_name, tissue_filter, ...}
  missing_inputs: string[];
  is_general_query: boolean;       // True for help, greetings, etc.
  suggested_response: string | null;  // Pre-computed response for general queries
  is_multi_agent: boolean;
  agent_pipeline: AgentIntent[];
}

interface AgentIntent {
  agent_type: AgentType;
  agent_name: string;
  confidence: number;
  reasoning: string;
  order: number;                   // 0-indexed execution order
}
```

---

## Appendix D: Welcome Screen Card Data

```typescript
const welcomeCards = [
  { icon: "🔍", title: "Cohort Retrieval", subtitle: "Search public databases", bg: "#f0fdf4" },
  { icon: "📊", title: "DEG Analysis", subtitle: "Differential expression", bg: "#eff6ff" },
  { icon: "🎯", title: "Gene Prioritization", subtitle: "Rank by relevance", bg: "#fef3c7" },
  { icon: "🛤️", title: "Pathway Analysis", subtitle: "Enrichment analysis", bg: "#ede9fe" },
  { icon: "🧬", title: "Deconvolution", subtitle: "Cell type estimation", bg: "#fce7f3" },
  { icon: "📊", title: "Structured Reports", subtitle: "Evidence-traced reports", bg: "#f0f9ff" },
];
```

Example prompt suggestions displayed below cards:
- "Find datasets for lupus disease"
- "Generate a structured report for my analysis"
