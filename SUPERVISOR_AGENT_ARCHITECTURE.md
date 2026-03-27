# Supervisor Agent — Complete Architecture & Functionality Reference

> **Scope:** Every module, class, method, data flow, and configuration in the supervisor agent system.
>
> **Audience:** Software leads, backend engineers, and integration teams who need to understand, extend, or deploy the supervisor agent.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Modules](#3-core-modules)
   - 3.1 [supervisor.py — Orchestrator](#31-supervisorpy--orchestrator)
   - 3.2 [state.py — Conversation & Session State](#32-statepy--conversation--session-state)
   - 3.3 [router.py — Intent Routing](#33-routerpy--intent-routing)
   - 3.4 [agent_registry.py — Agent Definitions](#34-agent_registrypy--agent-definitions)
   - 3.5 [llm_provider.py — LLM Abstraction](#35-llm_providerpy--llm-abstraction)
   - 3.6 [logging_utils.py — Structured Logging](#36-logging_utilspy--structured-logging)
4. [LangGraph Workflow](#4-langgraph-workflow)
   - 4.1 [Graph Topology](#41-graph-topology)
   - 4.2 [State Schema](#42-state-schema)
   - 4.3 [Node Implementations](#43-node-implementations)
   - 4.4 [Edge Functions](#44-edge-functions)
   - 4.5 [Entry Points](#45-entry-points)
   - 4.6 [Checkpointer](#46-checkpointer)
5. [Executors](#5-executors)
   - 5.1 [StatusUpdate System](#51-statusupdate-system)
   - 5.2 [Pipeline Executors](#52-pipeline-executors)
   - 5.3 [Output Validation](#53-output-validation)
6. [Response Module](#6-response-module)
   - 6.1 [Data Discovery](#61-data-discovery)
   - 6.2 [Query Functions](#62-query-functions)
   - 6.3 [Synthesizer & System Prompts](#63-synthesizer--system-prompts)
   - 6.4 [Document Renderer](#64-document-renderer)
   - 6.5 [Style Engine](#65-style-engine)
   - 6.6 [Enrichment Dispatcher](#66-enrichment-dispatcher)
7. [Data Layer](#7-data-layer)
   - 7.1 [Schemas](#71-schemas)
   - 7.2 [Manifest Builder](#72-manifest-builder)
   - 7.3 [Registry Builder](#73-registry-builder)
   - 7.4 [Module Adapters](#74-module-adapters)
8. [API Adapters](#8-api-adapters)
   - 8.1 [Base Adapter](#81-base-adapter)
   - 8.2 [TTL Cache](#82-ttl-cache)
   - 8.3 [Rate Limiter](#83-rate-limiter)
   - 8.4 [Configuration](#84-configuration)
   - 8.5 [Service Adapters](#85-service-adapters)
9. [Reporting Engine](#9-reporting-engine)
   - 9.1 [Report Planner](#91-report-planner)
   - 9.2 [Evidence Scoring & Conflict Detection](#92-evidence-scoring--conflict-detection)
   - 9.3 [Validation Guard](#93-validation-guard)
   - 9.4 [Section Builders](#94-section-builders)
   - 9.5 [Markdown Renderer](#95-markdown-renderer)
   - 9.6 [Report Enrichment](#96-report-enrichment)
10. [Causality Module](#10-causality-module)
    - 10.1 [Core Agent](#101-core-agent)
    - 10.2 [Adapter](#102-adapter)
    - 10.3 [LLM Bridge](#103-llm-bridge)
    - 10.4 [Tool Registry & Wiring](#104-tool-registry--wiring)
    - 10.5 [Constants](#105-constants)
11. [Data Flows](#11-data-flows)
    - 11.1 [Query to Routing Decision](#111-query-to-routing-decision)
    - 11.2 [Multi-Agent Pipeline Execution](#112-multi-agent-pipeline-execution)
    - 11.3 [Response Synthesis](#113-response-synthesis)
    - 11.4 [Structured Report Generation](#114-structured-report-generation)
    - 11.5 [Causality Analysis](#115-causality-analysis)
12. [Configuration Reference](#12-configuration-reference)
13. [File Type Detection Reference](#13-file-type-detection-reference)

---

## 1. System Overview

The Supervisor Agent is a **multi-agent LangGraph-based orchestrator** for bioinformatics analysis pipelines. It accepts natural-language queries, routes them to specialized analysis agents (DEG, pathway enrichment, drug discovery, etc.), manages conversation state across turns, and synthesizes results into chat responses or formatted reports (PDF/DOCX).

### Key Capabilities

- **Intent-based routing** — LLM-powered with keyword fallback
- **Multi-agent pipeline execution** — automatic dependency resolution and chain building
- **16 specialized agents** — cohort retrieval, DEG analysis, gene prioritization, pathway enrichment, deconvolution, temporal analysis, harmonization, MDP analysis, perturbation analysis, multi-omics integration, FASTQ processing, molecular report, CRISPR perturb-seq, CRISPR screening, CRISPR targeted, causality
- **Real-time progress streaming** — `AsyncGenerator[StatusUpdate]` for UI feedback
- **Conversation persistence** — LangGraph MemorySaver (dev) or PostgreSQL (prod) checkpointing
- **Structured report generation** — evidence-traced reports with conflict detection
- **API enrichment** — 9 external bioinformatics APIs (Ensembl, STRING, DGIdb, ChEMBL, OpenFDA, PubChem, Reactome, KEGG, ClinicalTrials.gov)
- **Dynamic document styling** — theme presets + LLM-generated CSS

### Execution Model

1. **Dual entry paths** — Legacy path (`SupervisorAgent.process_message()`) and LangGraph path (`run_supervisor_stream()` / `run_supervisor()`) controlled by `USE_LANGGRAPH_SUPERVISOR` env var
2. **Async generators** — All executors yield `StatusUpdate` objects for real-time UI updates
3. **Workflow state chaining** — Agent outputs accumulate in `workflow_state` dict, available as inputs to subsequent agents
4. **LLM failover** — AWS Bedrock (primary) with OpenAI GPT-4o fallback

---

## 2. Directory Structure

```
supervisor_agent/
├── __init__.py                     # Public exports: SupervisorAgent, SessionManager, ConversationState
├── supervisor.py                   # Main orchestrator (legacy path), file detection, chain builder
├── state.py                        # ConversationState, SessionManager, Message, UploadedFile
├── router.py                       # IntentRouter (LLM-based), KeywordRouter (fallback)
├── agent_registry.py               # 16 agent definitions, PIPELINE_ORDER, FILE_TYPE_TO_INPUT_KEY
├── llm_provider.py                 # Bedrock/OpenAI unified async LLM interface
├── logging_utils.py                # Structured logging with emoji indicators
├── utils.py                        # File-type detection, output collection helpers
│
├── executors/
│   ├── __init__.py                 # Re-exports StatusType, StatusUpdate, all execute_* functions
│   ├── base.py                     # StatusType enum, StatusUpdate dataclass, OutputValidation
│   └── pipeline_executors.py       # 15 async executor functions (one per agent type)
│
├── langgraph/
│   ├── __init__.py                 # Exports SupervisorGraphState, build_supervisor_graph, run_supervisor*
│   ├── state.py                    # SupervisorGraphState TypedDict, SupervisorResult, reducers
│   ├── graph.py                    # StateGraph construction and compilation
│   ├── nodes.py                    # 6 node implementations (intent, plan, router, executor, response, report)
│   ├── edges.py                    # Conditional edge functions (check_general_query, route_next)
│   ├── entry.py                    # Public async entry points (run_supervisor_stream, run_supervisor)
│   └── checkpointer.py            # Checkpointer factory (PostgreSQL or MemorySaver)
│
├── response/
│   ├── __init__.py                 # Re-exports all response functions
│   ├── data_discovery.py           # CSV discovery, scoring, query-aware data extraction
│   ├── query_functions.py          # Deterministic regex-based query dispatch (top genes, pathways, drugs)
│   ├── synthesizer.py              # 6 LLM system prompts + async synthesis functions
│   ├── document_renderer.py        # Markdown → HTML → PDF/DOCX rendering (WeasyPrint, htmldocx)
│   ├── style_engine.py             # Theme presets + LLM-generated CSS
│   └── enrichment_dispatcher.py    # Query-aware API enrichment (9 services)
│
├── data_layer/
│   ├── schemas/
│   │   ├── manifest.py             # RunManifest, ModuleRun, ModuleStatus
│   │   ├── registry.py             # ArtifactEntry, ArtifactIndex
│   │   ├── evidence.py             # EvidenceCard, ConflictRecord, Confidence, NarrativeContext
│   │   └── sections.py             # SectionBlock, TableBlock, SectionMeta, NarrativeMode, ReportingConfig
│   ├── adapters/
│   │   ├── base.py                 # BaseModuleAdapter (ABC)
│   │   ├── deg.py                  # DEGAdapter — discovers DEG CSV artifacts
│   │   ├── pathway.py              # PathwayAdapter — discovers pathway CSVs
│   │   └── drug.py                 # DrugDiscoveryAdapter — discovers drug pair CSVs
│   ├── manifest_builder.py         # Bridge: LangGraph state → RunManifest
│   └── registry_builder.py         # Bridge: RunManifest → ArtifactIndex
│
├── api_adapters/
│   ├── __init__.py                 # Imports all adapters (triggers auto-registration)
│   ├── base.py                     # BaseAPIAdapter with retry, cache, rate-limit
│   ├── config.py                   # APIConfig (Pydantic model)
│   ├── cache.py                    # TTLCache (LRU with per-entry expiry)
│   ├── rate_limiter.py             # AsyncRateLimiter (semaphore + interval tracking)
│   ├── chembl.py                   # ChEMBL drug/compound API
│   ├── dgidb.py                    # DGIdb gene-drug interactions
│   ├── kegg.py                     # KEGG pathways (academic-gated)
│   ├── openfda.py                  # OpenFDA adverse events
│   ├── pubchem.py                  # PubChem compound properties
│   ├── reactome.py                 # Reactome pathway database
│   ├── string_ppi.py               # STRING protein-protein interactions
│   ├── ensembl.py                  # Ensembl gene annotations
│   └── clinical_trials.py          # ClinicalTrials.gov
│
├── reporting_engine/
│   ├── __init__.py                 # Exports ReportPlanner, BUILDER_REGISTRY, etc.
│   ├── planner.py                  # ReportPlanner — section selection based on manifest
│   ├── evidence.py                 # score_findings(), detect_conflicts()
│   ├── validation.py               # ValidationGuard — pre-rendering validation
│   ├── enrichment.py               # ReportEnricher — optional API enrichment for reports
│   ├── builders/
│   │   ├── base.py                 # SectionBuilder ABC, BUILDER_REGISTRY
│   │   ├── deg_section.py          # DEGSectionBuilder
│   │   ├── pathway_section.py      # PathwaySectionBuilder
│   │   ├── drug_section.py         # DrugSectionBuilder
│   │   └── structural_sections.py  # ExecutiveSummary, RunSummary, DataOverview, IntegratedFindings,
│   │                               # Limitations, NextSteps, Appendix builders
│   └── renderers/
│       └── markdown_renderer.py    # Jinja2-based markdown rendering
│
├── causality/
│   ├── __init__.py                 # Exports CausalitySupervisorAgent, CausalityLLMBridge, execute_causality
│   ├── core_agent.py               # CausalitySupervisorAgent — intent parsing, gates, literature, DAG, analysis
│   ├── adapter.py                  # Async executor bridge to supervisor framework
│   ├── llm_bridge.py               # LLM access wrapper (Bedrock or local fallback)
│   ├── tool_registry.py            # Tool dispatch table (slot-based)
│   ├── tool_wiring.py              # Wire tool slots at runtime from workflow_state
│   ├── intelligence.py             # Intent parsing, literature search, evidence fusion
│   ├── routing_rules.py            # Intent-to-module routing rules
│   ├── routing_fallbacks.py        # Fallback routing when primary tools unavailable
│   ├── dag_registry.py             # DAG construction and persistence
│   ├── file_mapper.py              # Map workflow_state to causality input files
│   └── constants.py                # LLM defaults, literature endpoints, gate thresholds, evidence weights
│
└── tests/                          # 30+ test files covering all modules
```

---

## 3. Core Modules

### 3.1 supervisor.py — Orchestrator

The main entry point for the legacy (non-LangGraph) execution path. Contains file detection, input extraction, execution chain building, and the `SupervisorAgent` class.

#### Key Standalone Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_detect_file_type` | `(file_path: str) -> str` | Classifies files into `raw_counts`, `deg_results`, `prioritized_genes`, `pathway_results`, `multiomics_layer`, `patient_info`, `deconvolution_results`, `crispr_count_table`, `crispr_10x_data`, `fastq_file`, `fastq_directory`, `json_data`, `gene_list`, `unknown` |
| `_build_execution_chain` | `(target_agent: AgentType, available_keys: set, uploaded_file_type: str) -> tuple[list, list, str]` | Builds agent chain from available data to target agent. Returns `(agents_to_run, agents_skipped, info_message)` |
| `_dependency_closure` | `(target: AgentType) -> set` | Transitive dependency tree via `AGENT_REGISTRY.depends_on` |
| `_extract_contextual_inputs` | `(message: str, required_input_names: set) -> Dict[str, Any]` | Regex extraction of disease names, CRISPR params (protospacer, target_gene, region, reference_seq, project_id), file paths |
| `_validate_required_input_paths` | `(agent_info: AgentInfo, available_inputs: Dict) -> Optional[str]` | Validates that path inputs exist and meet content requirements (FASTQ reads, 10X triplet, etc.) |
| `_inspect_directory` | `(path: Path) -> Dict[str, Any]` | Scans directory for file counts, types, FASTQ pairs, 10X triplets, screening context |
| `_is_fastq_name` | `(name: str) -> bool` | Checks `.fastq`, `.fq`, `.fastq.gz`, `.fq.gz` extensions |
| `_paired_fastq_roots` | `(files: List[Path]) -> set[str]` | Finds paired FASTQ read roots (R1/R2 matching) |
| `_auto_generate_metadata` | `(counts_file, output_path, disease_name) -> Dict` | Auto-generates metadata CSV from counts file by classifying columns as Control/Disease |
| `_collect_generated_files` | `(output_dir, extensions) -> List[str]` | Recursively collects files by extension from output directory |
| `_hydrate_crispr_inputs` | `(agent_info, available_inputs, state) -> Dict` | Copies uploaded CRISPR files to runtime directories for Nextflow |

#### Disease Name Extraction Patterns

```python
# Priority order (first match wins):
r'disease\s+(?:is\s+)?["\']?([\w\s-]+)["\']?'   # "disease is lupus"
r'disease[:\s]+["\']?([\w\s-]+)["\']?'             # "disease: lupus"
r'for\s+([\w\s-]+?)\s+(?:analysis|study)'          # "for lupus analysis"
r'analyzing\s+([\w\s-]+)'                           # "analyzing lupus"
r'condition[:\s]+["\']?([\w\s-]+)["\']?'            # "condition: lupus"
r'^([\w\s-]+)$'                                     # bare disease name

# Stopwords filtered: is, the, a, an, my, this, that, yes, no, ok, okay
```

#### SupervisorAgent Class

```python
class SupervisorAgent:
    def __init__(self, session_manager=None, upload_dir="./uploads")

    async def process_message(
        self, user_message: str, session_id: str = None,
        user_id: str = None, uploaded_files: Dict[str, str] = None
    ) -> AsyncGenerator[StatusUpdate, None]

    # Internal methods:
    _register_agent_executors() -> Dict[AgentType, Callable]
    _resolve_dependencies(target_agent, available_inputs, uploaded_files) -> List[AgentInfo]
    _execute_multi_agent_pipeline(routing_decision, state) -> AsyncGenerator[StatusUpdate, None]
    _execute_pending_agent(agent_info, available_inputs, state) -> AsyncGenerator[StatusUpdate, None]
    _get_help_message() -> str
    _get_capabilities_summary() -> str
    _format_requirements(agent_info) -> str
    _format_missing_inputs_request(agent_info, missing, available) -> str
    _format_completion_message(agent_info, outputs) -> str
    _format_outputs(agent_info, outputs) -> str
    _format_pipeline_outputs(workflow_state) -> str
    _get_output_dir_for_agent(agent_name, workflow_state) -> Optional[str]
    _extract_inputs_from_message(message, required_inputs) -> Dict[str, Any]
```

#### process_message Flow (Legacy Path)

```
1. Register uploaded files → detect file types → yield PROGRESS status
2. Check USE_LANGGRAPH_SUPERVISOR → delegate if true
3. If waiting_for_input → handle follow-up input (MDP file assignment, missing params)
4. THINKING: "Understanding Your Request"
5. ROUTING: "Analyzing Intent" → call router.route()
6. Handle: general query → COMPLETED with suggested_response
          unknown intent → WAITING_INPUT with clarification
          multi-agent → _execute_multi_agent_pipeline()
7. Single agent:
   a. Extract contextual params (disease, CRISPR fields)
   b. Build execution chain → may expand to multi-agent pipeline
   c. VALIDATING: check requirements
   d. If missing → WAITING_INPUT with formatted request
   e. If all present → EXECUTING: start agent
   f. Execute via async generator → yield progress updates
   g. COMPLETED: format completion message with generated files
   h. On error → ERROR status
```

---

### 3.2 state.py — Conversation & Session State

#### Enums

| Enum | Values |
|------|--------|
| `MessageRole` | `USER`, `ASSISTANT`, `SYSTEM`, `AGENT_STATUS` |
| `MessageType` | `TEXT`, `FILE_UPLOAD`, `AGENT_ROUTING`, `AGENT_PROGRESS`, `AGENT_RESULT`, `ERROR`, `THINKING` |

#### Dataclasses

**Message**
| Field | Type | Default |
|-------|------|---------|
| `role` | `MessageRole` | required |
| `content` | `str` | required |
| `message_type` | `MessageType` | `TEXT` |
| `timestamp` | `float` | `time.time()` |
| `metadata` | `Dict[str, Any]` | `{}` |

**UploadedFile**
| Field | Type | Default |
|-------|------|---------|
| `filename` | `str` | required |
| `filepath` | `str` | required |
| `file_type` | `str` | required |
| `upload_time` | `float` | required |
| `size_bytes` | `int` | required |
| `description` | `Optional[str]` | `None` |

**AgentExecution**
| Field | Type | Default |
|-------|------|---------|
| `agent_name` | `str` | required |
| `agent_display_name` | `str` | required |
| `start_time` | `float` | required |
| `end_time` | `Optional[float]` | `None` |
| `status` | `str` | `"running"` |
| `inputs` | `Dict[str, Any]` | `{}` |
| `outputs` | `Dict[str, Any]` | `{}` |
| `error` | `Optional[str]` | `None` |
| `logs` | `List[str]` | `[]` |

**ConversationState** — complete session state tracking conversations, files, agent executions, and inter-agent data passing.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `session_id` | `str` | UUID | Unique session identifier |
| `user_id` | `Optional[str]` | `None` | Authenticated user |
| `created_at` | `float` | `time.time()` | Session creation time |
| `last_activity` | `float` | `time.time()` | Last interaction time |
| `messages` | `List[Message]` | `[]` | Conversation history |
| `uploaded_files` | `Dict[str, UploadedFile]` | `{}` | File registry |
| `agent_executions` | `List[AgentExecution]` | `[]` | Execution history |
| `workflow_state` | `Dict[str, Any]` | `{}` | Agent outputs (chaining) |
| `current_disease` | `Optional[str]` | `None` | Active disease context |
| `current_agent` | `Optional[str]` | `None` | Currently running agent |
| `waiting_for_input` | `bool` | `False` | Awaiting user input |
| `required_inputs` | `List[str]` | `[]` | What inputs are needed |
| `pending_agent_type` | `Optional[str]` | `None` | Agent waiting to resume |

#### Key Method: `get_available_inputs() -> Dict[str, Any]`

Merges all data sources into a single dict for agent input resolution:

1. **workflow_state** — outputs from completed agents
2. **uploaded_files** — mapped by filename patterns:
   - `"count"` or `"expression"` in name → `counts_file`, `bulk_file`
   - `"meta"` in name → `metadata_file`
   - `"deg"`, `"prioritized"`, `"filtered"` in name → `deg_input_file`, `prioritized_genes_path`, `deg_base_dir`
   - `"pathway"` in name → `pathway_consolidation_path`
   - `.h5ad` extension → `h5ad_file`
   - Any `.csv` → fallback for `counts_file`, `prioritized_genes_path`
3. **current_disease** → `disease_name`

#### SessionManager

In-memory session store (production would use Redis/DB).

| Method | Signature | Behavior |
|--------|-----------|----------|
| `create_session` | `(user_id=None) -> ConversationState` | Creates session, enforces `MAX_SESSIONS` limit with LRU eviction |
| `get_session` | `(session_id) -> Optional[ConversationState]` | Lookup by ID |
| `get_or_create_session` | `(session_id=None, user_id=None) -> ConversationState` | Tries session_id, then user_id, then creates new |
| `delete_session` | `(session_id)` | Removes session and user mapping |
| `cleanup_old_sessions` | `(max_age_hours=24)` | Removes sessions older than cutoff |

---

### 3.3 router.py — Intent Routing

#### IntentRouter (LLM-Based)

The primary router uses an LLM to classify user intent, extract parameters, and build multi-agent pipelines.

```python
class IntentRouter:
    async def route(user_query: str, conversation_state: ConversationState) -> RoutingDecision
```

**Routing prompt includes:**
- All 16 agent definitions with keywords, required inputs, dependencies
- Enforced dependency chain: `deg → gene_prioritization → pathway_enrichment → perturbation`
- Multi-agent query examples
- Topic-change detection rules
- MDP multi-disease handling
- CRISPR agent routing

**Pipeline ordering:** After LLM returns agent list, the router **re-sorts by `PIPELINE_ORDER`** to enforce correct dependencies (e.g., gene_prioritization always before pathway_enrichment even if LLM ordered them incorrectly).

#### RoutingDecision

| Field | Type | Purpose |
|-------|------|---------|
| `agent_type` | `Optional[AgentType]` | Primary agent (or first in pipeline) |
| `agent_name` | `Optional[str]` | Agent name string |
| `confidence` | `float` | 0.0 to 1.0 |
| `reasoning` | `str` | Human-readable explanation |
| `extracted_params` | `Dict[str, Any]` | Extracted from query: disease_name, tissue_filter, disease_names[], technique |
| `missing_inputs` | `List[str]` | Required inputs not yet available |
| `is_general_query` | `bool` | True for help, greetings, knowledge questions |
| `suggested_response` | `Optional[str]` | Pre-computed response for general queries |
| `is_multi_agent` | `bool` | True if pipeline detected |
| `agent_pipeline` | `List[AgentIntent]` | Ordered agent list for multi-agent execution |

#### KeywordRouter (Fallback)

Simple keyword-matching when LLM is unavailable. Builds keyword → agent_type map from all agents' keyword lists. Confidence = min(match_count / 3, 1.0).

---

### 3.4 agent_registry.py — Agent Definitions

#### AgentType Enum (16 values)

| Value | Pipeline Position | Dependencies |
|-------|-------------------|--------------|
| `COHORT_RETRIEVAL` | 0 | None |
| `DEG_ANALYSIS` | 1 | None |
| `GENE_PRIORITIZATION` | 2 | DEG_ANALYSIS |
| `PATHWAY_ENRICHMENT` | 3 | GENE_PRIORITIZATION |
| `PERTURBATION_ANALYSIS` | 4 | (needs prioritized_genes + pathway) |
| `MOLECULAR_REPORT` | 5 | (needs prioritized_genes + pathway) |
| `DECONVOLUTION` | Not in pipeline | None (standalone) |
| `TEMPORAL_ANALYSIS` | Not in pipeline | Varies |
| `HARMONIZATION` | Not in pipeline | Varies |
| `MDP_ANALYSIS` | Not in pipeline | Varies |
| `MULTIOMICS_INTEGRATION` | Not in pipeline | None (standalone) |
| `FASTQ_PROCESSING` | Not in pipeline | None (standalone) |
| `CRISPR_PERTURB_SEQ` | Not in pipeline | None (standalone) |
| `CRISPR_SCREENING` | Not in pipeline | None (standalone) |
| `CRISPR_TARGETED` | Not in pipeline | None (standalone) |
| `CAUSALITY` | Not in pipeline | Varies |

#### PIPELINE_ORDER

```python
[COHORT_RETRIEVAL, DEG_ANALYSIS, GENE_PRIORITIZATION, PATHWAY_ENRICHMENT, PERTURBATION_ANALYSIS, MOLECULAR_REPORT]
```

#### FILE_TYPE_TO_INPUT_KEY

| Detected File Type | Mapped Input Key(s) |
|--------------------|--------------------|
| `raw_counts` | `counts_file`, `bulk_file` |
| `deg_results` | `deg_input_file` |
| `prioritized_genes` | `prioritized_genes_path` |
| `pathway_results` | `pathway_consolidation_path` |
| `multiomics_layer` | `multiomics_layers` (accumulated into dict) |
| `patient_info` | `patient_info_path` |
| `deconvolution_results` | `xcell_path` |
| `crispr_10x_data` | `crispr_10x_input_dir` |
| `crispr_count_table` | `crispr_screening_input_dir` |
| `crispr_fastq_data` | `crispr_targeted_input_dir` |

#### MULTIOMICS_LAYER_NAMES

```python
frozenset({"genomics", "transcriptomics", "epigenomics", "proteomics", "metabolomics"})
```

#### Agent Input/Output Specifications

Each agent in `AGENT_REGISTRY` defines:
- **required_inputs** — `List[InputRequirement]` with name, description, file_type, is_file, required, can_come_from, example
- **optional_inputs** — same structure
- **outputs** — `List[OutputSpec]` with name, description, file_type, state_key
- **produces** — `List[str]` keys added to workflow_state
- **requires_one_of** — `List[str]` keys where at least one must be available
- **depends_on** — `List[AgentType]` upstream dependencies

---

### 3.5 llm_provider.py — LLM Abstraction

#### Dual Provider with Automatic Failover

```
USE_BEDROCK=true + AWS creds → Bedrock (Claude) primary → OpenAI fallback on error
USE_BEDROCK=false or no creds → OpenAI only
```

#### `llm_complete()` — Single Async Entry Point

```python
async def llm_complete(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    response_format: Optional[Dict] = None,  # {"type": "json_object"} for JSON mode
) -> LLMResult
```

Returns `LLMResult(text, provider, model, latency_ms, fallback_used, input_tokens, output_tokens)`.

#### `_safe_json_parse()` — Resilient JSON Extraction

Handles Claude's common output patterns:
1. Plain JSON
2. Markdown code fences (` ```json ... ``` `)
3. JSON embedded in text (extracts between first `{` and last `}`)

#### Bedrock Configuration

- Model: `BEDROCK_MODEL_ID` (default: `us.anthropic.claude-opus-4-5-20251101-v1:0`)
- API version: `bedrock-2023-05-31`
- Timeout: 300s read, 60s connect, 3 retries
- JSON mode: system prompt injection ("You must respond with valid JSON only")
- Concurrency: `LLM_MAX_CONCURRENCY` (default 10) via `ThreadPoolExecutor`

---

### 3.6 logging_utils.py — Structured Logging

Provides `SupervisorLogger` with emoji-annotated log methods:

| Method | Icon | Purpose |
|--------|------|---------|
| `thinking()` | 🧠 | Reasoning steps |
| `routing()` | 🎯 | Routing decisions with confidence |
| `validating()` | ✅ | Input validation checks |
| `executing()` | 🚀 | Agent start |
| `progress()` | ⏳ | Step progress with percentage |
| `completed()` | ✅ | Completion with duration and outputs |
| `error()` | ❌ | Errors with exception details |
| `user_input_needed()` | 📎 | Waiting for user input |
| `file_received()` | 📁 | File upload tracking |
| `session_event()` | 🔄 | Session lifecycle events |

Uses `contextvars.ContextVar` for `analysis_id` correlation across async tasks.

---

## 4. LangGraph Workflow

### 4.1 Graph Topology

```
START ──→ intent ──→ check_general_query ──→ general ──→ END
                │                          │
                │                          ├──→ follow_up ──→ response ──→ END
                │                          │
                │                          ├──→ structured_report ──→ END
                │                          │
                │                          └──→ needs_execution ──→ plan ──→ router ──┐
                │                                                                      │
                │                                                    ┌──────────────────┘
                │                                                    │
                │                                                    ▼
                │                                              route_next
                │                                              /   |   \   \
                │                                   execute_agent  │  report  done→END
                │                                        │         │    │
                │                                        ▼         │    ▼
                │                                    executor ──→ router  response→END
                │                                                  │
                │                                          structured_report→END
```

### 4.2 State Schema (SupervisorGraphState)

| Layer | Fields | Reducer |
|-------|--------|---------|
| **Intent** | `user_query`, `disease_name`, `uploaded_files`, `detected_file_types`, `routing_decision`, `is_general_query`, `general_response`, `needs_response`, `response_format`, `report_theme`, `style_instructions`, `molecular_report_format`, `requested_top_n`, `has_existing_results` | — |
| **Planning** | `execution_plan`, `agents_skipped`, `current_agent_index` | — |
| **Execution** | `workflow_outputs`, `agent_results`, `errors`, `retry_counts` | `_merge_dicts` for dicts, `_capped_add` for lists |
| **Conversation** | `conversation_history` | `_capped_add` (max 50) |
| **Response** | `final_response` | — |
| **Telemetry** | `llm_call_log` | `_capped_add` |
| **Infrastructure** | `session_id`, `analysis_id`, `user_id`, `output_root`, `status` | — |

**Reducers:**
- `_merge_dicts(left, right)` — shallow merge, filters strings >10k chars
- `_capped_add(left, right)` — append with MAX_CONVERSATION_HISTORY (default 50) cap

### 4.3 Node Implementations

#### intent_node

1. Detect uploaded file types via `_detect_file_type()`
2. Fast-path: short confirmations ("ok", "yes") replay prior response_format
3. Call `IntentRouter.route()` → RoutingDecision
4. Detect `needs_response`, `response_format` (pdf/docx/chat/none)
5. Handle molecular report intent (`_MOLECULAR_REPORT_RE`)
6. Extract `top_n`, `theme`, `style_instructions`
7. Return partial state with routing decision + metadata

**Regex patterns used:**
- `_REPORTING_SIGNALS` — "summarize", "results", "top N", "report", "findings"
- `_FOLLOWUP_SIGNALS` — "elaborate", "tell me more", "expand", "sources"
- `_EXEC_ONLY_START` — "run", "execute", "perform" at start of query
- `_FORMAT_PDF`, `_FORMAT_DOCX` — document format detection
- `_CONFIRMATION_RE` — short affirmative replies

#### plan_node

1. Extract agent types from routing decision
2. For multi-agent: skip agents whose outputs already exist in `workflow_outputs`
3. For single-agent: build upstream chain via `_build_execution_chain()`
4. Return `execution_plan[]`, `agents_skipped[]`, `current_agent_index=0`

#### agent_executor

1. Get current agent from `execution_plan[current_agent_index]`
2. Build `available_inputs` from `workflow_outputs` + uploaded files
3. Preflight check: validate required file paths exist
4. Call `EXECUTOR_MAP[agent_type](agent_info, inputs, state)`
5. Stream StatusUpdates via `progress_callback`
6. On success: update `workflow_outputs`, append to `agent_results`
7. On failure: classify error (transient → retry up to 2x, permanent → record)
8. Increment `current_agent_index`

#### response_node

1. Discover relevant CSVs from `workflow_outputs`
2. Extract entities (genes, drugs, pathways) from query + data
3. Call `enrich_for_response()` for API enrichment
4. Build prompt via `build_response_user_prompt()` or `build_document_user_prompt()`
5. Synthesize via `synthesize_chat_multipass()` (chat) or `synthesize_multipass()` (doc)
6. If pdf/docx: render via `render_pdf()` / `render_docx()`
7. Return `final_response`

#### report_generation_node

1. Build `RunManifest` from state via `manifest_from_state()`
2. Build `ArtifactIndex` via `build_artifact_index()`
3. Plan sections via `ReportPlanner.plan()`
4. For each section: run builder from `BUILDER_REGISTRY`
5. Score findings, detect conflicts
6. Validate via `ValidationGuard`
7. Render markdown via Jinja2 template
8. Optionally render PDF/DOCX
9. Return `final_response` with report path

### 4.4 Edge Functions

**`check_general_query(state) -> str`**

| Condition | Returns |
|-----------|---------|
| has_results AND structured report regex matches query | `"structured_report"` |
| is_general AND format is pdf/docx | `"follow_up"` |
| is_general AND needs_response | `"follow_up"` |
| is_general AND NOT needs_response | `"general"` (→ END) |
| NOT general AND has_results AND agent already produced outputs | `"follow_up"` |
| Default | `"needs_execution"` |

**`route_next(state) -> str`**

| Condition | Returns |
|-----------|---------|
| `current_agent_index < len(execution_plan)` | `"execute_agent"` |
| All done AND structured report in query | `"structured_report"` |
| All done AND `needs_response` | `"report"` |
| All done AND all failed AND no results | `"report"` |
| Default | `"done"` (→ END) |

### 4.5 Entry Points

#### `run_supervisor_stream()` — Streaming

```python
async def run_supervisor_stream(
    user_query: str, session_id: str, analysis_id: str = "",
    output_root: str = "", disease_name: str = "",
    uploaded_files: Dict = None, workflow_outputs: Dict = None,
    user_id: str = "",
) -> AsyncGenerator[StatusUpdate, None]
```

Uses `asyncio.Queue` with sentinel pattern to bridge graph callbacks to async generator.

#### `run_supervisor()` — Non-Streaming

```python
async def run_supervisor(...) -> SupervisorResult
```

Returns `SupervisorResult(status, final_response, output_dir, key_files, agent_results, errors, execution_time_ms)`.

### 4.6 Checkpointer

Controlled by `CHECKPOINTER_BACKEND` env var:
- `"postgres"` → `PostgresSaver` from `langgraph-checkpoint-postgres` (requires `CHECKPOINTER_DB_URL`)
- default → `MemorySaver` (in-memory, dev mode)

Thread ID = `session_id` for conversation persistence across invocations.

---

## 5. Executors

### 5.1 StatusUpdate System

```python
class StatusType(str, Enum):
    THINKING      = "thinking"       # Understanding/analyzing query
    ROUTING       = "routing"        # Matching to appropriate agent
    VALIDATING    = "validating"     # Checking requirements
    EXECUTING     = "executing"      # Running the agent
    PROGRESS      = "progress"       # Mid-execution progress update
    INFO          = "info"           # Informational message
    COMPLETED     = "completed"      # Successful completion
    ERROR         = "error"          # Execution failed
    WAITING_INPUT = "waiting_input"  # Need additional user input

@dataclass
class StatusUpdate:
    status_type: StatusType
    title: str                           # Short header (e.g., "🚀 Starting DEG Analysis")
    message: str                         # Markdown body
    details: Optional[str] = None        # Collapsible detail text
    progress: Optional[float] = None     # 0.0 to 1.0
    agent_name: Optional[str] = None     # e.g., "deg_analysis"
    timestamp: float = time.time()       # Auto-set
    generated_files: Optional[List] = None   # File paths on completion
    output_dir: Optional[str] = None     # Agent output directory
```

### 5.2 Pipeline Executors

All 15 executor functions follow the same pattern:

```python
async def execute_<agent_type>(
    agent_info: AgentInfo,
    inputs: Dict[str, Any],
    state: ConversationState
) -> AsyncGenerator[StatusUpdate, None]:
    # 1. yield StatusUpdate(EXECUTING, "Starting...")
    # 2. Set up output directory
    # 3. Call the agent's core function (sync via run_in_executor or async)
    # 4. yield StatusUpdate(PROGRESS, ...) at milestones
    # 5. Collect generated files
    # 6. Update state.workflow_state with outputs
    # 7. yield StatusUpdate(COMPLETED, ...) with generated_files
```

**Helper utilities:**
- `_ensure_conda_env_on_path()` — sets up PATH for Nextflow-based agents (CRISPR)
- `_cleanup_nextflow_output(output_dir)` — flattens Nextflow results directory
- `_get_output_dir(agent_name, session_id)` — creates `outputs/{agent_name}/{session_id}/`

### 5.3 Output Validation

```python
def validate_pipeline_output(output_dir: str, agent_display_name: str) -> OutputValidation
```

Scans output directory: counts files, calculates total size, categorizes by extension, detects empty subdirectories. Returns formatted summary string.

---

## 6. Response Module

### 6.1 Data Discovery

**CSV Scoring Algorithm:**
- +1 per query word found in CSV headers
- +3 for high-value filenames (`Final_Gene_Priorities`, `pathway_consolidation`, `Final_GeneDrug_Pairs`, etc.)
- -5 if file > 10MB
- -0.5 per depth level beyond 5 in directory tree
- Top 8 CSVs returned, sorted by score

**Query-Aware Data Extraction:**
- Natural language → column resolution via synonym dictionary (e.g., "disorder score" → `Non_CGC_Composite_Score`)
- Tier filters: "tier 1 genes only" → filter by Tier column
- Numeric filters: "composite > 0.5" → numeric comparison
- Smart sorting: ascending for rank/p-value, descending for scores

### 6.2 Query Functions

Deterministic regex dispatch for common queries:

| Pattern | Handler | CSV Type |
|---------|---------|----------|
| "top N genes" | `get_top_genes()` | genes |
| "top N pathways" | `get_top_pathways()` | pathways |
| "top N drugs" | `get_top_drugs()` | drugs |
| "details for GENE" | `get_gene_detail()` | genes |
| "summarize/findings" | `get_full_summary()` | any |

### 6.3 Synthesizer & System Prompts

Six specialized system prompts:

| Prompt | Purpose | Temperature |
|--------|---------|-------------|
| `RESPONSE_SYSTEM_PROMPT` | Chat responses calibrated to query type | 0.1 |
| `DOCUMENT_SYSTEM_PROMPT` | Formal analysis documents (5-12 pages) | 0.1 |
| `STRUCTURED_REPORT_PROMPT` | Evidence-traced narrative from NarrativeContext | 0.15 |
| `OUTLINE_SYSTEM_PROMPT` | Data-driven section outlines | 0.15 |
| `REVIEW_SYSTEM_PROMPT` | Scientific editing (data grounding, cross-refs) | 0.1 |
| `CHAT_REFINEMENT_PROMPT` | Transform analysis to conversational response | 0.15 |

**Synthesis Modes:**
- `synthesize_response()` — single-pass
- `synthesize_multipass()` — 3-4 pass: Outline → Draft → Review → (Polish)
- `synthesize_chat_multipass()` — 2 pass: Analysis → Refinement
- `augment_narrative()` — single-pass for NarrativeContext sections

### 6.4 Document Renderer

- **PDF:** WeasyPrint (HTML → PDF) with auto landscape for wide tables
- **DOCX:** htmldocx (HTML → DOCX) with post-processing: proportional column widths, header row repeat, font styling, heading colors from CSS

### 6.5 Style Engine

Three built-in themes:

| Theme | Primary Color | Font |
|-------|---------------|------|
| `default` | Teal `#028090` | Sans-serif |
| `clinical` | Navy `#1a365d` | Georgia (serif) |
| `minimal` | Gray `#333333` | Sans-serif |

LLM-generated CSS for custom styling requests (e.g., "make headings red", "use a dark theme").

### 6.6 Enrichment Dispatcher

**Entity Extraction (regex, zero LLM calls):**
- Genes: `[A-Z][A-Z0-9]{1,9}` from query + CSV gene columns (max 15)
- Drugs: from CSV drug columns (max 8)
- Pathways: from CSV pathway_id columns, filtered to `R-HSA-` or `hsa` prefix (max 5)
- Disease: from query context

**API Adapter → Entity Mapping:**

| Entity Type | Adapters Called | Data Returned |
|------------|----------------|---------------|
| genes | ensembl, string, dgidb | Annotations, PPI network, drug interactions |
| drugs | chembl, openfda, pubchem | Mechanisms, adverse events, properties |
| pathways | reactome, kegg | Pathway details |
| disease | clinical_trials | Recruiting trials |

All adapter calls run in parallel with configurable timeout (default 15s).

---

## 7. Data Layer

### 7.1 Schemas

**ModuleStatus:** `COMPLETED`, `FAILED`, `SKIPPED`, `NOT_RUN`

**Confidence:** `HIGH`, `MEDIUM`, `LOW`, `FLAGGED`

**NarrativeMode:** `DETERMINISTIC`, `LLM_AUGMENTED`

**EvidenceCard** — links a finding to its source artifact with confidence tier and ranking.

**ConflictRecord** — cross-module disagreement between two evidence cards.

**NarrativeContext** — boundary between deterministic extraction and optional LLM augmentation: `disease_name`, `section_title`, `evidence_cards[]`, `table_summaries{}`, `conflicts[]`, `extra{}`.

**ReportingConfig** — user-controllable report knobs: `narrative_mode`, `table_row_cap` (5-200, default 25), `include_appendix`, `score_threshold` (0.0-1.0), `sections_enabled{}`, `theme`.

### 7.2 Manifest Builder

`manifest_from_state(state) -> RunManifest` — discovers completed modules from `workflow_outputs` keys (suffix `_output_dir` or `_base_dir`), cross-references with `agent_results[]` and `errors[]`.

### 7.3 Registry Builder

`build_artifact_index(manifest) -> ArtifactIndex` — runs module-specific adapters (DEG, Pathway, Drug) to discover CSV artifacts with columns, row counts, and QC flags.

### 7.4 Module Adapters

| Adapter | Module | Discovers |
|---------|--------|-----------|
| `DEGAdapter` | deg_analysis | DEG tables, gene priorities |
| `PathwayAdapter` | pathway_enrichment | Pathway consolidation CSVs |
| `DrugDiscoveryAdapter` | perturbation_analysis | Drug-gene pairs, drug scores |

---

## 8. API Adapters

### 8.1 Base Adapter

All 9 service adapters extend `BaseAPIAdapter`:
- Auto-registration via `__init_subclass__` into `API_ADAPTER_REGISTRY`
- Built-in: httpx client, TTL cache, rate limiter, retry with exponential backoff
- `_request()` handles GET/POST, caching, rate-limiting, error classification

### 8.2 TTL Cache

LRU cache with per-entry expiry. Max size eviction (oldest-first). Deterministic cache keys from `(method, url, sorted_params)`.

### 8.3 Rate Limiter

Token-bucket-style: `asyncio.Semaphore` for concurrency + elapsed-time tracking for minimum inter-request interval.

### 8.4 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `kegg_enabled` | `False` | Academic-only gating |
| `default_timeout` | `30.0s` | HTTP request timeout |
| `max_retries` | `3` | Retry attempts on transient errors |
| `default_cache_ttl` | `3600s` | 1 hour cache |
| `cache_max_size` | `2048` | Max cached entries |
| Per-service RPS | 2.0–15.0 | Rate limits per service |

### 8.5 Service Adapters

| Service | Base URL | Key Methods |
|---------|----------|-------------|
| Ensembl | rest.ensembl.org | Gene lookup, xrefs |
| STRING | string-db.org | Network interactions |
| DGIdb | dgidb.org | Gene-drug interactions |
| ChEMBL | ebi.ac.uk/chembl | Molecule search, mechanisms, indications |
| OpenFDA | api.fda.gov | Adverse event queries |
| PubChem | pubchem.ncbi.nlm.nih.gov | Compound properties |
| Reactome | reactome.org | Pathway details |
| KEGG | rest.kegg.jp | Pathway info (academic-gated) |
| ClinicalTrials.gov | clinicaltrials.gov | Trial search by condition |

---

## 9. Reporting Engine

### 9.1 Report Planner

Plans which sections to include based on manifest:

| Section ID | Title | Required Module |
|-----------|-------|-----------------|
| `executive_summary` | Executive Summary | Always |
| `run_summary` | Run Summary | Always |
| `data_overview` | Data Overview | Always |
| `deg_findings` | Differential Expression Analysis | deg_analysis |
| `pathway_findings` | Pathway Enrichment Analysis | pathway_enrichment |
| `drug_findings` | Drug Discovery & Perturbation | perturbation_analysis |
| `integrated_findings` | Integrated Findings | Always |
| `limitations` | Limitations & Caveats | Always |
| `next_steps` | Recommended Next Steps | Always |
| `appendix` | Appendix | Always (toggleable) |

### 9.2 Evidence Scoring & Conflict Detection

- **Scoring:** Auto-promotes high-confidence module cards (deg_analysis, pathway_enrichment) if metric > 0. Ranks by confidence tier, then metric value.
- **Conflict detection:** Groups cards by section, compares across modules for directional disagreements (positive vs negative metric values).

### 9.3 Validation Guard

Pre-rendering checks: section IDs, titles, table column counts, evidence card artifact references. Raises `ValidationError(errors: List[str])` on hard errors, returns warnings list.

### 9.4 Section Builders

Each builder extends `SectionBuilder` ABC:
- `DEGSectionBuilder` — DEG table summaries with fold-change/p-value analysis
- `PathwaySectionBuilder` — pathway enrichment tables with scores
- `DrugSectionBuilder` — drug-gene pair tables with mechanisms
- Structural builders: ExecutiveSummary, RunSummary, DataOverview, IntegratedFindings, Limitations, NextSteps, Appendix

### 9.5 Markdown Renderer

Jinja2-based rendering of sections + evidence cards into markdown. Supports table formatting, evidence badges, conflict warnings.

### 9.6 Report Enrichment

`ReportEnricher` — optional API enrichment for report sections (calls relevant API adapters to add external context).

---

## 10. Causality Module

### 10.1 Core Agent

**Intent Classification (7 types):**

| ID | Name | Description |
|----|------|-------------|
| `I_01` | Causal Drivers Discovery | Find causal drivers of a phenotype |
| `I_02` | Directed Causality | Test X → Y causal relationship |
| `I_03` | Intervention/Actionability | Identify actionable interventions |
| `I_04` | Comparative Causality | Compare causal mechanisms across conditions |
| `I_05` | Counterfactual/What-If | Counterfactual analysis |
| `I_06` | Evidence Inspection | Explain existing evidence |
| `I_07` | Standard Association | Non-causal association analysis |

**Eligibility Gates:**

| Gate | Threshold | Status |
|------|-----------|--------|
| Min samples | < 30 | BLOCK |
| Warn samples | < 60 | WARN |
| Min MR instruments | < 3 | BLOCK |
| Edge confidence | < 0.70 | WARN |

**Evidence Stream Weights (6-stream fusion):**

| Stream | Weight |
|--------|--------|
| Genetic | 0.30 |
| Perturbation | 0.25 |
| Temporal | 0.20 |
| Network | 0.15 |
| Expression | 0.05 |
| Immunological | 0.05 |

**Pipeline:** Parse intent → File audit → Eligibility gates → Literature review → DAG construction → Causal analysis → Result synthesis.

### 10.2 Adapter

`execute_causality()` bridges the causality engine to the supervisor framework:
1. Map files from workflow_state
2. Wire tool slots from available data
3. Run `CausalitySupervisorAgent.run()` in thread
4. Yield StatusUpdates, populate workflow_state

### 10.3 LLM Bridge

`CausalityLLMBridge` wraps the supervisor's Bedrock client for causality LLM calls, with JSON parsing and local fallback.

### 10.4 Tool Registry & Wiring

Slot-based tool dispatch: `TOOL_REGISTRY` maps tool IDs to function pointers (initially None, populated at runtime via `tool_wiring.populate()` based on available workflow_state data).

### 10.5 Constants

```python
CAUSALITY_MODEL = "claude-sonnet-4-20250514"
LIT_MAX_PAPERS = 30, LIT_TOP_K = 15, LIT_TIMEOUT = 12.0
GATE_MIN_SAMPLES = 30, GATE_WARN_SAMPLES = 60
GATE_MIN_MR_INSTRUMENTS = 3, GATE_EDGE_CONFIDENCE = 0.70
```

---

## 11. Data Flows

### 11.1 Query to Routing Decision

```
User Query → supervisor.py (extract paths, params)
  → IntentRouter.route() [LLM analysis]
    → Parse JSON response → RoutingDecision
      → Re-sort pipeline by PIPELINE_ORDER
        → Return: agent_type, agent_pipeline[], extracted_params, confidence
```

### 11.2 Multi-Agent Pipeline Execution

```
RoutingDecision.agent_pipeline → plan_node (skip if outputs exist)
  → Loop: for each agent in execution_plan:
      1. Build available_inputs from workflow_outputs + uploads
      2. Preflight: validate file paths exist
      3. Execute: EXECUTOR_MAP[agent_type](agent_info, inputs, state)
      4. Stream: yield StatusUpdates to UI
      5. Update: workflow_outputs += agent outputs
      6. Error: classify → retry (transient, max 2x) or record (permanent)
  → Pipeline complete: yield COMPLETED with all generated files
```

### 11.3 Response Synthesis

```
workflow_outputs → discover_relevant_csvs() [score by query relevance]
  → extract_entities() [regex: genes, drugs, pathways from CSVs + query]
    → enrich_for_response() [parallel API calls: ensembl, string, chembl, ...]
      → build_*_user_prompt() [assemble context + data + enrichment]
        → synthesize_chat_multipass() [2-pass: analysis → refinement]
          OR synthesize_multipass() [3-4 pass: outline → draft → review → polish]
            → render_pdf/docx() [if document format requested]
              → Return: final_response + optional report file
```

### 11.4 Structured Report Generation

```
workflow_outputs → manifest_from_state() → RunManifest
  → build_artifact_index() → ArtifactIndex (via module adapters)
    → ReportPlanner.plan() → List[SectionBlock] (based on completed modules)
      → For each section:
          BUILDER_REGISTRY[section_type].build() → section body + tables
          score_findings() → ranked EvidenceCards
          detect_conflicts() → ConflictRecords
      → ValidationGuard.validate() → warnings / raise on hard errors
        → render_markdown() → Jinja2 template → markdown string
          → render_pdf/docx() [if requested]
            → Return: final_response + report path
```

### 11.5 Causality Analysis

```
User query + workflow_state → execute_causality()
  → file_mapper.map_files() → input file paths
    → tool_wiring.populate() → wire tool slots
      → CausalityLLMBridge(bedrock_fn) → LLM access
        → CausalitySupervisorAgent.run():
            1. Parse intent (I_01–I_07)
            2. Audit files (FileInspector)
            3. Check eligibility gates (sample size, MR instruments)
            4. Literature review (PubMed, EPMC, Semantic Scholar)
            5. DAG construction
            6. Causal analysis via tool slots
            7. Synthesize FinalResult
          → Return: headline, top_findings, tier1/2_candidates, evidence_quality
```

---

## 12. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_BEDROCK` | `"false"` | Enable AWS Bedrock as primary LLM |
| `BEDROCK_MODEL_ID` | `"us.anthropic.claude-opus-4-5-20251101-v1:0"` | Bedrock model identifier |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials for Bedrock |
| `AWS_REGION` | `"us-east-1"` | AWS region |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback or primary) |
| `LLM_MAX_CONCURRENCY` | `"10"` | Max parallel LLM calls |
| `USE_LANGGRAPH_SUPERVISOR` | `"false"` | Enable LangGraph execution path |
| `CHECKPOINTER_BACKEND` | `"memory"` | LangGraph checkpointer: `"postgres"` or `"memory"` |
| `CHECKPOINTER_DB_URL` | — | PostgreSQL connection string (when backend=postgres) |
| `MAX_SESSIONS` | `"1000"` | Maximum concurrent sessions |
| `MAX_CONVERSATION_HISTORY` | `"50"` | Messages retained per session in LangGraph |
| `KEGG_ENABLED` | `"false"` | Enable KEGG API adapter (academic use only) |
| `OPENFDA_API_KEY` | — | Optional FDA API key |
| `NCBI_API_KEY` | — | Optional NCBI API key |

### Output Directories

```
supervisor_agent/outputs/{agent_name}/{session_id}/     # Standard agent outputs
temp/supervisor_crispr/{session_id}/{agent_name}/       # CRISPR runtime files
./streamlit_uploads/                                     # UI file uploads
```

---

## 13. File Type Detection Reference

The `_detect_file_type()` function classifies input files in priority order:

| Priority | Type | Detection Method |
|----------|------|------------------|
| 1 | `fastq_file` | Extension: `.fastq`, `.fq`, `.fastq.gz`, `.fq.gz` |
| 2 | `crispr_10x_data` | Filename starts with: barcodes.tsv, features.tsv, genes.tsv, matrix.mtx |
| 3 | `multiomics_layer` | Filename contains: genomics, transcriptomics, epigenomics, proteomics, metabolomics |
| 4 | `json_data` | Extension: `.json` |
| 5 | `gene_list` | Extension: `.txt` with short alphanumeric lines |
| 6 | `crispr_count_table` | CSV with sgRNA/guide_rna/spacer columns |
| 7 | `pathway_results` | CSV with >=2 of: pathway_name, pathway_id, enrichment_score, gene_ratio, kegg_id, go_id, reactome |
| 8 | `prioritized_genes` | CSV with: composite_score, druggability_score, priority_score, final_score, ppi_degree |
| 9 | `deconvolution_results` | Filename contains: cibersort, xcell, deconvolution, deconv, cell_fractions |
| 10 | `patient_info` | Filename contains patient/subject/demographics OR >=2 clinical columns |
| 11 | `raw_counts` | Gene ID first column + all numeric columns + no DEG-specific column names |
| 12 | `deg_results` | Has both fold-change columns AND p-value columns |
| 13 | `unknown` | Could not determine |

**Directory detection:**
- 10X triplet (barcodes + features/genes + matrix) → `crispr_10x_data`
- Paired FASTQs with screening context → `crispr_count_table`
- Paired FASTQs without screening → `fastq_directory`
