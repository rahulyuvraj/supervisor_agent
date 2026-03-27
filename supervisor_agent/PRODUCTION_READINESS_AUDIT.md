# Production Readiness Audit — Supervisor Agent

**Auditor:** Senior Production Engineer & Code Auditor
**Date:** 2026-03-25
**Scope:** `supervisor_agent/` + `streamlit_app/` (109 Python files, ~24,400 LOC)
**Branch:** `supervisor_agent_V0.1`

---

## 1. EXECUTIVE SUMMARY

The Supervisor Agent is an impressively well-architected LangGraph orchestrator with solid fundamentals: clean node/edge topology, proper error classification with self-healing retries, robust API adapter base class with rate limiting and caching, and thoughtful multi-pass LLM synthesis. However, it has **8 critical and 14 high-priority issues** that must be addressed before production deployment. The most severe are: **committed `.env` file containing live API keys and AWS credentials** (immediate revocation required), **in-memory-only state with no persistence** (MemorySaver loses all sessions on restart), **XSS vulnerability in document renderer** (unescaped HTML in title/disease fields), **unbounded conversation_history growth** that will eventually OOM long-running sessions, **no authentication or authorization** on any endpoint, and **no timeout on any pipeline executor** (17 `run_in_executor()` calls can hang indefinitely). The codebase quality is otherwise strong — clean module boundaries, good type hints, and 505 passing tests — but the testing focuses on happy paths with zero coverage of the router, LLM provider, causality module, and LangGraph nodes.

---

## 2. CRITICAL ISSUES

These **must be fixed** before any production deployment.

| # | Issue | File:Line | Severity | Fix |
|---|-------|-----------|----------|-----|
| C1 | **MemorySaver is in-memory only — all state lost on restart/deploy** | `langgraph/graph.py:55` | CRITICAL | Replace `MemorySaver()` with `SqliteSaver` (single-node) or `PostgresSaver` (multi-node). LangGraph ships both. Wrap in env-gated factory: `get_checkpointer()` → `MemorySaver` in dev, `PostgresSaver` in prod. |
| C2 | **`conversation_history` grows unboundedly via `operator.add`** — accumulates across ALL turns with no eviction. A session with 50+ turns will stuff 100KB+ of history into every LLM prompt, causing OOM and token-limit blowouts. | `langgraph/state.py:52`, `langgraph/nodes.py:302,436-441` | CRITICAL | Add a sliding-window reducer: `def _capped_history(left, right): return (left + right)[-MAX_HISTORY:]` with `MAX_HISTORY=50`. Apply same pattern to `agent_results`, `errors`, and `llm_call_log`. |
| C3 | **OpenAI fallback has no error handling** — if Bedrock fails and then `_openai_call()` raises (rate limit, auth error, network), the exception propagates unhandled and crashes the graph node. | `llm_provider.py:157-158,185` | CRITICAL | Wrap the OpenAI call in try/except. Return a structured error `LLMResult` or raise a custom `LLMProviderError` that callers handle. Add `max_tokens` passthrough (currently silently dropped for OpenAI). |
| C4 | **No authentication or authorization** — the Streamlit app and any future API surface have zero auth. Any user can access any session_id, see other users' data, or trigger expensive LLM/pipeline operations. | `streamlit_app/app.py`, `langgraph/entry.py` | CRITICAL | Add session-scoped auth (at minimum: Streamlit `st.secrets` + login page). For API exposure: add JWT/API-key auth middleware. Validate that `session_id` belongs to the authenticated user. |
| C5 | **Global mutable singletons are not thread-safe under concurrent requests** — `_graph`, `_bedrock_client`, `_openai_client`, `_intent_router` are module-level globals set lazily with no locks. Under concurrent Streamlit sessions, two requests can race on initialization. | `langgraph/graph.py:62-70`, `llm_provider.py:29-30,70-83,168-171`, `langgraph/nodes.py:114-121` | CRITICAL | Use `threading.Lock()` for lazy-singleton init. Or initialize eagerly at module load. For `_bedrock_client`/`_openai_client`: these are internally thread-safe in `boto3`/`openai`, but the `global` + `if None` pattern is still racy — double-checked locking or `functools.lru_cache(maxsize=1)` is safer. |
| C6 | **Committed `.env` file contains live API keys and AWS credentials** — OpenAI API key (`sk-proj-...`), AWS Access Key ID (`AKIA3QI3...`), and AWS Secret Access Key are committed to the repository in plaintext. **Anyone with repo access can extract these.** | `supervisor_agent/.env:5,8,9` | CRITICAL | **Immediate actions:** (1) Revoke the OpenAI API key, (2) Rotate AWS credentials, (3) Add `.env` to `.gitignore`, (4) Use `git filter-repo` to purge `.env` from git history, (5) Create `.env.example` template with placeholder values. For production: use AWS Secrets Manager or HashiCorp Vault. |
| C7 | **XSS vulnerability in document renderer** — `title` and `disease` are inserted into HTML without escaping. `disease="Lupus</h1><script>alert('XSS')</script>"` would execute in the rendered PDF/DOCX HTML intermediate. | `response/document_renderer.py:36-38` | CRITICAL | Use `html.escape()`: `f"<h1>{html.escape(title)}</h1>"`. Apply to all user-controlled values in `_build_html()`. |
| C8 | **No timeout on any pipeline executor** — all 17 `run_in_executor()` calls in `pipeline_executors.py` can hang indefinitely. A single stuck Nextflow pipeline blocks the entire event loop thread. | `executors/pipeline_executors.py:376,654,735,819,980,1130,1268,1388,1524,1615,1687,1794` | CRITICAL | Wrap all executor calls: `await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=MAX_PIPELINE_TIMEOUT)`. Set `MAX_PIPELINE_TIMEOUT` to 1800s (30 min) or make configurable per agent type. |

---

## 3. HIGH PRIORITY

Should fix before launch.

| # | Issue | File:Line | Severity | Fix |
|---|-------|-----------|----------|-----|
| H1 | **`workflow_outputs` accumulates unboundedly** via `_merge_dicts` across turns. Large pipeline outputs (DataFrames serialized as dicts, file listings) pile up across sessions. | `langgraph/state.py:46` | HIGH | Implement a bounded merge that prunes stale keys (e.g., only keep latest 3 turns' outputs), or reset `workflow_outputs` at turn boundary in `_build_initial_state()` when not explicitly passed in. |
| H2 | **No LLM token tracking or cost monitoring** — `LLMResult.latency_ms` is logged, but token counts are never extracted from Bedrock/OpenAI responses. No budget enforcement. A single user can rack up thousands in API costs. | `llm_provider.py:34-43,126-127,185-187` | HIGH | Extract `input_tokens`/`output_tokens` from Bedrock response (`resp_body["usage"]`) and OpenAI response (`resp.usage`). Add them to `LLMResult`. Implement per-session and global budget limits via env vars (`MAX_TOKENS_PER_SESSION`, `MAX_DAILY_COST`). |
| H3 | **Report files written to relative path `agentic_ai_wf/shared/reports/`** with no cleanup, rotation, or size limits. In production, disk fills up. | `langgraph/nodes.py:971,1187` | HIGH | Use an absolute, configurable output directory (`REPORT_OUTPUT_DIR` env var). Add a background cleanup job (cron or async task) that purges reports older than N days. Set a per-session file-count limit. |
| H4 | **CSV loading has no size guard** — `pd.read_csv()` in `response_node` (line 830) loads entire CSVs into memory. A 2GB CSV will OOM the process. | `langgraph/nodes.py:830-833`, `response/data_discovery.py` | HIGH | Add `nrows=MAX_ROWS` (e.g., 50,000) to all `pd.read_csv()` calls in the response path. Or check file size before loading and skip/sample large files. |
| H5 | **API adapter sessions are created and destroyed per enrichment call** — `enrich_for_response()` calls `__aenter__`/`__aexit__` on every invocation, creating new `httpx.AsyncClient` instances each time. No connection pooling. | `response/enrichment_dispatcher.py:407-409,442-446` | HIGH | Create a singleton `AdapterPool` that initializes adapters once and reuses them. Use `httpx.AsyncClient` with connection pooling (`limits=httpx.Limits(max_connections=100)`). |
| H6 | **Enrichment timeout doesn't cancel hanging tasks properly** — `asyncio.wait()` with timeout returns pending tasks, which are cancelled but `await` is never called on them, leaking coroutines. | `response/enrichment_dispatcher.py:427-438` | HIGH | After cancelling timed-out tasks, `await asyncio.gather(*pending, return_exceptions=True)` to ensure cleanup. |
| H7 | **Exceptions swallowed silently in API adapter call wrappers** — every `_call_*` function in `enrichment_dispatcher.py` catches bare `except Exception: pass`, losing error context. | `enrichment_dispatcher.py:237-238,244-247,251-255,271-272,284-285,297-298,309-310` | HIGH | Log exceptions at DEBUG level: `except Exception as exc: logger.debug("ensembl lookup for %s failed: %s", g, exc)`. At minimum, aggregate a failure count for observability. |
| H8 | **`_bedrock_invoke` runs in default executor (ThreadPoolExecutor)** with no bound on concurrency. 50 concurrent requests = 50 threads making Bedrock calls. | `llm_provider.py:146-149` | HIGH | Use a bounded executor: `_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("LLM_MAX_CONCURRENCY", "10")))` and pass it to `run_in_executor()`. |
| H9 | **`validate_llm_config()` runs at module import time** (line 214), which means importing `llm_provider` in test environments triggers env-var reads and log output. | `llm_provider.py:214` | HIGH | Move `validate_llm_config()` to a lazy/explicit init function called by `llm_complete()` on first use, not at import time. |
| H10 | **No request-level correlation ID** — logs use `__name__` loggers but nothing ties log entries across intent → plan → execute → response for a single user request. | All files | HIGH | Add `analysis_id` (already in state) to a `logging.LoggerAdapter` or use `contextvars.ContextVar`. Pass it through config and include in all log messages. |
| H11 | **SessionManager (state.py:340-408) has no upper bound on sessions** — `_sessions` dict grows forever if `cleanup_old_sessions()` is never called (and no scheduler invokes it). | `state.py:340-408` | HIGH | Add a max-sessions limit with LRU eviction. Wire `cleanup_old_sessions()` into a periodic background task (e.g., every 5 minutes via `asyncio.create_task`). |
| H12 | **Path traversal in `_get_output_dir()`** — `session_id` is used directly in path construction without validation. A crafted `session_id="../../../etc"` creates directories outside the outputs folder. | `executors/pipeline_executors.py:111-115` | HIGH | Validate `session_id` contains only alphanumeric chars, hyphens, and underscores. Use `Path.resolve()` and verify the result is under the expected base directory. |
| H13 | **`asyncio.gather()` with no timeout in `ReportEnricher.enrich()`** — unlike `enrichment_dispatcher.py` which uses `asyncio.wait(timeout=15)`, the reporting engine enricher has zero timeout. A single hanging API call blocks the entire report. | `reporting_engine/enrichment.py:106-108` | HIGH | Add timeout: `done, pending = await asyncio.wait(tasks, timeout=30.0)` and cancel pending tasks. |
| H14 | **Bare `except:` clauses in supervisor.py** catch `SystemExit` and `KeyboardInterrupt` — three locations use bare `except:` (not `except Exception:`) which prevents graceful shutdown. | `supervisor.py:569,581,637` | HIGH | Change bare `except:` to `except Exception:` at all three locations. |

---

## 4. MEDIUM PRIORITY

Fix within the first sprint post-launch.

| # | Issue | File:Line | Severity | Fix |
|---|-------|-----------|----------|-----|
| M1 | **Backoff in `agent_executor` can be very long** — `5 ** attempt` for runtime errors = 25s on second retry, 125s on third. Combined with max_retries=2, worst case is 150s of sleeping. | `langgraph/nodes.py:716` | MEDIUM | Cap backoff: `min(5 ** attempt, 30)`. Consider using jitter: `backoff * (0.5 + random.random())`. |
| M2 | **`_safe_json_parse` fallback extracts between first `{` and last `}`** — this is fragile and could parse partial/incorrect JSON from LLM responses containing multiple JSON objects. | `llm_provider.py:61-66` | MEDIUM | Validate the parsed result against expected schema (e.g., `RoutingDecision` fields). Log a warning when fallback parsing is used. |
| M3 | **`detect_file_type()` in `utils.py` loads pandas for every file type check** — `import pandas as pd` inside the function, plus `pd.read_csv(path, nrows=10)` for every uploaded file. | `utils.py:35,74` | MEDIUM | Cache results by filepath+mtime. Consider using a lighter heuristic (read first line with `csv.reader`) before falling back to pandas. |
| M4 | **Report directory uses Unix timestamps as folder names** (e.g., `1743000000/`) — collisions possible if two reports generate in the same second. | `langgraph/nodes.py:967-978,1185-1192` | MEDIUM | Include `analysis_id` or a random suffix: `f"{slug}_{ts}_{analysis_id[:8]}"`. |
| M5 | **CSS sanitization in `style_engine.py` blocks `url()` but doesn't block `behavior:`** (IE-specific), `@charset`, or `@namespace`. | `style_engine.py:228-239` | MEDIUM | Add: `css = re.sub(r'behavior\s*:', '/* blocked */', css, flags=re.IGNORECASE)`. The current sanitization is good for most vectors but could be more comprehensive. |
| M6 | **`_GENE_SYMBOL_RE` in `enrichment_dispatcher.py` matches common English words** — `r"\b([A-Z][A-Z0-9]{1,9})\b"` matches "FOR", "AND", "THE", etc. in uppercase text. | `enrichment_dispatcher.py:38` | MEDIUM | Add a stopword filter: exclude common 2-3 letter words and acronyms that aren't genes. Or require minimum length of 3 and check against a gene symbol database. |
| M7 | **`collect_output_summaries()` and CSV discovery walk the filesystem** with `Path.rglob()` which can be slow on large output directories or network mounts. | `response/data_discovery.py` | MEDIUM | Set a max-depth limit on directory walking. Cache discovery results per session. Add a timeout guard. |
| M8 | **Structured report validation swallows `ValidationError`** — the guard catches it, logs a warning, but the report is still generated with potentially invalid data. | `langgraph/nodes.py:1117-1118` | MEDIUM | Include validation warnings in the report output (e.g., as a "Data Quality Notes" section) so users are aware. |
| M9 | **Logging uses emoji-heavy format strings** (`🧠`, `🎯`, `🚀`, etc.) — these render poorly in structured log aggregators (ELK, CloudWatch) and waste bytes. | `logging_utils.py:47-97` | MEDIUM | Use structured JSON logging in production. Keep emoji format for local dev only via a `LOG_FORMAT` env var toggle. |
| M10 | **`_ensure_conda_env_on_path()` mutates `os.environ["PATH"]` globally** — this affects all subprocesses in the process, not just the current executor. | `executors/pipeline_executors.py:26-69` | MEDIUM | Call this once at startup, not per-executor invocation. Or use `subprocess.run(env={...})` to pass modified env per-process. |
| M11 | **No health check endpoint** — there's no way to verify the system is operational (LLM providers reachable, adapters working, disk space available). | N/A | MEDIUM | Add a `/health` endpoint (or CLI command) that checks: Bedrock connectivity, OpenAI connectivity, disk free space, and returns structured status. |
| M12 | **`response_format` and `report_theme` leak from checkpoint** — `_build_initial_state()` resets `response_format` to `"none"` but the confirmation fast-path reads `prev_fmt` from the checkpoint. Edge case: user's second turn sees stale format. | `langgraph/entry.py:40`, `langgraph/nodes.py:273-276` | MEDIUM | The confirmation path is intentionally reading the checkpoint. Document this behavior. Add a staleness check: if the previous turn was >N minutes ago, don't treat short replies as confirmations. |

---

## 5. LOW PRIORITY

Nice-to-have improvements.

| # | Issue | File:Line | Severity | Fix |
|---|-------|-----------|----------|-----|
| L1 | **`import asyncio` at module bottom** in `logging_utils.py:155` — after using `asyncio.iscoroutinefunction` in the decorator. Works but is a code smell. | `logging_utils.py:147,155` | LOW | Move `import asyncio` to the top of the file. |
| L2 | **`_extract_top_n` caps at 100** but doesn't enforce a minimum. `top 0` or `top -5` would pass through. | `langgraph/nodes.py:214-220` | LOW | Add `return max(1, min(int(m.group(1)), 100))`. |
| L3 | **Duplicate `shutil` import** in `pipeline_executors.py` — imported at top (line 16) and again inside `_prepare_single_file_for_deg` (line 264). | `executors/pipeline_executors.py:16,264` | LOW | Remove the inner import. |
| L4 | **`_CONFIRMATION_RE` can false-positive** on queries like "yes, run the DEG analysis on this new file" — treating it as confirmation of prior action rather than a new request. | `langgraph/nodes.py:175-179,274` | LOW | Tighten: only match if the ENTIRE query (not just start) matches the pattern, or limit to queries under 15 chars. |
| L5 | **`TTLCache` does not proactively evict expired entries** — they're only removed on `get()`. Over time, expired entries consume memory until the max_size eviction kicks in. | `api_adapters/cache.py:29-31` | LOW | Add a periodic `_purge_expired()` call (e.g., every 100th `set()`). |
| L6 | **`build_css()` uses `%` string formatting** for CSS template which requires `%%` escaping for CSS percentage values. Fragile. | `style_engine.py:56,262-266` | LOW | Switch to `.format()` or f-strings for the template to avoid double-percent escaping. |
| L7 | **`test_supervisor.py` and `langgraph/test_langgraph_e2e.py` are outside the `tests/` directory** — inconsistent test organization. | `test_supervisor.py`, `langgraph/test_langgraph_e2e.py` | LOW | Move to `tests/` for consistent collection. |
| L8 | **`ConversationState.get_available_inputs()` has hardcoded filename heuristics** — `"count" in filename_lower`, `"meta" in filename_lower` — which will false-positive on files like "accountability_report.csv". | `state.py:276-279` | LOW | Use `_METADATA_RE` pattern (already in nodes.py) or more specific patterns. |
| L9 | **Multiple `except Exception: pass` blocks** in DOCX rendering post-processing. | `response/document_renderer.py:93-94` | LOW | Log at DEBUG level instead of silently passing. |

---

## 6. ARCHITECTURE RECOMMENDATIONS

### 6.1 Persistent Checkpointing (Required)
Replace `MemorySaver()` with `PostgresSaver` backed by a managed PostgreSQL instance. This provides:
- Session survival across deploys/restarts
- Multi-instance horizontal scaling
- Point-in-time session debugging
- Automatic cleanup via PostgreSQL TTL/partitioning

### 6.2 Observability Stack
1. **Structured Logging**: Switch to JSON logging (`python-json-logger`) with `analysis_id` correlation.
2. **Metrics**: Add Prometheus counters/histograms for: LLM calls (latency, tokens, cost), API adapter calls (success/failure/timeout per service), pipeline executions (duration, success rate), report generations.
3. **Tracing**: Integrate OpenTelemetry. LangGraph supports LangSmith tracing natively — enable it for production debugging.
4. **Alerting**: Alert on: LLM provider failures, adapter timeout rate > 20%, disk usage > 80%, session count growth rate.

### 6.3 Resource Governance
1. **LLM Budget**: Per-session token budget (e.g., 500K tokens). Per-day global budget. Circuit breaker that falls back to cached/shorter responses when approaching limits.
2. **Connection Pooling**: Singleton `httpx.AsyncClient` pool shared across adapter instances, with configurable limits.
3. **Report Cleanup**: Background async task that deletes reports older than 7 days. Configurable retention via env var.
4. **Memory Guard**: Monitor process RSS. If approaching limit, trim conversation_history and workflow_outputs aggressively.

### 6.4 Security Hardening
1. **Authentication**: Implement OAuth2/OIDC or API key auth before any external exposure.
2. **Input Validation**: Add Pydantic models for all user-facing inputs (`user_query` length limit, `uploaded_files` size limit, allowed file extensions).
3. **File Upload Safety**: Validate file paths are within expected directories. Reject symlinks. Sanitize filenames.
4. **Prompt Injection Defense**: The user_query flows directly into LLM prompts. Add input preprocessing to detect and defang injection attempts.
5. **Secret Management**: Move from `.env` files to a secrets manager (AWS Secrets Manager, HashiCorp Vault) for production API keys.

### 6.5 Graceful Degradation
1. **LLM Fallback Chain**: Bedrock → OpenAI → cached response → static error message (never crash).
2. **Adapter Circuit Breakers**: If an API adapter fails 3x in 5 minutes, disable it for 10 minutes instead of retrying on every request.
3. **Pipeline Timeout**: Add a global timeout for the entire graph invocation (e.g., 15 minutes) to prevent runaway executions.

---

## 7. DEPLOYMENT CHECKLIST

### Pre-Deployment (URGENT)
- [ ] **C6**: IMMEDIATELY revoke committed API keys and AWS credentials; rotate all secrets
- [ ] **C6**: Add `.env` to `.gitignore`; purge from git history with `git filter-repo`
- [ ] **C7**: HTML-escape title/disease in `document_renderer.py:_build_html()`
- [ ] **C8**: Add `asyncio.wait_for()` timeout to all 17 `run_in_executor()` calls

### Pre-Deployment (Critical)
- [ ] **C1**: Replace MemorySaver with PostgresSaver; test session persistence across restarts
- [ ] **C2**: Implement capped history reducer; load-test with 100+ turn sessions
- [ ] **C3**: Add error handling to OpenAI fallback; test with both providers down
- [ ] **C4**: Implement authentication (at minimum: API key or Streamlit login)
- [ ] **C5**: Add locks to all global singleton initializations
- [ ] **H2**: Extract and log LLM token counts; set per-session budget limit
- [ ] **H3**: Configure absolute report output directory; add cleanup cron
- [ ] **H4**: Add `nrows` limit to all response-path CSV loads
- [ ] **H10**: Implement correlation ID logging with `analysis_id`
- [ ] Pin all dependencies in `pyproject.toml` with exact versions
- [ ] Add Dockerfile with health check, resource limits, non-root user
- [ ] Set up structured JSON logging for production
- [ ] Configure log aggregation (CloudWatch/ELK)
- [ ] Add Prometheus metrics endpoint
- [ ] Set up alerting for LLM failures, OOM, disk full

### Testing
- [ ] Add integration test: full graph flow (intent → plan → execute → response)
- [ ] Add concurrency test: 10 simultaneous sessions
- [ ] Add chaos test: Bedrock returns 500, OpenAI returns 429
- [ ] Add memory test: 50-turn session stays under 500MB RSS
- [ ] Add load test: sustained 10 req/min for 1 hour
- [ ] Verify all API adapter error paths are tested

### Rollout
- [ ] Deploy to staging with feature flags (all adapters disabled initially)
- [ ] Enable adapters one at a time, monitoring error rates
- [ ] Run shadow-mode: process real queries but don't serve responses
- [ ] Enable for internal users (alpha)
- [ ] Monitor token costs for 1 week
- [ ] Enable for external users (beta) with rate limiting

---

## 8. ESTIMATED EFFORT

### Sprint 0 — Emergency Security (Day 1, 1 engineer)
| Task | Size | Est. |
|------|------|------|
| C6: Revoke API keys, rotate AWS creds, purge .env from git | S | 2 hours |
| C7: HTML-escape in document_renderer.py | S | 1 hour |
| C8: Add timeouts to all 17 run_in_executor() calls | S | 3 hours |
| H14: Fix bare except: clauses in supervisor.py | S | 30 min |

### Sprint 1 — Critical Fixes (1 week, 3 engineers)
| Task | Size | Est. |
|------|------|------|
| C1: PostgresSaver integration | M | 2 days |
| C2: Bounded history + state reducers | S | 1 day |
| C3: OpenAI fallback error handling | S | 0.5 days |
| C4: Basic authentication | M | 2 days |
| C5: Singleton thread safety | S | 0.5 days |
| H9: Lazy LLM config validation | S | 0.5 days |
| H12: Path traversal fix in _get_output_dir | S | 0.5 days |

### Sprint 2 — High Priority (1 week, 2 engineers)
| Task | Size | Est. |
|------|------|------|
| H2: LLM token tracking + budget | M | 2 days |
| H3: Report directory management + cleanup | S | 1 day |
| H4: CSV size guards | S | 0.5 days |
| H5: Adapter connection pooling | M | 1.5 days |
| H6-H7: Enrichment timeout + logging fixes | S | 1 day |
| H8: Bounded LLM executor | S | 0.5 days |
| H10: Correlation ID logging | M | 1.5 days |
| H11: Session cleanup scheduler | S | 1 day |

### Sprint 3 — Observability + Medium Priority (1 week, 2 engineers)
| Task | Size | Est. |
|------|------|------|
| M9+M11: Structured logging + health check | M | 2 days |
| Prometheus metrics integration | M | 2 days |
| M1: Backoff cap + jitter | S | 0.5 days |
| M2-M6: Assorted medium fixes | M | 2 days |
| Dockerfile + docker-compose | M | 1.5 days |
| Integration + load tests | L | 3 days |

### Sprint 4 — Hardening (1 week, 1 engineer)
| Task | Size | Est. |
|------|------|------|
| Security audit: input validation, prompt injection | M | 2 days |
| Circuit breakers for adapters | M | 2 days |
| Global graph timeout | S | 1 day |
| Documentation: env vars, deployment guide | M | 2 days |
| Low-priority fixes (L1-L9) | S | 1 day |

---

## APPENDIX A: LLM Call Map

A single user query can trigger up to **7 LLM calls** in the worst case:

| Call | Location | Purpose | Model |
|------|----------|---------|-------|
| 1 | `router.py` → `IntentRouter.route()` | Intent classification + routing | Bedrock Claude / GPT-4o |
| 2 | `nodes.py:886` | Multi-pass synthesis — outline | Bedrock Claude / GPT-4o |
| 3 | `nodes.py:922` | Multi-pass synthesis — draft | Bedrock Claude / GPT-4o |
| 4 | `nodes.py:922` | Multi-pass synthesis — review | Bedrock Claude / GPT-4o |
| 5 | `style_engine.py:248` | CSS generation (if styling requested) | Bedrock Claude / GPT-4o |
| 6 | `nodes.py:1144` | Narrative augmentation (structured report) | Bedrock Claude / GPT-4o |
| 7 | `nodes.py:1174` | Review pass (structured report) | Bedrock Claude / GPT-4o |

**Optimization opportunity**: Calls 2+3+4 (multi-pass synthesis) could be reduced to 2 passes for `"brief"` scope. Call 5 (CSS) is cached in `workflow_outputs` — ensure cache hit rate is high.

## APPENDIX B: Timeout Map

| Component | Timeout Value | Source |
|-----------|---------------|--------|
| Bedrock `read_timeout` | 300s | `llm_provider.py:80` |
| Bedrock `connect_timeout` | 60s | `llm_provider.py:80` |
| Bedrock retries | 3 attempts (boto3 built-in) | `llm_provider.py:81` |
| OpenAI timeout | Default (httpx 5s connect, 300s total) | `llm_provider.py:185` (no explicit timeout) |
| API adapter HTTP timeout | 30s | `api_adapters/config.py:44` |
| API adapter retries | 3 attempts | `api_adapters/config.py:45` |
| API adapter backoff cap | 16s | `api_adapters/base.py:162` |
| Enrichment dispatcher timeout | 15s | `enrichment_dispatcher.py:373,427` |
| Agent executor retry backoff | 2^n (API) or 5^n (runtime) — UNBOUNDED | `langgraph/nodes.py:716` |
| Max retries per agent | 2 (env: `SUPERVISOR_MAX_RETRIES`) | `langgraph/nodes.py:651` |
| Graph invocation | **No global timeout** | `langgraph/entry.py:93,183` |

**Key gap**: No global timeout on `graph.ainvoke()`. A hanging pipeline executor can block indefinitely.

## APPENDIX C: Testing Coverage Gaps

Based on analysis of 31 test files in `tests/`:

| Module | Tested? | Gap |
|--------|---------|-----|
| `langgraph/nodes.py` — intent_node | ✅ Partial | Confirmation fast-path, molecular report detection, style carry-forward untested |
| `langgraph/nodes.py` — agent_executor | ✅ Partial | Retry loop, error classification, _cleanup_partial_output untested |
| `langgraph/nodes.py` — response_node | ✅ Partial | PDF/DOCX rendering path, enrichment integration untested |
| `langgraph/nodes.py` — report_generation_node | ❌ | Not tested |
| `langgraph/edges.py` | ✅ Partial | structured_report routing untested |
| `langgraph/entry.py` | ❌ | `run_supervisor_stream` generator protocol untested |
| `llm_provider.py` — Bedrock→OpenAI fallback | ❌ | Fallback behavior not tested |
| `llm_provider.py` — `_safe_json_parse` | ❌ | Not tested |
| `router.py` — IntentRouter.route() | ❌ | Entire class untested — no mock LLM tests |
| `router.py` — KeywordRouter (dead code) | ❌ | Dead code — class defined but never used |
| `api_adapters/base.py` — retry logic | ✅ Partial | 429/503 paths tested; DNS failure, SSL error untested |
| `api_adapters/*` — individual adapters | ✅ Partial | Happy paths tested; error paths largely untested |
| `response/enrichment_dispatcher.py` | ❌ | Entity extraction tested, but full dispatch with timeout untested |
| `response/style_engine.py` — `sanitize_css` | ✅ | Good coverage |
| `response/document_renderer.py` | ✅ Partial | PDF tested; DOCX post-processing untested |
| `executors/pipeline_executors.py` | ✅ Partial | Happy paths; subprocess failures, hangs untested |
| `reporting_engine/*` | ✅ Partial | Builders tested; full pipeline integration untested |
| `causality/*` | ✅ Partial | Core agent tested; error paths untested |
| `data_layer/*` | ✅ | Good coverage |
| Concurrent execution | ❌ | No concurrency tests at all |
| State management edge cases | ❌ | No tests for checkpoint leakage, unbounded growth |

| `causality/*` (10 files) | ❌ | Entire causality subsystem has zero tests |
| `supervisor.py` — SupervisorAgent class | ❌ | Main orchestration class completely untested |
| `state.py` — SessionManager | ❌ | Session lifecycle untested |

**Critical untested paths**: LLM provider fallback, concurrent sessions, graph crash recovery, memory pressure, IntentRouter, causality module, SupervisorAgent orchestration.

## APPENDIX D: Dead Code Inventory

| Item | File:Line | Notes |
|------|-----------|-------|
| `KeywordRouter` class | `router.py:451-493` | Defined but never instantiated anywhere |
| `QUERY_INTENT_PROMPT` | `response/synthesizer.py:394-415` | Constant defined, never referenced |
| `ENTITY_EXTRACTION_PROMPT` | `response/synthesizer.py:417-438` | Constant defined, never referenced |
| `TABLE_FORMATTING_PROMPT` | `response/synthesizer.py:440-455` | Constant defined, never referenced |
| `ncbi_api_key` config field | `api_adapters/config.py:26` | Loaded from env but never used by any adapter |
| Logger instances in adapters | All adapter files | `logger = logging.getLogger(__name__)` created but never called in 9 adapter files |
