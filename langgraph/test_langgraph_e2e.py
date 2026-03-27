"""
LangGraph Supervisor — End-to-End Verification (7 test cases).

Tests every graph path before permanent enablement of USE_LANGGRAPH_SUPERVISOR.

Usage:
    cd /path/to/agenticaib
    USE_LANGGRAPH_SUPERVISOR=true python -m agentic_ai_wf.supervisor_agent.langgraph.test_langgraph_e2e

Requirements:
    - OPENAI_API_KEY in env (for IntentRouter + reporting_node LLM calls)
    - Existing DEG output CSVs in supervisor_agent/outputs/deg_analysis/ (auto-discovered)
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env if dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_PROJECT_ROOT / "agentic_ai_wf" / "supervisor_agent" / ".env")
except ImportError:
    pass

from agentic_ai_wf.supervisor_agent.executors.base import StatusType, StatusUpdate

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_langgraph_e2e")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Root of supervisor_agent on disk
_SUPERVISOR_ROOT = Path(__file__).resolve().parents[1]
_UPLOADS_DIR = _PROJECT_ROOT / "streamlit_uploads"


def _find_deg_output_dir() -> Optional[str]:
    """Auto-discover a DEG output dir containing *_DEGs.csv."""
    deg_root = _SUPERVISOR_ROOT / "outputs" / "deg_analysis"
    if not deg_root.is_dir():
        return None
    for csv in deg_root.glob("**/*_DEGs.csv"):
        parent = str(csv.parent)
        return parent
    return None


def _find_prioritized_genes_csv() -> Optional[str]:
    """Find a prioritized-genes CSV in streamlit_uploads/ or gene_prioritization outputs."""
    # Check streamlit_uploads first (static, always available)
    for p in sorted(_UPLOADS_DIR.glob("*prioritized*.csv")):
        return str(p)
    # Fallback: check gene_prioritization output dirs
    gp_root = _SUPERVISOR_ROOT / "outputs" / "gene_prioritization"
    if gp_root.is_dir():
        for csv in gp_root.glob("**/*prioritized*.csv"):
            return str(csv)
    return None


def _find_pathway_csv() -> Optional[str]:
    """Find a Pathways_Consolidated CSV in streamlit_uploads/ or pathway outputs."""
    for p in sorted(_UPLOADS_DIR.glob("*Pathways_Consolidated*.csv")):
        return str(p)
    pw_root = _SUPERVISOR_ROOT / "outputs" / "pathway_enrichment"
    if pw_root.is_dir():
        for csv in pw_root.glob("**/*Pathways_Consolidated*.csv"):
            return str(csv)
    return None


def _build_seeded_workflow_outputs() -> Dict[str, Any]:
    """Build comprehensive workflow_outputs to simulate completed pipeline.

    Seeds keys for DEG, gene_prioritization, and pathway so that
    check_general_query can see *all* common agent outputs as already
    present — routing follow-up queries to the reporting path.
    """
    deg_dir = _find_deg_output_dir()
    outputs: Dict[str, Any] = {}
    if deg_dir:
        outputs["deg_base_dir"] = deg_dir
        outputs["deg_output_dir"] = deg_dir

    # Gene prioritization output key
    gp_csv = _find_prioritized_genes_csv()
    if gp_csv:
        outputs["prioritized_genes_path"] = gp_csv

    # Pathway enrichment output key
    pw_csv = _find_pathway_csv()
    if pw_csv:
        outputs["pathway_consolidation_path"] = pw_csv

    if not outputs:
        # Last-resort fallback
        outputs["cohort_summary_text"] = (
            "Summary: 150 samples retrieved for lupus. 80 disease vs 70 control. "
            "Top DEGs include IFIH1, MX1, IFI44, OAS1, ISG15."
        )
    return outputs


async def _collect_stream(
    stream,
) -> Tuple[List[StatusUpdate], Optional[StatusUpdate]]:
    """Drain a run_supervisor_stream() generator; return (all_updates, terminal)."""
    updates: List[StatusUpdate] = []
    terminal: Optional[StatusUpdate] = None
    async for u in stream:
        updates.append(u)
        if u.status_type in (StatusType.COMPLETED, StatusType.ERROR):
            terminal = u
    return updates, terminal


# ---------------------------------------------------------------------------
# Test results bookkeeping
# ---------------------------------------------------------------------------
_results: List[Dict[str, Any]] = []


def _record(name: str, passed: bool, detail: str = ""):
    symbol = "✅ PASS" if passed else "❌ FAIL"
    _results.append({"name": name, "passed": passed, "detail": detail})
    logger.info(f"{symbol}: {name}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — General query (no prior results)
# Path: intent → general → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_1_general_query():
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    updates, terminal = await _collect_stream(
        run_supervisor_stream(
            user_query="What can you do?",
            session_id="e2e-test-1",
        )
    )

    try:
        assert terminal is not None, "No terminal StatusUpdate received"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )
        assert terminal.message, "Terminal message is empty"

        gs = getattr(terminal, "_graph_state", None)
        assert gs is not None, "_graph_state missing on terminal update"
        assert not gs.get("agent_results"), (
            f"Expected no agent_results, got {gs.get('agent_results')}"
        )
        _record("TEST 1 — General query", True, f"message={terminal.message[:80]}…")
    except AssertionError as e:
        _record("TEST 1 — General query", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Pipeline execution (mocked DEG executor)
# Path: intent → plan → router → executor → router → report → reporting → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_2_pipeline_mocked():
    from agentic_ai_wf.supervisor_agent.agent_registry import AgentType
    from agentic_ai_wf.supervisor_agent.langgraph import nodes as nodes_mod
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    # Build fake executor that simulates DEG completion
    async def _fake_deg_executor(agent_info, inputs, conv_state):
        yield StatusUpdate(StatusType.EXECUTING, "DEG analysis running…", "Step 1/3", agent_name="deg_analysis")
        # Simulate the output key that a real DEG executor would set
        fake_dir = str(_SUPERVISOR_ROOT / "outputs" / "deg_analysis" / "fake-e2e-test")
        conv_state.workflow_state["deg_output_dir"] = fake_dir
        conv_state.workflow_state["deg_base_dir"] = fake_dir
        yield StatusUpdate(StatusType.COMPLETED, "DEG analysis complete", "Done", agent_name="deg_analysis")

    # Patch EXECUTOR_MAP (dict swap — restored after test)
    original = nodes_mod.EXECUTOR_MAP.get(AgentType.DEG_ANALYSIS)
    nodes_mod.EXECUTOR_MAP[AgentType.DEG_ANALYSIS] = _fake_deg_executor

    # Uploaded files
    counts = str(_UPLOADS_DIR / "LP-20250628_counts.csv")
    meta = str(_UPLOADS_DIR / "LP-20250628_metadata.csv")
    uploaded = {}
    if Path(counts).exists():
        uploaded["LP-20250628_counts.csv"] = counts
    if Path(meta).exists():
        uploaded["LP-20250628_metadata.csv"] = meta

    try:
        updates, terminal = await _collect_stream(
            run_supervisor_stream(
                user_query="Run DEG analysis for lupus",
                session_id="e2e-test-2",
                disease_name="lupus",
                uploaded_files=uploaded or None,
            )
        )

        assert terminal is not None, "No terminal StatusUpdate"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )

        # Should have RUNNING + COMPLETED updates for deg_analysis
        deg_updates = [u for u in updates if getattr(u, "agent_name", None) == "deg_analysis"]
        assert len(deg_updates) >= 2, f"Expected ≥2 deg_analysis updates, got {len(deg_updates)}"

        gs = getattr(terminal, "_graph_state", None)
        assert gs is not None, "_graph_state missing"
        assert gs.get("agent_results"), "agent_results is empty"
        assert len(gs["agent_results"]) >= 1, "Expected at least 1 agent result"
        assert gs["agent_results"][0].get("agent") == "deg_analysis", (
            f"Expected 'deg_analysis', got {gs['agent_results'][0].get('agent')}"
        )

        _record("TEST 2 — Pipeline (mocked)", True,
                f"agent_results={len(gs['agent_results'])}, updates={len(updates)}")

    except AssertionError as e:
        _record("TEST 2 — Pipeline (mocked)", False, str(e))
    finally:
        # Restore
        if original is not None:
            nodes_mod.EXECUTOR_MAP[AgentType.DEG_ANALYSIS] = original


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Follow-up "top 10 genes" with existing results
# Path: intent → follow_up → reporting → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_3_followup_top10():
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    wo = _build_seeded_workflow_outputs()
    updates, terminal = await _collect_stream(
        run_supervisor_stream(
            user_query="Can you tell me my top 10 genes?",
            session_id="e2e-test-3",
            disease_name="lupus",
            workflow_outputs=wo,
        )
    )

    try:
        assert terminal is not None, "No terminal StatusUpdate"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )
        assert terminal.message, "Terminal message is empty"

        gs = getattr(terminal, "_graph_state", None)
        assert gs is not None, "_graph_state missing"
        # Workflow outputs should carry through the seeded keys
        gs_wo = gs.get("workflow_outputs", {})
        assert gs_wo, "workflow_outputs empty in _graph_state"

        _record("TEST 3 — Follow-up top 10", True,
                f"msg_len={len(terminal.message)}, msg_preview={terminal.message[:80]}…")
    except AssertionError as e:
        _record("TEST 3 — Follow-up top 10", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Multi-agent chain (mocked DEG + gene prioritization)
# Path: intent → plan → router → executor(DEG) → router → executor(GP) → router → report → reporting → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_4_multi_agent_mocked():
    from agentic_ai_wf.supervisor_agent.agent_registry import AgentType
    from agentic_ai_wf.supervisor_agent.langgraph import nodes as nodes_mod
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    # Fake DEG executor
    async def _fake_deg(agent_info, inputs, conv_state):
        yield StatusUpdate(StatusType.EXECUTING, "DEG running", "", agent_name="deg_analysis")
        conv_state.workflow_state["deg_output_dir"] = "/tmp/fake-deg-e2e"
        conv_state.workflow_state["deg_base_dir"] = "/tmp/fake-deg-e2e"
        yield StatusUpdate(StatusType.COMPLETED, "DEG done", "", agent_name="deg_analysis")

    # Fake gene-prioritization executor
    async def _fake_gp(agent_info, inputs, conv_state):
        yield StatusUpdate(StatusType.EXECUTING, "GP running", "", agent_name="gene_prioritization")
        conv_state.workflow_state["prioritized_genes_path"] = "/tmp/fake-gp/genes.csv"
        yield StatusUpdate(StatusType.COMPLETED, "GP done", "", agent_name="gene_prioritization")

    orig_deg = nodes_mod.EXECUTOR_MAP.get(AgentType.DEG_ANALYSIS)
    orig_gp = nodes_mod.EXECUTOR_MAP.get(AgentType.GENE_PRIORITIZATION)
    nodes_mod.EXECUTOR_MAP[AgentType.DEG_ANALYSIS] = _fake_deg
    nodes_mod.EXECUTOR_MAP[AgentType.GENE_PRIORITIZATION] = _fake_gp

    counts = str(_UPLOADS_DIR / "LP-20250628_counts.csv")
    meta = str(_UPLOADS_DIR / "LP-20250628_metadata.csv")
    uploaded = {}
    if Path(counts).exists():
        uploaded["LP-20250628_counts.csv"] = counts
    if Path(meta).exists():
        uploaded["LP-20250628_metadata.csv"] = meta

    try:
        updates, terminal = await _collect_stream(
            run_supervisor_stream(
                user_query="Run DEG and gene prioritization for lupus",
                session_id="e2e-test-4",
                disease_name="lupus",
                uploaded_files=uploaded or None,
            )
        )

        assert terminal is not None, "No terminal StatusUpdate"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )

        gs = getattr(terminal, "_graph_state", None)
        assert gs is not None, "_graph_state missing"
        results = gs.get("agent_results", [])
        assert len(results) >= 2, f"Expected ≥2 agent_results, got {len(results)}"
        agent_names = [r["agent"] for r in results]
        assert "deg_analysis" in agent_names, f"DEG missing from results: {agent_names}"
        assert "gene_prioritization" in agent_names, f"GP missing from results: {agent_names}"

        # Verify ordering — DEG should come before GP
        deg_idx = agent_names.index("deg_analysis")
        gp_idx = agent_names.index("gene_prioritization")
        assert deg_idx < gp_idx, f"DEG should precede GP: DEG@{deg_idx}, GP@{gp_idx}"

        _record("TEST 4 — Multi-agent (mocked)", True,
                f"agents={agent_names}, updates={len(updates)}")

    except AssertionError as e:
        _record("TEST 4 — Multi-agent (mocked)", False, str(e))
    finally:
        if orig_deg is not None:
            nodes_mod.EXECUTOR_MAP[AgentType.DEG_ANALYSIS] = orig_deg
        if orig_gp is not None:
            nodes_mod.EXECUTOR_MAP[AgentType.GENE_PRIORITIZATION] = orig_gp


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — PDF request (document generation)
# Path: intent → follow_up → response → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_5_pdf_request():
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    wo = _build_seeded_workflow_outputs()
    updates, terminal = await _collect_stream(
        run_supervisor_stream(
            user_query="I need a PDF report",
            session_id="e2e-test-5",
            disease_name="lupus",
            workflow_outputs=wo,
        )
    )

    try:
        assert terminal is not None, "No terminal StatusUpdate"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )
        assert terminal.message, "Terminal message is empty"
        assert "pdf" in terminal.message.lower(), (
            f"Expected PDF generation message, got: {terminal.message[:120]}"
        )

        _record("TEST 5 — PDF request", True, f"msg={terminal.message[:80]}…")
    except AssertionError as e:
        _record("TEST 5 — PDF request", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Dynamic top-N ("top 20 genes")
# Path: intent → follow_up → reporting → END
# ═══════════════════════════════════════════════════════════════════════════
async def test_6_top20():
    from agentic_ai_wf.supervisor_agent.langgraph.entry import run_supervisor_stream

    wo = _build_seeded_workflow_outputs()
    updates, terminal = await _collect_stream(
        run_supervisor_stream(
            user_query="Show me top 20 genes",
            session_id="e2e-test-6",
            disease_name="lupus",
            workflow_outputs=wo,
        )
    )

    try:
        assert terminal is not None, "No terminal StatusUpdate"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )
        assert terminal.message, "Terminal message is empty"

        gs = getattr(terminal, "_graph_state", None)
        assert gs is not None, "_graph_state missing"

        _record("TEST 6 — Top 20 genes", True,
                f"msg_len={len(terminal.message)}, msg_preview={terminal.message[:80]}…")
    except AssertionError as e:
        _record("TEST 6 — Top 20 genes", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Bridge method (_process_message_langgraph) with ConversationState sync-back
# ═══════════════════════════════════════════════════════════════════════════
async def test_7_bridge_syncback():
    from agentic_ai_wf.supervisor_agent.state import ConversationState
    from agentic_ai_wf.supervisor_agent.supervisor import SupervisorAgent

    supervisor = SupervisorAgent()
    conv_state = ConversationState(session_id="e2e-test-7")
    conv_state.current_disease = "lupus"

    # Pre-seed workflow_state with same data as Tests 3-6
    seeded = _build_seeded_workflow_outputs()
    conv_state.workflow_state.update(seeded)

    # Capture the seeded keys to verify they survive sync-back
    seeded_keys = set(seeded.keys())

    try:
        updates: List[StatusUpdate] = []
        terminal: Optional[StatusUpdate] = None

        async for update in supervisor._process_message_langgraph(
            user_message="What are my top genes?",
            session_id="e2e-test-7",
            uploaded_files=None,
            state=conv_state,
        ):
            updates.append(update)
            if update.status_type in (StatusType.COMPLETED, StatusType.ERROR):
                terminal = update

        assert terminal is not None, "No terminal StatusUpdate from bridge"
        assert terminal.status_type == StatusType.COMPLETED, (
            f"Expected COMPLETED, got {terminal.status_type}"
        )

        # Verify ConversationState was synced back
        for key in seeded_keys:
            assert key in conv_state.workflow_state, (
                f"Seeded key '{key}' lost from workflow_state after sync-back"
            )

        # Disease should be populated
        assert conv_state.current_disease, "current_disease is empty after sync-back"

        # An assistant message should have been added
        assistant_msgs = [
            m for m in conv_state.messages if m.role.value == "assistant"
        ]
        assert len(assistant_msgs) >= 1, (
            f"Expected ≥1 assistant message, got {len(assistant_msgs)}"
        )

        _record("TEST 7 — Bridge sync-back", True,
                f"workflow_state_keys={list(conv_state.workflow_state.keys())}, "
                f"disease={conv_state.current_disease}, msgs={len(conv_state.messages)}")

    except AssertionError as e:
        _record("TEST 7 — Bridge sync-back", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    # Pre-flight checks
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set — aborting")
        sys.exit(1)

    deg_dir = _find_deg_output_dir()
    logger.info(f"DEG output dir for Tests 3/5/6: {deg_dir or '(fallback to cohort_summary_text)'}")

    # Reset the graph singleton to ensure a clean MemorySaver
    import agentic_ai_wf.supervisor_agent.langgraph.graph as graph_mod
    graph_mod._graph = None

    # Log the seeded data that Tests 3/5/6/7 will use
    seeded = _build_seeded_workflow_outputs()
    logger.info(f"Seeded workflow_outputs keys: {list(seeded.keys())}")
    for k, v in seeded.items():
        logger.info(f"  {k} = {str(v)[:120]}")

    tests = [
        ("TEST 1", test_1_general_query),
        ("TEST 2", test_2_pipeline_mocked),
        ("TEST 3", test_3_followup_top10),
        ("TEST 4", test_4_multi_agent_mocked),
        ("TEST 5", test_5_pdf_request),
        ("TEST 6", test_6_top20),
        ("TEST 7", test_7_bridge_syncback),
    ]

    PER_TEST_TIMEOUT = 90  # seconds

    t0 = time.time()
    for label, fn in tests:
        logger.info(f"\n{'═' * 60}")
        logger.info(f"Running {label}…")
        logger.info(f"{'═' * 60}")
        try:
            await asyncio.wait_for(fn(), timeout=PER_TEST_TIMEOUT)
        except asyncio.TimeoutError:
            _record(label, False, f"TIMEOUT — exceeded {PER_TEST_TIMEOUT}s (likely hit a real executor)")
        except Exception as exc:
            _record(label, False, f"UNHANDLED EXCEPTION: {exc}\n{traceback.format_exc()}")
    elapsed = time.time() - t0

    # Summary
    passed = sum(1 for r in _results if r["passed"])
    total = len(_results)
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed}/{total} passed  ({elapsed:.1f}s)")
    print(f"{'═' * 60}")
    for r in _results:
        symbol = "✅" if r["passed"] else "❌"
        print(f"  {symbol} {r['name']}")
        if r["detail"]:
            print(f"     {r['detail'][:200]}")
    print(f"{'═' * 60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
