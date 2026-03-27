"""Supervisor LangGraph — graph construction and compilation.

Topology:
    START → intent → plan → router ↔ executor (loop) → response → END
    Follow-up queries (existing results): intent → response → END
    General queries short-circuit: intent → END
    Structured report requests: intent → structured_report → END
"""

import functools
import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from .checkpointer import get_checkpointer
from .state import SupervisorGraphState
from .nodes import (
    intent_node, plan_node, execution_router, agent_executor,
    response_node, report_generation_node,
)
from .edges import check_general_query, route_next

logger = logging.getLogger(__name__)


def build_supervisor_graph():
    """Build and compile the supervisor StateGraph."""
    graph = StateGraph(SupervisorGraphState)

    graph.add_node("intent", intent_node)
    graph.add_node("plan", plan_node)
    graph.add_node("router", execution_router)
    graph.add_node("executor", agent_executor)
    graph.add_node("response", response_node)
    graph.add_node("structured_report", report_generation_node)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", check_general_query, {
        "general": END,
        "follow_up": "response",
        "structured_report": "structured_report",
        "needs_execution": "plan",
    })
    graph.add_edge("plan", "router")
    graph.add_conditional_edges("router", route_next, {
        "execute_agent": "executor",
        "structured_report": "structured_report",
        "report": "response",
        "done": END,
    })
    graph.add_edge("executor", "router")
    graph.add_edge("response", END)
    graph.add_edge("structured_report", END)

    compiled = graph.compile(checkpointer=get_checkpointer())
    logger.info("Supervisor LangGraph compiled")
    return compiled


@functools.lru_cache(maxsize=1)
def get_supervisor_graph():
    return build_supervisor_graph()
