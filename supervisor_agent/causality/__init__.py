"""
Causality pipeline integration package.

Public API consumed by the supervisor executor::

    from supervisor_agent.causality import CausalityAdapter
"""
from __future__ import annotations

from .core_agent import CausalitySupervisorAgent
from .llm_bridge import CausalityLLMBridge
from .adapter import execute_causality
