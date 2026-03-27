"""
tool_wiring.py — Wire supervisor workflow_state into TOOL_REGISTRY slots.

``populate()`` reads output directories from the supervisor's workflow_state
and creates thin closure wrappers that satisfy the
``fn(artifact_store, audits, intent, output_dir) -> dict`` signature expected
by the execution loop in core_agent.py.

Ordering contract (enforced by adapter.py):
    1. check_preconditions() validates that required state keys exist.
    2. tool_wiring.populate() reads those keys to fill TOOL_REGISTRY.
    3. CausalitySupervisorAgent.run() dispatches via TOOL_REGISTRY.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("causal_platform")

# Mapping: tool_id -> workflow_state key that supplies the directory / file.
_TOOL_STATE_MAP: Dict[str, str] = {
    "T_00":  "cohort_output_dir",
    "T_01":  "harmonized_counts_file",
    "T_02":  "deg_base_dir",
    "T_03":  "pathway_consolidation_path",
    "T_04":  "deconvolution_output_dir",
    "T_04b": "single_cell_output_dir",
    "T_05":  "temporal_output_dir",
    "T_06":  "perturbation_output_dir",
    "T_07":  "prior_network_dir",
    "T_08":  "gwas_output_dir",
}


def populate(
    workflow_state: Dict[str, Any],
    get_available_inputs: Optional[Callable[[], Dict[str, Any]]] = None,
) -> int:
    """Fill ``TOOL_REGISTRY`` slots from *workflow_state* directories.

    Each populated slot is a closure that returns a dict with
    ``{"tool_id": ..., "source_dir": ..., "status": "pass-through"}``.

    Parameters
    ----------
    workflow_state : dict
        The supervisor's shared state bag.
    get_available_inputs : callable, optional
        Additional source of run-time input directories.

    Returns
    -------
    int
        Number of slots successfully wired.
    """
    from .tool_registry import TOOL_REGISTRY

    extra = get_available_inputs() if get_available_inputs else {}
    wired = 0

    for tool_id, state_key in _TOOL_STATE_MAP.items():
        path_str = workflow_state.get(state_key) or extra.get(state_key)
        if not path_str:
            log.debug("tool_wiring: no key '%s' for %s — slot stays None", state_key, tool_id)
            continue
        resolved = Path(str(path_str))
        if not resolved.exists():
            log.debug("tool_wiring: path %s missing for %s — slot stays None", resolved, tool_id)
            continue

        # Create a closure that captures tool_id and resolved
        def _make_wrapper(_tid: str, _path: Path):
            def _wrapper(artifact_store: dict, audits: list, intent, output_dir: str) -> dict:
                return {
                    "tool_id": _tid,
                    "source_dir": str(_path),
                    "output_dir": output_dir,
                    "status": "pass-through",
                }
            return _wrapper

        TOOL_REGISTRY[tool_id] = _make_wrapper(tool_id, resolved)
        wired += 1
        log.debug("tool_wiring: wired %s -> %s", tool_id, resolved)

    log.info("tool_wiring: %d / %d slots populated", wired, len(_TOOL_STATE_MAP))
    return wired
