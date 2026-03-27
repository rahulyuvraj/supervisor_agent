"""Build a RunManifest from SupervisorGraphState.

This is the bridge between the LangGraph state dict (execution layer)
and the data_layer's typed schema world.  Zero domain entities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .schemas.manifest import ModuleRun, ModuleStatus, RunManifest

logger = logging.getLogger(__name__)

# Mapping from workflow_outputs keys → module names.
# Pattern: "<module>_output_dir" → module name.
# This is structural — it encodes the *key naming convention*, not domain data.
_OUTPUT_DIR_SUFFIX = "_output_dir"
_BASE_DIR_SUFFIX = "_base_dir"


def manifest_from_state(state: Dict[str, Any]) -> RunManifest:
    """Derive a RunManifest from a SupervisorGraphState-shaped dict.

    Reads `session_id`, `analysis_id`, `disease_name`, `output_root`,
    `agent_results`, `errors`, and `workflow_outputs`.
    """
    wo = state.get("workflow_outputs", {}) or {}

    manifest = RunManifest(
        session_id=state.get("session_id", ""),
        analysis_id=state.get("analysis_id", ""),
        disease_name=state.get("disease_name", ""),
        output_root=state.get("output_root", ""),
    )

    # Index successful results by agent name
    completed = {
        r["agent"]: r for r in (state.get("agent_results") or [])
    }
    failed = {
        e["agent"]: e for e in (state.get("errors") or [])
    }

    # Discover modules from workflow_outputs dir keys
    seen_modules: set[str] = set()
    for key, val in wo.items():
        if not isinstance(val, str):
            continue
        module_name = ""
        if key.endswith(_OUTPUT_DIR_SUFFIX):
            module_name = key[: -len(_OUTPUT_DIR_SUFFIX)]
        elif key.endswith(_BASE_DIR_SUFFIX):
            module_name = key[: -len(_BASE_DIR_SUFFIX)]
        if not module_name or module_name in seen_modules:
            continue
        seen_modules.add(module_name)

        result = completed.get(module_name, {})
        error = failed.get(module_name, {})
        status = (
            ModuleStatus.COMPLETED if module_name in completed
            else ModuleStatus.FAILED if module_name in failed
            else ModuleStatus.COMPLETED  # dir exists → assume completed
        )

        # Collect all workflow_output keys that belong to this module
        output_files = {}
        prefix = module_name + "_"
        for k, v in wo.items():
            if isinstance(v, str) and k.startswith(prefix):
                output_files[k] = v

        manifest.modules.append(ModuleRun(
            module_name=module_name,
            status=status,
            output_dir=val,
            duration_s=result.get("duration_s"),
            error_message=error.get("error"),
            output_files=output_files,
        ))

    return manifest
