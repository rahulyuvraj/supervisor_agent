"""Run manifest — describes which modules ran and their output locations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModuleStatus(str, Enum):
    """Outcome of a pipeline module execution."""
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"


class ModuleRun(BaseModel):
    """Record of a single module's execution within a pipeline run."""
    module_name: str = Field(description="Agent type value, e.g. 'deg_analysis'")
    status: ModuleStatus = ModuleStatus.NOT_RUN
    output_dir: Optional[str] = None
    duration_s: Optional[float] = None
    error_message: Optional[str] = None
    output_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Logical label → absolute file path",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution parameters captured from routing decision",
    )


class RunManifest(BaseModel):
    """Snapshot of a complete pipeline run — inputs, modules, and context."""
    session_id: str
    analysis_id: str = ""
    disease_name: str = ""
    output_root: str = ""
    modules: List[ModuleRun] = Field(default_factory=list)

    def completed_modules(self) -> List[ModuleRun]:
        return [m for m in self.modules if m.status == ModuleStatus.COMPLETED]

    def get_module(self, name: str) -> Optional[ModuleRun]:
        return next((m for m in self.modules if m.module_name == name), None)
