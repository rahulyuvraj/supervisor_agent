"""Section and report configuration schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TableBlock(BaseModel):
    """A capped, column-selected data table embedded in a section."""
    caption: str = ""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    source_artifact: str = Field(default="", description="Artifact label for traceability")
    truncated: bool = False
    total_rows: int = 0


class SectionMeta(BaseModel):
    """Metadata about a report section."""
    module: str = Field(default="", description="Module that sourced this section")
    artifact_labels: List[str] = Field(default_factory=list)
    omitted_reason: str = Field(default="", description="Why section was skipped")


class SectionBlock(BaseModel):
    """One section of a generated report."""
    id: str = Field(description="Stable anchor ID, e.g. 'deg_findings'")
    title: str
    level: int = Field(default=2, description="Heading level (2=##, 3=###)")
    order: int = 0
    body: str = Field(default="", description="Markdown body text")
    tables: List[TableBlock] = Field(default_factory=list)
    meta: SectionMeta = Field(default_factory=SectionMeta)


class NarrativeMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_AUGMENTED = "llm_augmented"


class ReportingConfig(BaseModel):
    """User-controllable knobs for report generation."""
    narrative_mode: NarrativeMode = NarrativeMode.DETERMINISTIC
    table_row_cap: int = Field(default=25, ge=5, le=200)
    include_appendix: bool = True
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    sections_enabled: Dict[str, bool] = Field(
        default_factory=dict,
        description="section_id → enabled. Empty = all enabled.",
    )
    theme: str = "default"
