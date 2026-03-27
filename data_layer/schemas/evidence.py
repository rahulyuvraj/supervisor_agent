"""Evidence and narrative context schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    """Evidence confidence tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FLAGGED = "flagged"


class EvidenceCard(BaseModel):
    """Structured evidence item linking a finding to its source artifact."""
    finding: str = Field(description="One-sentence factual statement")
    module: str = Field(description="Originating module, e.g. 'deg_analysis'")
    artifact_label: str = Field(description="Label in ArtifactIndex that backs this finding")
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    confidence: Confidence = Confidence.MEDIUM
    section: str = Field(default="", description="Section this card is assigned to")
    rank: int = 0


class ConflictRecord(BaseModel):
    """Cross-module disagreement between two evidence cards."""
    card_a: str = Field(description="Finding text from first card")
    card_b: str = Field(description="Finding text from second card")
    module_a: str
    module_b: str
    description: str = Field(description="Human-readable conflict explanation")


class NarrativeContext(BaseModel):
    """Data boundary between deterministic extraction and optional LLM second-pass.

    The reporting engine populates this from evidence cards and section scaffolds.
    If LLM narrative augmentation is enabled, this object is the ONLY input the
    LLM receives — never raw file paths or artifacts.
    """
    disease_name: str = ""
    section_title: str = ""
    evidence_cards: List[EvidenceCard] = Field(default_factory=list)
    table_summaries: Dict[str, str] = Field(
        default_factory=dict,
        description="table_label → pipe-delimited summary string",
    )
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Section-specific context (counts, thresholds, etc.)",
    )
