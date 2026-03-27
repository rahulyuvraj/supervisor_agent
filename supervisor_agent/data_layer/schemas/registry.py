"""Artifact registry — index of discovered output files with structural metadata."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ArtifactEntry(BaseModel):
    """A single pipeline output file with structural metadata."""
    path: str
    module: str = Field(description="Agent type value that produced this artifact")
    label: str = Field(description="Structural label, e.g. 'deg_table', 'pathway_consolidation'")
    file_type: str = "csv"
    columns: List[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    qc_flags: List[str] = Field(
        default_factory=list,
        description="Quality warnings, e.g. 'missing_padj', 'zero_scores'",
    )


class ArtifactIndex(BaseModel):
    """Collection of all discovered artifacts from a pipeline run."""
    artifacts: List[ArtifactEntry] = Field(default_factory=list)
    _by_label: Dict[str, ArtifactEntry] = {}

    def model_post_init(self, __context) -> None:
        self._by_label = {a.label: a for a in self.artifacts}

    def get(self, label: str) -> Optional[ArtifactEntry]:
        return self._by_label.get(label)

    def for_module(self, module: str) -> List[ArtifactEntry]:
        return [a for a in self.artifacts if a.module == module]
