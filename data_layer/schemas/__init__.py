"""Pydantic v2 data contracts for pipeline output ingestion."""

from .manifest import RunManifest, ModuleRun, ModuleStatus
from .registry import ArtifactEntry, ArtifactIndex
from .evidence import EvidenceCard, ConflictRecord, NarrativeContext, Confidence
from .sections import SectionBlock, SectionMeta, TableBlock, ReportingConfig

__all__ = [
    "RunManifest", "ModuleRun", "ModuleStatus",
    "ArtifactEntry", "ArtifactIndex",
    "EvidenceCard", "ConflictRecord", "NarrativeContext", "Confidence",
    "SectionBlock", "SectionMeta", "TableBlock", "ReportingConfig",
]
