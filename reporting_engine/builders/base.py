"""Base section builder interface and registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Type

import pandas as pd

from ...data_layer.schemas.manifest import RunManifest
from ...data_layer.schemas.registry import ArtifactEntry, ArtifactIndex
from ...data_layer.schemas.sections import ReportingConfig, SectionBlock, TableBlock
from ...data_layer.schemas.evidence import EvidenceCard

logger = logging.getLogger(__name__)

BUILDER_REGISTRY: Dict[str, Type["SectionBuilder"]] = {}


class SectionBuilder(ABC):
    """Base class for report section builders.

    Each subclass handles one section_id and knows how to read the relevant
    artifacts and produce markdown body + tables + evidence cards.
    """

    section_id: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.section_id:
            BUILDER_REGISTRY[cls.section_id] = cls

    def __init__(
        self,
        manifest: RunManifest,
        index: ArtifactIndex,
        config: ReportingConfig,
    ):
        self.manifest = manifest
        self.index = index
        self.config = config

    @abstractmethod
    def build(self, section: SectionBlock) -> SectionBlock:
        """Populate a SectionBlock skeleton with body text, tables, etc."""

    def collect_evidence(self, section: SectionBlock) -> List[EvidenceCard]:
        """Override in subclasses to produce evidence cards for this section."""
        return []

    # ── Shared helpers ──

    def _read_artifact(self, label: str) -> Optional[pd.DataFrame]:
        """Load a CSV artifact by label. Returns None on any failure."""
        entry = self.index.get(label)
        if not entry:
            return None
        try:
            return pd.read_csv(entry.path, encoding="utf-8", encoding_errors="replace")
        except Exception as exc:
            logger.debug("Failed to read artifact '%s': %s", label, exc)
            return None

    def _df_to_table(
        self, df: pd.DataFrame, caption: str = "", label: str = "",
        max_rows: Optional[int] = None, columns: Optional[List[str]] = None,
    ) -> TableBlock:
        """Convert a DataFrame to a capped TableBlock."""
        cap = max_rows or self.config.table_row_cap
        total = len(df)
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available] if available else df
        if len(df.columns) > 8:
            df = df.iloc[:, :8]
        truncated = total > cap
        df = df.head(cap)
        return TableBlock(
            caption=caption,
            headers=list(df.columns),
            rows=[[str(v) for v in row] for row in df.values],
            source_artifact=label,
            truncated=truncated,
            total_rows=total,
        )

    def _get_qc_flags(self, label: str) -> List[str]:
        entry = self.index.get(label)
        return entry.qc_flags if entry else []
