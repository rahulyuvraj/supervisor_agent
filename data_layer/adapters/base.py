"""Base adapter interface and registry for module output readers."""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

import pandas as pd

from ..schemas.manifest import ModuleRun, ModuleStatus
from ..schemas.registry import ArtifactEntry

logger = logging.getLogger(__name__)

# Global registry populated via __init_subclass__
ADAPTER_REGISTRY: Dict[str, Type["BaseModuleAdapter"]] = {}


class BaseModuleAdapter(ABC):
    """Read a module's output directory and produce ArtifactEntry objects.

    Subclasses declare `module_name` and implement `discover()`.
    Each adapter stays under 200 LOC and encodes *structural* knowledge
    (column names, file patterns) — never domain entities.
    """

    module_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.module_name:
            ADAPTER_REGISTRY[cls.module_name] = cls

    def __init__(self, module_run: ModuleRun):
        self.run = module_run
        self.output_dir = Path(module_run.output_dir) if module_run.output_dir else None

    # ── Public API ──

    @abstractmethod
    def discover(self) -> List[ArtifactEntry]:
        """Scan the module output dir and return all meaningful artifacts."""

    def is_available(self) -> bool:
        return (
            self.run.status == ModuleStatus.COMPLETED
            and self.output_dir is not None
            and self.output_dir.is_dir()
        )

    # ── Helpers for subclasses ──

    def _find_csvs(self, pattern: str = "**/*.csv") -> List[Path]:
        if not self.output_dir:
            return []
        return sorted(self.output_dir.glob(pattern))

    def _read_csv_meta(self, path: Path) -> tuple[List[str], int]:
        """Return (columns, row_count) for a CSV without loading it fully."""
        columns: List[str] = []
        row_count = 0
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    columns = header
                    row_count = sum(1 for _ in reader)
        except Exception as exc:
            logger.debug("Failed to read CSV meta for %s: %s", path, exc)
        return columns, row_count

    def _make_entry(
        self,
        path: Path,
        label: str,
        columns: Optional[List[str]] = None,
        row_count: Optional[int] = None,
        qc_flags: Optional[List[str]] = None,
    ) -> ArtifactEntry:
        cols, rows = (columns, row_count) if columns is not None else self._read_csv_meta(path)
        return ArtifactEntry(
            path=str(path),
            module=self.module_name,
            label=label,
            file_type=path.suffix.lstrip("."),
            columns=cols or [],
            row_count=rows if rows is not None else 0,
            qc_flags=qc_flags or [],
        )

    def _score_match(self, path: Path, patterns: Dict[str, str]) -> Optional[str]:
        """Match a file path against label→regex patterns. Returns the label or None."""
        import re
        name = path.name.lower()
        for label, regex in patterns.items():
            if re.search(regex, name, re.IGNORECASE):
                return label
        return None
