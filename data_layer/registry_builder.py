"""Build an ArtifactIndex by running adapters against a RunManifest."""

from __future__ import annotations

import logging
from typing import List

from .adapters.base import ADAPTER_REGISTRY
from .schemas.manifest import RunManifest
from .schemas.registry import ArtifactEntry, ArtifactIndex

logger = logging.getLogger(__name__)


def build_artifact_index(manifest: RunManifest) -> ArtifactIndex:
    """Run all registered adapters for completed modules, collect artifacts."""
    all_entries: List[ArtifactEntry] = []

    for module_run in manifest.completed_modules():
        adapter_cls = ADAPTER_REGISTRY.get(module_run.module_name)
        if not adapter_cls:
            logger.debug("No adapter registered for module '%s'", module_run.module_name)
            continue

        adapter = adapter_cls(module_run)
        if not adapter.is_available():
            logger.debug("Adapter for '%s' not available (dir missing?)", module_run.module_name)
            continue

        try:
            entries = adapter.discover()
            logger.info(
                "Adapter '%s' discovered %d artifacts", module_run.module_name, len(entries),
            )
            all_entries.extend(entries)
        except Exception:
            logger.exception("Adapter '%s' failed during discovery", module_run.module_name)

    return ArtifactIndex(artifacts=all_entries)
