"""
file_mapper.py — Discover files from workflow_state for causality file inspection.

``map_files()`` walks the output directories recorded in workflow_state and
returns de-duplicated file paths that the ``FileInspector`` will classify.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("causal_platform")

# Extensions that FileInspector can classify.
_SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".parquet", ".h5ad", ".json"}

# workflow_state keys whose values point to directories / files.
_DIR_KEYS = [
    "cohort_output_dir",
    "harmonized_counts_file",
    "deg_base_dir",
    "pathway_consolidation_path",
    "deconvolution_output_dir",
    "single_cell_output_dir",
    "temporal_output_dir",
    "perturbation_output_dir",
    "prior_network_dir",
    "gwas_output_dir",
]


def map_files(workflow_state: Dict[str, Any]) -> List[str]:
    """Return sorted, de-duplicated file paths from *workflow_state* directories.

    Only files whose suffix is in ``_SUPPORTED_EXTS`` are returned.
    """
    seen: set[str] = set()
    result: list[str] = []

    for key in _DIR_KEYS:
        raw = workflow_state.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_file():
            _maybe_add(p, seen, result)
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file():
                    _maybe_add(child, seen, result)
        else:
            log.debug("file_mapper: %s -> %s does not exist", key, p)

    log.info("file_mapper: discovered %d files from workflow_state", len(result))
    return sorted(result)


def _maybe_add(path: Path, seen: set, result: list) -> None:
    canonical = str(path.resolve())
    if canonical in seen:
        return
    if path.suffix.lower() not in _SUPPORTED_EXTS:
        return
    seen.add(canonical)
    result.append(str(path))
