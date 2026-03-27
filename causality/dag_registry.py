"""
dag_registry.py — DAG lifecycle helpers (discover, load, save).

Provides lightweight utilities for the causality pipeline to locate, persist
and reload causal DAG artifacts (JSON edge-lists) produced by M12/M13.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("causal_platform")

_DAG_PATTERN = "*dag*.json"


def discover(output_dir: str) -> List[str]:
    """Return sorted list of DAG JSON files under *output_dir*."""
    p = Path(output_dir)
    if not p.is_dir():
        log.debug("dag_registry.discover: %s is not a directory", output_dir)
        return []
    matches = sorted(str(f) for f in p.glob(_DAG_PATTERN) if f.is_file())
    log.info("dag_registry: discovered %d DAG file(s) in %s", len(matches), output_dir)
    return matches


def load(dag_path: str) -> Dict[str, Any]:
    """Load a DAG from a JSON file.

    Returns a dict with at least ``{"edges": [...], "nodes": [...]}``.
    Missing keys default to empty lists.
    """
    p = Path(dag_path)
    if not p.is_file():
        raise FileNotFoundError(f"DAG file not found: {dag_path}")
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        data = {"edges": data if isinstance(data, list) else [], "nodes": []}
    data.setdefault("edges", [])
    data.setdefault("nodes", [])
    log.info("dag_registry: loaded DAG from %s (%d edges, %d nodes)",
             dag_path, len(data["edges"]), len(data["nodes"]))
    return data


def save_checkpoint(
    dag: Dict[str, Any],
    output_dir: str,
    label: str = "checkpoint",
) -> str:
    """Persist *dag* as a timestamped JSON file and return the path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"dag_{label}_{ts}.json"
    dest = p / fname
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(dag, fh, indent=2, default=str)
    log.info("dag_registry: saved DAG checkpoint -> %s", dest)
    return str(dest)
