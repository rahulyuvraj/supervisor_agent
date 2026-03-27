"""Reporting engine — structured report generation from pipeline outputs.

Uses data_layer/ schemas and adapters for all data access.
"""

from .planner import ReportPlanner
from .builders import BUILDER_REGISTRY
from .enrichment import ReportEnricher
from .evidence import score_findings, detect_conflicts
from .validation import ValidationGuard, ValidationError
from .renderers.markdown_renderer import render_markdown

__all__ = [
    "ReportPlanner",
    "BUILDER_REGISTRY",
    "ReportEnricher",
    "score_findings",
    "detect_conflicts",
    "ValidationGuard",
    "ValidationError",
    "render_markdown",
]
