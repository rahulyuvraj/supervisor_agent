"""Markdown renderer — converts SectionBlocks to a complete Markdown report.

Uses Jinja2 for template rendering with stable headings and anchors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...data_layer.schemas.evidence import ConflictRecord, EvidenceCard
from ...data_layer.schemas.sections import SectionBlock, TableBlock

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def _table_to_markdown(table: TableBlock) -> str:
    """Render a TableBlock as a Markdown pipe-delimited table."""
    if not table.headers:
        return ""
    lines = []
    if table.caption:
        lines.append(f"**{table.caption}**\n")
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    if table.truncated:
        lines.append(f"\n*Showing {len(table.rows)} of {table.total_rows} rows.*")
    return "\n".join(lines)


def render_markdown(
    sections: List[SectionBlock],
    evidence_cards: Optional[List[EvidenceCard]] = None,
    conflicts: Optional[List[ConflictRecord]] = None,
    title: str = "Analysis Report",
    disease: str = "",
) -> str:
    """Render a complete Markdown report from SectionBlocks.

    Tries Jinja2 template first; falls back to programmatic rendering.
    """
    try:
        return _render_with_jinja(sections, evidence_cards, conflicts, title, disease)
    except Exception as exc:
        logger.debug("Jinja2 template rendering failed (%s), using fallback", exc)
        return _render_fallback(sections, evidence_cards, conflicts, title, disease)


def _render_with_jinja(
    sections: List[SectionBlock],
    evidence_cards: Optional[List[EvidenceCard]],
    conflicts: Optional[List[ConflictRecord]],
    title: str,
    disease: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["table_md"] = _table_to_markdown
    template = env.get_template("report.md.j2")
    return template.render(
        title=title,
        disease=disease,
        sections=sections,
        evidence_cards=evidence_cards or [],
        conflicts=conflicts or [],
    )


def _render_fallback(
    sections: List[SectionBlock],
    evidence_cards: Optional[List[EvidenceCard]],
    conflicts: Optional[List[ConflictRecord]],
    title: str,
    disease: str,
) -> str:
    """Programmatic Markdown rendering — no template dependency."""
    lines: list[str] = [f"# {title}\n"]
    if disease:
        lines.append(f"*Disease context: {disease}*\n")

    for section in sorted(sections, key=lambda s: s.order):
        heading = "#" * section.level
        lines.append(f"\n{heading} {section.title}\n")
        if section.body:
            lines.append(section.body)
        for table in section.tables:
            lines.append("")
            lines.append(_table_to_markdown(table))

    if conflicts:
        lines.append("\n## Cross-Module Conflicts\n")
        for c in conflicts:
            lines.append(f"- **{c.module_a}** vs **{c.module_b}**: {c.description}")

    return "\n".join(lines) + "\n"
