"""Tests for the Markdown renderer."""

from __future__ import annotations

import pytest

from ..data_layer.schemas.evidence import ConflictRecord, EvidenceCard
from ..data_layer.schemas.sections import SectionBlock, TableBlock
from ..reporting_engine.renderers.markdown_renderer import (
    _table_to_markdown,
    render_markdown,
)


# ─── _table_to_markdown ───


class TestTableToMarkdown:
    def test_basic_table(self):
        tbl = TableBlock(headers=["Gene", "FC"], rows=[["TP53", "2.5"], ["BRCA1", "-1.3"]])
        md = _table_to_markdown(tbl)
        assert "| Gene | FC |" in md
        assert "| --- | --- |" in md
        assert "| TP53 | 2.5 |" in md

    def test_empty_headers(self):
        tbl = TableBlock(headers=[], rows=[])
        assert _table_to_markdown(tbl) == ""

    def test_caption(self):
        tbl = TableBlock(headers=["A"], rows=[["1"]], caption="My Table")
        md = _table_to_markdown(tbl)
        assert "**My Table**" in md

    def test_truncated(self):
        tbl = TableBlock(
            headers=["A"], rows=[["1"], ["2"]],
            truncated=True, total_rows=100,
        )
        md = _table_to_markdown(tbl)
        assert "Showing 2 of 100 rows" in md

    def test_not_truncated(self):
        tbl = TableBlock(headers=["A"], rows=[["1"]])
        md = _table_to_markdown(tbl)
        assert "Showing" not in md


# ─── render_markdown (integration via Jinja2 or fallback) ───


class TestRenderMarkdown:
    def test_contains_title(self, sample_sections):
        md = render_markdown(sample_sections, title="Test Report")
        assert "# Test Report" in md

    def test_contains_disease(self, sample_sections):
        md = render_markdown(sample_sections, disease="Lupus")
        assert "Lupus" in md

    def test_sections_rendered(self, sample_sections):
        md = render_markdown(sample_sections)
        for section in sample_sections:
            assert section.title in md

    def test_section_body_rendered(self):
        sections = [SectionBlock(
            id="test", title="Test Section",
            body="Some detailed analysis text.",
        )]
        md = render_markdown(sections)
        assert "Some detailed analysis text." in md

    def test_section_table_rendered(self):
        sections = [SectionBlock(
            id="t", title="T",
            tables=[TableBlock(headers=["X"], rows=[["val"]])],
        )]
        md = render_markdown(sections)
        assert "| X |" in md
        assert "| val |" in md

    def test_conflicts_rendered(self, sample_sections):
        conflicts = [ConflictRecord(
            card_a="up", card_b="down",
            module_a="mod1", module_b="mod2",
            description="Directional disagreement",
        )]
        md = render_markdown(sample_sections, conflicts=conflicts)
        assert "Conflicts" in md
        assert "mod1" in md
        assert "mod2" in md

    def test_no_conflicts_section_when_empty(self, sample_sections):
        md = render_markdown(sample_sections, conflicts=[])
        assert "Conflicts" not in md

    def test_evidence_cards_optional(self, sample_sections):
        md = render_markdown(sample_sections, evidence_cards=None)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_empty_sections(self):
        md = render_markdown([], title="Empty Report")
        assert "# Empty Report" in md

    def test_section_ordering(self):
        """Sections should appear in the order they are passed."""
        sections = [
            SectionBlock(id="a", title="First", order=1),
            SectionBlock(id="b", title="Second", order=2),
        ]
        md = render_markdown(sections)
        idx_first = md.index("First")
        idx_second = md.index("Second")
        assert idx_first < idx_second


class TestFallbackRenderer:
    """Test fallback rendering works when Jinja2 template is unavailable."""

    def test_fallback_produces_output(self, sample_sections, monkeypatch):
        from ..reporting_engine.renderers import markdown_renderer as mod

        # Force Jinja2 rendering to fail so fallback is used
        monkeypatch.setattr(
            mod, "_render_with_jinja",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no template")),
        )
        md = render_markdown(sample_sections, title="Fallback Test")
        assert "# Fallback Test" in md
        for section in sample_sections:
            assert section.title in md

    def test_fallback_disease_context(self, monkeypatch):
        from ..reporting_engine.renderers import markdown_renderer as mod

        monkeypatch.setattr(
            mod, "_render_with_jinja",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        md = render_markdown([], disease="SLE", title="R")
        assert "SLE" in md
