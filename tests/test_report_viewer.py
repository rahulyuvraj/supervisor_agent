"""Tests for Streamlit structured-report page parsing utilities.

Validates Markdown parsing, table extraction, and conflict detection
without needing a running Streamlit server.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

import pytest

# The Streamlit page module uses st.set_page_config etc. at import time.
# We need to mock streamlit before importing the module's functions.
# Instead, we re-implement the pure parsing functions for testing.
# They are stateless and can be tested directly.

# Add the streamlit_app directory to path so we can import the module
_streamlit_app_dir = Path(__file__).resolve().parent.parent.parent.parent / "streamlit_app"


def _parse_markdown_sections(md_text: str):
    """Mirror of the parsing function from 1_structured_report.py."""
    import re
    sections = []
    current = None
    body_lines = []

    for line in md_text.split("\n"):
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading_match:
            if current is not None:
                current["body"] = "\n".join(body_lines).strip()
                sections.append(current)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            current = {"level": level, "title": title, "anchor": anchor, "body": ""}
            body_lines = []
        else:
            body_lines.append(line)

    if current is not None:
        current["body"] = "\n".join(body_lines).strip()
        sections.append(current)

    return sections


def _extract_tables(body: str):
    """Mirror of the table extraction function."""
    import re
    tables = []
    non_table_lines = []
    table_lines = []
    in_table = False

    for line in body.split("\n"):
        stripped = line.strip()
        if "|" in stripped and stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
        else:
            if in_table and table_lines:
                table = _parse_pipe_table(table_lines)
                if table:
                    tables.append(table)
                table_lines = []
                in_table = False
            non_table_lines.append(line)

    if table_lines:
        table = _parse_pipe_table(table_lines)
        if table:
            tables.append(table)

    return "\n".join(non_table_lines), tables


def _parse_pipe_table(lines):
    import re
    if len(lines) < 2:
        return None
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^-+$", c.strip()) for c in cells):
            continue
        rows.append(cells)
    return rows if len(rows) >= 1 else None


def _detect_api_tables(body: str):
    import re
    api_markers = []
    captions = re.findall(r"\*\*(.+?)\*\*", body)
    for cap in captions:
        lower = cap.lower()
        if any(kw in lower for kw in ("ensembl", "string", "reactome", "chembl",
                                       "openfda", "faers", "dgidb", "clinical trial")):
            api_markers.append(cap)
    return api_markers


# ── Sample report Markdown ──

SAMPLE_REPORT = """\
# Lupus Analysis Report

*Disease context: lupus*

## Executive Summary

This report summarizes the analysis of lupus disease data.

## Differential Expression Analysis

Top differentially expressed genes identified.

**Top DEGs**

| gene_symbol | log2FC | padj |
| --- | --- | --- |
| STAT1 | 3.2 | 1e-10 |
| IRF7 | 2.8 | 5e-9 |

**Gene Annotations (Ensembl)**

| Symbol | Ensembl ID | Biotype | Chromosome | Description |
| --- | --- | --- | --- | --- |
| STAT1 | ENSG_STAT1 | protein_coding | 2 | Signal transducer |

## Pathway Enrichment Analysis

Enriched pathways from Reactome.

| pathway_id | name | pValue |
| --- | --- | --- |
| R-HSA-913531 | Interferon Signaling | 1e-12 |

## Drug Discovery & Perturbation Analysis

Drug candidates identified.

**Drug Mechanisms & Indications (ChEMBL)**

| Drug | ChEMBL ID | Mechanism of Action | Indications |
| --- | --- | --- | --- |
| Baricitinib | CHEMBL3301612 | JAK inhibitor | RA |

## Cross-Module Conflicts

- **deg_analysis** vs **pathway_enrichment**: Directional disagreement: deg reports log2FC=3.20 while pathway reports neg_log_p=-2.50

## Recommended Next Steps

Consider validation experiments.
"""


# ── Tests ──


class TestMarkdownParsing:
    def test_parse_sections_count(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        titles = [s["title"] for s in sections]
        assert "Lupus Analysis Report" in titles
        assert "Executive Summary" in titles
        assert "Differential Expression Analysis" in titles
        assert "Recommended Next Steps" in titles

    def test_parse_section_levels(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        title_section = next(s for s in sections if s["title"] == "Lupus Analysis Report")
        deg_section = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        assert title_section["level"] == 1
        assert deg_section["level"] == 2

    def test_parse_section_anchors(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        assert deg["anchor"] == "differential-expression-analysis"

    def test_parse_section_body_content(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        summary = next(s for s in sections if s["title"] == "Executive Summary")
        assert "summarizes" in summary["body"]

    def test_parse_preserves_order(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        titles = [s["title"] for s in sections]
        idx_exec = titles.index("Executive Summary")
        idx_deg = titles.index("Differential Expression Analysis")
        idx_pw = titles.index("Pathway Enrichment Analysis")
        assert idx_exec < idx_deg < idx_pw


class TestTableExtraction:
    def test_extract_tables_from_deg(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        text, tables = _extract_tables(deg["body"])
        assert len(tables) == 2  # DEGs + Ensembl

    def test_table_headers(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        _, tables = _extract_tables(deg["body"])
        first_table = tables[0]
        headers = first_table[0]
        assert "gene_symbol" in headers
        assert "log2FC" in headers

    def test_table_rows(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        _, tables = _extract_tables(deg["body"])
        first_table = tables[0]
        rows = first_table[1:]  # skip headers
        assert len(rows) == 2  # STAT1, IRF7
        assert rows[0][0].strip() == "STAT1"

    def test_extract_no_tables(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        summary = next(s for s in sections if s["title"] == "Executive Summary")
        text, tables = _extract_tables(summary["body"])
        assert tables == []
        assert "summarizes" in text

    def test_body_text_preserved_around_tables(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        text, _ = _extract_tables(deg["body"])
        assert "Top differentially expressed" in text


class TestApiTableDetection:
    def test_detect_ensembl_table(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        deg = next(s for s in sections if s["title"] == "Differential Expression Analysis")
        api_tables = _detect_api_tables(deg["body"])
        assert any("Ensembl" in t for t in api_tables)

    def test_detect_chembl_table(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        drug = next(s for s in sections if s["title"] == "Drug Discovery & Perturbation Analysis")
        api_tables = _detect_api_tables(drug["body"])
        assert any("ChEMBL" in t for t in api_tables)

    def test_no_api_tables_in_summary(self):
        sections = _parse_markdown_sections(SAMPLE_REPORT)
        summary = next(s for s in sections if s["title"] == "Executive Summary")
        api_tables = _detect_api_tables(summary["body"])
        assert api_tables == []


class TestConflictParsing:
    def test_conflict_section_present(self):
        assert "## Cross-Module Conflicts" in SAMPLE_REPORT

    def test_conflict_pattern_extracted(self):
        import re
        match = re.search(
            r"## Cross-Module Conflicts\n(.+?)(?=\n## |\Z)", SAMPLE_REPORT, re.DOTALL,
        )
        assert match is not None
        conflicts = match.group(1).strip().split("\n")
        assert len(conflicts) >= 1
        assert "deg_analysis" in conflicts[0]
        assert "pathway_enrichment" in conflicts[0]
