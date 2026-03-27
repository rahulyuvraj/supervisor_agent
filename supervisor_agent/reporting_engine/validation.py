"""Validation guard — hard gate before rendering.

Checks schema compliance, reference integrity, and data bounds.
Single module, <150 LOC.
"""

from __future__ import annotations

import logging
from typing import List

from ..data_layer.schemas.evidence import EvidenceCard
from ..data_layer.schemas.registry import ArtifactIndex
from ..data_layer.schemas.sections import SectionBlock, TableBlock

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails with hard errors."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


class ValidationGuard:
    """Validates a report before rendering — fail-fast on hard errors, warn on soft issues."""

    def __init__(self, index: ArtifactIndex):
        self.index = index
        self.warnings: List[str] = []

    def validate(
        self, sections: List[SectionBlock], cards: List[EvidenceCard],
    ) -> List[str]:
        """Run all validation checks. Returns list of warnings. Raises on hard errors."""
        errors: List[str] = []
        self.warnings = []

        for section in sections:
            self._check_section(section, errors)
        self._check_evidence(cards, errors)
        self._check_unique_ids(sections, errors)

        if errors:
            raise ValidationError(errors)

        return self.warnings

    def _check_section(self, section: SectionBlock, errors: List[str]) -> None:
        if not section.id:
            errors.append("Section missing 'id' field")
        if not section.title:
            errors.append(f"Section '{section.id}' missing title")
        for table in section.tables:
            self._check_table(section.id, table, errors)

    def _check_table(self, section_id: str, table: TableBlock, errors: List[str]) -> None:
        if table.headers and table.rows:
            expected_cols = len(table.headers)
            for i, row in enumerate(table.rows):
                if len(row) != expected_cols:
                    self.warnings.append(
                        f"Section '{section_id}': table row {i} has {len(row)} cols, "
                        f"expected {expected_cols}"
                    )
        # Reference integrity: source artifact should exist in index
        if table.source_artifact and not self.index.get(table.source_artifact):
            self.warnings.append(
                f"Section '{section_id}': table references artifact "
                f"'{table.source_artifact}' not found in index"
            )

    def _check_evidence(self, cards: List[EvidenceCard], errors: List[str]) -> None:
        for card in cards:
            if not card.finding:
                errors.append("Evidence card has empty 'finding'")
            if card.artifact_label and not self.index.get(card.artifact_label):
                self.warnings.append(
                    f"Evidence card references artifact '{card.artifact_label}' "
                    f"not in index"
                )

    def _check_unique_ids(self, sections: List[SectionBlock], errors: List[str]) -> None:
        seen: set[str] = set()
        for s in sections:
            if s.id in seen:
                errors.append(f"Duplicate section ID: '{s.id}'")
            seen.add(s.id)
