"""Evidence scoring and conflict detection — two functions, one module.

score_findings: Rank and assign confidence to evidence cards.
detect_conflicts: Identify cross-module disagreements.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..data_layer.schemas.evidence import Confidence, ConflictRecord, EvidenceCard

logger = logging.getLogger(__name__)

# Confidence scoring rules — maps (module, metric_name) patterns to tier logic.
# These are structural, not domain-specific.
_HIGH_CONFIDENCE_MODULES = {"deg_analysis", "pathway_enrichment"}


def score_findings(cards: List[EvidenceCard]) -> List[EvidenceCard]:
    """Score, rank, and assign confidence tiers to evidence cards.

    Cards are ranked by confidence tier (HIGH > MEDIUM > LOW > FLAGGED),
    then by metric_value descending within each tier.
    """
    tier_order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2, Confidence.FLAGGED: 3}

    for card in cards:
        # Auto-promote if from a high-confidence module with a metric
        if (card.module in _HIGH_CONFIDENCE_MODULES
                and card.metric_value is not None
                and card.metric_value > 0
                and card.confidence == Confidence.MEDIUM):
            card.confidence = Confidence.HIGH

    # Sort: tier ascending, then metric_value descending
    cards.sort(key=lambda c: (
        tier_order.get(c.confidence, 2),
        -(c.metric_value or 0),
    ))

    # Assign ranks
    for i, card in enumerate(cards):
        card.rank = i + 1

    return cards


def detect_conflicts(cards: List[EvidenceCard]) -> List[ConflictRecord]:
    """Detect cross-module disagreements in evidence cards.

    Currently checks for directional conflicts: if one module reports upregulation
    and another reports downregulation for overlapping entities, that's flagged.

    Uses a dict-based tracker — no NetworkX dependency.
    """
    conflicts: List[ConflictRecord] = []

    # Group cards by section for within-section cross-module comparison
    by_section: Dict[str, List[EvidenceCard]] = {}
    for card in cards:
        by_section.setdefault(card.section, []).append(card)

    for section, section_cards in by_section.items():
        if len(section_cards) < 2:
            continue

        # Compare every pair across different modules
        for i, a in enumerate(section_cards):
            for b in section_cards[i + 1:]:
                if a.module == b.module:
                    continue
                conflict = _check_directional_conflict(a, b)
                if conflict:
                    conflicts.append(conflict)

    return conflicts


def _check_directional_conflict(a: EvidenceCard, b: EvidenceCard) -> ConflictRecord | None:
    """Check if two cards from different modules have contradictory directions."""
    # Both need metric values to compare direction
    if a.metric_value is None or b.metric_value is None:
        return None

    # Simple heuristic: if one is positive and the other negative, flag it
    if (a.metric_value > 0) != (b.metric_value > 0):
        return ConflictRecord(
            card_a=a.finding,
            card_b=b.finding,
            module_a=a.module,
            module_b=b.module,
            description=(
                f"Directional disagreement: {a.module} reports "
                f"{a.metric_name}={a.metric_value:.2f} while {b.module} "
                f"reports {b.metric_name}={b.metric_value:.2f}"
            ),
        )
    return None
