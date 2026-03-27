"""Section builders — transform artifacts into populated SectionBlocks."""

from .base import SectionBuilder, BUILDER_REGISTRY
from .deg_section import DEGSectionBuilder
from .pathway_section import PathwaySectionBuilder
from .drug_section import DrugSectionBuilder
from .structural_sections import (
    ExecutiveSummaryBuilder,
    RunSummaryBuilder,
    DataOverviewBuilder,
    IntegratedFindingsBuilder,
    LimitationsBuilder,
    NextStepsBuilder,
    AppendixBuilder,
)

__all__ = [
    "SectionBuilder", "BUILDER_REGISTRY",
    "DEGSectionBuilder", "PathwaySectionBuilder", "DrugSectionBuilder",
    "ExecutiveSummaryBuilder", "RunSummaryBuilder", "DataOverviewBuilder",
    "IntegratedFindingsBuilder", "LimitationsBuilder", "NextStepsBuilder",
    "AppendixBuilder",
]
