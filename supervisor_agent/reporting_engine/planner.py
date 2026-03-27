"""Report planner — decides which sections to include based on available artifacts."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..data_layer.schemas.manifest import RunManifest, ModuleStatus
from ..data_layer.schemas.registry import ArtifactIndex
from ..data_layer.schemas.sections import ReportingConfig, SectionBlock, SectionMeta

logger = logging.getLogger(__name__)

# Ordered list of (section_id, title, required_module_or_None, heading_level).
# "required_module" of None means always-included structural sections.
_SECTION_PLAN = [
    ("executive_summary", "Executive Summary", None, 2),
    ("run_summary", "Run Summary", None, 2),
    ("data_overview", "Data Overview", None, 2),
    ("deg_findings", "Differential Expression Analysis", "deg_analysis", 2),
    ("pathway_findings", "Pathway Enrichment Analysis", "pathway_enrichment", 2),
    ("drug_findings", "Drug Discovery & Perturbation Analysis", "perturbation_analysis", 2),
    ("integrated_findings", "Integrated Findings", None, 2),
    ("limitations", "Limitations & Caveats", None, 2),
    ("next_steps", "Recommended Next Steps", None, 2),
    ("appendix", "Appendix", None, 2),
]


class ReportPlanner:
    """Given a manifest + artifact index + config, produce an ordered section plan."""

    def __init__(
        self,
        manifest: RunManifest,
        index: ArtifactIndex,
        config: Optional[ReportingConfig] = None,
    ):
        self.manifest = manifest
        self.index = index
        self.config = config or ReportingConfig()

    def plan(self) -> List[SectionBlock]:
        """Return ordered SectionBlocks — populated skeletons ready for builders."""
        completed = {m.module_name for m in self.manifest.completed_modules()}
        sections: List[SectionBlock] = []

        for order, (sid, title, req_module, level) in enumerate(_SECTION_PLAN):
            # Check if section is disabled in config
            if self.config.sections_enabled and not self.config.sections_enabled.get(sid, True):
                continue

            # Check appendix toggle
            if sid == "appendix" and not self.config.include_appendix:
                continue

            meta = SectionMeta(module=req_module or "")

            # Module-specific section: skip if module didn't run
            if req_module and req_module not in completed:
                module_run = self.manifest.get_module(req_module)
                if module_run and module_run.status == ModuleStatus.FAILED:
                    meta.omitted_reason = f"Module '{req_module}' failed during execution"
                elif module_run and module_run.status == ModuleStatus.SKIPPED:
                    meta.omitted_reason = f"Module '{req_module}' was skipped"
                else:
                    meta.omitted_reason = f"Module '{req_module}' was not part of this run"

                # Include a stub section noting the omission
                sections.append(SectionBlock(
                    id=sid,
                    title=title,
                    level=level,
                    order=order,
                    body=f"*{meta.omitted_reason}.*",
                    meta=meta,
                ))
                continue

            # Attach artifact labels for this module
            if req_module:
                meta.artifact_labels = [
                    a.label for a in self.index.for_module(req_module)
                ]

            sections.append(SectionBlock(
                id=sid, title=title, level=level, order=order, meta=meta,
            ))

        return sections
