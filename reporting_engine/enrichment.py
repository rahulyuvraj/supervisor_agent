"""Report enrichment — augments locally-built sections with live API data.

Sits between the Build and Validation steps in the reporting pipeline.
Calls Track B API adapters to inject supplementary context into sections:
  - DEG section: gene annotations (Ensembl), PPI network (STRING)
  - Pathway section: hierarchy + drugs (Reactome)
  - Drug section: mechanisms (ChEMBL), adverse events (OpenFDA),
    gene-drug interactions (DGIdb), active trials (ClinicalTrials.gov)

All enrichment is **optional** — API failures degrade gracefully to
the local-only section content.  Zero imports from outside supervisor_agent/.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..api_adapters.base import API_ADAPTER_REGISTRY, BaseAPIAdapter
from ..api_adapters.config import APIConfig
from ..data_layer.schemas.evidence import Confidence, EvidenceCard
from ..data_layer.schemas.sections import SectionBlock, TableBlock

logger = logging.getLogger(__name__)

# Limits to keep enrichment fast and bounded
_MAX_GENES = 20
_MAX_DRUGS = 10
_MAX_PATHWAYS = 5
_MAX_TRIALS = 10


class ReportEnricher:
    """Async enricher that fans out API calls to augment report sections.

    Usage::

        enricher = ReportEnricher(api_config)
        sections, extra_cards = await enricher.enrich(
            sections, evidence_cards, disease="lupus"
        )
    """

    def __init__(self, config: Optional[APIConfig] = None) -> None:
        self.config = config or APIConfig.from_env()

    async def enrich(
        self,
        sections: List[SectionBlock],
        evidence_cards: List[EvidenceCard],
        disease: str = "",
    ) -> Tuple[List[SectionBlock], List[EvidenceCard]]:
        """Enrich sections and return (updated_sections, new_evidence_cards).

        Failures in any enrichment path are logged and silently swallowed —
        the original section content is preserved.
        """
        extra_cards: List[EvidenceCard] = []
        section_map = {s.id: s for s in sections}

        # Extract entities from existing sections
        genes = self._extract_genes(section_map, evidence_cards)
        drugs = self._extract_drugs(section_map, evidence_cards)
        pathway_ids = self._extract_pathway_ids(section_map, evidence_cards)

        if not any([genes, drugs, pathway_ids]):
            logger.debug("No enrichment entities found — skipping API calls")
            return sections, extra_cards

        # Fan out enrichment calls
        tasks: Dict[str, asyncio.Task] = {}

        async with self._adapters() as adapters:
            if genes and "ensembl" in adapters:
                tasks["gene_annotations"] = asyncio.create_task(
                    self._enrich_genes(adapters["ensembl"], genes)
                )
            if genes and "string" in adapters:
                tasks["ppi_network"] = asyncio.create_task(
                    self._enrich_ppi(adapters["string"], genes)
                )
            if pathway_ids and "reactome" in adapters:
                tasks["pathway_detail"] = asyncio.create_task(
                    self._enrich_pathways(adapters["reactome"], pathway_ids)
                )
            if drugs and "chembl" in adapters:
                tasks["chembl_mechanisms"] = asyncio.create_task(
                    self._enrich_drug_mechanisms(adapters["chembl"], drugs)
                )
            if drugs and "openfda" in adapters:
                tasks["fda_events"] = asyncio.create_task(
                    self._enrich_adverse_events(adapters["openfda"], drugs)
                )
            if genes and "dgidb" in adapters:
                tasks["dgidb_interactions"] = asyncio.create_task(
                    self._enrich_gene_drug(adapters["dgidb"], genes)
                )
            if disease and "clinical_trials" in adapters:
                trial_query = self._build_trial_query(disease, drugs)
                tasks["clinical_trials"] = asyncio.create_task(
                    self._enrich_trials(adapters["clinical_trials"], trial_query)
                )

            # Gather results — each returns (section_id, body_addendum, tables, cards)
            done, pending = await asyncio.wait(
                list(tasks.values()), timeout=30.0, return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            results = [t.result() if not t.cancelled() and t.exception() is None else t.exception() or Exception("cancelled")
                       for t in tasks.values()]

        enrichments: Dict[str, Any] = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("Enrichment '%s' failed: %s", key, result)
                continue
            enrichments[key] = result

        # Inject enrichments into sections
        self._apply_gene_annotations(section_map, enrichments)
        self._apply_ppi_network(section_map, enrichments)
        self._apply_pathway_detail(section_map, enrichments)
        self._apply_drug_mechanisms(section_map, enrichments)
        self._apply_adverse_events(section_map, enrichments)
        self._apply_gene_drug(section_map, enrichments)
        trial_cards = self._apply_trials(section_map, enrichments)
        extra_cards.extend(trial_cards)

        return sections, extra_cards

    # ── Entity extraction ──

    def _extract_genes(
        self,
        section_map: Dict[str, SectionBlock],
        cards: List[EvidenceCard],
    ) -> List[str]:
        """Pull gene names from DEG section tables and evidence cards."""
        genes: List[str] = []
        deg = section_map.get("deg_findings")
        if deg:
            for table in deg.tables:
                gene_col = self._find_column(table, ("gene", "gene_symbol", "symbol", "gene_name"))
                if gene_col is not None:
                    genes.extend(row[gene_col] for row in table.rows if row[gene_col])

        # Also from evidence card findings that look like gene symbols
        for card in cards:
            if card.module == "deg_analysis" and card.finding:
                # Extract words that look like gene symbols (all caps, 2-10 chars)
                for word in card.finding.split():
                    cleaned = word.strip(",.;:()")
                    if (
                        cleaned.isupper()
                        and 2 <= len(cleaned) <= 10
                        and cleaned[0:1].isalpha()
                        and cleaned.isalnum()
                    ):
                        genes.append(cleaned)

        # Deduplicate preserving order, cap at limit
        seen: set = set()
        unique: List[str] = []
        for g in genes:
            upper = g.upper()
            if upper not in seen:
                seen.add(upper)
                unique.append(g)
        return unique[:_MAX_GENES]

    def _extract_drugs(
        self,
        section_map: Dict[str, SectionBlock],
        cards: List[EvidenceCard],
    ) -> List[str]:
        """Pull drug names from drug section tables and evidence cards."""
        drugs: List[str] = []
        drug_sec = section_map.get("drug_findings")
        if drug_sec:
            for table in drug_sec.tables:
                drug_col = self._find_column(table, ("drug", "drug_name", "compound", "molecule_name"))
                if drug_col is not None:
                    drugs.extend(row[drug_col] for row in table.rows if row[drug_col])

        for card in cards:
            if card.module == "perturbation_analysis" and card.finding:
                for word in card.finding.split():
                    cleaned = word.strip(",.;:()")
                    if len(cleaned) > 3 and cleaned[0].isupper():
                        drugs.append(cleaned)

        seen: set = set()
        unique: List[str] = []
        for d in drugs:
            lower = d.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(d)
        return unique[:_MAX_DRUGS]

    def _extract_pathway_ids(
        self,
        section_map: Dict[str, SectionBlock],
        cards: List[EvidenceCard],
    ) -> List[str]:
        """Pull Reactome/KEGG pathway IDs from pathway section tables."""
        ids: List[str] = []
        pw_sec = section_map.get("pathway_findings")
        if pw_sec:
            for table in pw_sec.tables:
                id_col = self._find_column(table, ("pathway_id", "stId", "id", "pathway"))
                if id_col is not None:
                    for row in table.rows:
                        val = row[id_col].strip()
                        if val.startswith("R-HSA-") or val.startswith("hsa"):
                            ids.append(val)
        seen: set = set()
        unique: List[str] = []
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique[:_MAX_PATHWAYS]

    # ── API enrichment coroutines ──

    async def _enrich_genes(
        self, adapter: BaseAPIAdapter, genes: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch Ensembl annotations for gene symbols."""
        results = []
        for gene in genes:
            try:
                info = await adapter.lookup_symbol(gene)
                results.append({
                    "symbol": gene,
                    "ensembl_id": info.get("id", ""),
                    "biotype": info.get("biotype", ""),
                    "description": info.get("description", ""),
                    "chromosome": info.get("seq_region_name", ""),
                    "start": info.get("start", ""),
                    "end": info.get("end", ""),
                })
            except Exception as exc:
                logger.debug("Ensembl lookup failed for %s: %s", gene, exc)
        return results

    async def _enrich_ppi(
        self, adapter: BaseAPIAdapter, genes: List[str],
    ) -> List[Dict[str, str]]:
        """Fetch STRING PPI network for top genes."""
        try:
            return await adapter.get_network(genes)
        except Exception as exc:
            logger.debug("STRING PPI enrichment failed: %s", exc)
            return []

    async def _enrich_pathways(
        self, adapter: BaseAPIAdapter, pathway_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch Reactome pathway details and associated drugs."""
        results = []
        for pid in pathway_ids:
            try:
                info = await adapter.get_pathway(pid)
                drugs = await adapter.get_pathway_drugs(pid)
                results.append({
                    "pathway_id": pid,
                    "name": info.get("displayName", pid),
                    "species": info.get("speciesName", ""),
                    "drug_count": len(drugs),
                    "drugs": [d.get("displayName", "") for d in drugs[:5]],
                })
            except Exception as exc:
                logger.debug("Reactome enrichment failed for %s: %s", pid, exc)
        return results

    async def _enrich_drug_mechanisms(
        self, adapter: BaseAPIAdapter, drugs: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch ChEMBL mechanisms of action for drugs."""
        results = []
        for drug in drugs:
            try:
                molecules = await adapter.search_molecule(drug, limit=1)
                if not molecules:
                    continue
                chembl_id = molecules[0].get("molecule_chembl_id", "")
                if not chembl_id:
                    continue
                mechanisms = await adapter.get_mechanisms(chembl_id)
                indications = await adapter.get_indications(chembl_id)
                results.append({
                    "drug": drug,
                    "chembl_id": chembl_id,
                    "mechanisms": [
                        m.get("mechanism_of_action", "") for m in mechanisms[:3]
                    ],
                    "indications": [
                        i.get("mesh_heading", "") for i in indications[:5]
                    ],
                })
            except Exception as exc:
                logger.debug("ChEMBL enrichment failed for %s: %s", drug, exc)
        return results

    async def _enrich_adverse_events(
        self, adapter: BaseAPIAdapter, drugs: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch OpenFDA FAERS top adverse events per drug."""
        results = []
        for drug in drugs:
            try:
                resp = await adapter.count_adverse_events(
                    f'patient.drug.medicinalproduct:"{drug}"', limit=5,
                )
                events = resp.get("results", [])
                results.append({
                    "drug": drug,
                    "top_events": [
                        {"term": e.get("term", ""), "count": e.get("count", 0)}
                        for e in events
                    ],
                })
            except Exception as exc:
                logger.debug("OpenFDA enrichment failed for %s: %s", drug, exc)
        return results

    async def _enrich_gene_drug(
        self, adapter: BaseAPIAdapter, genes: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch DGIdb gene-drug interactions."""
        try:
            return await adapter.get_gene_interactions(genes)
        except Exception as exc:
            logger.debug("DGIdb enrichment failed: %s", exc)
            return []

    async def _enrich_trials(
        self, adapter: BaseAPIAdapter, query: str,
    ) -> List[Dict[str, Any]]:
        """Fetch active clinical trials."""
        try:
            studies = await adapter.search_all(
                query,
                max_results=_MAX_TRIALS,
                filter_overall_status=["RECRUITING", "NOT_YET_RECRUITING"],
                fields=["NCTId", "BriefTitle", "OverallStatus",
                         "StartDate", "LeadSponsorName", "EnrollmentCount"],
            )
            return studies
        except Exception as exc:
            logger.debug("ClinicalTrials enrichment failed: %s", exc)
            return []

    # ── Enrichment injection ──

    def _apply_gene_annotations(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("gene_annotations")
        if not data:
            return
        deg = sections.get("deg_findings")
        if not deg:
            return
        table = TableBlock(
            caption="Gene Annotations (Ensembl)",
            headers=["Symbol", "Ensembl ID", "Biotype", "Chromosome", "Description"],
            rows=[
                [g["symbol"], g["ensembl_id"], g["biotype"],
                 str(g["chromosome"]), g.get("description", "")[:80]]
                for g in data
            ],
            source_artifact="api:ensembl",
        )
        deg.tables.append(table)

    def _apply_ppi_network(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("ppi_network")
        if not data:
            return
        target = sections.get("integrated_findings") or sections.get("deg_findings")
        if not target:
            return
        rows = []
        for edge in data[:15]:
            rows.append([
                edge.get("preferredName_A", edge.get("stringId_A", "")),
                edge.get("preferredName_B", edge.get("stringId_B", "")),
                edge.get("score", ""),
            ])
        if rows:
            table = TableBlock(
                caption="Protein-Protein Interactions (STRING)",
                headers=["Protein A", "Protein B", "Score"],
                rows=rows,
                source_artifact="api:string",
            )
            target.tables.append(table)
            if target.body:
                target.body += "\n\n"
            target.body += (
                f"STRING PPI analysis identified **{len(data)}** interactions "
                f"among the top differentially expressed genes."
            )

    def _apply_pathway_detail(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("pathway_detail")
        if not data:
            return
        pw = sections.get("pathway_findings")
        if not pw:
            return
        drug_rows = [
            [p["name"], str(p["drug_count"]), ", ".join(p["drugs"])]
            for p in data if p["drug_count"] > 0
        ]
        if drug_rows:
            table = TableBlock(
                caption="Pathway-Associated Drugs (Reactome)",
                headers=["Pathway", "Drug Count", "Drugs"],
                rows=drug_rows,
                source_artifact="api:reactome",
            )
            pw.tables.append(table)

    def _apply_drug_mechanisms(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("chembl_mechanisms")
        if not data:
            return
        drug_sec = sections.get("drug_findings")
        if not drug_sec:
            return
        rows = [
            [d["drug"], d["chembl_id"],
             "; ".join(d["mechanisms"]) or "—",
             "; ".join(d["indications"][:3]) or "—"]
            for d in data
        ]
        if rows:
            table = TableBlock(
                caption="Drug Mechanisms & Indications (ChEMBL)",
                headers=["Drug", "ChEMBL ID", "Mechanism of Action", "Indications"],
                rows=rows,
                source_artifact="api:chembl",
            )
            drug_sec.tables.append(table)

    def _apply_adverse_events(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("fda_events")
        if not data:
            return
        drug_sec = sections.get("drug_findings") or sections.get("limitations")
        if not drug_sec:
            return
        rows = []
        for entry in data:
            if entry["top_events"]:
                top = entry["top_events"][0]
                rows.append([
                    entry["drug"],
                    top["term"],
                    str(top["count"]),
                    str(len(entry["top_events"])),
                ])
        if rows:
            table = TableBlock(
                caption="Top Adverse Events (OpenFDA FAERS)",
                headers=["Drug", "Top Event", "Count", "Total Event Types"],
                rows=rows,
                source_artifact="api:openfda",
            )
            drug_sec.tables.append(table)

    def _apply_gene_drug(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> None:
        data = enrichments.get("dgidb_interactions")
        if not data or not isinstance(data, list):
            return
        target = sections.get("drug_findings") or sections.get("integrated_findings")
        if not target:
            return
        rows = []
        for node in data:
            gene_name = node.get("name", "")
            interactions = node.get("interactions", [])
            for ix in interactions[:3]:
                drug = ix.get("drug", {})
                rows.append([
                    gene_name,
                    drug.get("name", ""),
                    ix.get("interactionTypes", [{}])[0].get("type", "")
                    if ix.get("interactionTypes") else "",
                    str(ix.get("interactionScore") or "—"),
                ])
        if rows:
            table = TableBlock(
                caption="Gene-Drug Interactions (DGIdb)",
                headers=["Gene", "Drug", "Interaction Type", "Score"],
                rows=rows[:15],
                source_artifact="api:dgidb",
            )
            target.tables.append(table)

    def _apply_trials(
        self, sections: Dict[str, SectionBlock], enrichments: Dict[str, Any],
    ) -> List[EvidenceCard]:
        """Inject clinical trial data into the drug or next_steps section."""
        data = enrichments.get("clinical_trials")
        cards: List[EvidenceCard] = []
        if not data:
            return cards
        target = sections.get("next_steps") or sections.get("drug_findings")
        if not target:
            return cards
        rows = []
        for study in data:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            sponsor = proto.get("sponsorCollaboratorsModule", {})
            lead = sponsor.get("leadSponsor", {})
            rows.append([
                ident.get("nctId", ""),
                (ident.get("briefTitle", "") or "")[:80],
                status_mod.get("overallStatus", ""),
                str(design.get("enrollmentInfo", {}).get("count", "—")),
                lead.get("name", "")[:40],
            ])
        if rows:
            table = TableBlock(
                caption=f"Active Clinical Trials ({len(rows)} recruiting)",
                headers=["NCT ID", "Title", "Status", "Enrollment", "Sponsor"],
                rows=rows,
                source_artifact="api:clinical_trials",
            )
            target.tables.append(table)

            # Evidence card for active trials
            cards.append(EvidenceCard(
                finding=f"{len(rows)} actively recruiting clinical trials found",
                module="clinical_trials_enrichment",
                artifact_label="api:clinical_trials",
                metric_name="active_trials",
                metric_value=float(len(rows)),
                confidence=Confidence.MEDIUM,
                section=target.id,
            ))

        return cards

    # ── Helpers ──

    @staticmethod
    def _find_column(table: TableBlock, candidates: Tuple[str, ...]) -> Optional[int]:
        """Find the index of the first matching column header (case-insensitive)."""
        lower_headers = [h.lower() for h in table.headers]
        for candidate in candidates:
            if candidate.lower() in lower_headers:
                return lower_headers.index(candidate.lower())
        return None

    @staticmethod
    def _build_trial_query(disease: str, drugs: List[str]) -> str:
        parts = [disease] if disease else []
        if drugs:
            parts.extend(drugs[:3])
        return " OR ".join(parts) if parts else ""

    class _AdapterContextManager:
        """Open/close a set of API adapters as async context managers."""

        def __init__(self, config: APIConfig):
            self.config = config
            self.adapters: Dict[str, BaseAPIAdapter] = {}

        async def __aenter__(self) -> Dict[str, BaseAPIAdapter]:
            for name, cls in API_ADAPTER_REGISTRY.items():
                try:
                    adapter = cls(self.config)
                    await adapter.__aenter__()
                    self.adapters[name] = adapter
                except Exception as exc:
                    logger.debug("Failed to init adapter %s: %s", name, exc)
            return self.adapters

        async def __aexit__(self, *exc: object) -> None:
            for adapter in self.adapters.values():
                try:
                    await adapter.__aexit__(*exc)
                except Exception:
                    pass

    def _adapters(self) -> _AdapterContextManager:
        return self._AdapterContextManager(self.config)
