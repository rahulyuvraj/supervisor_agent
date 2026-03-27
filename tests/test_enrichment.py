"""Tests for ReportEnricher — integration wiring between Track A and Track B.

Uses httpx.MockTransport for all API calls — zero real network traffic.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List

import httpx
import pytest
import pytest_asyncio

from supervisor_agent.api_adapters.base import API_ADAPTER_REGISTRY, BaseAPIAdapter
from supervisor_agent.api_adapters.config import APIConfig

# Import adapter modules to trigger __init_subclass__ registration
import supervisor_agent.api_adapters.ensembl  # noqa: F401
import supervisor_agent.api_adapters.string_ppi  # noqa: F401
import supervisor_agent.api_adapters.reactome  # noqa: F401
import supervisor_agent.api_adapters.chembl  # noqa: F401
import supervisor_agent.api_adapters.openfda  # noqa: F401
import supervisor_agent.api_adapters.dgidb  # noqa: F401
import supervisor_agent.api_adapters.clinical_trials  # noqa: F401
from supervisor_agent.data_layer.schemas.evidence import Confidence, EvidenceCard
from supervisor_agent.data_layer.schemas.sections import (
    ReportingConfig,
    SectionBlock,
    SectionMeta,
    TableBlock,
)
from supervisor_agent.reporting_engine.enrichment import ReportEnricher


# ── Fixtures ──


@pytest.fixture
def api_config():
    return APIConfig(kegg_enabled=False, default_timeout=5.0, max_retries=1)


def _make_sections() -> List[SectionBlock]:
    """Create a minimal set of populated sections for enrichment."""
    return [
        SectionBlock(
            id="deg_findings",
            title="Differential Expression Analysis",
            level=2,
            order=3,
            body="Top differentially expressed genes identified.",
            tables=[
                TableBlock(
                    caption="Top DEGs",
                    headers=["gene_symbol", "log2FC", "padj"],
                    rows=[
                        ["STAT1", "3.2", "1e-10"],
                        ["IRF7", "2.8", "5e-9"],
                        ["MX1", "2.5", "1e-8"],
                    ],
                    source_artifact="deg_results",
                ),
            ],
            meta=SectionMeta(module="deg_analysis"),
        ),
        SectionBlock(
            id="pathway_findings",
            title="Pathway Enrichment Analysis",
            level=2,
            order=4,
            body="Enriched pathways from Reactome.",
            tables=[
                TableBlock(
                    caption="Top Pathways",
                    headers=["pathway_id", "name", "pValue"],
                    rows=[
                        ["R-HSA-913531", "Interferon Signaling", "1e-12"],
                        ["R-HSA-168256", "Immune System", "5e-8"],
                    ],
                    source_artifact="pathway_results",
                ),
            ],
            meta=SectionMeta(module="pathway_enrichment"),
        ),
        SectionBlock(
            id="drug_findings",
            title="Drug Discovery & Perturbation Analysis",
            level=2,
            order=5,
            body="Drug candidates identified.",
            tables=[
                TableBlock(
                    caption="Candidate Drugs",
                    headers=["drug_name", "score", "source"],
                    rows=[
                        ["Baricitinib", "0.87", "DGIdb"],
                        ["Anifrolumab", "0.75", "OpenTargets"],
                    ],
                    source_artifact="drug_results",
                ),
            ],
            meta=SectionMeta(module="perturbation_analysis"),
        ),
        SectionBlock(
            id="integrated_findings",
            title="Integrated Findings",
            level=2,
            order=6,
            body="",
            meta=SectionMeta(),
        ),
        SectionBlock(
            id="next_steps",
            title="Recommended Next Steps",
            level=2,
            order=8,
            body="Consider the following next steps.",
            meta=SectionMeta(),
        ),
    ]


def _make_evidence() -> List[EvidenceCard]:
    return [
        EvidenceCard(
            finding="STAT1 upregulated (log2FC=3.2, padj=1e-10)",
            module="deg_analysis",
            artifact_label="deg_results",
            metric_name="log2FC",
            metric_value=3.2,
            confidence=Confidence.HIGH,
            section="deg_findings",
        ),
        EvidenceCard(
            finding="Interferon Signaling enriched (p=1e-12)",
            module="pathway_enrichment",
            artifact_label="pathway_results",
            metric_name="neg_log_p",
            metric_value=12.0,
            confidence=Confidence.HIGH,
            section="pathway_findings",
        ),
    ]


# ── Mock transport that routes to mock API responses ──


def _mock_transport() -> httpx.MockTransport:
    """Create a mock transport that handles all enrichment API paths."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = request.url.path

        # Ensembl — gene lookup
        if "rest.ensembl.org" in url and "/lookup/symbol" in path:
            symbol = path.split("/")[-1]
            return httpx.Response(200, json={
                "id": f"ENSG_{symbol}",
                "biotype": "protein_coding",
                "description": f"Signal transducer {symbol}",
                "seq_region_name": "2",
                "start": 100000,
                "end": 200000,
            })

        # STRING — PPI network
        if "string-db.org" in url and "network" in path:
            return httpx.Response(200, text=(
                "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\n"
                "9606.STAT1\t9606.IRF7\tSTAT1\tIRF7\t0.95\n"
                "9606.STAT1\t9606.MX1\tSTAT1\tMX1\t0.88\n"
            ), headers={"content-type": "text/plain"})

        # Reactome — pathway detail
        if "reactome.org" in url and "/ContentService/data/" in path:
            if "participants" in path:
                return httpx.Response(200, json=[])
            if "complexes" in path:
                return httpx.Response(200, json=[])
            # get_pathway
            pathway_id = path.split("/")[-1]
            return httpx.Response(200, json={
                "stId": pathway_id,
                "displayName": "Interferon Signaling",
                "speciesName": "Homo sapiens",
            })

        # Reactome — pathway drugs (hasComponent)
        if "reactome.org" in url and "hasComponent" in str(request.url):
            return httpx.Response(200, json=[
                {"schemaClass": "Drug", "displayName": "Ruxolitinib"},
            ])

        # Reactome analysis
        if "reactome.org" in url and "AnalysisService" in path:
            return httpx.Response(200, json={})

        # ChEMBL — molecule search
        if "ebi.ac.uk" in url and "/molecule.json" in path:
            return httpx.Response(200, json={
                "molecules": [{"molecule_chembl_id": "CHEMBL3301612", "pref_name": "BARICITINIB"}],
            })

        # ChEMBL — mechanisms
        if "ebi.ac.uk" in url and "/mechanism.json" in path:
            return httpx.Response(200, json={
                "mechanisms": [{"mechanism_of_action": "Janus kinase (JAK) inhibitor"}],
            })

        # ChEMBL — indications
        if "ebi.ac.uk" in url and "/drug_indication.json" in path:
            return httpx.Response(200, json={
                "drug_indications": [{"mesh_heading": "Rheumatoid Arthritis"}],
            })

        # OpenFDA — adverse events count
        if "api.fda.gov" in url and "event.json" in path:
            return httpx.Response(200, json={
                "results": [
                    {"term": "Nausea", "count": 150},
                    {"term": "Headache", "count": 120},
                ],
            })

        # DGIdb — GraphQL
        if "dgidb.org" in url and "/api/graphql" in path:
            return httpx.Response(200, json={
                "data": {"genes": {"nodes": [
                    {
                        "name": "STAT1",
                        "interactions": [{
                            "drug": {"name": "Ruxolitinib"},
                            "interactionScore": 5.0,
                            "interactionTypes": [{"type": "inhibitor"}],
                        }],
                    },
                ]}},
            })

        # ClinicalTrials.gov — search
        if "clinicaltrials.gov" in url:
            return httpx.Response(200, json={
                "studies": [{
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT05000001",
                            "briefTitle": "A Study of Baricitinib in Lupus",
                        },
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {"enrollmentInfo": {"count": 200}},
                        "sponsorCollaboratorsModule": {
                            "leadSponsor": {"name": "Lilly"},
                        },
                    },
                }],
            })

        # PubChem / Ensembl fallbacks
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


# ── Entity extraction tests ──


class TestEntityExtraction:
    """Test that the enricher correctly extracts genes, drugs, pathway IDs."""

    def test_extract_genes_from_table(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        genes = enricher._extract_genes(section_map, [])
        assert "STAT1" in genes
        assert "IRF7" in genes
        assert "MX1" in genes

    def test_extract_genes_from_evidence(self, api_config):
        enricher = ReportEnricher(api_config)
        cards = _make_evidence()
        section_map = {}
        genes = enricher._extract_genes(section_map, cards)
        assert "STAT1" in genes

    def test_extract_drugs_from_table(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        drugs = enricher._extract_drugs(section_map, [])
        assert "Baricitinib" in drugs
        assert "Anifrolumab" in drugs

    def test_extract_pathway_ids(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        ids = enricher._extract_pathway_ids(section_map, [])
        assert "R-HSA-913531" in ids
        assert "R-HSA-168256" in ids

    def test_extract_genes_dedup(self, api_config):
        """Duplicate genes (case-insensitive) are removed."""
        enricher = ReportEnricher(api_config)
        section = SectionBlock(
            id="deg_findings", title="DEG", level=2, order=0,
            tables=[TableBlock(
                headers=["gene_symbol", "val"],
                rows=[["STAT1", "1"], ["stat1", "2"], ["IRF7", "3"]],
            )],
        )
        genes = enricher._extract_genes({"deg_findings": section}, [])
        assert len(genes) == 2

    def test_extract_empty_sections(self, api_config):
        """No entities extracted from empty sections."""
        enricher = ReportEnricher(api_config)
        genes = enricher._extract_genes({}, [])
        drugs = enricher._extract_drugs({}, [])
        pathway_ids = enricher._extract_pathway_ids({}, [])
        assert genes == []
        assert drugs == []
        assert pathway_ids == []


# ── Enrichment injection tests ──


class TestEnrichmentInjection:
    """Test that enrichment data is correctly injected into sections."""

    def test_apply_gene_annotations(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{"symbol": "STAT1", "ensembl_id": "ENSG_STAT1",
                 "biotype": "protein_coding", "chromosome": "2", "description": "STAT1"}]
        enricher._apply_gene_annotations(section_map, {"gene_annotations": data})
        deg = section_map["deg_findings"]
        assert len(deg.tables) == 2  # original + new
        assert deg.tables[-1].caption == "Gene Annotations (Ensembl)"
        assert deg.tables[-1].source_artifact == "api:ensembl"

    def test_apply_ppi_network(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{"preferredName_A": "STAT1", "preferredName_B": "IRF7", "score": "0.95"}]
        enricher._apply_ppi_network(section_map, {"ppi_network": data})
        integrated = section_map["integrated_findings"]
        assert len(integrated.tables) == 1
        assert "STRING" in integrated.tables[0].caption
        assert "interactions" in integrated.body

    def test_apply_drug_mechanisms(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{"drug": "Baricitinib", "chembl_id": "CHEMBL3301612",
                 "mechanisms": ["JAK inhibitor"], "indications": ["RA"]}]
        enricher._apply_drug_mechanisms(section_map, {"chembl_mechanisms": data})
        drug_sec = section_map["drug_findings"]
        assert any("ChEMBL" in t.caption for t in drug_sec.tables)

    def test_apply_adverse_events(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{"drug": "Baricitinib", "top_events": [{"term": "Nausea", "count": 150}]}]
        enricher._apply_adverse_events(section_map, {"fda_events": data})
        drug_sec = section_map["drug_findings"]
        assert any("OpenFDA" in t.caption for t in drug_sec.tables)

    def test_apply_trials(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{
            "protocolSection": {
                "identificationModule": {"nctId": "NCT05000001", "briefTitle": "Lupus Trial"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"enrollmentInfo": {"count": 200}},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Lilly"}},
            },
        }]
        cards = enricher._apply_trials(section_map, {"clinical_trials": data})
        assert len(cards) == 1
        assert cards[0].module == "clinical_trials_enrichment"
        assert cards[0].metric_value == 1.0
        # Table appended to next_steps
        ns = section_map["next_steps"]
        assert any("Clinical Trials" in t.caption for t in ns.tables)

    def test_apply_gene_drug_interactions(self, api_config):
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        data = [{
            "name": "STAT1",
            "interactions": [{
                "drug": {"name": "Ruxolitinib"},
                "interactionScore": 5.0,
                "interactionTypes": [{"type": "inhibitor"}],
            }],
        }]
        enricher._apply_gene_drug(section_map, {"dgidb_interactions": data})
        drug_sec = section_map["drug_findings"]
        assert any("DGIdb" in t.caption for t in drug_sec.tables)

    def test_apply_empty_enrichments(self, api_config):
        """No-op when enrichment data is empty or missing."""
        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        section_map = {s.id: s for s in sections}
        original_table_count = sum(len(s.tables) for s in sections)
        enricher._apply_gene_annotations(section_map, {})
        enricher._apply_ppi_network(section_map, {})
        enricher._apply_drug_mechanisms(section_map, {})
        enricher._apply_adverse_events(section_map, {})
        enricher._apply_gene_drug(section_map, {})
        after_count = sum(len(s.tables) for s in sections)
        assert after_count == original_table_count


# ── Full async enrichment tests (with mocked HTTP) ──


@pytest.mark.asyncio
class TestFullEnrichment:
    """End-to-end enrichment with mocked API calls."""

    async def test_enrich_adds_tables(self, api_config, monkeypatch):
        """Full enrich() call adds API-sourced tables to sections."""
        transport = _mock_transport()

        # Monkey-patch all adapters to use the mock transport
        original_aenter = BaseAPIAdapter.__aenter__

        async def mock_aenter(self):
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=api_config.default_timeout,
                transport=transport,
                headers={"Accept": "application/json"},
            )
            return self

        monkeypatch.setattr(BaseAPIAdapter, "__aenter__", mock_aenter)

        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        cards = _make_evidence()
        original_tables = sum(len(s.tables) for s in sections)

        sections, extra_cards = await enricher.enrich(sections, cards, disease="lupus")

        new_tables = sum(len(s.tables) for s in sections)
        # At least some enrichment tables should have been added
        assert new_tables > original_tables

    async def test_enrich_no_entities(self, api_config):
        """Enrichment is a no-op when sections contain no extractable entities."""
        enricher = ReportEnricher(api_config)
        sections = [
            SectionBlock(id="executive_summary", title="Summary", level=2, order=0),
        ]
        sections_out, extra = await enricher.enrich(sections, [], disease="")
        assert len(sections_out) == 1
        assert extra == []

    async def test_enrich_graceful_on_api_failure(self, api_config, monkeypatch):
        """If all API adapters fail, sections are returned unmodified."""

        async def failing_aenter(self):
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=0.001,
                transport=httpx.MockTransport(lambda r: httpx.Response(500)),
                headers={"Accept": "application/json"},
            )
            return self

        monkeypatch.setattr(BaseAPIAdapter, "__aenter__", failing_aenter)

        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        cards = _make_evidence()
        original_bodies = [s.body for s in sections]

        sections_out, extra = await enricher.enrich(sections, cards, disease="lupus")

        # Original content preserved
        for s, orig_body in zip(sections_out, original_bodies):
            assert s.body.startswith(orig_body) or s.body == orig_body

    async def test_clinical_trials_evidence_card(self, api_config, monkeypatch):
        """Clinical trial enrichment produces evidence cards."""
        transport = _mock_transport()

        async def mock_aenter(self):
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=api_config.default_timeout,
                transport=transport,
                headers={"Accept": "application/json"},
            )
            return self

        monkeypatch.setattr(BaseAPIAdapter, "__aenter__", mock_aenter)

        enricher = ReportEnricher(api_config)
        sections = _make_sections()
        cards = _make_evidence()

        _, extra_cards = await enricher.enrich(sections, cards, disease="lupus")

        trial_cards = [c for c in extra_cards if c.module == "clinical_trials_enrichment"]
        assert len(trial_cards) >= 1
        assert trial_cards[0].metric_name == "active_trials"


# ── Helpers test ──


class TestHelpers:
    def test_find_column(self, api_config):
        enricher = ReportEnricher(api_config)
        table = TableBlock(headers=["Gene_Symbol", "log2FC", "padj"])
        idx = enricher._find_column(table, ("gene", "gene_symbol", "symbol"))
        assert idx == 0

    def test_find_column_not_found(self, api_config):
        enricher = ReportEnricher(api_config)
        table = TableBlock(headers=["colA", "colB"])
        idx = enricher._find_column(table, ("gene",))
        assert idx is None

    def test_build_trial_query(self, api_config):
        q = ReportEnricher._build_trial_query("lupus", ["Baricitinib", "Anifrolumab"])
        assert "lupus" in q
        assert "Baricitinib" in q
        assert "OR" in q

    def test_build_trial_query_empty(self, api_config):
        q = ReportEnricher._build_trial_query("", [])
        assert q == ""
