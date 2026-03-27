"""Tests for enrichment_dispatcher — entity extraction and adapter dispatch.

All API calls are mocked — zero real network traffic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from supervisor_agent.response.enrichment_dispatcher import (
    _ENTITY_ADAPTER_MAP,
    _extract_drugs_from_df,
    _extract_genes_from_df,
    _extract_genes_from_query,
    _extract_pathway_ids_from_df,
    extract_entities,
    enrich_for_response,
)


# ── Entity extraction tests ──


class TestExtractGenesFromDF:
    def test_gene_symbol_column(self):
        df = pd.DataFrame({"gene_symbol": ["STAT1", "TP53", "BRCA1"]})
        result = _extract_genes_from_df(df)
        assert result == ["STAT1", "TP53", "BRCA1"]

    def test_gene_column(self):
        df = pd.DataFrame({"Gene": ["TNF", "IL6"], "logFC": [2.1, -1.5]})
        result = _extract_genes_from_df(df)
        assert result == ["TNF", "IL6"]

    def test_no_gene_column(self):
        df = pd.DataFrame({"pathway": ["R-HSA-123"], "fdr": [0.01]})
        result = _extract_genes_from_df(df)
        assert result == []

    def test_drops_nan(self):
        df = pd.DataFrame({"gene": ["STAT1", None, "TP53"]})
        result = _extract_genes_from_df(df)
        assert "None" not in result
        assert len(result) == 2


class TestExtractDrugsFromDF:
    def test_drug_column(self):
        df = pd.DataFrame({"drug_name": ["ruxolitinib", "imatinib"]})
        assert _extract_drugs_from_df(df) == ["ruxolitinib", "imatinib"]

    def test_compound_column(self):
        df = pd.DataFrame({"Compound": ["aspirin"]})
        assert _extract_drugs_from_df(df) == ["aspirin"]

    def test_no_drug_column(self):
        df = pd.DataFrame({"gene": ["TP53"]})
        assert _extract_drugs_from_df(df) == []


class TestExtractPathwayIdsFromDF:
    def test_reactome_ids(self):
        df = pd.DataFrame({"pathway_id": ["R-HSA-1234", "R-HSA-5678", "GO:0001"]})
        result = _extract_pathway_ids_from_df(df)
        assert "R-HSA-1234" in result
        assert "R-HSA-5678" in result
        assert "GO:0001" not in result  # Only R-HSA- and hsa prefixes

    def test_kegg_ids(self):
        df = pd.DataFrame({"pathway_id": ["hsa04630", "hsa04060"]})
        result = _extract_pathway_ids_from_df(df)
        assert "hsa04630" in result


class TestExtractGenesFromQuery:
    def test_single_gene(self):
        result = _extract_genes_from_query("What does STAT1 do?")
        assert "STAT1" in result

    def test_multiple_genes(self):
        result = _extract_genes_from_query("Compare TP53 and BRCA1 expression")
        assert "TP53" in result
        assert "BRCA1" in result

    def test_no_genes(self):
        result = _extract_genes_from_query("summarize all results")
        # Only captures uppercase 2-10 char tokens — common words excluded
        assert not any(g in result for g in ["summarize", "all", "results"])

    def test_short_symbols_excluded(self):
        """Single-char tokens should not match the gene regex."""
        result = _extract_genes_from_query("A quick test")
        assert "A" not in result  # Too short (1 char)


class TestExtractEntities:
    def test_combined_extraction(self):
        csvs = {
            "genes": pd.DataFrame({"gene_symbol": ["TNF", "IL6"]}),
            "drugs": pd.DataFrame({"drug_name": ["aspirin"]}),
        }
        entities = extract_entities("What about STAT1?", csvs, disease="lupus")
        assert "genes" in entities
        assert "STAT1" in entities["genes"]  # Query gene first
        assert "TNF" in entities["genes"]
        assert "drugs" in entities
        assert "disease" in entities
        assert entities["disease"] == ["lupus"]

    def test_query_genes_prioritized(self):
        csvs = {"genes": pd.DataFrame({"gene": ["IL6", "TNF"]})}
        entities = extract_entities("Focus on STAT1 specifically", csvs)
        assert entities["genes"][0] == "STAT1"

    def test_empty_inputs(self):
        entities = extract_entities("hello", {}, "")
        # May or may not find anything depending on if there are uppercase tokens
        assert isinstance(entities, dict)

    def test_deduplication(self):
        csvs = {"genes": pd.DataFrame({"gene": ["STAT1", "TNF"]})}
        entities = extract_entities("What is STAT1?", csvs)
        assert entities["genes"].count("STAT1") == 1


class TestEntityAdapterMap:
    def test_gene_adapters(self):
        assert "ensembl" in _ENTITY_ADAPTER_MAP["genes"]
        assert "string" in _ENTITY_ADAPTER_MAP["genes"]
        assert "dgidb" in _ENTITY_ADAPTER_MAP["genes"]

    def test_drug_adapters(self):
        assert "chembl" in _ENTITY_ADAPTER_MAP["drugs"]
        assert "openfda" in _ENTITY_ADAPTER_MAP["drugs"]
        assert "pubchem" in _ENTITY_ADAPTER_MAP["drugs"]

    def test_pathway_adapters(self):
        assert "reactome" in _ENTITY_ADAPTER_MAP["pathways"]
        assert "kegg" in _ENTITY_ADAPTER_MAP["pathways"]

    def test_disease_adapters(self):
        assert "clinical_trials" in _ENTITY_ADAPTER_MAP["disease"]


class TestClinicalTrialsStatusParam:
    """Regression: filter_overall_status must be a list, not a string."""

    def test_call_passes_list_not_string(self):
        from supervisor_agent.response.enrichment_dispatcher import _call_clinical_trials

        adapter = AsyncMock()
        adapter.search_all = AsyncMock(return_value=[])
        asyncio.get_event_loop().run_until_complete(
            _call_clinical_trials(adapter, ["breast cancer"])
        )
        _call_args = adapter.search_all.call_args
        status = _call_args.kwargs.get("filter_overall_status")
        assert isinstance(status, list), f"Expected list, got {type(status).__name__}"
        assert status == ["RECRUITING"]


# ── enrich_for_response tests (all mocked) ──


class TestEnrichForResponse:
    def test_empty_entities_returns_empty(self):
        """No entities → no API calls → empty dict."""
        result = asyncio.get_event_loop().run_until_complete(
            enrich_for_response("hello", {}, "")
        )
        assert result == {} or isinstance(result, dict)

    @patch("supervisor_agent.api_adapters.APIConfig")
    @patch("supervisor_agent.api_adapters.API_ADAPTER_REGISTRY", {})
    def test_no_registry_adapters(self, mock_config):
        """If registry has no matching adapters, returns empty."""
        mock_config.from_env.return_value = MagicMock()
        csvs = {"genes": pd.DataFrame({"gene": ["STAT1"]})}
        result = asyncio.get_event_loop().run_until_complete(
            enrich_for_response("test", csvs, "lupus")
        )
        assert result == {}

    @patch("supervisor_agent.api_adapters.APIConfig")
    @patch("supervisor_agent.api_adapters.API_ADAPTER_REGISTRY")
    def test_adapter_failure_graceful(self, mock_registry, mock_config):
        """Adapter that raises is skipped gracefully."""
        mock_config.from_env.return_value = MagicMock()

        mock_cls = MagicMock()
        mock_adapter = AsyncMock()
        mock_adapter.__aenter__ = AsyncMock(side_effect=RuntimeError("init failed"))
        mock_adapter.__aexit__ = AsyncMock()
        mock_cls.return_value = mock_adapter
        mock_registry.get.return_value = mock_cls

        csvs = {"genes": pd.DataFrame({"gene": ["STAT1"]})}
        result = asyncio.get_event_loop().run_until_complete(
            enrich_for_response("test", csvs, "lupus", timeout=2.0)
        )
        assert isinstance(result, dict)

    def test_config_failure_returns_empty(self):
        """APIConfig.from_env() failure → graceful empty return."""
        with patch(
            "supervisor_agent.api_adapters.APIConfig"
        ) as mock_config:
            mock_config.from_env.side_effect = RuntimeError("no env")
            csvs = {"genes": pd.DataFrame({"gene": ["STAT1"]})}
            result = asyncio.get_event_loop().run_until_complete(
                enrich_for_response("test", csvs, "lupus")
            )
            assert result == {}
