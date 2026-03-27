"""Tests for APIConfig."""

import os

import pytest

from supervisor_agent.api_adapters.config import APIConfig


class TestAPIConfig:
    def test_defaults(self):
        cfg = APIConfig()
        assert cfg.kegg_enabled is False
        assert cfg.openfda_api_key is None
        assert cfg.default_timeout == 30.0
        assert cfg.max_retries == 3

    def test_from_env_reads_vars(self, monkeypatch):
        monkeypatch.setenv("KEGG_ENABLED", "true")
        monkeypatch.setenv("OPENFDA_API_KEY", "test-key-123")
        monkeypatch.setenv("NCBI_API_KEY", "ncbi-key")
        cfg = APIConfig.from_env()
        assert cfg.kegg_enabled is True
        assert cfg.openfda_api_key == "test-key-123"
        assert cfg.ncbi_api_key == "ncbi-key"

    def test_from_env_defaults_without_vars(self, monkeypatch):
        monkeypatch.delenv("KEGG_ENABLED", raising=False)
        monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
        cfg = APIConfig.from_env()
        assert cfg.kegg_enabled is False
        assert cfg.openfda_api_key is None

    def test_rate_limits_configurable(self):
        cfg = APIConfig(reactome_rps=1.0, chembl_rps=10.0)
        assert cfg.reactome_rps == 1.0
        assert cfg.chembl_rps == 10.0
