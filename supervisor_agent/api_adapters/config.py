"""Centralized configuration for all API adapters.

Reads API keys from environment variables, exposes per-service feature
flags and rate-limit settings.  Instantiate once and pass to adapters.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """Immutable runtime configuration for the adapter layer."""

    # ── Feature flags ──
    kegg_enabled: bool = Field(
        default=False,
        description="KEGG REST API is free for academic use only; gate behind explicit opt-in.",
    )

    # ── API keys (all optional; adapters degrade gracefully) ──
    openfda_api_key: Optional[str] = Field(default=None)
    ncbi_api_key: Optional[str] = Field(default=None)

    # ── Rate limits (requests per second) ──
    reactome_rps: float = Field(default=3.0)
    kegg_rps: float = Field(default=2.0)
    chembl_rps: float = Field(default=5.0)
    openfda_rps: float = Field(default=4.0)
    dgidb_rps: float = Field(default=5.0)
    string_rps: float = Field(default=4.0)
    pubchem_rps: float = Field(default=5.0)
    ensembl_rps: float = Field(default=15.0)
    clinical_trials_rps: float = Field(default=3.0)

    # ── Cache TTLs (seconds) ──
    default_cache_ttl: int = Field(default=3600)
    cache_max_size: int = Field(default=2048)

    # ── HTTP defaults ──
    default_timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Build config from environment variables."""
        return cls(
            kegg_enabled=os.getenv("KEGG_ENABLED", "").lower() in ("1", "true", "yes"),
            openfda_api_key=os.getenv("OPENFDA_API_KEY"),
            ncbi_api_key=os.getenv("NCBI_API_KEY"),
        )
