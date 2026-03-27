"""Self-contained async API adapter layer.

Every adapter calls public REST/GraphQL endpoints directly via httpx.
Zero imports from tools/, drug_agent/, gene_prioritization/, or
reporting_pipeline_agent/.
"""

from .base import API_ADAPTER_REGISTRY, BaseAPIAdapter
from .config import APIConfig

# Import adapter modules so __init_subclass__ populates the registry
from . import (  # noqa: F401
    chembl,
    clinical_trials,
    dgidb,
    ensembl,
    kegg,
    openfda,
    pubchem,
    reactome,
    string_ppi,
)

__all__ = ["API_ADAPTER_REGISTRY", "APIConfig", "BaseAPIAdapter"]
