"""Module output adapters — read pipeline artifacts into structured schemas."""

from .base import BaseModuleAdapter, ADAPTER_REGISTRY
from .deg import DEGAdapter
from .pathway import PathwayAdapter
from .drug import DrugDiscoveryAdapter

__all__ = [
    "BaseModuleAdapter", "ADAPTER_REGISTRY",
    "DEGAdapter", "PathwayAdapter", "DrugDiscoveryAdapter",
]
