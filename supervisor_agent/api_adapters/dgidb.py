"""DGIdb GraphQL API adapter.

Single endpoint: POST https://dgidb.org/api/graphql
Batch gene queries in groups of 50 to stay within server limits.
``interactionScore`` may be null in DGIdb 5.0 — handled gracefully.

Docs: https://dgidb.org/api
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)

_GENES_QUERY = """
query GeneInteractions($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      conceptId
      interactions {
        interactionScore
        interactionTypes {
          type
          directionality
        }
        drug {
          name
          conceptId
          approved
        }
        interactionAttributes {
          name
          value
        }
        publications {
          pmid
        }
        sources {
          fullName
        }
      }
    }
  }
}
"""

_DRUGS_QUERY = """
query DrugInteractions($names: [String!]!) {
  drugs(names: $names) {
    nodes {
      name
      conceptId
      approved
      interactions {
        interactionScore
        interactionTypes {
          type
          directionality
        }
        gene {
          name
          conceptId
        }
        publications {
          pmid
        }
        sources {
          fullName
        }
      }
    }
  }
}
"""

_BATCH_SIZE = 50


class DGIdbAdapter(BaseAPIAdapter):
    service_name = "dgidb"
    base_url = "https://dgidb.org"

    # ── Gene → drug interactions ──

    async def get_gene_interactions(
        self,
        gene_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Query drug-gene interactions for a list of gene symbols.

        Results are batched in groups of 50.  Returns the merged ``nodes``
        list across all batches.
        """
        return await self._batched_query(
            _GENES_QUERY, "genes", gene_names
        )

    # ── Drug → gene interactions ──

    async def get_drug_interactions(
        self,
        drug_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Query gene interactions for a list of drug names."""
        return await self._batched_query(
            _DRUGS_QUERY, "drugs", drug_names
        )

    # ── Internal helpers ──

    async def _batched_query(
        self,
        query: str,
        root_key: str,
        names: List[str],
    ) -> List[Dict[str, Any]]:
        all_nodes: List[Dict[str, Any]] = []
        for i in range(0, len(names), _BATCH_SIZE):
            batch = names[i : i + _BATCH_SIZE]
            data = await self._graphql(query, {"names": batch})
            nodes = (data.get("data", {}).get(root_key, {}).get("nodes") or [])
            all_nodes.extend(self._normalise_scores(nodes))
        return all_nodes

    async def _graphql(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/graphql",
            json={"query": query, "variables": variables},
            skip_cache=True,
        )

    @staticmethod
    def _normalise_scores(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure interactionScore is always a float (DGIdb 5.0 may return null)."""
        for node in nodes:
            for ix in node.get("interactions", []):
                if ix.get("interactionScore") is None:
                    ix["interactionScore"] = 0.0
                else:
                    ix["interactionScore"] = float(ix["interactionScore"])
        return nodes
