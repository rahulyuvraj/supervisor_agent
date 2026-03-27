"""PubChem PUG-REST API adapter.

Base: https://pubchem.ncbi.nlm.nih.gov/rest/pug/
Rate limit: 5 req/s (unauthenticated).

Docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)


class PubChemAdapter(BaseAPIAdapter):
    service_name = "pubchem"
    base_url = "https://pubchem.ncbi.nlm.nih.gov"

    # ── Compound by name ──

    async def get_compound_by_name(self, name: str) -> Dict[str, Any]:
        """Look up a compound by common or IUPAC name."""
        data = await self._request(
            "GET",
            f"/rest/pug/compound/name/{name}/JSON",
        )
        compounds = data.get("PC_Compounds", [])
        return compounds[0] if compounds else {}

    async def get_compound_properties(
        self,
        name: str,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve computed properties for a named compound."""
        props = ",".join(properties or [
            "MolecularFormula", "MolecularWeight", "CanonicalSMILES",
            "IUPACName", "XLogP", "TPSA", "HBondDonorCount",
            "HBondAcceptorCount",
        ])
        data = await self._request(
            "GET",
            f"/rest/pug/compound/name/{name}/property/{props}/JSON",
        )
        table = data.get("PropertyTable", {}).get("Properties", [])
        return table[0] if table else {}

    # ── Compound by CID ──

    async def get_compound_by_cid(self, cid: int) -> Dict[str, Any]:
        """Fetch a compound by PubChem CID."""
        data = await self._request(
            "GET",
            f"/rest/pug/compound/cid/{cid}/JSON",
        )
        compounds = data.get("PC_Compounds", [])
        return compounds[0] if compounds else {}

    # ── Similarity / substructure ──

    async def similarity_search(
        self,
        smiles: str,
        *,
        threshold: int = 90,
        max_records: int = 10,
    ) -> List[int]:
        """2D fingerprint similarity search. Returns list of CIDs."""
        data = await self._request(
            "GET",
            "/rest/pug/compound/similarity/smiles/JSON",
            params={
                "smiles": smiles,
                "Threshold": threshold,
                "MaxRecords": max_records,
            },
            skip_cache=True,
        )
        return data.get("IdentifierList", {}).get("CID", [])

    # ── Bioactivity ──

    async def get_bioassay_summary(self, cid: int) -> Dict[str, Any]:
        """Bioassay summary for a compound."""
        return await self._request(
            "GET",
            f"/rest/pug/compound/cid/{cid}/assaysummary/JSON",
        )

    # ── Synonyms ──

    async def get_synonyms(self, name: str) -> List[str]:
        """All synonyms for a named compound."""
        data = await self._request(
            "GET",
            f"/rest/pug/compound/name/{name}/synonyms/JSON",
        )
        info_list = data.get("InformationList", {}).get("Information", [])
        if info_list:
            return info_list[0].get("Synonym", [])
        return []
