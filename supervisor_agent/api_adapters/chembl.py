"""ChEMBL REST API adapter.

Calls www.ebi.ac.uk/chembl/api/data/ directly — no chembl_webresource_client.
All endpoints return JSON when the URL ends with ``.json``.

Docs: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIAdapter

logger = logging.getLogger(__name__)


class ChEMBLAdapter(BaseAPIAdapter):
    service_name = "chembl"
    base_url = "https://www.ebi.ac.uk"

    # ── Molecule search ──

    async def search_molecule(
        self,
        name: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search molecules by preferred name (case-insensitive exact match)."""
        data = await self._request(
            "GET",
            "/chembl/api/data/molecule.json",
            params={"pref_name__iexact": name, "limit": limit},
        )
        return data.get("molecules", [])

    async def get_molecule(self, chembl_id: str) -> Dict[str, Any]:
        """Fetch a single molecule by ChEMBL ID."""
        return await self._request(
            "GET", f"/chembl/api/data/molecule/{chembl_id}.json"
        )

    async def search_molecule_by_synonym(
        self,
        synonym: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search molecules by synonym (brand/generic name, case-insensitive)."""
        data = await self._request(
            "GET",
            "/chembl/api/data/molecule.json",
            params={
                "molecule_synonyms__molecule_synonym__icontains": synonym,
                "limit": limit,
            },
        )
        return data.get("molecules", [])

    # ── Mechanism of action ──

    async def get_mechanisms(
        self,
        chembl_id: str,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Mechanism-of-action records for a molecule."""
        data = await self._request(
            "GET",
            "/chembl/api/data/mechanism.json",
            params={"molecule_chembl_id": chembl_id, "limit": limit},
        )
        return data.get("mechanisms", [])

    async def get_mechanisms_by_target(
        self,
        target_chembl_id: str,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Mechanisms for a given target."""
        data = await self._request(
            "GET",
            "/chembl/api/data/mechanism.json",
            params={"target_chembl_id": target_chembl_id, "limit": limit},
        )
        return data.get("mechanisms", [])

    # ── Drug indications ──

    async def get_indications(
        self,
        chembl_id: str,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Clinical indications for a drug molecule."""
        data = await self._request(
            "GET",
            "/chembl/api/data/drug_indication.json",
            params={"molecule_chembl_id": chembl_id, "limit": limit},
        )
        return data.get("drug_indications", [])

    # ── Target search ──

    async def search_target(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search targets by preferred name."""
        data = await self._request(
            "GET",
            "/chembl/api/data/target.json",
            params={"pref_name__icontains": query, "limit": limit},
        )
        return data.get("targets", [])

    async def get_target(self, target_chembl_id: str) -> Dict[str, Any]:
        """Single target by ChEMBL ID."""
        return await self._request(
            "GET", f"/chembl/api/data/target/{target_chembl_id}.json"
        )

    # ── Bioactivity ──

    async def get_activities(
        self,
        chembl_id: str,
        *,
        target_chembl_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Bioactivity data for a molecule (optionally filtered by target/type)."""
        params: Dict[str, Any] = {"molecule_chembl_id": chembl_id, "limit": limit}
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id
        if activity_type:
            params["standard_type__iexact"] = activity_type
        data = await self._request(
            "GET", "/chembl/api/data/activity.json", params=params
        )
        return data.get("activities", [])
