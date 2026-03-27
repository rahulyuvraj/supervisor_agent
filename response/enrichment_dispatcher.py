"""Query-aware API enrichment dispatcher for the response path.

Extracts biomedical entities from user queries and CSV data, determines
which API adapters are relevant, calls them in parallel with a timeout,
and returns formatted markdown sections for the LLM context.

Independent of the reporting_engine — entity extraction logic is self-contained.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Adapter connection pool (H5) ──
_adapter_cache: dict = {}
_adapter_lock = asyncio.Lock()


async def _get_adapters(needed: set, api_config) -> dict:
    async with _adapter_lock:
        for name in needed:
            if name not in _adapter_cache:
                from ..api_adapters import API_ADAPTER_REGISTRY
                cls = API_ADAPTER_REGISTRY.get(name)
                if cls:
                    try:
                        adapter = cls(api_config)
                        await adapter.__aenter__()
                        _adapter_cache[name] = adapter
                    except Exception as exc:
                        logger.debug("Adapter %s init failed: %s", name, exc)
    return {n: _adapter_cache[n] for n in needed if n in _adapter_cache}


async def shutdown_adapters() -> None:
    for adapter in list(_adapter_cache.values()):
        try:
            await adapter.__aexit__(None, None, None)
        except Exception:
            pass
    _adapter_cache.clear()

# ── Entity-type → adapter mapping ──────────────────────────────────────────
# Driven by what entities are found, not hardcoded per query.

_ENTITY_ADAPTER_MAP: Dict[str, List[str]] = {
    "genes":    ["ensembl", "string", "dgidb"],
    "drugs":    ["chembl", "openfda", "pubchem"],
    "pathways": ["reactome", "kegg"],
    "disease":  ["clinical_trials"],
}

_MAX_GENES = 15
_MAX_DRUGS = 8
_MAX_PATHWAYS = 5

# ── Entity extraction (regex-based, zero LLM calls) ───────────────────────

_GENE_COL_NAMES = {"gene", "gene_symbol", "symbol", "gene_name"}
_DRUG_COL_NAMES = {"drug", "drug_name", "compound", "molecule_name"}
_PATHWAY_COL_NAMES = {"pathway_id", "stid", "id", "pathway"}
_GENE_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")


def _extract_genes_from_df(df: pd.DataFrame) -> List[str]:
    """Pull gene symbols from a DataFrame's gene-like columns."""
    for col in df.columns:
        if col.lower() in _GENE_COL_NAMES:
            return [str(v) for v in df[col].dropna().unique() if str(v).strip()]
    return []


def _extract_drugs_from_df(df: pd.DataFrame) -> List[str]:
    for col in df.columns:
        if col.lower() in _DRUG_COL_NAMES:
            return [str(v) for v in df[col].dropna().unique() if str(v).strip()]
    return []


def _extract_pathway_ids_from_df(df: pd.DataFrame) -> List[str]:
    for col in df.columns:
        if col.lower() in _PATHWAY_COL_NAMES:
            vals = df[col].dropna().astype(str)
            return [v for v in vals.unique()
                    if v.startswith("R-HSA-") or v.startswith("hsa")]
    return []


def _extract_genes_from_query(query: str) -> List[str]:
    """Find likely gene symbols in query text (2-10 uppercase alphanumeric)."""
    return [m.group(1) for m in _GENE_SYMBOL_RE.finditer(query)
            if len(m.group(1)) >= 2]


def _dedupe(items: List[str], limit: int) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        key = item.upper()
        if key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def extract_entities(
    query: str,
    csvs_by_type: Dict[str, pd.DataFrame],
    disease: str = "",
) -> Dict[str, List[str]]:
    """Extract genes, drugs, pathways, and disease from query + CSV data."""
    genes: List[str] = []
    drugs: List[str] = []
    pathways: List[str] = []

    # From CSVs
    if "genes" in csvs_by_type:
        genes.extend(_extract_genes_from_df(csvs_by_type["genes"]))
    if "drugs" in csvs_by_type:
        drugs.extend(_extract_drugs_from_df(csvs_by_type["drugs"]))
    if "pathways" in csvs_by_type:
        pathways.extend(_extract_pathway_ids_from_df(csvs_by_type["pathways"]))

    # From query text — prepend so user-mentioned entities get priority
    query_genes = _extract_genes_from_query(query)
    genes = query_genes + [g for g in genes if g.upper() not in {qg.upper() for qg in query_genes}]

    entities: Dict[str, List[str]] = {}
    genes = _dedupe(genes, _MAX_GENES)
    drugs = _dedupe(drugs, _MAX_DRUGS)
    pathways = _dedupe(pathways, _MAX_PATHWAYS)

    if genes:
        entities["genes"] = genes
    if drugs:
        entities["drugs"] = drugs
    if pathways:
        entities["pathways"] = pathways
    if disease:
        entities["disease"] = [disease]

    return entities


# ── Adapter result formatters ──────────────────────────────────────────────

def _fmt_gene_annotations(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["Gene | Biotype | Location | Description"]
    for r in results[:_MAX_GENES]:
        symbol = r.get("display_name", r.get("id", "?"))
        biotype = r.get("biotype", "?")
        region = r.get("seq_region_name", "?")
        start = r.get("start", "")
        desc = r.get("description", "").split("[")[0].strip() or "—"
        lines.append(f"{symbol} | {biotype} | chr{region}:{start} | {desc}")
    return "\n".join(lines)


def _fmt_ppi_network(edges: List[Dict[str, str]]) -> str:
    if not edges:
        return ""
    lines = ["Protein A | Protein B | Score"]
    for e in sorted(edges, key=lambda x: float(x.get("score", 0)), reverse=True)[:15]:
        lines.append(f"{e.get('preferredName_A', '?')} | {e.get('preferredName_B', '?')} | {e.get('score', '?')}")
    return "\n".join(lines)


def _fmt_gene_drug_interactions(nodes: List[Dict[str, Any]]) -> str:
    if not nodes:
        return ""
    lines = ["Gene | Drug | Interaction | Score"]
    for node in nodes:
        gene = node.get("name", "?")
        for ix in (node.get("interactions") or [])[:3]:
            drug = (ix.get("drug") or {}).get("name", "?")
            itype = ", ".join(t.get("type", "") for t in (ix.get("interactionTypes") or []))
            score = ix.get("interactionScore", "—")
            lines.append(f"{gene} | {drug} | {itype or '—'} | {score}")
    return "\n".join(lines)


def _fmt_drug_mechanisms(results: List[Tuple[str, Dict]]) -> str:
    if not results:
        return ""
    lines = ["Drug | ChEMBL ID | Mechanisms | Indications"]
    for drug_name, data in results:
        cid = data.get("chembl_id", "?")
        mechs = ", ".join(m.get("mechanism_of_action", "") for m in (data.get("mechanisms") or [])[:3]) or "—"
        indics = ", ".join(i.get("mesh_heading", "") for i in (data.get("indications") or [])[:3]) or "—"
        lines.append(f"{drug_name} | {cid} | {mechs} | {indics}")
    return "\n".join(lines)


def _fmt_adverse_events(results: List[Tuple[str, List[Dict]]]) -> str:
    if not results:
        return ""
    lines = ["Drug | Adverse Event | Count"]
    for drug_name, events in results:
        for ev in (events or [])[:5]:
            term = ev.get("term", "?")
            count = ev.get("count", "?")
            lines.append(f"{drug_name} | {term} | {count}")
    return "\n".join(lines)


def _fmt_compound_properties(results: List[Tuple[str, Dict]]) -> str:
    if not results:
        return ""
    lines = ["Compound | MW | LogP | TPSA | Formula"]
    for name, props in results:
        mw = props.get("MolecularWeight", "?")
        logp = props.get("XLogP", "?")
        tpsa = props.get("TPSA", "?")
        formula = props.get("MolecularFormula", "?")
        lines.append(f"{name} | {mw} | {logp} | {tpsa} | {formula}")
    return "\n".join(lines)


def _fmt_clinical_trials(studies: List[Dict[str, Any]]) -> str:
    if not studies:
        return ""
    lines = ["NCT ID | Title | Status | Phase | Enrollment"]
    for s in studies[:10]:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        nct = ident.get("nctId", "?")
        title = (ident.get("briefTitle", "?") or "?")[:80]
        status = status_mod.get("overallStatus", "?")
        phases = ", ".join(design.get("phases", []) or ["—"])
        enroll = (design.get("enrollmentInfo") or {}).get("count", "?")
        lines.append(f"{nct} | {title} | {status} | {phases} | {enroll}")
    return "\n".join(lines)


def _fmt_pathway_detail(results: List[Tuple[str, Dict]]) -> str:
    if not results:
        return ""
    lines = ["Pathway ID | Name | Species"]
    for pid, data in results:
        name = data.get("displayName", data.get("name", "?"))
        species = data.get("speciesName", "Homo sapiens")
        lines.append(f"{pid} | {name} | {species}")
    return "\n".join(lines)


# ── Adapter call wrappers ─────────────────────────────────────────────────

async def _call_ensembl(adapter, genes: List[str]) -> str:
    results = []
    for g in genes[:_MAX_GENES]:
        try:
            data = await adapter.lookup_symbol(g)
            if data:
                results.append(data)
        except Exception as exc:
            logger.debug("Ensembl lookup failed: %s", exc)
    return _fmt_gene_annotations(results)


async def _call_string(adapter, genes: List[str]) -> str:
    try:
        edges = await adapter.get_network(genes[:_MAX_GENES])
        return _fmt_ppi_network(edges)
    except Exception as exc:
        logger.debug("STRING lookup failed: %s", exc)
        return ""


async def _call_dgidb(adapter, genes: List[str]) -> str:
    try:
        nodes = await adapter.get_gene_interactions(genes[:_MAX_GENES])
        return _fmt_gene_drug_interactions(nodes)
    except Exception as exc:
        logger.debug("DGIdb lookup failed: %s", exc)
        return ""


async def _call_chembl(adapter, drugs: List[str]) -> str:
    results = []
    for d in drugs[:_MAX_DRUGS]:
        try:
            molecules = await adapter.search_molecule(d, limit=1)
            if molecules:
                mol = molecules[0]
                cid = mol.get("molecule_chembl_id", "")
                mechs, indics = [], []
                if cid:
                    mechs = await adapter.get_mechanisms(cid)
                    indics = await adapter.get_indications(cid)
                results.append((d, {"chembl_id": cid, "mechanisms": mechs, "indications": indics}))
        except Exception as exc:
            logger.debug("ChEMBL lookup failed: %s", exc)
    return _fmt_drug_mechanisms(results)


async def _call_openfda(adapter, drugs: List[str]) -> str:
    results = []
    for d in drugs[:_MAX_DRUGS]:
        try:
            query = f'patient.drug.openfda.generic_name:"{d}"'
            resp = await adapter.count_adverse_events(query, limit=5)
            events = resp.get("results", []) if resp else []
            results.append((d, events))
        except Exception as exc:
            logger.debug("OpenFDA lookup failed: %s", exc)
    return _fmt_adverse_events(results)


async def _call_pubchem(adapter, drugs: List[str]) -> str:
    results = []
    for d in drugs[:_MAX_DRUGS]:
        try:
            props = await adapter.get_compound_properties(d)
            if props:
                results.append((d, props))
        except Exception as exc:
            logger.debug("PubChem lookup failed: %s", exc)
    return _fmt_compound_properties(results)


async def _call_reactome(adapter, pathway_ids: List[str]) -> str:
    results = []
    for pid in pathway_ids[:_MAX_PATHWAYS]:
        try:
            data = await adapter.get_pathway(pid)
            if data:
                results.append((pid, data))
        except Exception as exc:
            logger.debug("Reactome lookup failed: %s", exc)
    return _fmt_pathway_detail(results)


async def _call_kegg(adapter, pathway_ids: List[str]) -> str:
    results = []
    for pid in pathway_ids[:_MAX_PATHWAYS]:
        try:
            data = await adapter.get_pathway(pid.replace("hsa", ""))
            if data:
                results.append((pid, data))
        except Exception as exc:
            logger.debug("KEGG lookup failed: %s", exc)
    return _fmt_pathway_detail(results)


async def _call_clinical_trials(adapter, diseases: List[str]) -> str:
    if not diseases:
        return ""
    try:
        studies = await adapter.search_all(
            diseases[0],
            max_results=10,
            filter_overall_status=["RECRUITING"],
        )
        return _fmt_clinical_trials(studies)
    except Exception as exc:
        logger.debug("ClinicalTrials lookup failed: %s", exc)
        return ""


# Dispatch: adapter_name → (call_function, entity_key)
_ADAPTER_CALLS = {
    "ensembl":         (_call_ensembl, "genes"),
    "string":          (_call_string, "genes"),
    "dgidb":           (_call_dgidb, "genes"),
    "chembl":          (_call_chembl, "drugs"),
    "openfda":         (_call_openfda, "drugs"),
    "pubchem":         (_call_pubchem, "drugs"),
    "reactome":        (_call_reactome, "pathways"),
    "kegg":            (_call_kegg, "pathways"),
    "clinical_trials": (_call_clinical_trials, "disease"),
}

# Human-readable section labels for enrichment results
_ADAPTER_LABELS = {
    "ensembl":         "Gene Annotations (Ensembl)",
    "string":          "Protein-Protein Interactions (STRING)",
    "dgidb":           "Gene-Drug Interactions (DGIdb)",
    "chembl":          "Drug Mechanisms (ChEMBL)",
    "openfda":         "Adverse Events (FDA FAERS)",
    "pubchem":         "Compound Properties (PubChem)",
    "reactome":        "Pathway Details (Reactome)",
    "kegg":            "Pathway Details (KEGG)",
    "clinical_trials": "Active Clinical Trials",
}


# ── Main entry point ──────────────────────────────────────────────────────

async def enrich_for_response(
    query: str,
    csvs_by_type: Dict[str, pd.DataFrame],
    disease: str = "",
    timeout: float = 15.0,
) -> Dict[str, str]:
    """Query-aware API enrichment for the response path.

    Returns a dict of {source_label: markdown_table} ready to pass as
    enrichment_data to prompt builders. Empty dict if no entities found
    or all adapters fail.
    """
    entities = extract_entities(query, csvs_by_type, disease)
    if not entities:
        return {}

    # Determine which adapters to call based on entity types present
    needed_adapters: set = set()
    for entity_type in entities:
        needed_adapters.update(_ENTITY_ADAPTER_MAP.get(entity_type, []))

    if not needed_adapters:
        return {}

    # Initialize only the required adapters
    from ..api_adapters import APIConfig

    try:
        api_config = APIConfig.from_env()
    except Exception as exc:
        logger.debug("APIConfig.from_env() failed: %s — skipping enrichment", exc)
        return {}

    adapters = await _get_adapters(needed_adapters, api_config)

    if not adapters:
        return {}

    # Build async tasks for each adapter
    tasks: Dict[str, asyncio.Task] = {}
    for adapter_name, adapter_inst in adapters.items():
        call_fn, entity_key = _ADAPTER_CALLS.get(adapter_name, (None, None))
        entity_list = entities.get(entity_key, [])
        if call_fn and entity_list:
            tasks[adapter_name] = asyncio.ensure_future(call_fn(adapter_inst, entity_list))

    enrichment: Dict[str, str] = {}
    try:
        if tasks:
            done, _ = await asyncio.wait(
                tasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED,
            )
            for adapter_name, task in tasks.items():
                if task in done and not task.cancelled() and task.exception() is None:
                    result = task.result()
                    if result and result.strip():
                        label = _ADAPTER_LABELS.get(adapter_name, adapter_name)
                        enrichment[label] = result
                elif task not in done:
                    task.cancel()
                    logger.debug("Adapter %s timed out after %.0fs", adapter_name, timeout)
            pending = [t for t in tasks.values() if t not in done]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    except Exception as exc:
        logger.warning("Enrichment gather failed: %s", exc)

    return enrichment
