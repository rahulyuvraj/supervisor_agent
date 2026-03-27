"""Deterministic query functions for extracting structured data from DataFrames.

All functions are pure — they accept DataFrames and return pipe-delimited strings.
The regex_fast_path orchestrator matches user intent to the right query function.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Intent patterns ──

_INTENT_TOP_GENES = re.compile(
    r'\b(?:top|best|highest|leading)\s*(\d+)?\s*(?:genes?|markers?|candidates?)\b', re.IGNORECASE,
)
_INTENT_TOP_PATHWAYS = re.compile(
    r'\b(?:top|best|highest|leading)\s*(\d+)?\s*(?:pathways?|signaling|biological\s+process)\b',
    re.IGNORECASE,
)
_INTENT_TOP_DRUGS = re.compile(
    r'\b(?:top|best|highest|leading)\s*(\d+)?\s*(?:drugs?|compounds?|therapeutics?|molecules?)\b',
    re.IGNORECASE,
)
_INTENT_GENE_DETAIL = re.compile(
    r'\b(?:tell me about|details? (?:of|for|on|about)|info (?:on|about)|what (?:is|about))\s+([A-Za-z][A-Za-z0-9]{1,12})\b',
    re.IGNORECASE,
)
_INTENT_SUMMARY = re.compile(
    r'\b(summar|overview|recap|findings|results|what did you find|what was found)\b',
    re.IGNORECASE,
)

# ── Column alias resolution ──

_GENE_SORT_ALIASES: Dict[str, List[str]] = {
    "composite_score": ["composite score", "composite", "overall score", "score"],
    "disorder_score": ["disorder score", "disorder", "disease score"],
    "gene_score": ["gene score"],
    "rank": ["rank", "ranked"],
}
_PATHWAY_SORT_ALIASES: Dict[str, List[str]] = {
    "p_value": ["p-value", "pvalue", "p value", "significance"],
    "combined_score": ["combined score", "score"],
    "overlap_count": ["overlap", "gene count"],
}
_DRUG_SORT_ALIASES: Dict[str, List[str]] = {
    "score": ["score", "drug score", "overall score"],
    "confidence": ["confidence"],
    "rank": ["rank"],
}


def _resolve_col(df: pd.DataFrame, aliases: Dict[str, List[str]],
                 query_lower: str = "") -> Optional[str]:
    """Resolve a column from alias dict, optionally matching query terms.
    Only substring-matches aliases with length > 4 to avoid greedy false matches."""
    col_lower_map = {c.lower().replace("_", " "): c for c in df.columns}
    if query_lower:
        best, best_len = None, 0
        for canonical, synonyms in aliases.items():
            for syn in synonyms:
                if len(syn) > 4 and syn in query_lower and len(syn) > best_len:
                    norm_key = canonical.lower().replace("_", " ")
                    if norm_key in col_lower_map:
                        best, best_len = col_lower_map[norm_key], len(syn)
        if best:
            return best
    for canonical in aliases:
        norm_key = canonical.lower().replace("_", " ")
        if norm_key in col_lower_map:
            return col_lower_map[norm_key]
    return None


_ID_COLS = ["Gene", "Pathway", "Drug", "Term", "Compound", "Name"]
_META_COLS = ["Tier", "Rank", "Status"]
_MAX_TABLE_COLS = 8


def _select_columns(df: pd.DataFrame, sort_col: str = "") -> pd.DataFrame:
    """Pick the most informative columns for document-ready output."""
    if len(df.columns) <= _MAX_TABLE_COLS:
        return df
    col_lower = {c.lower(): c for c in df.columns}
    keep: list[str] = []
    for label in _ID_COLS + _META_COLS:
        actual = col_lower.get(label.lower())
        if actual and actual not in keep:
            keep.append(actual)
    if sort_col and sort_col in df.columns and sort_col not in keep:
        keep.append(sort_col)
    # Fill remaining budget with score columns
    for c in df.columns:
        if len(keep) >= _MAX_TABLE_COLS:
            break
        if c not in keep and c.lower().endswith(("_score", "score")):
            keep.append(c)
    # Still room? add remaining columns in order
    for c in df.columns:
        if len(keep) >= _MAX_TABLE_COLS:
            break
        if c not in keep:
            keep.append(c)
    return df[keep]


def df_to_table(df: pd.DataFrame, sort_col: str = "") -> str:
    """Convert DataFrame to pipe-delimited text table with smart column selection."""
    if df.empty:
        return ""
    df = _select_columns(df, sort_col)
    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)


# ── 5 Deterministic query functions ──

def get_top_genes(df: pd.DataFrame, n: int, query: str) -> str:
    sort_col = _resolve_col(df, _GENE_SORT_ALIASES, query.lower())
    if not sort_col:
        sort_col = _resolve_col(df, _GENE_SORT_ALIASES)
    if sort_col and sort_col in df.columns:
        ascending = sort_col.lower() in ("rank",)
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")
    return df_to_table(df.head(n), sort_col or "")


def get_top_pathways(df: pd.DataFrame, n: int, query: str) -> str:
    sort_col = _resolve_col(df, _PATHWAY_SORT_ALIASES, query.lower())
    if not sort_col:
        sort_col = _resolve_col(df, _PATHWAY_SORT_ALIASES)
    if sort_col and sort_col in df.columns:
        ascending = sort_col.lower() in ("p_value", "p value", "pvalue")
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")
    return df_to_table(df.head(n), sort_col or "")


def get_top_drugs(df: pd.DataFrame, n: int, query: str) -> str:
    sort_col = _resolve_col(df, _DRUG_SORT_ALIASES, query.lower())
    if not sort_col:
        sort_col = _resolve_col(df, _DRUG_SORT_ALIASES)
    if sort_col and sort_col in df.columns:
        ascending = sort_col.lower() in ("rank",)
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")
    return df_to_table(df.head(n), sort_col or "")


def get_gene_detail(df: pd.DataFrame, gene_name: str) -> str:
    gene_col = next((c for c in df.columns if c.lower() == "gene"), None)
    if not gene_col:
        return ""
    mask = df[gene_col].astype(str).str.upper() == gene_name.upper()
    matched = df[mask]
    if matched.empty:
        return f"Gene '{gene_name}' not found in the dataset."
    return df_to_table(matched)


def get_full_summary(df: pd.DataFrame, n: int) -> str:
    return df_to_table(df.head(n))


# ── Intent-to-function dispatcher ──

# Maps (intent_regex, csv_type) → (handler, needs_n_arg)
_DISPATCH = [
    (_INTENT_GENE_DETAIL, "genes",    get_gene_detail,  False),
    (_INTENT_TOP_GENES,   "genes",    get_top_genes,    True),
    (_INTENT_TOP_PATHWAYS,"pathways", get_top_pathways,  True),
    (_INTENT_TOP_DRUGS,   "drugs",    get_top_drugs,     True),
]


def regex_fast_path(
    query: str,
    csvs_by_type: Dict[str, pd.DataFrame],
    top_n: int = 10,
    best_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Try to answer common queries deterministically via regex.

    *csvs_by_type*: {"genes": DataFrame, "pathways": DataFrame, ...}
    *best_df*: fallback DataFrame for summary intent.
    Returns pipe-delimited text or None if no regex matches.
    """
    # Gene detail: "tell me about BRCA1"
    m = _INTENT_GENE_DETAIL.search(query)
    if m:
        gene_name = m.group(1).upper()
        df = csvs_by_type.get("genes")
        if df is not None:
            return get_gene_detail(df, gene_name)

    # Top-N dispatches
    for pattern, csv_type, handler, _ in _DISPATCH[1:]:  # skip gene_detail (handled above)
        m = pattern.search(query)
        if m:
            n = int(m.group(1)) if m.group(1) else top_n
            df = csvs_by_type.get(csv_type)
            if df is not None:
                return handler(df, n, query)

    # Summary
    if _INTENT_SUMMARY.search(query):
        df = best_df if best_df is not None else next(iter(csvs_by_type.values()), None)
        if df is not None:
            return get_full_summary(df, top_n)

    return None
