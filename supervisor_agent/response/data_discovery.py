"""CSV discovery, scoring, and query-aware extraction for response synthesis."""

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(100 * 1024 * 1024)))

# Files known to be high-value summaries across agents
_HIGH_VALUE_RE = re.compile(
    r'(Final_Gene_Priorities|pathway_consolidation|Final_GeneDrug_Pairs|'
    r'CIBERSORT_results|_DEGs|prioritized)', re.IGNORECASE,
)

# Natural-language → CSV column mapping
_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "disorder_score":       ["disorder score", "disorder", "disease score"],
    "gene_score":           ["gene score"],
    "composite_score":      ["composite score", "composite"],
    "non_cgc_composite_score": ["non cgc composite", "non-cgc composite"],
    "ppi_score":            ["ppi score", "ppi", "protein interaction"],
    "ppi_degree":           ["ppi degree", "interaction degree"],
    "tier":                 ["tier"],
    "non_cgc_tier":         ["non cgc tier", "non-cgc tier"],
    "rank":                 ["rank", "ranked"],
    "score":                ["score", "overall score"],
    "confidence":           ["confidence"],
    "patient_lfc_mean":     ["fold change", "log2fc", "lfc", "expression"],
    "dcs":                  ["dcs"],
    "csc":                  ["csc"],
    "cfc":                  ["cfc"],
    "jl_score":             ["jl score", "jl_score"],
    "novel":                ["novel"],
    "status":               ["status", "unique", "shared"],
    "trend_consensus":      ["trend", "consensus"],
    "disorder_type":        ["disorder type", "disease type"],
}

_CONTEXT_COLS = ["Gene", "Tier", "Rank"]

_FILTER_RE = re.compile(
    r'(?:where|with|having|whose)?\s*'
    r'(?P<col>[a-z_ ]+?)\s*'
    r'(?P<op>above|below|greater than|less than|over|under|[><=!]+)\s*'
    r'(?P<val>[\d.]+)',
    re.IGNORECASE,
)
_TIER_FILTER_RE = re.compile(r'\btier\s*(\d)\b', re.IGNORECASE)
_ONLY_RE = re.compile(r'\bonly\b', re.IGNORECASE)


def read_csv_head(path: str, max_rows: int, columns: Optional[List[str]] = None) -> str:
    """Read a CSV's header + top N rows as a pipe-delimited table string."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return ""
            cols = [c for c in columns if c in reader.fieldnames] if columns else None
            use_cols = cols if cols else list(reader.fieldnames)
            lines = [" | ".join(use_cols)]
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                lines.append(" | ".join(str(row.get(c, "")) for c in use_cols))
            return "\n".join(lines)
    except Exception:
        return ""


def resolve_column(query_lower: str, csv_columns: List[str]) -> Optional[str]:
    """Match natural-language references in the query to an actual CSV column."""
    col_lower_map = {c.lower(): c for c in csv_columns}
    best, best_len = None, 0
    for canonical, synonyms in _COLUMN_SYNONYMS.items():
        for syn in synonyms:
            if syn in query_lower and len(syn) > best_len:
                for csv_col_lower, csv_col in col_lower_map.items():
                    if canonical == csv_col_lower or canonical.replace("_", "") == csv_col_lower.replace("_", ""):
                        best, best_len = csv_col, len(syn)
    return best


def query_csv_data(csv_path: str, user_query: str, top_n: int) -> str:
    """Pandas-driven query layer: detect column + operation from user query,
    return a focused, pipe-delimited table the LLM can reason over."""
    try:
        if Path(csv_path).stat().st_size > _MAX_CSV_BYTES:
            logger.debug("CSV %s exceeds size limit — skipping", csv_path)
            return ""
        df = pd.read_csv(csv_path, nrows=50_000, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return ""
    if df.empty:
        return ""

    q = user_query.lower()
    target_col = resolve_column(q, list(df.columns))

    # Tier filter: "tier 1 genes only", "show tier 2"
    tier_m = _TIER_FILTER_RE.search(q)
    if tier_m and _ONLY_RE.search(q):
        tier_val = f"Tier {tier_m.group(1)}"
        tier_col = next((c for c in df.columns if c.lower() == "tier"), None)
        if tier_col:
            df = df[df[tier_col].astype(str).str.strip() == tier_val]

    # Numeric filter: "disorder score above 10", "composite > 0.5"
    filter_m = _FILTER_RE.search(q)
    if filter_m and target_col and target_col in df.columns:
        op_str, val = filter_m.group("op").lower(), float(filter_m.group("val"))
        numeric = pd.to_numeric(df[target_col], errors="coerce")
        if op_str in ("above", "greater than", "over", ">", ">="):
            df = df[numeric >= val]
        elif op_str in ("below", "less than", "under", "<", "<="):
            df = df[numeric <= val]

    # Sort by target column (descending for scores, ascending for rank)
    if target_col and target_col in df.columns:
        numeric = pd.to_numeric(df[target_col], errors="coerce")
        ascending = target_col.lower() in ("rank",)
        df = df.assign(_sort_key=numeric).sort_values(
            "_sort_key", ascending=ascending, na_position="last"
        ).drop(columns="_sort_key")

    df = df.head(top_n)
    if df.empty:
        return ""

    # Select columns: Gene + target + context — keep it focused
    keep = list(dict.fromkeys(
        [c for c in _CONTEXT_COLS if c in df.columns]
        + ([target_col] if target_col and target_col in df.columns else [])
    ))
    if not keep:
        keep = list(df.columns[:8])
    extras = [c for c in df.columns if c.lower().endswith(("_score", "score")) and c not in keep]
    keep.extend(extras[:3])

    df = df[keep]
    lines = [" | ".join(df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)


def discover_relevant_csvs(
    workflow_outputs: Dict[str, Any], user_query: str,
    uploaded_files: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, float]]:
    """Find CSV files across all agent output dirs, scored by query relevance."""
    query_words = set(re.findall(r'[a-z]+', user_query.lower()))
    csv_paths: set[str] = set()

    for key, val in workflow_outputs.items():
        if not isinstance(val, str):
            continue
        if val.endswith(".csv") and Path(val).is_file():
            csv_paths.add(val)
        elif key.endswith(("_output_dir", "_base_dir")) and Path(val).is_dir():
            csv_paths.update(str(p) for p in Path(val).glob("**/*.csv"))

    if uploaded_files:
        for fpath in uploaded_files.values():
            if isinstance(fpath, str) and fpath.endswith(".csv") and Path(fpath).is_file():
                csv_paths.add(fpath)

    scored: List[Tuple[str, float]] = []
    for csv_path in csv_paths:
        p = Path(csv_path)
        try:
            size = p.stat().st_size
        except OSError:
            continue

        score = 0.0
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), [])
            header_lower = {c.lower() for c in header}
            score += sum(1 for w in query_words if any(w in col for col in header_lower))
        except Exception:
            header = []

        if _HIGH_VALUE_RE.search(p.name):
            score += 3
        if size > 10_000_000:
            score -= 5
        depth = len(p.parts)
        if depth > 5:
            score -= 0.5 * (depth - 5)

        scored.append((csv_path, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:8]


def classify_csv_type(csv_path: str) -> str:
    """Classify a CSV as genes/pathways/drugs based on filename and columns."""
    name_lower = Path(csv_path).name.lower()
    if "drug" in name_lower:
        return "drugs"
    if "pathway" in name_lower:
        return "pathways"
    if any(kw in name_lower for kw in ("gene", "priorit", "deg", "ranked")):
        return "genes"
    try:
        cols = set(pd.read_csv(csv_path, nrows=0).columns.str.lower())
    except Exception:
        return "unknown"
    if "gene" in cols or "gene_score" in cols:
        return "genes"
    if "pathway" in cols or "term" in cols:
        return "pathways"
    if "drug" in cols or "compound" in cols:
        return "drugs"
    return "unknown"


def collect_output_summaries(
    workflow_outputs: Dict[str, Any],
    user_query: str,
    top_n: int,
    uploaded_files: Optional[Dict[str, str]] = None,
    budget: int = 8000,
) -> Dict[str, str]:
    """Read relevant CSV head data from all completed agent outputs."""
    summaries: Dict[str, str] = {}
    total_chars = 0
    char_budget = budget

    cohort_text = workflow_outputs.get("cohort_summary_text", "")
    if cohort_text:
        summaries["cohort_summary"] = cohort_text[:500]
        total_chars += len(summaries["cohort_summary"])

    for csv_path, _score in discover_relevant_csvs(workflow_outputs, user_query, uploaded_files):
        if total_chars >= char_budget:
            break
        data = query_csv_data(csv_path, user_query, top_n)
        if not data:
            data = read_csv_head(csv_path, top_n)
        if data:
            key = Path(csv_path).name
            summaries[key] = data
            total_chars += len(data)

    return summaries
