"""
Utility functions for the Supervisor Agent.

This module provides helper functions for:
- File type detection
- Output directory management
- Execution chain building
- File collection
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_file_type(file_path: str) -> str:
    """
    Detect the type of input file based on its columns or content.
    
    Detection order (most specific first):
    1. json_data - JSON file with pathway or gene data
    2. gene_list - Plain text file with gene names (one per line)
    3. prioritized_genes - CSV with gene prioritization columns
    4. deg_results - CSV with DEG analysis columns
    5. raw_counts - CSV with expression counts matrix
    6. metadata - CSV with sample metadata
    
    Returns:
        str: One of 'raw_counts', 'deg_results', 'prioritized_genes', 
             'metadata', 'json_data', 'gene_list', or 'unknown'
    """
    import pandas as pd
    
    path = Path(file_path)
    if not path.exists():
        return "unknown"
    
    ext = path.suffix.lower()
    
    # Handle JSON files
    if ext == ".json":
        try:
            import json
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if any(k in data for k in ["pathways", "genes", "gene_symbol", "enrichment"]):
                        return "json_data"
        except Exception:
            pass
        return "json_data"
    
    # Handle plain text files (gene lists)
    if ext == ".txt":
        try:
            with open(path, "r") as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                # Check if it looks like a gene list (short lines, no tabs/commas)
                if all(len(line.split()) <= 2 for line in first_lines if line):
                    return "gene_list"
        except Exception:
            pass
        return "gene_list"
    
    # Handle CSV/TSV files
    if ext not in [".csv", ".tsv"]:
        return "unknown"
    
    try:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, nrows=10, sep=sep)
        columns_lower = [c.lower() for c in df.columns]
        
        # Check for prioritized genes (has priority/score columns)
        priority_indicators = ['priority', 'priority_score', 'combined_score', 
                              'disease_score', 'rank', 'final_score']
        if any(ind in columns_lower for ind in priority_indicators):
            # Also check for gene column
            gene_cols = ['gene', 'gene_symbol', 'gene_name', 'symbol', 'gene_id']
            if any(gc in columns_lower for gc in gene_cols):
                return "prioritized_genes"
        
        # Check for DEG results (has log2fc and pvalue/padj)
        deg_indicators = ['log2foldchange', 'log2fc', 'logfc', 'log_fc', 'foldchange']
        pval_indicators = ['pvalue', 'p_value', 'padj', 'adj_p', 'fdr', 'qvalue']
        if any(ind in columns_lower for ind in deg_indicators):
            if any(pv in columns_lower for pv in pval_indicators):
                return "deg_results"
        
        # Check for raw counts (numeric matrix with gene column)
        gene_cols = ['gene', 'gene_id', 'gene_symbol', 'ensembl', 'ensembl_id']
        has_gene_col = any(gc in columns_lower for gc in gene_cols)
        
        # Check if most columns are numeric (counts)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= len(df.columns) * 0.7 and has_gene_col:
            return "raw_counts"
        
        # Check for metadata (has sample/condition columns)
        meta_indicators = ['sample', 'sample_id', 'condition', 'group', 'treatment',
                          'disease', 'control', 'patient', 'subject']
        if any(mi in columns_lower for mi in meta_indicators):
            return "metadata"
        
        # If has gene column but couldn't determine type, treat as potential counts
        if has_gene_col:
            return "raw_counts"
        
    except Exception as e:
        logger.warning(f"Error detecting file type for {path}: {e}")
    
    return "unknown"


def get_agent_output_dir(agent_name: str = "") -> Path:
    """
    Get the output directory for an agent.
    
    Args:
        agent_name: Optional agent name for subdirectory
        
    Returns:
        Path to the output directory
    """
    base_dir = Path(__file__).parent / "outputs"
    if agent_name:
        output_dir = base_dir / agent_name
    else:
        output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_execution_chain(
    target_agent: 'AgentType',
    available_keys: Set[str],
    detected_file_type: Optional[str] = None
) -> Tuple[List['AgentType'], List['AgentType'], str]:
    """
    Build an execution chain for the target agent based on available inputs.
    
    This function determines what agents need to run in sequence to satisfy
    the target agent's requirements. It handles:
    - Direct execution when inputs are satisfied
    - Prepending dependency agents when inputs are missing
    - Skipping agents when their outputs already exist
    
    Args:
        target_agent: The AgentType that was requested
        available_keys: Set of input keys already available in workflow_state
        detected_file_type: Optional detected type of uploaded file
        
    Returns:
        Tuple of:
        - agents_to_run: List of AgentTypes to execute in order
        - agents_skipped: List of AgentTypes that were skipped
        - skip_info: Human-readable explanation of skips
    """
    from .agent_registry import AgentType, AGENT_REGISTRY, PIPELINE_ORDER
    
    agents_to_run = []
    agents_skipped = []
    skip_info_parts = []
    
    # Get target agent info
    target_info = AGENT_REGISTRY.get(target_agent)
    if not target_info:
        return [target_agent], [], ""
    
    # Check what the target agent needs
    required_inputs = {inp.name for inp in target_info.required_inputs}
    
    # Determine starting point based on detected file type
    start_agent = None
    if detected_file_type == "raw_counts":
        start_agent = AgentType.DEG_ANALYSIS
        skip_info_parts.append("📊 Raw counts detected → starting with DEG Analysis")
    elif detected_file_type == "deg_results":
        start_agent = AgentType.GENE_PRIORITIZATION
        skip_info_parts.append("📈 DEG results detected → starting with Gene Prioritization")
    elif detected_file_type == "prioritized_genes":
        start_agent = AgentType.PATHWAY_ENRICHMENT
        skip_info_parts.append("🎯 Prioritized genes detected → starting with Pathway Enrichment")
    
    # Determine the pipeline segment needed
    if target_agent in PIPELINE_ORDER:
        target_idx = PIPELINE_ORDER.index(target_agent)
        
        # Find starting index based on detected file or available inputs
        start_idx = 0
        if start_agent and start_agent in PIPELINE_ORDER:
            start_idx = PIPELINE_ORDER.index(start_agent)
        
        # Add agents from start to target
        for idx in range(start_idx, target_idx + 1):
            agent = PIPELINE_ORDER[idx]
            agent_info = AGENT_REGISTRY.get(agent)
            
            if agent_info:
                # Check if this agent's outputs are already available
                output_keys = {out.state_key or out.name for out in agent_info.outputs if out.state_key}
                if output_keys and output_keys.issubset(available_keys) and agent != target_agent:
                    agents_skipped.append(agent)
                    skip_info_parts.append(f"⏭️ Skipping {agent_info.display_name} (outputs already available)")
                else:
                    agents_to_run.append(agent)
    else:
        # Non-pipeline agent - just run it directly
        agents_to_run = [target_agent]
    
    skip_info = "\n".join(skip_info_parts)
    return agents_to_run, agents_skipped, skip_info


def collect_generated_files(output_dir: str) -> List[str]:
    """
    Collect file paths of generated files in an output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        List of file path strings
    """
    if not output_dir:
        return []
    
    path = Path(output_dir)
    if not path.exists():
        return []
    
    files = []
    
    # Common output file extensions to look for
    output_extensions = {'.csv', '.tsv', '.xlsx', '.json', '.png', '.pdf', '.html', '.txt'}
    
    try:
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in output_extensions:
                files.append(str(file_path))
        
        # Also check subdirectories (one level deep)
        for subdir in path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in output_extensions:
                        files.append(str(file_path))
    except Exception as e:
        logger.warning(f"Error collecting files from {output_dir}: {e}")
    
    return files


def extract_patient_prefix(filename: str) -> str:
    """
    Extract patient/analysis prefix from filename.
    
    Examples:
        'RW-20251119_Sarcoidosis.csv' -> 'RW-20251119'
        'MC-20250909_Lupus_counts.csv' -> 'MC-20250909'
        'patient_data.csv' -> 'patient_data'
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try common patterns: PREFIX_Disease or PREFIX-Disease
    # Match patterns like 'RW-20251119', 'MC-20250909', etc.
    match = re.match(r'^([A-Z]{2,3}[-_]\d{6,8})', name)
    if match:
        return match.group(1)
    
    # Fallback: take first part before underscore
    parts = re.split(r'[_\-]', name)
    if parts:
        return parts[0]
    
    return name
