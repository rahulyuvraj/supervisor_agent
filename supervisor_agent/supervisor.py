"""
Supervisor Agent - Main orchestrator for routing and executing agent workflows

This module provides:
1. The main SupervisorAgent class that handles user interactions
2. Real-time status updates and logging
3. Agent execution with progress tracking
4. Detailed reasoning and explanations for users
"""

import os
import asyncio
import logging
import re
import shlex
import time
import shutil
from typing import Dict, Any, Optional, AsyncGenerator, Callable, List, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv
# Load from project root .env (not local supervisor_agent/.env)
project_root = Path(__file__).resolve().parent.parent.parent  # supervisor_agent -> agentic_ai_wf -> agenticaib
load_dotenv(project_root / ".env")

_USE_LANGGRAPH = os.environ.get("USE_LANGGRAPH_SUPERVISOR", "false").lower() in ("true", "1", "yes")

from .agent_registry import AGENT_REGISTRY, AgentType, AgentInfo, InputRequirement, get_agent_by_name, PIPELINE_ORDER, FILE_TYPE_TO_INPUT_KEY, MULTIOMICS_LAYER_NAMES
from .router import IntentRouter, RoutingDecision
from .state import (
    ConversationState, SessionManager, Message,
    MessageRole, MessageType
)
from agentic_ai_wf.crispr_screening_extraction import extract_crispr_screening_params_sync
from agentic_ai_wf.crispr_targeted_extraction import extract_crispr_targeted_params_sync

logger = logging.getLogger(__name__)

_FASTQ_SUFFIXES = ('.fastq', '.fq', '.fastq.gz', '.fq.gz')
_TABULAR_SUFFIXES = ('.csv', '.csv.gz', '.tsv', '.tsv.gz', '.txt', '.txt.gz')
_PATH_INPUT_KEYS = frozenset({
    'fastq_input_dir',
    'crispr_10x_input_dir',
    'crispr_screening_input_dir',
    'crispr_targeted_input_dir',
})
_MESSAGE_PATH_RE = re.compile(r'([~/.][^\s,;]+(?:\s[^\s,;]+)*)')
_PROTOSPACER_RE = re.compile(r'\bprotospacer\b\s*(?:is|=|:)?\s*([ACGTU]{18,40})\b', re.IGNORECASE)
_TARGET_GENE_RE = re.compile(r'\btarget(?:[_\s-]?gene)?\b\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)')
_REGION_RE = re.compile(r'\b((?:chr)?[A-Za-z0-9._-]+:\d+-\d+)\b', re.IGNORECASE)
_REFERENCE_SEQ_RE = re.compile(r'\breference(?:[_\s-]?seq(?:uence)?)?\b\s*(?:is|=|:)?\s*([ACGTU]{30,})\b', re.IGNORECASE)
_CRISPR_AGENT_TYPES = frozenset({
    AgentType.CRISPR_PERTURB_SEQ,
    AgentType.CRISPR_SCREENING,
    AgentType.CRISPR_TARGETED,
})
_CRISPR_FILE_TYPES = frozenset({'crispr_10x_data', 'crispr_count_table', 'crispr_fastq_data'})


def _iter_existing_message_paths(message: str) -> List[str]:
    paths: List[str] = []
    seen = set()

    for token in shlex.split(message):
        if token.startswith(('~', '/', './', '../')) and token not in seen and Path(token).expanduser().exists():
            seen.add(token)
            paths.append(token)

    for match in _MESSAGE_PATH_RE.findall(message):
        candidate = match.strip('"\'')
        if candidate in seen:
            continue
        if Path(candidate).expanduser().exists():
            seen.add(candidate)
            paths.append(candidate)

    return paths


def _is_fastq_name(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(suffix) for suffix in _FASTQ_SUFFIXES)


def _strip_known_suffix(name: str, suffixes: tuple[str, ...]) -> str:
    name = name.lower()
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _paired_fastq_roots(files: List[Path]) -> set[str]:
    r1_roots, r2_roots = set(), set()
    patterns = (
        (re.compile(r'(.+?)(?:_r?1)$', re.IGNORECASE), r1_roots),
        (re.compile(r'(.+?)(?:_r?2)$', re.IGNORECASE), r2_roots),
        (re.compile(r'(.+?)(?:[._-]1)$', re.IGNORECASE), r1_roots),
        (re.compile(r'(.+?)(?:[._-]2)$', re.IGNORECASE), r2_roots),
    )

    for file_path in files:
        stem = _strip_known_suffix(file_path.name, _FASTQ_SUFFIXES)
        for pattern, bucket in patterns:
            match = pattern.match(stem)
            if match:
                bucket.add(match.group(1))
                break

    return r1_roots & r2_roots


def _inspect_directory(path: Union[str, Path]) -> Dict[str, Any]:
    directory = Path(path).expanduser()
    files = [p for p in directory.rglob('*') if p.is_file()]
    names = [p.name.lower() for p in files]
    tabular_files = [
        p for p in files
        if any(p.name.lower().endswith(suffix) for suffix in _TABULAR_SUFFIXES)
    ]
    paired_roots = _paired_fastq_roots(files)
    screening_context = any(
        any(token in p.stem.lower() for token in ('contrast', 'design', 'sample', 'library', 'essential', 'nonessential'))
        for p in tabular_files
    )

    return {
        'files': files,
        'tabular_files': tabular_files,
        'fastq_count': sum(1 for p in files if _is_fastq_name(p.name)),
        'paired_fastq_roots': paired_roots,
        'has_10x_triplet': (
            any('barcode' in name for name in names)
            and any(token in name for name in names for token in ('feature', 'gene'))
            and any('matrix' in name for name in names)
        ),
        'has_screening_counts': any(
            _detect_file_type(str(p)) == 'crispr_count_table'
            for p in tabular_files[:12]
        ),
        'has_screening_context': screening_context,
    }


def _match_path_to_inputs(path: Union[str, Path], required_input_names: Optional[set[str]] = None) -> Dict[str, str]:
    required = required_input_names or set()
    resolved = str(Path(path).expanduser().resolve())
    target = Path(resolved)

    if target.is_file():
        file_type = _detect_file_type(resolved)
        keys = [key for key in FILE_TYPE_TO_INPUT_KEY.get(file_type, []) if not required or key in required]
        return {key: resolved for key in keys}

    if not target.is_dir():
        return {}

    detected = _detect_file_type(resolved)
    if detected in FILE_TYPE_TO_INPUT_KEY:
        keys = [key for key in FILE_TYPE_TO_INPUT_KEY[detected] if not required or key in required]
        if keys:
            return {key: resolved for key in keys}

    inspection = _inspect_directory(target)
    if inspection['paired_fastq_roots']:
        if 'crispr_targeted_input_dir' in required and not inspection['has_screening_context']:
            return {'crispr_targeted_input_dir': resolved}
        if 'crispr_screening_input_dir' in required and inspection['has_screening_context']:
            return {'crispr_screening_input_dir': resolved}
        if 'fastq_input_dir' in required:
            return {'fastq_input_dir': resolved}

    return {}


def _extract_contextual_inputs(message: str, required_input_names: Optional[set[str]] = None) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    required = required_input_names or set()
    message_lower = message.lower().strip()

    targeted_keys = {
        'project_id', 'target_gene', 'protospacer', 'region', 'reference_seq',
        'extract_metadata', 'download_fastq', 'crispr_targeted_input_dir', 'crispr_targeted_source',
    }
    if not required or required & targeted_keys:
        targeted = extract_crispr_targeted_params_sync(message)
        if targeted.get('project_id'):
            extracted['project_id'] = targeted['project_id']
            extracted['extract_metadata'] = targeted.get('extract_metadata', True)
            extracted['download_fastq'] = targeted.get('download_fastq', True)
        if targeted.get('target_gene'):
            extracted['target_gene'] = targeted['target_gene']
        if targeted.get('region'):
            extracted['region'] = targeted['region']
        if targeted.get('reference_seq'):
            extracted['reference_seq'] = targeted['reference_seq']
        if targeted.get('protospacer'):
            extracted['protospacer'] = targeted['protospacer']

    screening_keys = {'modes', 'crispr_screening_input_dir', 'crispr_screening_source'}
    if not required or required & screening_keys:
        screening = extract_crispr_screening_params_sync(message)
        if screening.get('modes'):
            extracted['modes'] = screening['modes']

    disease_patterns = [
        r'disease\s+(?:is\s+)?["\']?([\w\s-]+)["\']?',
        r'disease[:\s]+["\']?([\w\s-]+)["\']?',
        r'for\s+([\w\s-]+?)\s+(?:analysis|study)',
        r'analyzing\s+([\w\s-]+)',
        r'condition[:\s]+["\']?([\w\s-]+)["\']?',
        r'^([\w\s-]+)$',
    ]
    for pattern in disease_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        value = match.group(1).strip()
        if value and value not in {'is', 'the', 'a', 'an', 'my', 'this', 'that', 'yes', 'no', 'ok', 'okay'}:
            extracted['disease_name'] = value.title()
            break

    for message_path in _iter_existing_message_paths(message):
        extracted.update(_match_path_to_inputs(message_path, required))

    if not required or 'protospacer' in required:
        match = _PROTOSPACER_RE.search(message)
        if match:
            extracted['protospacer'] = match.group(1).upper().replace('U', 'T')

    if not required or 'target_gene' in required:
        match = _TARGET_GENE_RE.search(message)
        if match:
            extracted['target_gene'] = match.group(1)

    if not required or 'region' in required:
        match = _REGION_RE.search(message)
        if match:
            extracted['region'] = match.group(1)

    if not required or 'reference_seq' in required:
        match = _REFERENCE_SEQ_RE.search(message)
        if match:
            extracted['reference_seq'] = match.group(1).upper().replace('U', 'T')

    return extracted


def _validate_required_input_paths(agent_info: AgentInfo, available_inputs: Dict[str, Any]) -> Optional[str]:
    for inp in agent_info.required_inputs:
        value = available_inputs.get(inp.name)
        if not value or not isinstance(value, str):
            continue

        target = Path(value).expanduser()
        if inp.name in _PATH_INPUT_KEYS:
            if not target.is_dir():
                return f"Required input '{inp.name}' must be a directory path: {value}"

            inspection = _inspect_directory(target)
            if inp.name == 'crispr_10x_input_dir' and not inspection['has_10x_triplet']:
                return "CRISPR perturb-seq input must contain matrix, barcode, and feature or gene files."
            if inp.name == 'crispr_screening_input_dir' and not (
                inspection['has_screening_counts']
                or (inspection['paired_fastq_roots'] and inspection['has_screening_context'])
            ):
                return "CRISPR screening input must contain either screening count data or FASTQs with matching screening metadata."
            if (
                inp.name == 'crispr_targeted_input_dir'
                and not available_inputs.get('project_id')
                and not inspection['paired_fastq_roots']
            ):
                return "CRISPR targeted input must contain paired FASTQ reads."
            if inp.name == 'fastq_input_dir' and inspection['fastq_count'] == 0:
                return "FASTQ input directory must contain sequencing reads."
        elif inp.is_file and not target.exists():
            return f"Required input '{inp.name}' not found at: {value}"

    return None


def _collect_uploaded_paths(state: ConversationState) -> List[Path]:
    return [Path(uploaded.filepath) for uploaded in state.uploaded_files.values() if uploaded.filepath]


def _supervisor_runtime_dir(session_id: str, name: str) -> Path:
    runtime_dir = project_root / "temp" / "supervisor_crispr" / session_id / name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _copy_runtime_files(file_paths: List[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in file_paths:
        if file_path.exists() and file_path.is_file():
            shutil.copy2(file_path, destination / file_path.name)


def _hydrate_crispr_inputs(agent_info: AgentInfo, available_inputs: Dict[str, Any], state: ConversationState) -> Dict[str, Any]:
    hydrated = dict(available_inputs)
    uploaded_paths = _collect_uploaded_paths(state)

    if agent_info.agent_type == AgentType.CRISPR_TARGETED:
        if hydrated.get('project_id') and not hydrated.get('crispr_targeted_input_dir'):
            runtime_dir = _supervisor_runtime_dir(state.session_id, 'crispr_targeted_input')
            (runtime_dir / 'fastq').mkdir(exist_ok=True)
            hydrated['crispr_targeted_input_dir'] = str(runtime_dir)
            hydrated.setdefault('extract_metadata', True)
            hydrated.setdefault('download_fastq', True)
        elif not hydrated.get('crispr_targeted_input_dir'):
            fastq_files = [path for path in uploaded_paths if _is_fastq_name(path.name)]
            if fastq_files:
                runtime_dir = _supervisor_runtime_dir(state.session_id, 'crispr_targeted_input')
                _copy_runtime_files(fastq_files, runtime_dir / 'fastq')
                hydrated['crispr_targeted_input_dir'] = str(runtime_dir)

    elif agent_info.agent_type == AgentType.CRISPR_SCREENING and not (
        hydrated.get('crispr_screening_input_dir') and Path(hydrated['crispr_screening_input_dir']).is_dir()
    ):
        screening_files = [
            path for path in uploaded_paths
            if (
                _detect_file_type(str(path)) == 'crispr_count_table'
                or _is_fastq_name(path.name)
                or any(token in path.stem.lower() for token in ('contrast', 'design', 'sample', 'library', 'essential', 'nonessential'))
            )
        ]
        if screening_files:
            runtime_dir = _supervisor_runtime_dir(state.session_id, 'crispr_screening_input')
            counts_dir = runtime_dir / 'counts'
            metadata_dir = runtime_dir / 'metadata'
            full_test_dir = runtime_dir / 'full_test'
            fastq_dir = runtime_dir / 'fastq'
            for path in screening_files:
                name_lower = path.name.lower()
                if _is_fastq_name(path.name):
                    target_dir = fastq_dir
                elif 'count_table' in name_lower or 'contrast' in name_lower or 'design' in name_lower:
                    target_dir = counts_dir
                elif any(token in name_lower for token in ('drug', 'treatment')):
                    target_dir = full_test_dir
                else:
                    target_dir = metadata_dir
                _copy_runtime_files([path], target_dir)
            hydrated['crispr_screening_input_dir'] = str(runtime_dir)

    elif agent_info.agent_type == AgentType.CRISPR_PERTURB_SEQ and not hydrated.get('crispr_10x_input_dir'):
        perturb_files = [
            path for path in uploaded_paths
            if any(token in path.name.lower() for token in ('barcode', 'feature', 'gene', 'matrix', 'identity', 'metadata'))
        ]
        if perturb_files:
            runtime_dir = _supervisor_runtime_dir(state.session_id, 'crispr_perturb_input')
            _copy_runtime_files(perturb_files, runtime_dir)
            hydrated['crispr_10x_input_dir'] = str(runtime_dir)

    if hydrated != available_inputs:
        state.workflow_state.update({k: v for k, v in hydrated.items() if v is not None})

    return hydrated


def _get_effective_missing_inputs(agent_info: AgentInfo, available_inputs: Dict[str, Any]) -> List[InputRequirement]:
    if agent_info.agent_type == AgentType.CRISPR_TARGETED:
        missing: List[InputRequirement] = []
        has_source = bool(available_inputs.get('project_id') or available_inputs.get('crispr_targeted_input_dir'))
        if not has_source:
            missing.append(InputRequirement(
                name='crispr_targeted_source',
                description='A public project ID or a local directory/uploaded paired FASTQ files for targeted CRISPR analysis',
                is_file=False,
                required=True,
                example='PRJNA1240319 or /data/crispr_targeted_run/'
            ))
        if not available_inputs.get('protospacer'):
            missing.append(InputRequirement(
                name='protospacer',
                description='Guide RNA sequence for the targeted edit',
                is_file=False,
                required=True,
                example='GGTGGATCCTATTCTAAACG'
            ))
        if not any(available_inputs.get(key) for key in ('target_gene', 'region', 'reference_seq')):
            missing.append(InputRequirement(
                name='target_gene',
                description='Target gene, genomic region, or reference sequence for the targeted analysis',
                is_file=False,
                required=True,
                example='RAB11A'
            ))
        return missing

    if agent_info.agent_type == AgentType.CRISPR_SCREENING:
        if available_inputs.get('crispr_screening_input_dir'):
            return []
        return [InputRequirement(
            name='crispr_screening_source',
            description='A screening input directory, uploaded count-table bundle, or uploaded FASTQ plus screening metadata files',
            is_file=False,
            required=True,
            example='/data/crispr_screen/ or uploaded screening files'
        )]

    if agent_info.agent_type == AgentType.CRISPR_PERTURB_SEQ:
        if available_inputs.get('crispr_10x_input_dir'):
            return []
        return [InputRequirement(
            name='crispr_10x_input_dir',
            description='A local directory or uploaded structured matrix files for perturb-seq analysis',
            is_file=False,
            required=True,
            example='/data/perturb_seq/GSE12345/'
        )]

    return agent_info.get_missing_inputs(available_inputs)


def _should_handle_crispr_in_legacy_path(message: str, state: ConversationState) -> bool:
    if any(uploaded.file_type in _CRISPR_FILE_TYPES for uploaded in state.uploaded_files.values()):
        return True

    lowered = message.lower()
    for agent_type in _CRISPR_AGENT_TYPES:
        agent_info = AGENT_REGISTRY.get(agent_type)
        if agent_info and any(keyword.lower() in lowered for keyword in agent_info.keywords):
            return True

    targeted = extract_crispr_targeted_params_sync(message)
    return bool(targeted.get('project_id'))


def _auto_generate_metadata(counts_file: str, output_path: str, disease_name: str = "Disease") -> Dict[str, Any]:
    """
    Auto-generate metadata CSV from counts file by classifying sample columns.
    
    Classifies samples as Control if column name contains keywords like 
    'control', 'ctrl', 'wt', 'normal', 'healthy'; otherwise classifies as Disease.
    """
    import pandas as pd
    
    try:
        counts_df = pd.read_csv(counts_file)
        sample_names = counts_df.columns[1:].tolist()  # Skip gene column
        
        control_keywords = ['control', 'ctrl', 'wt', 'normal', 'healthy', 'wildtype', 'baseline']
        control_samples, disease_samples = [], []
        
        for sample in sample_names:
            if any(kw in sample.lower() for kw in control_keywords):
                control_samples.append(sample)
            else:
                disease_samples.append(sample)
        
        # Build metadata rows
        rows = [{'sample': s, 'condition': 'Control'} for s in control_samples]
        rows += [{'sample': s, 'condition': disease_name or 'Disease'} for s in disease_samples]
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        
        return {'success': True, 'num_control': len(control_samples), 'num_disease': len(disease_samples)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _extract_patient_prefix(filename: str) -> str:
    """
    Extract patient/analysis prefix from filename.
    
    Examples:
        'RW-20251119_Sarcoidosis.csv' -> 'RW-20251119'
        'MC-20250909_Lupus_counts.csv' -> 'MC-20250909'
        'patient_data.csv' -> 'patient_data'
    """
    import re
    
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


def _detect_file_type(file_path: str) -> str:
    """
    Detect the type of input file based on its columns or content.
    
    Detection order (most specific first):
    1. multiomics_layer - Filename contains a recognised omics layer name
    2. json_data - JSON file with pathway or gene data
    3. gene_list - Plain text file with gene names (one per line)
    4. prioritized_genes - Has scoring columns (composite_score, etc.)
    5. pathway_results - Has pathway-specific columns
    6. raw_counts - Gene column + ALL numeric sample columns (no DEG terms)
    7. deg_results - Has BOTH fold-change AND p-value columns
    8. unknown - Could not determine
    
    Returns:
        'multiomics_layer' - Omics layer file for multi-omics integration
        'raw_counts' - Expression matrix with gene column + sample columns
        'deg_results' - DEG results with log2FC, p-value columns
        'prioritized_genes' - Prioritized gene list with scoring columns
        'pathway_results' - Pathway enrichment results
        'gene_list' - Plain text file with gene names
        'json_data' - JSON file with structured data
        'unknown' - Could not determine file type
    """
    import pandas as pd
    import json

    path = Path(file_path).expanduser()
    if path.is_dir():
        inspection = _inspect_directory(path)
        if inspection['has_10x_triplet']:
            return 'crispr_10x_data'
        if inspection['has_screening_counts']:
            return 'crispr_count_table'
        if inspection['paired_fastq_roots']:
            return 'fastq_directory'
        return 'unknown'
    
    file_ext = path.suffix.lower()

    # === STEP -1: Extension-based FASTQ detection ===
    # FASTQ files are binary sequencing data; no CSV column inspection needed.
    fastq_extensions = {'.fastq', '.fq'}
    file_name_lower = path.name.lower()
    if file_ext in fastq_extensions or (
        file_ext == '.gz' and any(
            file_name_lower.endswith(ext + '.gz') for ext in fastq_extensions
        )
    ):
        return 'fastq_file'

    # === STEP -0.5: CRISPR 10X matrix file detection ===
    crispr_10x_markers = ['barcodes.tsv', 'features.tsv', 'genes.tsv', 'matrix.mtx']
    if any(file_name_lower.startswith(m) for m in crispr_10x_markers):
        return 'crispr_10x_data'

    # === STEP 0: Filename-based omics layer detection ===
    # Layer files use 'feature' as first column (not gene/symbol) so column-based
    # detection can't identify them. We rely on filename patterns instead.
    filename_lower = path.stem.lower()
    if file_ext in ('.csv', '.tsv'):
        for layer_name in MULTIOMICS_LAYER_NAMES:
            if layer_name in filename_lower:
                return 'multiomics_layer'
    
    # Handle JSON files
    if file_ext == '.json':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Check if it looks like pathway data or gene data
            if isinstance(data, dict) and any(k in str(data.keys()).lower() for k in ['pathway', 'gene', 'enrichment']):
                return 'json_data'
            return 'json_data'
        except Exception:
            return 'unknown'
    
    # Handle TXT files (gene lists)
    if file_ext == '.txt':
        try:
            with open(file_path, 'r') as f:
                lines = [l.strip() for l in f.readlines()[:20] if l.strip()]
            # Gene list: lines look like gene symbols (short, alphanumeric)
            if lines and all(len(l) < 30 and l.replace('-', '').replace('_', '').isalnum() for l in lines[:10]):
                return 'gene_list'
            return 'gene_list'  # Default for txt
        except Exception:
            return 'unknown'
    
    try:
        df = pd.read_csv(file_path, nrows=10)
        cols = df.columns.tolist()
        cols_lower = [c.lower() for c in cols]
        
        # === STEP 0.5: CRISPR screening sgRNA count table ===
        sgrna_indicators = ['sgrna', 'guide_rna', 'spacer', 'guide_sequence', 'guide_id']
        if any(ind in col for col in cols_lower for ind in sgrna_indicators):
            return 'crispr_count_table'

        # === STEP 1: Check for PATHWAY RESULTS first (highly distinctive columns) ===
        pathway_indicators = ['pathway_name', 'pathway_id', 'pathway_source', 'enrichment_score', 
                              'gene_ratio', 'kegg_id', 'go_id', 'reactome', 'db_id']
        if sum(1 for col in cols_lower if any(ind in col for ind in pathway_indicators)) >= 2:
            return 'pathway_results'
        
        # === STEP 2: Check for PRIORITIZED GENES ===
        prioritized_indicators = ['composite_score', 'druggability_score', 'priority_score', 
                                  'final_score', 'ppi_degree']
        if any(ind in col for col in cols_lower for ind in prioritized_indicators):
            return 'prioritized_genes'
        
        # === STEP 2.5: Check for DECONVOLUTION RESULTS ===
        deconv_filename_patterns = ['cibersort', 'xcell', 'deconvolution', 'deconv',
                                    'cell_fractions', 'cell_proportions', 'immune_infiltration']
        if any(pat in filename_lower for pat in deconv_filename_patterns):
            return 'deconvolution_results'

        # === STEP 2.6: Check for PATIENT INFO ===
        patient_filename_patterns = ['patient', 'subject', 'demographics', 'clinical_info']
        patient_col_indicators = ['patient_name', 'patient_id', 'date_of_birth', 'dob',
                                  'gender', 'sex', 'age', 'diagnosis', 'specimen', 'mrn']
        is_patient_by_name = any(pat in filename_lower for pat in patient_filename_patterns)
        patient_col_hits = sum(1 for col in cols_lower
                               if any(ind in col for ind in patient_col_indicators))
        if is_patient_by_name or patient_col_hits >= 2:
            return 'patient_info'

        # === STEP 3: Check for RAW COUNTS (expression matrix) ===
        # Pattern: First col = gene identifier, rest = numeric sample columns
        # Sample columns are named like Control1, Sample_X, GSM123, Patient_ID - NOT log2fc/pvalue
        gene_id_patterns = ['gene', 'symbol', 'ensembl', 'geneid', 'entrez']
        first_col_lower = cols[0].lower()
        is_gene_col = any(pat in first_col_lower for pat in gene_id_patterns)
        
        if is_gene_col and len(cols) > 2:
            non_gene_cols = cols[1:]
            non_gene_cols_lower = [c.lower() for c in non_gene_cols]
            
            # Check if ALL non-gene columns are numeric
            try:
                numeric_cols = df[non_gene_cols].select_dtypes(include=['number']).columns
                all_numeric = len(numeric_cols) == len(non_gene_cols)
            except Exception:
                all_numeric = False
            
            # DEG-specific column name patterns (should NOT be present in raw counts)
            deg_col_patterns = ['log2fc', 'logfc', 'foldchange', 'pvalue', 'p_value', 'p-value',
                                'padj', 'adj_p', 'fdr', 'basemean', 'lfcse', 'stat']
            has_deg_named_cols = any(
                any(pat in col for pat in deg_col_patterns) 
                for col in non_gene_cols_lower
            )
            
            # Raw counts: all numeric + no DEG-specific column names
            if all_numeric and not has_deg_named_cols:
                return 'raw_counts'
        
        # === STEP 4: Check for DEG RESULTS ===
        # Must have BOTH fold-change type AND p-value type columns
        fc_patterns = ['log2fc', 'logfc', 'log2foldchange', 'foldchange']
        pval_patterns = ['pvalue', 'p_value', 'p-value', 'padj', 'adj_p', 'fdr', 'adj_pvalue']
        
        has_fc = any(any(pat in col for pat in fc_patterns) for col in cols_lower)
        has_pval = any(any(pat in col for pat in pval_patterns) for col in cols_lower)
        
        if has_fc and has_pval:
            return 'deg_results'
        
        return 'unknown'
    except Exception as e:
        logger.warning(f"Could not detect file type for {file_path}: {e}")
        return 'unknown'


def _get_agent_output_dir(agent_name: str, session_id: str) -> Path:
    """
    Get the standardized output directory for an agent.
    
    All agent outputs go to: supervisor_agent/outputs/{agent_name}/{session_id}/
    
    Args:
        agent_name: Name of the agent (e.g., 'gene_prioritization', 'deg_analysis')
        session_id: Full session identifier (UUID)
    
    Returns:
        Path to the output directory (created if not exists)
    """
    base_dir = Path(__file__).parent / "outputs" / agent_name / session_id
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _dependency_closure(target: AgentType) -> set:
    """Transitive dependency tree for an agent via AGENT_REGISTRY.depends_on."""
    closure = {target}
    info = AGENT_REGISTRY.get(target)
    if info:
        for dep in info.depends_on:
            closure |= _dependency_closure(dep)
    return closure


def _build_execution_chain(
    target_agent: AgentType,
    available_keys: set,
    uploaded_file_type: Optional[str] = None
) -> tuple[list[AgentType], list[AgentType], str]:
    """
    Build the execution chain needed to reach target agent.
    
    Dynamically determines which agents to run based on:
    1. What inputs are already available (from workflow_state or uploads)
    2. What the target agent requires
    3. What intermediate agents produce
    
    Args:
        target_agent: The agent the user wants to run
        available_keys: Set of keys already available in workflow_state
        uploaded_file_type: Detected type of uploaded file (raw_counts, deg_results, etc.)
    
    Returns:
        Tuple of (agents_to_run, agents_skipped, info_message)
    """
    # If uploaded file provides an input, add all applicable keys
    if uploaded_file_type and uploaded_file_type in FILE_TYPE_TO_INPUT_KEY:
        input_keys = FILE_TYPE_TO_INPUT_KEY[uploaded_file_type]
        available_keys = available_keys | set(input_keys)
    
    # Get target agent's position in pipeline
    if target_agent not in PIPELINE_ORDER:
        # Not part of standard pipeline (e.g., deconvolution), run directly
        return [target_agent], [], ""
    
    target_idx = PIPELINE_ORDER.index(target_agent)
    
    # Only include agents that are actual ancestors of the target
    dep_closure = _dependency_closure(target_agent)
    
    # Find earliest agent we need to start from
    start_idx = 0
    for idx in range(target_idx, -1, -1):
        agent_type = PIPELINE_ORDER[idx]
        if agent_type not in dep_closure:
            continue
        agent_info = AGENT_REGISTRY[agent_type]
        
        # Check if any of the required inputs are available
        if agent_info.requires_one_of:
            has_required = any(key in available_keys for key in agent_info.requires_one_of)
            if has_required:
                start_idx = idx
                break
    
    # Build the chain from start_idx to target_idx
    agents_to_run = []
    agents_skipped = []
    
    for idx in range(target_idx + 1):
        agent_type = PIPELINE_ORDER[idx]
        agent_info = AGENT_REGISTRY[agent_type]
        
        if idx < start_idx or agent_type not in dep_closure:
            agents_skipped.append(agent_type)
        else:
            agents_to_run.append(agent_type)
            # Add what this agent produces to available keys for next iteration
            available_keys = available_keys | set(agent_info.produces)
    
    # Build info message about skipped agents
    info_message = ""
    if agents_skipped:
        skipped_names = [AGENT_REGISTRY[a].display_name for a in agents_skipped]
        if uploaded_file_type:
            info_message = (
                f"Your uploaded file appears to be **{uploaded_file_type.replace('_', ' ')}**. "
                f"Skipping: {', '.join(skipped_names)}."
            )
        else:
            info_message = f"Skipping (data already available): {', '.join(skipped_names)}."
    
    return agents_to_run, agents_skipped, info_message


class StatusType(str, Enum):
    """Types of status updates for the UI"""
    THINKING = "thinking"
    ROUTING = "routing"
    VALIDATING = "validating"
    EXECUTING = "executing"
    PROGRESS = "progress"
    INFO = "info"  # Informational status updates
    COMPLETED = "completed"
    ERROR = "error"
    WAITING_INPUT = "waiting_input"


@dataclass
class StatusUpdate:
    """A status update to send to the UI"""
    status_type: StatusType
    title: str
    message: str
    details: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    agent_name: Optional[str] = None
    timestamp: float = None
    generated_files: Optional[List[str]] = None  # Files generated by this step (plots, CSVs, etc.)
    output_dir: Optional[str] = None  # Output directory for this agent
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.generated_files is None:
            self.generated_files = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status_type": self.status_type.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "progress": self.progress,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "generated_files": self.generated_files or [],
            "output_dir": self.output_dir
        }


def _collect_generated_files(output_dir: Union[str, Path], extensions: List[str] = None) -> List[str]:
    """
    Collect all generated files from an output directory.
    Returns list of file paths, categorized by type.
    """
    if not output_dir:
        return []
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    # Default extensions to look for
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.csv', '.tsv', '.xlsx', '.json', '.pdf', '.html']
    
    files = []
    for ext in extensions:
        files.extend([str(f) for f in output_path.glob(f"**/*{ext}")])
    
    return sorted(files)


class SupervisorAgent:
    """
    Main supervisor agent that:
    1. Understands user queries
    2. Routes to appropriate specialized agents
    3. Manages conversation state
    4. Provides detailed feedback and reasoning
    """
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        upload_dir: str = "./uploads"
    ):
        self.session_manager = session_manager or SessionManager()
        self.router = IntentRouter()
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent execution functions - using executors from the executors package
        self._agent_executors = self._register_agent_executors()
        
        logger.info("🎯 SupervisorAgent initialized")
    
    def _register_agent_executors(self) -> Dict[AgentType, Callable]:
        """Register the actual execution functions for each agent"""
        from .executors import (
            execute_cohort_retrieval,
            execute_deg_analysis,
            execute_gene_prioritization,
            execute_pathway_enrichment,
            execute_deconvolution,
            execute_temporal_analysis,
            execute_harmonization,
            execute_mdp_analysis,
            execute_perturbation_analysis,
            execute_multiomics_integration,
            execute_fastq_processing,
            execute_molecular_report,
            execute_crispr_perturb_seq,
            execute_crispr_screening,
            execute_crispr_targeted,
        )
        return {
            AgentType.COHORT_RETRIEVAL: execute_cohort_retrieval,
            AgentType.DEG_ANALYSIS: execute_deg_analysis,
            AgentType.GENE_PRIORITIZATION: execute_gene_prioritization,
            AgentType.PATHWAY_ENRICHMENT: execute_pathway_enrichment,
            AgentType.DECONVOLUTION: execute_deconvolution,
            AgentType.TEMPORAL_ANALYSIS: execute_temporal_analysis,
            AgentType.HARMONIZATION: execute_harmonization,
            AgentType.MDP_ANALYSIS: execute_mdp_analysis,
            AgentType.PERTURBATION_ANALYSIS: execute_perturbation_analysis,
            AgentType.MULTIOMICS_INTEGRATION: execute_multiomics_integration,
            AgentType.FASTQ_PROCESSING: execute_fastq_processing,
            AgentType.MOLECULAR_REPORT: execute_molecular_report,
            AgentType.CRISPR_PERTURB_SEQ: execute_crispr_perturb_seq,
            AgentType.CRISPR_SCREENING: execute_crispr_screening,
            AgentType.CRISPR_TARGETED: execute_crispr_targeted,
        }
    
    def _resolve_dependencies(
        self, 
        target_agent: AgentInfo, 
        available_inputs: Dict[str, Any],
        uploaded_files: Optional[Dict[str, Any]] = None
    ) -> List[AgentInfo]:
        """
        Check if missing inputs can be auto-produced by other agents.
        Returns list of agents to prepend (in order) to satisfy dependencies.
        """
        missing = target_agent.get_missing_inputs(available_inputs)
        if not missing:
            return []
        
        prepend_agents = []
        simulated_inputs = available_inputs.copy()
        
        # Helper: check if agent's file requirements can be satisfied by uploaded files
        def can_potentially_run(agent: AgentInfo) -> bool:
            for inp in agent.required_inputs:
                if inp.name in simulated_inputs and simulated_inputs[inp.name]:
                    continue
                if inp.is_file and inp.file_type and uploaded_files:
                    # Check if any uploaded file matches the required type
                    if any(f.lower().endswith(f".{inp.file_type}") for f in uploaded_files.keys()):
                        continue
                if not inp.is_file:
                    # Non-file inputs (like disease_name) - check if available
                    if inp.name in simulated_inputs:
                        continue
                return False
            return True
        
        for inp in missing:
            if not inp.can_come_from:
                continue
            
            producing_agent = get_agent_by_name(inp.can_come_from)
            if not producing_agent:
                continue
            
            # Recursively resolve dependencies of the producing agent
            nested_deps = self._resolve_dependencies(producing_agent, simulated_inputs, uploaded_files)
            for dep in nested_deps:
                if dep not in prepend_agents:
                    prepend_agents.append(dep)
                    for out in dep.outputs:
                        simulated_inputs[out.state_key or out.name] = True
            
            # Check if producing agent can potentially run with available files
            if can_potentially_run(producing_agent) and producing_agent not in prepend_agents:
                prepend_agents.append(producing_agent)
                for out in producing_agent.outputs:
                    simulated_inputs[out.state_key or out.name] = True
        
        return prepend_agents
    
    async def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        uploaded_files: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[StatusUpdate, None]:
        """
        Process a user message and yield status updates.
        
        This is the main entry point for the Streamlit UI.
        It yields StatusUpdate objects that can be displayed in real-time.
        
        Args:
            user_message: The user's message/query
            session_id: Session ID for continuity
            user_id: User ID
            uploaded_files: Dict of filename -> filepath for any uploaded files
            
        Yields:
            StatusUpdate objects for real-time UI updates
        """
        # Get or create session
        state = self.session_manager.get_or_create_session(session_id, user_id)
        
        # Register uploaded files and detect file types
        new_files_registered = []
        detected_file_type = None  # Track the detected type of most recent CSV upload
        
        if uploaded_files:
            for filename, filepath in uploaded_files.items():
                # Only register if not already registered
                if filename not in state.uploaded_files:
                    file_size = Path(filepath).stat().st_size if Path(filepath).exists() else 0
                    file_type = Path(filename).suffix.lstrip(".")
                    state.add_uploaded_file(filename, filepath, file_type, file_size)
                    new_files_registered.append(filename)
                    
                    content_type = _detect_file_type(filepath) if Path(filepath).exists() else None
                    if content_type and content_type != 'unknown':
                        detected_file_type = content_type

                        if content_type == 'multiomics_layer':
                            layer_name = self._detect_omics_layer_name(filename)
                            if layer_name:
                                layers = state.workflow_state.setdefault("multiomics_layers", {})
                                layers[layer_name] = filepath
                                logger.info(f"📁 Detected omics layer '{layer_name}' → accumulated into multiomics_layers")
                        elif content_type in FILE_TYPE_TO_INPUT_KEY:
                            input_keys = FILE_TYPE_TO_INPUT_KEY[content_type]
                            for input_key in input_keys:
                                state.workflow_state[input_key] = filepath
                            logger.info(f"📁 Detected file type '{content_type}' → mapped to {input_keys}")

                    yield StatusUpdate(
                        status_type=StatusType.PROGRESS,
                        title="📁 File Registered",
                        message=f"Received file: **{filename}**",
                        details=f"Size: {file_size:,} bytes | Type: {file_type.upper()}" + 
                               (f" | Detected: **{content_type.replace('_', ' ')}**" if content_type else "")
                    )
        
        # Store detected file type in state for chain building
        if detected_file_type:
            state.workflow_state["_detected_file_type"] = detected_file_type

        # LangGraph path — delegate to the compiled graph when enabled
        if _USE_LANGGRAPH and not state.waiting_for_input and not _should_handle_crispr_in_legacy_path(user_message, state):
            state.add_user_message(user_message)
            async for update in self._process_message_langgraph(
                user_message, session_id, uploaded_files, state
            ):
                yield update
            return

        # Check if we're waiting for input and have a pending agent
        if state.waiting_for_input and state.pending_agent_type:
            # Check for MDP file assignment response
            if "mdp_file_assignment" in state.required_inputs:
                pending_diseases = state.workflow_state.get("_mdp_pending_diseases", [])
                pending_files = state.workflow_state.get("_mdp_pending_files", [])
                
                # Try to parse user's response (number or disease name)
                selected_disease = None
                response = user_message.strip().lower()
                
                # Check if response is a number
                if response.isdigit():
                    idx = int(response) - 1
                    if 0 <= idx < len(pending_diseases):
                        selected_disease = pending_diseases[idx]
                else:
                    # Check if response matches a disease name
                    for d in pending_diseases:
                        if d.lower() in response or response in d.lower():
                            selected_disease = d
                            break
                
                if selected_disease:
                    # Build file assignments: selected disease gets the file, others use KG
                    file_assignments = {}
                    for f in pending_files:
                        # Get file path from uploaded_files (UploadedFile object has .filepath)
                        uploaded = next((v for k, v in state.uploaded_files.items() if k == f), None)
                        if uploaded:
                            file_assignments[selected_disease] = uploaded.filepath
                    
                    state.workflow_state["file_assignments"] = file_assignments
                    state.workflow_state["disease_names"] = pending_diseases
                    
                    # Clear pending state
                    del state.workflow_state["_mdp_pending_diseases"]
                    del state.workflow_state["_mdp_pending_files"]
                    state.waiting_for_input = False
                    state.required_inputs = []
                    
                    yield StatusUpdate(
                        status_type=StatusType.PROGRESS,
                        title="✅ File Assignment Confirmed",
                        message=f"File assigned to **{selected_disease}**. Other diseases will use Knowledge Graph.",
                        details="Starting MDP analysis now..."
                    )
                    
                    state.add_user_message(user_message)
                    
                    # Execute MDP with the assignments
                    agent_info = AGENT_REGISTRY.get(AgentType(state.pending_agent_type))
                    available_inputs = state.get_available_inputs()
                    available_inputs["file_assignments"] = file_assignments
                    available_inputs["disease_names"] = pending_diseases
                    
                    async for update in self._execute_pending_agent(agent_info, available_inputs, state):
                        yield update
                    return
                else:
                    # Couldn't parse response - ask again
                    yield StatusUpdate(
                        status_type=StatusType.WAITING_INPUT,
                        title="❓ Please Choose a Disease",
                        message=f"Please reply with a number (1-{len(pending_diseases)}) or the disease name.",
                        details=""
                    )
                    state.add_user_message(user_message)
                    return
            
            # Check if we now have all required inputs
            available_inputs = state.get_available_inputs()
            
            # Extract any additional inputs from the message
            extracted = self._extract_inputs_from_message(user_message, state.required_inputs)
            if extracted:
                available_inputs.update(extracted)
                state.workflow_state.update(extracted)
                # Update disease in state if extracted
                if "disease_name" in extracted:
                    state.current_disease = extracted["disease_name"]
                    logger.info(f"🔬 Disease set to: {state.current_disease}")
            
            # Get the pending agent info
            agent_info = AGENT_REGISTRY.get(AgentType(state.pending_agent_type))
            if agent_info:
                available_inputs = _hydrate_crispr_inputs(agent_info, available_inputs, state)
                missing_inputs = _get_effective_missing_inputs(agent_info, available_inputs)
                
                logger.info(f"📋 Available inputs: {list(available_inputs.keys())}")
                logger.info(f"❓ Missing inputs: {[inp.name for inp in missing_inputs]}")
                
                if not missing_inputs:
                    # All inputs available! Continue with the agent
                    yield StatusUpdate(
                        status_type=StatusType.PROGRESS,
                        title="✅ All Inputs Received",
                        message=f"Great! Continuing with {agent_info.display_name}...",
                        details="Starting analysis now"
                    )
                    
                    state.waiting_for_input = False
                    state.required_inputs = []
                    
                    # Add user message to history before executing
                    state.add_user_message(user_message)
                    
                    # Execute the agent directly
                    async for update in self._execute_pending_agent(agent_info, available_inputs, state):
                        yield update
                    return
                else:
                    # Still missing some inputs - tell user what's still needed
                    missing_msg = self._format_missing_inputs_request(agent_info, missing_inputs, available_inputs)
                    yield StatusUpdate(
                        status_type=StatusType.WAITING_INPUT,
                        title="📎 Still Need More Input",
                        message=missing_msg,
                        details=""
                    )
                    state.add_user_message(user_message)
                    state.add_assistant_message(missing_msg)
                    return
        
        # Add user message to history
        state.add_user_message(user_message)
        
        # Reset executed agents tracking for this new query
        state.workflow_state["_executed_agents_this_run"] = []
        
        # ===== STEP 1: Understanding the query =====
        yield StatusUpdate(
            status_type=StatusType.THINKING,
            title="🧠 Understanding Your Request",
            message="Analyzing your query to determine the best course of action...",
            details="I'm examining keywords, context, and your conversation history."
        )
        
        await asyncio.sleep(0.3)  # Small delay for UI feedback
        
        # ===== STEP 2: Route to appropriate agent =====
        yield StatusUpdate(
            status_type=StatusType.ROUTING,
            title="🔍 Analyzing Intent",
            message="Matching your request to specialized bioinformatics agents...",
            details="Comparing against: Cohort Retrieval, DEG Analysis, Gene Prioritization, Pathway Enrichment, Deconvolution"
        )
        
        routing_decision = await self.router.route(user_message, state)
        
        # ===== STEP 3: Handle routing decision =====
        if routing_decision.is_general_query:
            # Handle general queries (help, greetings, etc.)
            yield StatusUpdate(
                status_type=StatusType.COMPLETED,
                title="💬 General Query",
                message=routing_decision.suggested_response or self._get_help_message(),
                details=f"Reasoning: {routing_decision.reasoning}"
            )
            state.add_assistant_message(routing_decision.suggested_response or self._get_help_message())
            return
        
        if not routing_decision.agent_type:
            # Couldn't determine intent
            yield StatusUpdate(
                status_type=StatusType.WAITING_INPUT,
                title="❓ Clarification Needed",
                message="I'm not sure which analysis you'd like to perform. Could you be more specific?",
                details=f"Reasoning: {routing_decision.reasoning}"
            )
            state.add_assistant_message(
                "I'm not sure which analysis you'd like to perform. Here's what I can help with:\n\n" +
                self._get_capabilities_summary()
            )
            return
        
        # ===== STEP 4: Check for multi-agent pipeline =====
        if routing_decision.is_multi_agent and len(routing_decision.agent_pipeline) > 1:
            # Execute multi-agent pipeline
            async for update in self._execute_multi_agent_pipeline(routing_decision, state):
                yield update
            return
        
        # ===== STEP 5: Single agent - Explain routing decision =====
        agent_info = AGENT_REGISTRY[routing_decision.agent_type]
        required_input_names = {
            inp.name for inp in (agent_info.required_inputs + agent_info.optional_inputs)
        }
        deterministic_params = _extract_contextual_inputs(user_message, required_input_names)
        routing_decision.extracted_params = {
            **deterministic_params,
            **routing_decision.extracted_params,
        }
        
        yield StatusUpdate(
            status_type=StatusType.ROUTING,
            title=f"🎯 Routing to {agent_info.display_name}",
            message=f"I've identified this as a **{agent_info.name.replace('_', ' ').title()}** task.",
            details=f"**Why this agent?**\n{routing_decision.reasoning}\n\n**Confidence:** {routing_decision.confidence:.0%}",
            agent_name=agent_info.name
        )
        
        await asyncio.sleep(0.5)
        
        # Update extracted parameters in state
        if routing_decision.extracted_params.get("disease_name"):
            state.current_disease = routing_decision.extracted_params["disease_name"]
        
        # Store disease_names list for multi-disease agents like MDP
        disease_names = routing_decision.extracted_params.get("disease_names", [])
        if disease_names:
            state.workflow_state["disease_names"] = disease_names
        
        # ===== MDP FILE/DISEASE CLARIFICATION CHECK =====
        if routing_decision.agent_type == AgentType.MDP_ANALYSIS and len(disease_names) > 1:
            # Count uploaded files
            uploaded_file_count = len([f for f in state.uploaded_files.values() if f])
            
            # Check if clarification needed: N diseases but M files where 0 < M < N
            if 0 < uploaded_file_count < len(disease_names):
                # Check if we already have file assignments
                existing_assignments = state.workflow_state.get("file_assignments", {})
                
                if not existing_assignments:
                    # Need clarification: which disease is the file for?
                    file_names = list(state.uploaded_files.keys())
                    
                    clarification_msg = (
                        f"I noticed you mentioned **{len(disease_names)} diseases** ({', '.join(disease_names)}) "
                        f"but uploaded **{uploaded_file_count} file(s)** ({', '.join(file_names)}).\n\n"
                        f"**Which disease is the uploaded file for?**\n\n"
                        + "\n".join([f"• Reply '{i+1}' for **{d}**" for i, d in enumerate(disease_names)])
                        + "\n\n*(The other disease(s) will use our Knowledge Graph for pathway data)*"
                    )
                    
                    yield StatusUpdate(
                        status_type=StatusType.WAITING_INPUT,
                        title="🔍 Clarification Needed for MDP",
                        message=clarification_msg,
                        details=""
                    )
                    
                    state.waiting_for_input = True
                    state.required_inputs = ["mdp_file_assignment"]
                    state.pending_agent_type = routing_decision.agent_type.value
                    state.workflow_state["_mdp_pending_diseases"] = disease_names
                    state.workflow_state["_mdp_pending_files"] = file_names
                    state.add_assistant_message(clarification_msg)
                    return
        
        # ===== STEP 6: Build execution chain (handles dependencies automatically) =====
        available_inputs = state.get_available_inputs()
        available_inputs.update(routing_decision.extracted_params)
        available_inputs = _hydrate_crispr_inputs(agent_info, available_inputs, state)
        
        # Get available keys from workflow_state for chain building
        available_keys = set(k for k, v in state.workflow_state.items() if v and not k.startswith('_'))
        
        # Get detected file type if any
        detected_file_type = state.workflow_state.get("_detected_file_type")
        
        # Build execution chain using new chain builder
        agents_to_run, agents_skipped, skip_info = _build_execution_chain(
            routing_decision.agent_type,
            available_keys,
            detected_file_type
        )
        
        # Show what we're planning to do
        if len(agents_to_run) > 1:
            # Multi-agent pipeline needed
            agent_names = [AGENT_REGISTRY[a].display_name for a in agents_to_run]
            pipeline_str = " → ".join(agent_names)
            
            yield StatusUpdate(
                status_type=StatusType.ROUTING,
                title="🔗 Building Execution Pipeline",
                message=f"Your request requires **{len(agents_to_run)} agents**:\n\n**{pipeline_str}**",
                details=(skip_info + "\n\n" if skip_info else "") + 
                       f"Each agent's output will feed into the next agent."
            )
            
            await asyncio.sleep(0.5)
            
            # Build pipeline for multi-agent execution
            from .router import AgentIntent
            
            pipeline = [
                AgentIntent(
                    agent_type=agent_type,
                    agent_name=AGENT_REGISTRY[agent_type].name,
                    confidence=0.9 if agent_type != routing_decision.agent_type else routing_decision.confidence,
                    reasoning=f"Required in pipeline for {agent_info.display_name}" if agent_type != routing_decision.agent_type else routing_decision.reasoning,
                    order=i
                ) for i, agent_type in enumerate(agents_to_run)
            ]
            
            # Create pipeline routing decision
            from .router import RoutingDecision as RD
            pipeline_decision = RD(
                agent_type=routing_decision.agent_type,
                agent_name=routing_decision.agent_name,
                confidence=routing_decision.confidence,
                reasoning=routing_decision.reasoning,
                extracted_params=routing_decision.extracted_params,
                missing_inputs=[],
                is_general_query=False,
                suggested_response=None,
                is_multi_agent=True,
                agent_pipeline=pipeline
            )
            
            async for update in self._execute_multi_agent_pipeline(pipeline_decision, state):
                yield update
            return
        
        # Single agent - check if inputs are satisfied
        yield StatusUpdate(
            status_type=StatusType.VALIDATING,
            title="📋 Checking Requirements",
            message=f"Verifying that all required inputs are available for {agent_info.display_name}...",
            details=self._format_requirements(agent_info)
        )
        
        if skip_info:
            yield StatusUpdate(
                status_type=StatusType.INFO,
                title="ℹ️ Smart Routing",
                message=skip_info,
                progress=0.05
            )
        
        missing_inputs = _get_effective_missing_inputs(agent_info, available_inputs)
        
        if missing_inputs:
            # Can't auto-resolve - request missing inputs from user
            missing_msg = self._format_missing_inputs_request(agent_info, missing_inputs, available_inputs)
            
            yield StatusUpdate(
                status_type=StatusType.WAITING_INPUT,
                title="📎 Additional Input Needed",
                message=missing_msg,
                details=""
            )
            
            state.waiting_for_input = True
            state.required_inputs = [inp.name for inp in missing_inputs]
            state.pending_agent_type = routing_decision.agent_type.value  # Store pending agent
            state.add_assistant_message(missing_msg)
            return
        
        # ===== STEP 7: Execute the agent =====
        state.waiting_for_input = False
        state.required_inputs = []

        preflight_err = _validate_required_input_paths(agent_info, available_inputs)
        if preflight_err:
            yield StatusUpdate(
                status_type=StatusType.WAITING_INPUT,
                title="📎 Additional Input Needed",
                message=preflight_err,
                details=""
            )
            state.waiting_for_input = True
            state.required_inputs = [inp.name for inp in agent_info.required_inputs if inp.name not in available_inputs or not available_inputs.get(inp.name)]
            state.pending_agent_type = routing_decision.agent_type.value
            state.add_assistant_message(preflight_err)
            return
        
        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title=f"🚀 Starting {agent_info.display_name}",
            message=f"All requirements satisfied! Beginning analysis...",
            details=f"**Estimated time:** {agent_info.estimated_time}\n\n**What's happening:**\n{agent_info.detailed_description[:300]}...",
            agent_name=agent_info.name,
            progress=0.0
        )
        
        # Start agent execution tracking
        state.start_agent_execution(
            agent_name=agent_info.name,
            agent_display_name=agent_info.display_name,
            inputs=available_inputs
        )
        
        # Execute the agent and yield progress updates
        try:
            executor = self._agent_executors.get(routing_decision.agent_type)
            if not executor:
                raise ValueError(f"No executor registered for {routing_decision.agent_type}")
            
            async for update in executor(agent_info, available_inputs, state):
                yield update
            
            # Agent completed successfully
            state.complete_agent_execution(state.workflow_state)
            
            # Track this agent as executed in the current run
            executed_this_run = state.workflow_state.get("_executed_agents_this_run", [])
            if agent_info.name not in executed_this_run:
                executed_this_run.append(agent_info.name)
                state.workflow_state["_executed_agents_this_run"] = executed_this_run
            
            # Generate completion message
            completion_msg = self._format_completion_message(agent_info, state.workflow_state)
            
            # Collect generated files from output directory
            output_dir = self._get_output_dir_for_agent(agent_info.name, state.workflow_state)
            generated_files = _collect_generated_files(output_dir) if output_dir else []
            
            yield StatusUpdate(
                status_type=StatusType.COMPLETED,
                title=f"✅ {agent_info.display_name} Complete!",
                message=completion_msg,
                details=self._format_outputs(agent_info, state.workflow_state),
                agent_name=agent_info.name,
                progress=1.0,
                generated_files=generated_files,
                output_dir=output_dir
            )
            
            state.add_assistant_message(completion_msg)
            
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            state.fail_agent_execution(str(e))
            
            yield StatusUpdate(
                status_type=StatusType.ERROR,
                title=f"❌ {agent_info.display_name} Failed",
                message=f"An error occurred during execution: {str(e)}",
                details="Please check your inputs and try again. If the problem persists, the data format may be incompatible.",
                agent_name=agent_info.name
            )
            
            state.add_assistant_message(f"I encountered an error: {str(e)}\n\nPlease check your inputs and try again.")
    
    # =========================================================================
    # MULTI-AGENT PIPELINE EXECUTOR
    # =========================================================================
    
    async def _execute_multi_agent_pipeline(
        self,
        routing_decision: 'RoutingDecision',
        state: ConversationState
    ) -> AsyncGenerator[StatusUpdate, None]:
        """Execute a pipeline of multiple agents sequentially with smart step skipping"""
        from .router import RoutingDecision
        
        pipeline = routing_decision.agent_pipeline
        total_agents = len(pipeline)
        
        # Show pipeline overview
        agent_names = [AGENT_REGISTRY[a.agent_type].display_name for a in pipeline]
        pipeline_str = " → ".join(agent_names)
        
        yield StatusUpdate(
            status_type=StatusType.ROUTING,
            title=f"🔗 Multi-Agent Pipeline Detected",
            message=f"Your request requires **{total_agents} agents** working together:\n\n**{pipeline_str}**",
            details=f"I'll execute each agent in sequence, passing outputs as inputs to the next agent.\n\n**Reasoning:** {routing_decision.reasoning}"
        )
        
        await asyncio.sleep(0.5)
        
        # Update extracted parameters in state
        if routing_decision.extracted_params.get("disease_name"):
            state.current_disease = routing_decision.extracted_params["disease_name"]
        
        available_inputs = state.get_available_inputs()
        available_inputs.update(routing_decision.extracted_params)
        
        completed_agents = []
        skipped_agents = []
        failed_agent = None
        
        for idx, agent_intent in enumerate(pipeline):
            agent_info = AGENT_REGISTRY[agent_intent.agent_type]
            current_num = idx + 1
            
            # Show progress header for this agent
            yield StatusUpdate(
                status_type=StatusType.EXECUTING,
                title=f"📍 Pipeline Progress: Agent {current_num}/{total_agents}",
                message=f"Now running: **{agent_info.display_name}**",
                details=f"**Why:** {agent_intent.reasoning}\n\n**Previously completed:** {', '.join(completed_agents) if completed_agents else 'None yet'}" + (f"\n**Skipped:** {', '.join(skipped_agents)}" if skipped_agents else ""),
                agent_name=agent_info.name,
                progress=idx / total_agents
            )
            
            await asyncio.sleep(0.3)
            
            # Check if this agent has all required inputs
            available_inputs = _hydrate_crispr_inputs(agent_info, available_inputs, state)
            missing_inputs = _get_effective_missing_inputs(agent_info, available_inputs)
            
            if missing_inputs:
                # For multi-agent, we may need to ask for missing inputs but continue with first agent
                if idx == 0 or (idx > 0 and not completed_agents and not skipped_agents):
                    # First actual agent (accounting for skips) missing inputs - need to ask user
                    missing_msg = self._format_missing_inputs_request(agent_info, missing_inputs, available_inputs)
                    
                    yield StatusUpdate(
                        status_type=StatusType.WAITING_INPUT,
                        title=f"📎 Input Needed for {agent_info.display_name}",
                        message=missing_msg,
                        details=f"This is the first agent in the pipeline. Please provide the required inputs to continue."
                    )
                    
                    state.waiting_for_input = True
                    state.required_inputs = [inp.name for inp in missing_inputs]
                    state.pending_agent_type = agent_intent.agent_type.value
                    # Store pending pipeline
                    state.workflow_state["_pending_pipeline"] = [a.to_dict() for a in pipeline]
                    state.workflow_state["_pipeline_index"] = idx
                    state.add_assistant_message(missing_msg)
                    return
                else:
                    # Later agent missing inputs - likely needs output from previous
                    # This shouldn't happen if pipeline is correct, but handle gracefully
                    yield StatusUpdate(
                        status_type=StatusType.ERROR,
                        title=f"⚠️ Missing Dependencies",
                        message=f"Agent **{agent_info.display_name}** requires: {', '.join([inp.name for inp in missing_inputs])}",
                        details="The previous agent may not have produced the expected outputs. Please check the data."
                    )
                    failed_agent = agent_info.display_name
                    break
            
            # Execute this agent
            yield StatusUpdate(
                status_type=StatusType.EXECUTING,
                title=f"🚀 Running {agent_info.display_name} ({current_num}/{total_agents})",
                message=f"Starting analysis...",
                details=f"**Estimated time:** {agent_info.estimated_time}",
                agent_name=agent_info.name,
                progress=(idx + 0.1) / total_agents
            )
            
            state.start_agent_execution(
                agent_name=agent_info.name,
                agent_display_name=agent_info.display_name,
                inputs=available_inputs
            )
            
            try:
                executor = self._agent_executors.get(agent_intent.agent_type)
                if not executor:
                    raise ValueError(f"No executor registered for {agent_intent.agent_type}")
                
                async for update in executor(agent_info, available_inputs, state):
                    # Adjust progress for pipeline context
                    if update.progress is not None:
                        update.progress = (idx + update.progress) / total_agents
                    # Add pipeline context to title
                    if update.title and not update.title.startswith("📍"):
                        update.title = f"[{current_num}/{total_agents}] {update.title}"
                    yield update
                
                # Agent completed
                state.complete_agent_execution(state.workflow_state)
                completed_agents.append(agent_info.display_name)
                
                # Track this agent as executed in the current run
                executed_this_run = state.workflow_state.get("_executed_agents_this_run", [])
                if agent_info.name not in executed_this_run:
                    executed_this_run.append(agent_info.name)
                    state.workflow_state["_executed_agents_this_run"] = executed_this_run
                
                # Update available inputs with new outputs from this agent
                available_inputs = state.get_available_inputs()
                available_inputs.update(routing_decision.extracted_params)
                
                yield StatusUpdate(
                    status_type=StatusType.PROGRESS,
                    title=f"✅ {agent_info.display_name} Complete ({current_num}/{total_agents})",
                    message=f"Moving to next agent in pipeline...",
                    agent_name=agent_info.name,
                    progress=(idx + 1) / total_agents
                )
                
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.exception(f"Agent {agent_info.name} failed in pipeline: {e}")
                state.fail_agent_execution(str(e))
                
                yield StatusUpdate(
                    status_type=StatusType.ERROR,
                    title=f"❌ {agent_info.display_name} Failed ({current_num}/{total_agents})",
                    message=f"Error: {str(e)}",
                    details=f"Pipeline stopped. Completed agents: {', '.join(completed_agents) if completed_agents else 'None'}",
                    agent_name=agent_info.name
                )
                
                failed_agent = agent_info.display_name
                break
        
        # Pipeline completion summary
        if not failed_agent:
            completion_msg = f"✅ **Multi-Agent Pipeline Complete!**\n\n"
            completion_msg += f"Successfully executed {total_agents} agents:\n"
            for agent_name in completed_agents:
                completion_msg += f"- {agent_name} ✓\n"
            
            # Collect all generated files from pipeline outputs
            all_generated_files = []
            all_output_dirs = []
            for agent_name in ["cohort_retrieval", "deg_analysis", "gene_prioritization", "pathway_enrichment", "deconvolution", "perturbation_analysis", "multiomics_integration"]:
                output_dir = self._get_output_dir_for_agent(agent_name, state.workflow_state)
                if output_dir:
                    all_output_dirs.append(output_dir)
                    all_generated_files.extend(_collect_generated_files(output_dir))
            
            yield StatusUpdate(
                status_type=StatusType.COMPLETED,
                title="🎉 Pipeline Complete!",
                message=completion_msg,
                details=self._format_pipeline_outputs(state.workflow_state),
                progress=1.0,
                generated_files=all_generated_files,
                output_dir=all_output_dirs[0] if all_output_dirs else None
            )
            
            state.add_assistant_message(completion_msg)
        else:
            error_msg = f"❌ **Pipeline Stopped**\n\n"
            error_msg += f"Failed at: {failed_agent}\n"
            if completed_agents:
                error_msg += f"Completed: {', '.join(completed_agents)}\n"
            
            state.add_assistant_message(error_msg)
    
    def _format_pipeline_outputs(self, workflow_state: Dict[str, Any]) -> str:
        """Format all outputs from a multi-agent pipeline"""
        output_keys = {
            "cohort_output_dir": "📁 Cohort Data Directory",
            "deg_base_dir": "📊 DEG Results Directory", 
            "prioritized_genes_path": "🎯 Prioritized Genes File",
            "pathway_consolidation_path": "🛤️ Pathway Analysis Results",
            "deconvolution_output_dir": "🧬 Deconvolution Results",
            "perturbation_output_dir": "💊 Perturbation Analysis Results",
            "multiomics_output_dir": "🧬 Multi-Omics Integration Results"
        }
        
        parts = ["**Generated Outputs:**\n"]
        for key, label in output_keys.items():
            if key in workflow_state and workflow_state[key]:
                parts.append(f"- {label}: `{workflow_state[key]}`")
        
        return "\n".join(parts) if len(parts) > 1 else "No output files generated."
    
    # HELPER METHODS
    # =========================================================================
    
    @staticmethod
    def _detect_omics_layer_name(filename: str) -> Optional[str]:
        """
        Extract omics layer name from a filename.
        
        Matches any of the MULTIOMICS_LAYER_NAMES that appears in the
        filename stem.  Returns the first match or None.
        """
        stem = Path(filename).stem.lower()
        for layer in MULTIOMICS_LAYER_NAMES:
            if layer in stem:
                return layer
        return None
    
    def _get_help_message(self) -> str:
        """Generate a help message explaining available capabilities"""
        return """👋 **Hello! I'm your Bioinformatics Analysis Assistant.**

I can help you with various transcriptome analysis tasks:

🔍 **Cohort Retrieval** - Search GEO/ArrayExpress for public datasets
📊 **DEG Analysis** - Find differentially expressed genes
🎯 **Gene Prioritization** - Rank genes by disease relevance  
🛤️ **Pathway Enrichment** - Identify affected biological pathways
🧬 **Deconvolution** - Estimate cell type composition
💊 **Perturbation Analysis** - Drug perturbation via DEPMAP + L1000
🧬 **Multi-Omics Integration** - Integrate genomics, proteomics, metabolomics & more
🧬 **FASTQ Processing** - QC, trim and quantify raw sequencing reads

**How to get started:**
1. Tell me what analysis you want to perform
2. Specify the disease/condition you're studying
3. Upload any required data files

**Example queries:**
- "Find datasets for lupus disease"
- "Run DEG analysis on my count data for breast cancer"
- "What pathways are enriched in my gene list?"
- "Run perturbation analysis on my prioritized genes for lupus"

What would you like to do?"""
    
    def _get_capabilities_summary(self) -> str:
        """Get a brief summary of capabilities"""
        lines = ["**Available Analyses:**\n"]
        for agent in AGENT_REGISTRY.values():
            lines.append(f"• **{agent.display_name}** - {agent.description}")
        return "\n".join(lines)
    
    def _format_requirements(self, agent_info: AgentInfo) -> str:
        """Format agent requirements for display"""
        lines = [f"**Required for {agent_info.display_name}:**\n"]
        for inp in agent_info.required_inputs:
            lines.append(f"• **{inp.name}**: {inp.description}")
        
        if agent_info.optional_inputs:
            lines.append("\n**Optional:**")
            for inp in agent_info.optional_inputs:
                lines.append(f"• {inp.name}: {inp.description}")
        
        return "\n".join(lines)
    
    def _format_missing_inputs_request(
        self,
        agent_info: AgentInfo,
        missing: list,
        available: Dict[str, Any]
    ) -> str:
        """Format a request for missing inputs"""
        lines = [f"To run **{agent_info.display_name}**, I need:\n"]
        
        for inp in missing:
            if inp.is_file:
                lines.append(f"📎 **{inp.name}** - {inp.description}")
                if inp.example:
                    lines.append(f"   Example: {inp.example}")
            else:
                lines.append(f"✏️ **{inp.name}** - {inp.description}")
                if inp.example:
                    lines.append(f"   Example: {inp.example}")
        
        lines.append("\n💡 **Tip:** Upload files in the sidebar, then send me a message (e.g., \"disease is lupus\") and I'll start automatically!")
        
        return "\n".join(lines)
    
    def _format_input_instructions(self, missing: list) -> str:
        """Format detailed instructions for providing inputs - simplified to avoid clutter"""
        # Return empty to avoid showing redundant instructions in UI
        return ""
    
    def _format_completion_message(self, agent_info: AgentInfo, outputs: Dict[str, Any]) -> str:
        """Format a completion message with results"""
        lines = [f"🎉 **{agent_info.display_name}** completed successfully!\n"]
        
        if agent_info.agent_type == AgentType.COHORT_RETRIEVAL:
            summary = outputs.get("cohort_summary_text", "")
            lines.append(summary[:500] if summary else "Datasets retrieved successfully.")
        
        elif agent_info.agent_type == AgentType.DEG_ANALYSIS:
            lines.append("Differential expression analysis complete.")
            if outputs.get("deg_base_dir"):
                lines.append(f"\n📁 Results saved to: `{outputs['deg_base_dir']}`")
        
        elif agent_info.agent_type == AgentType.GENE_PRIORITIZATION:
            lines.append("Gene prioritization complete.")
            if outputs.get("prioritized_genes_path"):
                lines.append(f"\n📁 Prioritized genes: `{Path(outputs['prioritized_genes_path']).name}`")
        
        elif agent_info.agent_type == AgentType.PATHWAY_ENRICHMENT:
            summary = outputs.get("pathway_analysis_summary", "")
            lines.append(summary[:500] if summary else "Pathway enrichment complete.")
            if outputs.get("pathway_consolidation_path"):
                lines.append(f"\n📁 Results: `{Path(outputs['pathway_consolidation_path']).name}`")
        
        elif agent_info.agent_type == AgentType.DECONVOLUTION:
            lines.append(f"Cell type deconvolution complete using {outputs.get('deconvolution_technique', 'xCell')}.")
            if outputs.get("deconvolution_output_dir"):
                lines.append(f"\n📁 Results: `{outputs['deconvolution_output_dir']}`")
        
        elif agent_info.agent_type == AgentType.PERTURBATION_ANALYSIS:
            lines.append("Drug perturbation analysis complete (DEPMAP + L1000 + Integration).")
            if outputs.get("perturbation_output_dir"):
                lines.append(f"\n📁 Results: `{outputs['perturbation_output_dir']}`")
        
        elif agent_info.agent_type == AgentType.MULTIOMICS_INTEGRATION:
            lines.append("Multi-omics integration complete (integration + biomarkers + cross-omics + literature).")
            if outputs.get("multiomics_output_dir"):
                lines.append(f"\n📁 Results: `{outputs['multiomics_output_dir']}`")
        
        # Suggest next steps
        lines.append("\n**What's next?**")
        if agent_info.agent_type == AgentType.DEG_ANALYSIS:
            lines.append("• Run gene prioritization to rank the DEGs by disease relevance")
        elif agent_info.agent_type == AgentType.GENE_PRIORITIZATION:
            lines.append("• Run pathway enrichment to understand affected biological processes")
        elif agent_info.agent_type == AgentType.COHORT_RETRIEVAL:
            lines.append("• Run DEG analysis on the downloaded datasets")
        elif agent_info.agent_type == AgentType.PATHWAY_ENRICHMENT:
            lines.append("• Run perturbation analysis to identify drug targets and essential genes")
        
        return "\n".join(lines)
    
    def _format_outputs(self, agent_info: AgentInfo, outputs: Dict[str, Any]) -> str:
        """Format output details"""
        lines = ["**Generated Outputs:**\n"]
        
        for output_spec in agent_info.outputs:
            if output_spec.state_key in outputs:
                value = outputs[output_spec.state_key]
                if value:
                    # Truncate long paths
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = "..." + display_value[-57:]
                    lines.append(f"• {output_spec.name}: `{display_value}`")
        
        return "\n".join(lines)
    
    async def _process_message_langgraph(
        self,
        user_message: str,
        session_id: Optional[str],
        uploaded_files: Optional[Dict[str, str]],
        state: ConversationState,
    ) -> AsyncGenerator[StatusUpdate, None]:
        """Bridge to LangGraph supervisor, enriching updates and syncing state back."""
        from .langgraph.entry import run_supervisor_stream

        flat_files = {name: uf.filepath for name, uf in state.uploaded_files.items()}
        if uploaded_files:
            flat_files.update(uploaded_files)

        last_update = None
        async for update in run_supervisor_stream(
            user_query=user_message,
            session_id=session_id or state.session_id,
            disease_name=state.current_disease or "",
            uploaded_files=flat_files or None,
            workflow_outputs=state.workflow_state or None,
        ):
            # Enrich pipeline-agent completions with generated files for Streamlit
            if (update.status_type == StatusType.COMPLETED
                    and update.agent_name
                    and update.agent_name != "response"):
                output_dir = self._get_output_dir_for_agent(update.agent_name, state.workflow_state)
                if output_dir:
                    update.generated_files = _collect_generated_files(output_dir)
                    update.output_dir = output_dir

            yield update
            last_update = update

        # Sync graph results back to ConversationState for cross-call continuity
        if last_update:
            graph_state = getattr(last_update, '_graph_state', None)
            if graph_state:
                state.workflow_state.update(graph_state.get("workflow_outputs", {}))
                if graph_state.get("disease_name"):
                    state.current_disease = graph_state["disease_name"]
            if last_update.message and last_update.status_type == StatusType.COMPLETED:
                state.add_assistant_message(last_update.message)

    def _get_output_dir_for_agent(self, agent_name: str, workflow_state: Dict[str, Any]) -> Optional[str]:
        """Get the output directory for a specific agent from workflow state"""
        # Map agent names to their output directory keys
        output_dir_keys = {
            "cohort_retrieval": "cohort_output_dir",
            "deg_analysis": "deg_base_dir",
            "gene_prioritization": "prioritized_genes_path",  # File path - get parent dir
            "pathway_enrichment": "pathway_consolidation_path",  # File path - get parent dir
            "deconvolution": "deconvolution_output_dir",
            "perturbation_analysis": "perturbation_output_dir",
            "multiomics_integration": "multiomics_output_dir",
            "fastq_processing": "fastq_output_dir",
            "molecular_report": "report_output_dir",
            "crispr_perturb_seq": "crispr_perturb_seq_output_dir",
            "crispr_screening": "crispr_screening_output_dir",
            "crispr_targeted": "crispr_targeted_output_dir",
        }
        
        key = output_dir_keys.get(agent_name)
        if not key or key not in workflow_state:
            return None
        
        path_value = workflow_state[key]
        if not path_value:
            return None
        
        path = Path(path_value)
        # If it's a file, return parent directory
        if path.is_file():
            return str(path.parent)
        return str(path)
    
    def _extract_inputs_from_message(
        self, 
        message: str, 
        required_inputs: List[str]
    ) -> Dict[str, Any]:
        """Extract input values from user message"""
        return _extract_contextual_inputs(message, set(required_inputs))
    
    async def _execute_pending_agent(
        self,
        agent_info: AgentInfo,
        inputs: Dict[str, Any],
        state: ConversationState
    ) -> AsyncGenerator[StatusUpdate, None]:
        """Execute a pending agent after inputs are provided"""
        
        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title=f"🚀 Starting {agent_info.display_name}",
            message=f"All requirements satisfied! Beginning analysis...",
            details=f"**Estimated time:** {agent_info.estimated_time}",
            agent_name=agent_info.name,
            progress=0.0
        )
        
        # Start agent execution tracking
        state.start_agent_execution(
            agent_name=agent_info.name,
            agent_display_name=agent_info.display_name,
            inputs=inputs
        )
        
        # Clear pending state
        state.pending_agent_type = None
        
        # Execute the agent and yield progress updates
        try:
            executor = self._agent_executors.get(agent_info.agent_type)
            if not executor:
                raise ValueError(f"No executor registered for {agent_info.agent_type}")
            
            last_status_type = None
            async for update in executor(agent_info, inputs, state):
                last_status_type = update.status_type
                yield update
            
            # Don't mark as completed if agent is waiting for user input
            if state.waiting_for_input:
                logger.info(f"[{agent_info.name}] Awaiting user input, not marking as completed")
                return
            
            # Agent completed successfully
            state.complete_agent_execution(state.workflow_state)
            
            # Track this agent as executed in the current run
            executed_this_run = state.workflow_state.get("_executed_agents_this_run", [])
            if agent_info.name not in executed_this_run:
                executed_this_run.append(agent_info.name)
                state.workflow_state["_executed_agents_this_run"] = executed_this_run
            
            # Generate completion message
            completion_msg = self._format_completion_message(agent_info, state.workflow_state)
            
            yield StatusUpdate(
                status_type=StatusType.COMPLETED,
                title=f"✅ {agent_info.display_name} Complete!",
                message=completion_msg,
                details=self._format_outputs(agent_info, state.workflow_state),
                agent_name=agent_info.name,
                progress=1.0
            )
            
            state.add_assistant_message(completion_msg)
            
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            state.fail_agent_execution(str(e))
            
            yield StatusUpdate(
                status_type=StatusType.ERROR,
                title=f"❌ {agent_info.display_name} Failed",
                message=f"An error occurred: {str(e)}",
                details="",  # Don't include details that might render as HTML
                agent_name=agent_info.name
            )
            
            state.add_assistant_message(f"I encountered an error: {str(e)}\n\nPlease check your inputs and try again.")
