"""
Pipeline Executor Functions

This module contains executor functions for each agent type.
Each executor:
1. Takes agent info, inputs, and state
2. Calls the actual agent pipeline function
3. Yields StatusUpdate objects for real-time UI feedback
4. Updates workflow state with outputs
"""

import asyncio
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime

from .base import StatusType, StatusUpdate

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _ensure_conda_env_on_path() -> None:
    """Ensure conda env bin dirs and JVM are visible to subprocesses.

    Resolves 'samtools', 'java', etc. not found when running from .venv
    by prepending the conda env ``bin/`` to PATH and setting NXF_JAVA_HOME.
    The .venv bin is kept first so its nextflow (v25+) shadows any older
    conda-installed version.
    """
    # --- 1. Discover conda env bin dir containing samtools/java ---
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    bin_candidates = [
        Path(conda_prefix) / "bin" if conda_prefix else None,
        *sorted(Path.home().glob("miniconda3/envs/*/bin")),
        *sorted(Path.home().glob("anaconda3/envs/*/bin")),
    ]
    conda_bin: str | None = None
    for bin_dir in bin_candidates:
        if bin_dir and bin_dir.is_dir():
            if (bin_dir / "samtools").is_file() or (bin_dir / "java").is_file():
                conda_bin = str(bin_dir)
                break

    # --- 2. Rebuild PATH: .venv/bin first, then conda, then the rest ---
    venv_bin = str(Path(sys.prefix) / "bin")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    # Strip both to avoid duplicates
    parts = [p for p in parts if p not in (venv_bin, conda_bin)]
    head = [d for d in (venv_bin, conda_bin) if d and Path(d).is_dir()]
    os.environ["PATH"] = os.pathsep.join(head + parts)
    if conda_bin:
        logger.info("PATH: %s (venv) > %s (conda) > system", venv_bin, conda_bin)

    # --- 3. Set NXF_JAVA_HOME for Nextflow ---
    if not (os.environ.get("NXF_JAVA_HOME") or os.environ.get("JAVA_HOME")):
        jvm_candidates = [
            Path(conda_prefix) / "lib" / "jvm" if conda_prefix else None,
            *sorted(Path.home().glob("miniconda3/envs/*/lib/jvm")),
            *sorted(Path.home().glob("anaconda3/envs/*/lib/jvm")),
        ]
        for jvm_dir in jvm_candidates:
            if jvm_dir and (jvm_dir / "bin" / "java").is_file():
                os.environ["NXF_JAVA_HOME"] = str(jvm_dir)
                logger.info("NXF_JAVA_HOME set to %s", jvm_dir)
                break


def _cleanup_nextflow_output(output_dir: Path) -> None:
    """Flatten results/ into output_dir and remove Nextflow working dirs.

    Produces a clean output matching the pipeline's canonical structure
    (crisprseq_targeted/, report, samplesheet) without intermediate dirs.
    """
    results_dir = output_dir / "results"
    if results_dir.is_dir():
        for child in results_dir.iterdir():
            dest = output_dir / child.name
            if not dest.exists():
                child.rename(dest)
        # Remove results/ if now empty
        try:
            results_dir.rmdir()
        except OSError:
            pass

    for name in ("work", "tmp", ".nextflow"):
        target = output_dir / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
    nxf_log = output_dir / ".nextflow.log"
    if nxf_log.is_file():
        nxf_log.unlink(missing_ok=True)


def _can_prepare_fasta_index(fasta_path: Path) -> bool:
    if Path(f"{fasta_path}.fai").is_file():
        return True
    if shutil.which("samtools"):
        return True
    try:
        import pysam  # type: ignore
    except Exception:
        return False
    return pysam is not None


def _get_output_dir(agent_name: str, session_id: str) -> Path:
    """Get output directory for an agent using session_id"""
    if not _SAFE_ID_RE.match(session_id) or not _SAFE_ID_RE.match(agent_name):
        raise ValueError("Invalid characters in session_id or agent_name")
    base_dir = Path(__file__).parent.parent / "outputs" / agent_name / session_id
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _get_user_query_from_state(state: 'ConversationState') -> Optional[str]:
    """Extract the last user message from state"""
    from ..state import MessageRole
    for msg in reversed(state.messages):
        if msg.role == MessageRole.USER:
            return msg.content
    return None


def _extract_cohort_summary(result: Any, disease_name: str) -> str:
    """Extract summary string from cohort result"""
    if isinstance(result, str):
        return result[:500]
    
    if isinstance(result, dict):
        if "summary" in result and isinstance(result["summary"], str):
            return result["summary"][:500]
        
        # Navigate nested result structure
        res = result.get("result", {})
        if isinstance(res, dict):
            inner = res.get("result")
            if hasattr(inner, "total_datasets_found"):
                return f"Found {inner.total_datasets_found} datasets, downloaded {inner.total_datasets_downloaded}"
            if isinstance(inner, dict):
                found = inner.get("total_datasets_found", "N/A")
                downloaded = inner.get("total_datasets_downloaded", "N/A")
                return f"Found {found} datasets, downloaded {downloaded}"
    
    if hasattr(result, "total_datasets_found"):
        return f"Found {result.total_datasets_found} datasets, downloaded {result.total_datasets_downloaded}"
    
    return f"Cohort retrieval completed for {disease_name}"


def _collect_files(output_dir: Path) -> List[str]:
    """Collect generated files from output directory"""
    if not output_dir or not output_dir.exists():
        return []
    extensions = ['.png', '.jpg', '.jpeg', '.csv', '.tsv', '.xlsx', '.json', '.pdf', '.html']
    files = []
    for ext in extensions:
        files.extend([str(f) for f in output_dir.glob(f"**/*{ext}")])
    return sorted(files)


# =============================================================================
# COHORT RETRIEVAL EXECUTOR
# =============================================================================

async def execute_cohort_retrieval(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the cohort retrieval agent"""
    from agentic_ai_wf.cohort_retrieval_agent.cohortagent import cohortagent
    
    disease_name = inputs.get("disease_name", "unknown_disease")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🔍 Searching Public Databases",
        message=f"Querying GEO and ArrayExpress for **{disease_name}** datasets...",
        details="This may take 1-5 minutes depending on the number of matching datasets.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    # Use session_id for output directory - cohortagent adds disease subfolder internally
    output_dir = _get_output_dir("cohort_retrieval", state.session_id)
    
    # Get actual user query for proper LLM filter extraction
    user_query = _get_user_query_from_state(state) or f"Find datasets for {disease_name}"
    
    try:
        result = await cohortagent(
            user_query=user_query,
            output_dir=str(output_dir)
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Processing Results",
            message="Analyzing and filtering retrieved datasets...",
            progress=0.7
        )
        
        # Find actual output directory (cohortagent creates disease subfolder)
        actual_output_dir = output_dir
        disease_safe = disease_name.lower().replace(" ", "_")
        disease_folder = output_dir / disease_safe
        if disease_folder.exists():
            actual_output_dir = disease_folder
        
        state.workflow_state["cohort_output_dir"] = str(actual_output_dir)
        state.workflow_state["cohort_result"] = result
        
        summary_text = _extract_cohort_summary(result, disease_name)
        state.workflow_state["cohort_summary_text"] = summary_text
        
        generated_files = _collect_files(actual_output_dir)
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Cohort Retrieval Complete",
            message=f"Retrieved datasets for {disease_name}",
            details=summary_text,
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(actual_output_dir),
            generated_files=generated_files
        )
        
    except Exception as e:
        logger.exception(f"Cohort retrieval failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Cohort Retrieval Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# DEG ANALYSIS EXECUTOR
# =============================================================================

def _prepare_single_file_for_deg(counts_file: str, metadata_file: Optional[str], 
                                  output_dir: Path, disease_name: str,
                                  session_id: str) -> Path:
    """
    Prepare a single uploaded counts file for DEG analysis.
    Creates the directory structure and metadata that DEG agent expects.
    
    Args:
        counts_file: Path to the uploaded counts CSV file
        metadata_file: Optional path to metadata file
        output_dir: Base output directory
        disease_name: Disease name for sample naming
        session_id: Session ID to use as folder name (for column naming consistency)
        
    Returns:
        Path to the prepared input directory
    """
    import shutil
    import pandas as pd
    
    counts_path = Path(counts_file)
    
    # Create sample directory structure: input_data/{session_id}/
    # Using session_id as folder name ensures column naming matches (like the working pipeline)
    sample_dir = output_dir / "input_data" / session_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy counts file to sample directory
    dest_counts = sample_dir / counts_path.name
    shutil.copy2(counts_file, dest_counts)
    logger.info(f"Copied counts file to: {dest_counts}")
    
    # Handle metadata
    if metadata_file and Path(metadata_file).exists():
        # Copy provided metadata
        dest_meta = sample_dir / Path(metadata_file).name
        shutil.copy2(metadata_file, dest_meta)
        logger.info(f"Copied metadata file to: {dest_meta}")
    else:
        # Auto-generate metadata from counts file columns
        try:
            # Read counts to get sample column names
            counts_df = pd.read_csv(dest_counts, index_col=0, nrows=0)
            sample_ids = list(counts_df.columns)
            
            # Infer conditions from sample names (like DEG agent does internally)
            conditions = []
            control_keywords = ['control', 'ctrl', 'normal', 'wt', 'wildtype', 'healthy']
            for sample in sample_ids:
                sample_lower = sample.lower()
                if any(kw in sample_lower for kw in control_keywords):
                    conditions.append('Control')
                else:
                    conditions.append('Disease')
            
            # Create metadata DataFrame
            metadata_df = pd.DataFrame({
                'sample_id': sample_ids,
                'condition': conditions
            })
            
            # Save metadata file
            metadata_path = sample_dir / "metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            logger.info(f"Auto-generated metadata with {len(sample_ids)} samples: {metadata_path}")
            logger.info(f"  - Control samples: {conditions.count('Control')}")
            logger.info(f"  - Disease samples: {conditions.count('Disease')}")
            
        except Exception as e:
            logger.warning(f"Could not auto-generate metadata: {e}")
    
    # Return the parent directory (input_data), not the sample directory
    # DEG agent will walk into sample_dir and find the files
    return output_dir / "input_data"


async def execute_deg_analysis(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the DEG (Differential Expression Gene) analysis agent"""
    from agentic_ai_wf.deg_pipeline_agent import DEGPipelineAgent, DEGPipelineConfig
    
    disease_name = inputs.get("disease_name", "Disease")
    counts_file = inputs.get("counts_file") or inputs.get("bulk_file")
    metadata_file = inputs.get("metadata_file")
    geo_dir = inputs.get("cohort_output_dir") or inputs.get("geo_dir")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="📊 Running DEG Analysis",
        message=f"Analyzing differential gene expression for **{disease_name}**...",
        details="This typically takes 2-5 minutes depending on data size.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("deg_analysis", state.session_id)
    
    # Prepare input directory for DEG agent
    analysis_dir = None
    if counts_file and Path(counts_file).is_file():
        # Single file upload - prepare directory structure with auto-generated metadata
        analysis_dir = _prepare_single_file_for_deg(
            counts_file, metadata_file, output_dir, disease_name, state.session_id
        )
    elif counts_file and Path(counts_file).is_dir():
        analysis_dir = Path(counts_file)
    
    try:
        # Configure the DEG pipeline - requires either geo_dir or analysis_transcriptome_dir
        config = DEGPipelineConfig(
            disease_name=disease_name,
            output_dir=str(output_dir),
            geo_dir=geo_dir,
            analysis_transcriptome_dir=str(analysis_dir) if analysis_dir else None
        )
        
        # Create and run the agent
        agent = DEGPipelineAgent(config)
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Processing Data",
            message="Running statistical analysis...",
            progress=0.4
        )
        
        # Run pipeline in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run_pipeline(
                geo_dir=geo_dir,
                analysis_transcriptome_dir=str(analysis_dir) if analysis_dir else None,
                disease_name=disease_name
            )
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📈 Generating Results",
            message="Creating output files and visualizations...",
            progress=0.8
        )
        
        # Update state with outputs
        state.workflow_state["deg_base_dir"] = str(output_dir)
        state.workflow_state["deg_result"] = result
        
        # Look for DEG output files (case-insensitive patterns for *_DEGs.csv)
        deg_files = (
            list(output_dir.glob("**/*_DEGs.csv")) +  # Standard format: sample_DEGs.csv
            list(output_dir.glob("**/*DEGs*.csv")) +  # Any file with DEGs in name
            list(output_dir.glob("**/*_degs.csv"))    # Lowercase variant
        )
        if deg_files:
            state.workflow_state["deg_input_file"] = str(deg_files[0])
            logger.info(f"Found DEG output file: {deg_files[0]}")
        else:
            logger.warning(f"No DEG files found in {output_dir}")
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ DEG Analysis Complete",
            message=f"Differential expression analysis finished for {disease_name}",
            details=f"Results saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logger.exception(f"DEG analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ DEG Analysis Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# GENE PRIORITIZATION EXECUTOR
# =============================================================================

async def execute_gene_prioritization(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the gene prioritization agent"""
    from agentic_ai_wf.gene_prioritization.deg_filtering import run_deg_filtering
    
    disease_name = inputs.get("disease_name", "Disease")
    deg_base_dir = inputs.get("deg_base_dir")
    deg_input_file = inputs.get("deg_input_file")
    
    # If we have the DEG file path, use its parent directory as deg_base_dir
    # The gene prioritization agent expects DEG files directly in the base_dir
    # Also extract analysis_id from the sample folder name for column matching
    analysis_id = None
    if deg_input_file and Path(deg_input_file).exists():
        deg_file_path = Path(deg_input_file)
        deg_base_dir = str(deg_file_path.parent)
        # Extract sample name from parent folder (e.g., "RW-20251119_ Sarcoidosis_bis")
        analysis_id = deg_file_path.parent.name
        logger.info(f"Using DEG file's parent directory: {deg_base_dir}")
        logger.info(f"Using analysis_id: {analysis_id}")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🎯 Prioritizing Genes",
        message=f"Ranking genes by disease relevance for **{disease_name}**...",
        details="This typically takes 2-5 minutes.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("gene_prioritization", state.session_id)
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Analyzing Gene Relevance",
            message="Scoring genes based on disease associations...",
            progress=0.3
        )
        
        # Run gene prioritization in thread pool
        loop = asyncio.get_event_loop()
        result_path = await loop.run_in_executor(
            None,
            lambda: run_deg_filtering(
                patient_prefix=analysis_id,
                deg_base_dir=Path(deg_base_dir) if deg_base_dir else None,
                disease_name=disease_name,
                output_dir=Path(output_dir),
                analysis_id=analysis_id
            )
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Generating Output",
            message="Creating prioritized gene list...",
            progress=0.8
        )
        
        # Update state with outputs
        state.workflow_state["prioritized_genes_path"] = str(result_path) if result_path else None
        state.workflow_state["gene_prioritization_output_dir"] = str(output_dir)
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Gene Prioritization Complete",
            message=f"Genes prioritized for {disease_name}",
            details=f"Output: {result_path.name if result_path else 'N/A'}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logger.exception(f"Gene prioritization failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Gene Prioritization Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# PATHWAY ENRICHMENT EXECUTOR
# =============================================================================

async def execute_pathway_enrichment(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the pathway enrichment agent with full pipeline (enrichment → consolidation)"""
    from agentic_ai_wf.pathway_agent.agent_runner import run_autonomous_analysis
    
    disease_name = inputs.get("disease_name", "Disease")
    gene_file_path = inputs.get("prioritized_genes_path") or inputs.get("gene_file_path")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🛤️ Running Pathway Analysis",
        message=f"Analyzing biological pathways for **{disease_name}**...",
        details="This runs enrichment, deduplication, categorization, literature analysis, and consolidation.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    # Create session-specific output dir; use 'enrichment' subfolder so consolidation
    # (which uses .parent) stays within the session folder
    session_output_dir = _get_output_dir("pathway_enrichment", state.session_id)
    output_dir = session_output_dir / "enrichment"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = state.session_id
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Running Full Pathway Pipeline",
            message="Enrichment → Deduplication → Categorization → Literature → Consolidation...",
            progress=0.3
        )
        
        # Run the full autonomous pathway analysis (all 5 stages)
        result = await run_autonomous_analysis(
            deg_file_path=Path(gene_file_path),
            disease_name=disease_name,
            patient_prefix=analysis_id,
            output_dir=Path(output_dir),
            use_simple_workflow=True  # Use simple workflow for reliability
        )
        
        # Extract outputs from result
        consolidation_path = result.get("output_file")  # Final consolidated file
        enrichment_path = result.get("enrichment_output")  # Enrichment file
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Finalizing Results",
            message="Pathway consolidation complete...",
            progress=0.9
        )
        
        # Update state with outputs (consolidation is the main output)
        state.workflow_state["pathway_consolidation_path"] = str(consolidation_path) if consolidation_path else None
        state.workflow_state["pathway_enrichment_path"] = str(enrichment_path) if enrichment_path else None
        state.workflow_state["pathway_output_dir"] = str(session_output_dir)
        
        # Determine which file to report
        final_file = consolidation_path or enrichment_path
        file_name = Path(final_file).name if final_file else "N/A"
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Pathway Analysis Complete",
            message=f"Pathway analysis finished for {disease_name}",
            details=f"Consolidated pathways: {file_name}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(session_output_dir)
        )
        
    except Exception as e:
        logger.exception(f"Pathway analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Pathway Analysis Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# DECONVOLUTION EXECUTOR
# =============================================================================

async def execute_deconvolution(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the cell type deconvolution agent"""
    from agentic_ai_wf.deconv_pipeline_agent.single_cell_deconv import run_pipeline
    
    disease_name = inputs.get("disease_name", "Disease")
    bulk_file = inputs.get("bulk_file") or inputs.get("counts_file")
    metadata_file = inputs.get("metadata_file")
    # Let orchestrator decide technique (None) unless user explicitly specified
    technique = inputs.get("deconvolution_technique")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🧬 Running Cell Deconvolution",
        message=f"Estimating cell type composition for **{disease_name}**...",
        details="Technique will be auto-selected by orchestrator" if not technique else f"Using technique: {technique}",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("deconvolution", state.session_id)
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Processing Expression Data",
            message="Running deconvolution algorithm...",
            progress=0.3
        )
        
        # Get SC reference base directory from supervisor agent location
        sc_base_dir = Path(__file__).parent.parent
        
        # Run deconvolution in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                bulk_file=bulk_file,
                metadata=metadata_file,
                output_dir=str(output_dir),
                technique=technique,
                disease_name=disease_name,
                sc_base_dir=str(sc_base_dir)
            )
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Generating Results",
            message="Creating cell composition estimates...",
            progress=0.8
        )
        
        # Update state with outputs - technique may have been auto-selected
        state.workflow_state["deconvolution_output_dir"] = str(output_dir)
        state.workflow_state["deconvolution_technique"] = technique or "auto-selected"
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Deconvolution Complete",
            message=f"Cell type analysis finished for {disease_name}",
            details=f"Results saved to: {output_dir.name}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logger.exception(f"Deconvolution failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Deconvolution Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# TEMPORAL ANALYSIS EXECUTOR
# =============================================================================

async def execute_temporal_analysis(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the temporal analysis agent"""
    from agentic_ai_wf.temporal_pipeline_agent.temporal_bulk.runner import run_temporal_analysis
    
    disease_name = inputs.get("disease_name", "Disease")
    counts_file = inputs.get("counts_file")
    metadata_file = inputs.get("metadata_file")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="⏱️ Running Temporal Analysis",
        message=f"Analyzing time-series expression data for **{disease_name}**...",
        details="This may take several minutes.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("temporal_analysis", state.session_id)
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Processing Time Points",
            message="Analyzing temporal expression patterns...",
            progress=0.3
        )
        
        # Run temporal analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_temporal_analysis(
                output_dir=str(output_dir),
                counts=counts_file,
                metadata=metadata_file
            )
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Generating Results",
            message="Creating temporal analysis output...",
            progress=0.8
        )
        
        # Update state with outputs
        state.workflow_state["temporal_output_dir"] = str(output_dir)
        state.workflow_state["temporal_result"] = result
        
        # Collect generated files for download tracking
        generated_files = []
        for ext in ['*.csv', '*.png', '*.html', '*.pdf']:
            generated_files.extend([str(f) for f in output_dir.glob(f'**/{ext}')])
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Temporal Analysis Complete",
            message=f"Time-series analysis finished for {disease_name}",
            details=f"Results saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir),
            generated_files=generated_files[:50]  # Limit to 50 files
        )
        
    except Exception as e:
        logger.exception(f"Temporal analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Temporal Analysis Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# HARMONIZATION EXECUTOR
# =============================================================================

async def execute_harmonization(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the data harmonization agent"""
    from agentic_ai_wf.harmonization_pipeline_agent.harmonizer.harmonizer import harmonize_single
    
    disease_name = inputs.get("disease_name", "Disease")
    counts_file = inputs.get("counts_file")
    metadata_file = inputs.get("metadata_file")
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🔄 Running Data Harmonization",
        message=f"Harmonizing expression data for **{disease_name}**...",
        details="Standardizing gene IDs and normalizing expression values.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("harmonization", state.session_id)
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Processing Data",
            message="Converting gene identifiers and normalizing...",
            progress=0.3
        )
        
        # Run harmonization in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: harmonize_single(
                counts_path=counts_file,
                meta_path=metadata_file,
                output_dir=str(output_dir)
            )
        )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Generating Output",
            message="Creating harmonized data files...",
            progress=0.8
        )
        
        # Update state with outputs
        state.workflow_state["harmonization_output_dir"] = str(output_dir)
        state.workflow_state["harmonization_result"] = result
        
        # Look for harmonized output files
        harmonized_files = list(output_dir.glob("*harmonized*.csv"))
        if harmonized_files:
            state.workflow_state["harmonized_counts_file"] = str(harmonized_files[0])
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Harmonization Complete",
            message=f"Data harmonization finished for {disease_name}",
            details=f"Results saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logger.exception(f"Harmonization failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Harmonization Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# MDP ANALYSIS EXECUTOR
# =============================================================================

def _setup_mdp_environment():
    """Setup environment for MDP pipeline subprocess calls.
    
    The MDP pipeline spawns subprocesses that need PYTHONPATH to be set
    correctly to find the agentic_ai_wf module.
    """
    # Get the project root (parent of agentic_ai_wf)
    project_root = Path(__file__).resolve().parents[3]  # Up from executors -> supervisor_agent -> agentic_ai_wf -> agenticaib
    project_root_str = str(project_root)
    
    # Ensure project root is in PYTHONPATH for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root_str not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{project_root_str}:{current_pythonpath}" if current_pythonpath else project_root_str
    
    # Also ensure it's in sys.path for direct imports
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Save original cwd and change to project root for subprocess compatibility
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    return original_cwd


async def execute_mdp_analysis(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the MDP (Molecular Disease Portrait) analysis agent
    
    Supports multiple input modes:
    - Single disease with counts file
    - Single disease with DEG/gene list file  
    - Multi-disease with items list (each can have file or use KG)
    """
    from agentic_ai_wf.mdp_pipeline_agent.main import mdp_pipeline_direct
    
    # Setup environment for MDP subprocesses
    original_cwd = _setup_mdp_environment()
    
    # Extract inputs
    disease_name = inputs.get("disease_name", "")
    disease_names = inputs.get("disease_names", [])  # Multi-disease list
    file_assignments = inputs.get("file_assignments", {})  # {disease: file_path}
    
    # Input files (priority: counts > degs > gene list)
    # Handle both raw paths and UploadedFile objects
    def _get_filepath(val):
        if val is None:
            return None
        if hasattr(val, 'filepath'):
            return val.filepath  # UploadedFile object
        return str(val)  # Already a path string
    
    counts_file = _get_filepath(inputs.get("counts_file") or inputs.get("bulk_file"))
    deg_file = _get_filepath(inputs.get("deg_file") or inputs.get("prioritized_genes_path"))
    gene_list_file = _get_filepath(inputs.get("gene_list_file"))
    tissue = inputs.get("tissue", "")
    
    # Determine primary input file
    input_file = counts_file or deg_file or gene_list_file
    
    # Build disease list (handle both single and multi-disease)
    all_diseases = disease_names if disease_names else ([disease_name] if disease_name else [])
    
    yield StatusUpdate(
        status_type=StatusType.EXECUTING,
        title="🎨 Running MDP Analysis",
        message=f"Creating molecular disease portrait for **{', '.join(all_diseases) or 'Disease'}**...",
        details="Analyzing pathway activity and cross-disease comparison.",
        agent_name=agent_info.name,
        progress=0.1
    )
    
    output_dir = _get_output_dir("mdp_analysis", state.session_id)
    
    try:
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Analyzing Molecular Profile",
            message="Building disease portrait from gene data...",
            progress=0.3
        )
        
        # Build items list for MDP pipeline
        # Format: "name=disease,input=/path/to/file" or "name=disease" (for KG fallback)
        items = []
        
        if len(all_diseases) > 1:
            # Multi-disease mode: build items list
            for disease in all_diseases:
                file_val = file_assignments.get(disease)
                file_path = _get_filepath(file_val)
                if file_path:
                    items.append(f"name={disease},input={file_path}")
                else:
                    items.append(f"name={disease}")  # Uses Neo4j KG
        elif len(all_diseases) == 1 and input_file:
            # Single disease with file
            items.append(f"name={all_diseases[0]},input={input_file}")
        elif len(all_diseases) == 1:
            # Single disease, no file (KG fallback)
            items.append(f"name={all_diseases[0]}")
        
        # Run MDP pipeline in thread pool
        loop = asyncio.get_event_loop()
        
        if items:
            result = await loop.run_in_executor(
                None,
                lambda: mdp_pipeline_direct(
                    mode="auto",
                    items=items,
                    output_dir=str(output_dir),
                    tissue=tissue,
                    run_enzymes=True,
                    run_analyses=True,
                    report_q_cutoff=0.05,  # Default cutoff for report generation
                    report_no_llm=False,
                    skip_report=False
                )
            )
        else:
            # Fallback: direct mode if no items built
            result = await loop.run_in_executor(
                None,
                lambda: mdp_pipeline_direct(
                    mode="counts" if counts_file else "degs" if deg_file else "gl",
                    input_path=input_file,
                    output_dir=str(output_dir),
                    disease_name=disease_name,
                    tissue=tissue,
                    run_enzymes=True,
                    run_analyses=True,
                    report_q_cutoff=0.05,
                    report_no_llm=False,
                    skip_report=False
                )
            )
        
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📊 Generating Portrait",
            message="Creating visualizations and reports...",
            progress=0.8
        )
        
        # Update state with outputs
        state.workflow_state["mdp_output_dir"] = str(output_dir)
        state.workflow_state["mdp_result"] = result
        
        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ MDP Analysis Complete",
            message=f"Molecular disease portrait created for {', '.join(all_diseases) or disease_name}",
            details=f"Results saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logger.exception(f"MDP analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ MDP Analysis Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


# =============================================================================
# PERTURBATION ANALYSIS EXECUTOR
# =============================================================================

async def execute_perturbation_analysis(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """
    Execute perturbation analysis pipeline (DEPMAP + L1000 + Integration).

    Required inputs:
      - prioritized_genes_path: CSV with prioritized genes
      - pathway_consolidation_path: CSV with consolidated pathway results
      - disease_name: disease context string

    Optional inputs:
      - dep_map_mode, genes_selection, l1000_tissue, l1000_drug
    """
    original_cwd = os.getcwd()

    try:
        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="💊 Starting Perturbation Analysis",
            message="Initializing DEPMAP + L1000 perturbation pipeline...",
            agent_name=agent_info.name,
            progress=0.0
        )

        # ---- Resolve inputs ----
        deg_path = inputs.get("prioritized_genes_path") or state.workflow_state.get("prioritized_genes_path")
        pathway_path = inputs.get("pathway_consolidation_path") or state.workflow_state.get("pathway_consolidation_path")
        disease_name = inputs.get("disease_name") or state.workflow_state.get("disease_name", "unknown_disease")

        if not deg_path:
            raise ValueError(
                "Missing prioritized genes file. Please upload a _DEGs_prioritized.csv "
                "or run gene prioritization first."
            )
        if not pathway_path:
            raise ValueError(
                "Missing pathway consolidation file. Please upload a _Pathways_Consolidated.csv "
                "or run pathway enrichment first."
            )

        deg_path = str(deg_path)
        pathway_path = str(pathway_path)

        # Validate files exist
        if not Path(deg_path).exists():
            raise FileNotFoundError(f"Prioritized genes file not found: {deg_path}")
        if not Path(pathway_path).exists():
            raise FileNotFoundError(f"Pathway consolidation file not found: {pathway_path}")

        output_dir = _get_output_dir("perturbation_analysis", state.session_id)

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title=" Running Perturbation Pipeline",
            message=f"Analyzing {disease_name} with DEPMAP + L1000 (this may take 15-45 min)...",
            progress=0.1
        )

        # ---- Build optional addons ----
        dep_map_addons = {
            "mode_model": inputs.get("dep_map_mode"),
            "genes_selection": inputs.get("genes_selection", "all"),
            "top_up": inputs.get("top_up"),
            "top_down": inputs.get("top_down"),
        }
        l1000_addons = {
            "tissue": inputs.get("l1000_tissue"),
            "drug": inputs.get("l1000_drug"),
            "time_points": inputs.get("l1000_time_points"),
            "cell_lines": inputs.get("l1000_cell_lines"),
        }

        # ---- Import and run the pipeline ----
        from agentic_ai_wf.perturbation_pipeline_agent.main import perturbation_pipeline

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: perturbation_pipeline(
                deg_path=deg_path,
                pathway_path=pathway_path,
                output_dir=str(output_dir),
                disease=disease_name,
                dep_map_addons=dep_map_addons,
                l1000_addons=l1000_addons,
                parallel=True
            )
        )

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📦 Collecting Results",
            message="Gathering perturbation analysis outputs...",
            progress=0.9
        )

        # ---- Store results in state ----
        state.workflow_state["perturbation_output_dir"] = str(output_dir)
        state.workflow_state["perturbation_result"] = result

        # Build summary message
        summary_parts = [f"Disease: {disease_name}"]
        if isinstance(result, dict):
            if result.get("status") == "success":
                summary_parts.append("Status: Completed successfully")
            if "results" in result and isinstance(result["results"], dict):
                for step_name, step_result in result["results"].items():
                    if isinstance(step_result, dict) and step_result.get("status") == "success":
                        summary_parts.append(f"  ✓ {step_name}")
        summary = " | ".join(summary_parts)

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Perturbation Analysis Complete",
            message=f"DEPMAP + L1000 perturbation analysis finished for {disease_name}",
            details=f"{summary}\nResults saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"Perturbation analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Perturbation Analysis Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise
    finally:
        os.chdir(original_cwd)


# =============================================================================
# MULTIOMICS INTEGRATION EXECUTOR
# =============================================================================

async def execute_multiomics_integration(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """
    Execute multi-omics integration pipeline.

    Required inputs (resolved from state):
      - multiomics_layers: dict mapping layer name → file path (≥1 layer)
      - disease_name: disease context string

    Optional inputs:
      - metadata_path, label_column, query_term
    """
    try:
        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🧬 Starting Multi-Omics Integration",
            message="Initializing multi-omics pipeline...",
            agent_name=agent_info.name,
            progress=0.0
        )

        # ---- Resolve layers ----
        layers = (
            inputs.get("multiomics_layers")
            or state.workflow_state.get("multiomics_layers")
        )
        if not layers or not isinstance(layers, dict) or len(layers) == 0:
            raise ValueError(
                "No omics layer files found. Please upload at least one layer file "
                "(e.g., genomics, transcriptomics, proteomics, metabolomics, epigenomics)."
            )

        # ---- Resolve other parameters ----
        disease_name = (
            inputs.get("disease_name")
            or state.workflow_state.get("disease_name", "unknown_disease")
        )
        metadata_path = (
            inputs.get("metadata_path")
            or state.workflow_state.get("metadata_path")
        )
        label_column = (
            inputs.get("label_column")
            or state.workflow_state.get("label_column")
        )
        query_term = (
            inputs.get("query_term")
            or state.workflow_state.get("query_term")
            or f"{disease_name} multi-omics biomarkers"
        )

        # ---- Prepare output directory ----
        output_dir = _get_output_dir("multiomics_integration", state.session_id)

        layer_summary = ", ".join(f"{k} ({Path(v).name})" for k, v in layers.items())
        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📂 Layers Identified",
            message=f"Running integration on {len(layers)} layer(s): {layer_summary}",
            progress=0.1
        )

        # ---- Import and run pipeline ----
        from agentic_ai_wf.multiomics_pipeline_agent.main import multiomics_pipeline_direct

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="⚙️ Pipeline Running",
            message="Executing multi-omics integration (ingestion → preprocessing → integration → ML biomarkers → cross-omics → literature)...",
            progress=0.2
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: multiomics_pipeline_direct(
                output_dir=str(output_dir),
                layers=layers,
                metadata_path=metadata_path,
                label_column=label_column,
                disease_term=disease_name,
                query_term=query_term,
            )
        )

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📦 Collecting Results",
            message="Gathering multi-omics integration outputs...",
            progress=0.9
        )

        # ---- Store results in state ----
        state.workflow_state["multiomics_output_dir"] = str(output_dir)
        state.workflow_state["multiomics_result"] = result

        # Build summary
        summary_parts = [f"Disease: {disease_name}", f"Layers: {len(layers)}"]
        if isinstance(result, dict):
            final_status = result.get("final_status", "unknown")
            steps_completed = result.get("steps_completed", [])
            summary_parts.append(f"Status: {final_status}")
            if steps_completed:
                for step in steps_completed:
                    summary_parts.append(f"  ✓ {step}")
        summary = " | ".join(summary_parts)

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Multi-Omics Integration Complete",
            message=f"Multi-omics integration finished for {disease_name} ({len(layers)} layers)",
            details=f"{summary}\nResults saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"Multi-omics integration failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Multi-Omics Integration Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# FASTQ PROCESSING EXECUTOR
# =============================================================================

async def execute_fastq_processing(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the FASTQ processing pipeline.

    The user provides a local filesystem path to a directory containing FASTQ
    files.  The pipeline scans the directory internally to discover samples.
    We run it via run_in_executor because it calls asyncio.run() internally.
    """
    from agentic_ai_wf.fastq_pipeline_agent.fastq.main_module import run_pipeline
    from .base import validate_pipeline_output

    try:
        # ---- Resolve FASTQ input directory ----
        fastq_input_dir = (
            inputs.get("fastq_input_dir")
            or state.workflow_state.get("fastq_input_dir")
        )
        if not fastq_input_dir or not Path(fastq_input_dir).is_dir():
            raise ValueError(
                "No FASTQ directory provided. Please enter the path to a local "
                "folder containing .fastq, .fq, .fastq.gz, or .fq.gz files."
            )

        fastq_input_dir = str(Path(fastq_input_dir).resolve())

        # Quick scan to count files and give user feedback
        fq_suffixes = ('.fastq', '.fq', '.fastq.gz', '.fq.gz')
        fq_count = sum(
            1 for f in Path(fastq_input_dir).rglob("*")
            if f.is_file() and f.name.lower().endswith(fq_suffixes)
        )
        if fq_count == 0:
            raise ValueError(
                f"No FASTQ files found in: {fastq_input_dir}\n"
                "Expected .fastq, .fq, .fastq.gz, or .fq.gz files."
            )

        disease_name = (
            inputs.get("disease_name")
            or state.workflow_state.get("disease_name", "unknown_disease")
        )

        # ---- Prepare output directory ----
        output_dir = _get_output_dir("fastq_processing", state.session_id)

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🧬 FASTQ Pipeline Starting",
            message=f"Processing **{fq_count}** FASTQ file(s) for **{disease_name}**...",
            details=f"Input: {fastq_input_dir}\n"
                    "Steps: FastQC → Trimming → Salmon → Combine → MultiQC.\n"
                    "This may take 10-60+ minutes depending on file sizes.",
            agent_name=agent_info.name,
            progress=0.1
        )

        # ---- Run the synchronous pipeline in a thread ----
        loop = asyncio.get_event_loop()
        result_dir = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                input_path=fastq_input_dir,
                results_root=str(output_dir),
                disease_name=disease_name,
            )
        )

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📦 Collecting Results",
            message="Gathering FASTQ pipeline outputs...",
            progress=0.9
        )

        # ---- Store results in state ----
        state.workflow_state["fastq_output_dir"] = str(result_dir)

        # ---- Validate output ----
        validation = validate_pipeline_output(str(result_dir), "FASTQ Processing")
        logger.info(f"FASTQ output validation:\n{validation.summary}")

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ FASTQ Processing Complete",
            message=f"FASTQ pipeline finished for {disease_name} ({fq_count} files)",
            details=f"{validation.summary}\nResults saved to: {result_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(result_dir)
        )

    except Exception as e:
        logger.exception(f"FASTQ processing failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ FASTQ Processing Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


async def execute_molecular_report(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the molecular report pipeline.

    Wraps the synchronous ReportingPipelineAgent in run_in_executor.
    Output format defaults to PDF; respects explicit user format request.
    """
    from agentic_ai_wf.reporting_pipeline_agent.pipeline_agent import ReportingPipelineAgent
    from agentic_ai_wf.reporting_pipeline_agent.llm_factory import create_llm_client
    from agentic_ai_wf.reporting_pipeline_agent.llm_knowledge import PatientInfoParser

    try:
        genes_path = inputs.get("prioritized_genes_path")
        pathways_path = inputs.get("pathway_consolidation_path")
        disease_name = inputs.get("disease_name", "Disease")

        if not genes_path or not Path(genes_path).is_file():
            raise ValueError("Prioritized genes file is required for molecular report generation.")
        if not pathways_path or not Path(pathways_path).is_file():
            raise ValueError("Pathway consolidation file is required for molecular report generation.")

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="📋 Generating Molecular Report",
            message=f"Starting report pipeline for **{disease_name}**...",
            details="Steps: data loading → disease context → gene/pathway mapping → "
                    "narrative generation → document assembly.\n"
                    "This typically takes 2-10 minutes.",
            agent_name=agent_info.name,
            progress=0.05
        )

        output_dir = _get_output_dir("molecular_report", state.session_id)

        # ── Resolve optional deconvolution data ──
        xcell_path = inputs.get("xcell_path")
        if not xcell_path:
            deconv_dir = inputs.get("deconvolution_output_dir")
            if deconv_dir and Path(deconv_dir).is_dir():
                for pattern in ("CIBERSORT_results*.csv", "xCell_results*.csv", "*deconv*.csv"):
                    hits = list(Path(deconv_dir).rglob(pattern))
                    if hits:
                        xcell_path = str(hits[0])
                        break

        # ── Resolve optional patient info ──
        patient_info = None
        patient_info_path = inputs.get("patient_info_path")
        if patient_info_path and Path(patient_info_path).is_file():
            try:
                llm_for_parse = create_llm_client()
                parser = PatientInfoParser(llm_client=llm_for_parse)
                patient_info = parser.parse(patient_info_path, disease_name=disease_name)
            except Exception as e:
                logger.warning(f"Patient info parsing failed (non-fatal): {e}")

        # ── Drug inclusion — driven by user query keywords ──
        user_query = (inputs.get("user_query") or "").lower()
        include_drugs = any(kw in user_query for kw in (
            "with drug", "drug recommendation", "therapeutic", "include drug",
        ))

        # ── Report output format — default PDF, respect explicit request ──
        report_format = inputs.get("report_output_format", "pdf")

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="🔬 Analyzing Data & Generating Narratives",
            message="Running LLM-powered analysis pipeline...",
            agent_name=agent_info.name,
            progress=0.2
        )

        llm_client = create_llm_client()

        agent = ReportingPipelineAgent(
            genes_path=Path(genes_path),
            pathways_path=Path(pathways_path),
            disease_name=disease_name,
            output_dir=output_dir,
            llm_client=llm_client,
            xcell_path=Path(xcell_path) if xcell_path else None,
            patient_info=patient_info,
            report_id=state.session_id,
            use_drug_agent=include_drugs,
            include_therapeutic_section=include_drugs,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: agent.run(output_format="docx"))

        yield StatusUpdate(
            status_type=StatusType.PROGRESS,
            title="📦 Collecting Report Files",
            message="Finalizing output files...",
            agent_name=agent_info.name,
            progress=0.9
        )

        # The pipeline always generates DOCX (and DocxGenerator auto-converts to PDF).
        # Expose whichever format the user requested.
        docx_path = result["output_files"].get("docx") or result["output_files"].get("docx_internal", "")
        pdf_path = str(Path(docx_path).with_suffix(".pdf")) if docx_path else ""

        if report_format == "docx":
            state.workflow_state["report_docx_path"] = docx_path
        else:
            state.workflow_state["report_pdf_path"] = pdf_path if Path(pdf_path).is_file() else docx_path

        state.workflow_state["report_output_dir"] = str(output_dir)
        state.workflow_state["report_summary"] = result.get("summary", {})

        summary = result.get("summary", {})
        detail_lines = [f"Disease: {disease_name}"]
        if summary.get("disease_genes_found"):
            detail_lines.append(f"Genes mapped: {summary['disease_genes_found']}")
        if summary.get("disease_pathways_found"):
            detail_lines.append(f"Pathways mapped: {summary['disease_pathways_found']}")

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Molecular Report Complete",
            message=f"Report generated for {disease_name}",
            details="\n".join(detail_lines),
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"Molecular report generation failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Molecular Report Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# CRISPR PERTURB-SEQ
# =============================================================================

async def execute_crispr_perturb_seq(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute the CRISPR Perturb-seq pipeline (13 stages + scRNA + report)."""
    from agentic_ai_wf.crispr_pipeline_agent.crispr import run_pipeline, discover_pipeline_inputs

    try:
        input_dir = (
            inputs.get("crispr_10x_input_dir")
            or state.workflow_state.get("crispr_10x_input_dir")
        )
        if not input_dir or not Path(input_dir).is_dir():
            raise ValueError(
                "No 10X scRNA-seq directory provided. Please provide a path to a "
                "directory containing barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz."
            )

        input_dir = str(Path(input_dir).resolve())
        disease_name = inputs.get("disease_name") or state.workflow_state.get("disease_name", "unknown_disease")
        output_dir = _get_output_dir("crispr_perturb_seq", state.session_id)

        discovery = discover_pipeline_inputs(input_dir)
        sample_count = len(discovery.get("samples", []))

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🧬 CRISPR Perturb-seq Starting",
            message=f"Processing **{sample_count}** sample(s) for **{disease_name}**...",
            details="Stages: ingestion → perturbation calling → mixscape → DEG → ML → causal → report",
            agent_name=agent_info.name,
            progress=0.1
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                input_gse_dirs=input_dir,
                output_dir=output_dir,
                generate_report=True,
            )
        )

        state.workflow_state["crispr_perturb_seq_output_dir"] = str(output_dir)

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Perturb-seq Complete",
            message=f"Perturb-seq pipeline finished for {disease_name} ({sample_count} samples)",
            details=f"Results saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"CRISPR Perturb-seq failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ Perturb-seq Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# CRISPR SCREENING
# =============================================================================

async def execute_crispr_screening(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute CRISPR genetic screening via nf-core/crisprseq."""
    from agentic_ai_wf.crispr_pipeline_agent.screening_crispr import run_screening

    try:
        input_dir = (
            inputs.get("crispr_screening_input_dir")
            or state.workflow_state.get("crispr_screening_input_dir")
        )
        if not input_dir or not Path(input_dir).is_dir():
            raise ValueError(
                "No screening input directory provided. Please provide a path containing "
                "count_table.tsv and rra_contrasts.txt."
            )

        input_dir = str(Path(input_dir).resolve())
        disease_name = inputs.get("disease_name") or state.workflow_state.get("disease_name", "unknown_disease")
        modes = inputs.get("modes") or state.workflow_state.get("modes") or [3]
        output_dir = _get_output_dir("crispr_screening", state.session_id)

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🔬 CRISPR Screening Starting",
            message=f"Running genetic screening analysis for **{disease_name}**...",
            details="Running nf-core/crisprseq with MAGeCK, BAGEL2, and directional scoring",
            agent_name=agent_info.name,
            progress=0.1
        )

        _ensure_conda_env_on_path()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_screening(
                input_dir=input_dir,
                output_dir=str(output_dir),
                modes=tuple(modes),
                generate_report=True,
                profile="docker" if shutil.which("docker") else "singularity",
            )
        )

        state.workflow_state["crispr_screening_output_dir"] = str(output_dir)
        _cleanup_nextflow_output(output_dir)

        detail_msg = result.message if result.success else f"Partial: {result.message}"
        if result.report_path:
            detail_msg += f"\nReport: {result.report_path}"

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ CRISPR Screening Complete",
            message=f"Screening analysis finished for {disease_name}",
            details=detail_msg,
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"CRISPR Screening failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ CRISPR Screening Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise


# =============================================================================
# CRISPR TARGETED
# =============================================================================

async def execute_crispr_targeted(
    agent_info: 'AgentInfo',
    inputs: Dict[str, Any],
    state: 'ConversationState'
) -> AsyncGenerator[StatusUpdate, None]:
    """Execute targeted CRISPR editing analysis via nf-core/crisprseq."""
    from agentic_ai_wf.crispr_pipeline_agent.targeted import run_targeted_pipeline

    try:
        input_dir = (
            inputs.get("crispr_targeted_input_dir")
            or state.workflow_state.get("crispr_targeted_input_dir")
        )
        if not input_dir or not Path(input_dir).is_dir():
            raise ValueError(
                "No targeted CRISPR input directory provided. Please provide a path "
                "containing paired FASTQ files."
            )

        protospacer = inputs.get("protospacer") or state.workflow_state.get("protospacer")
        if not protospacer:
            raise ValueError(
                "No protospacer sequence provided. Please specify the guide RNA "
                "target sequence (e.g. ATCGATCGATCGATCGATCG)."
            )

        input_dir = str(Path(input_dir).resolve())
        disease_name = inputs.get("disease_name") or state.workflow_state.get("disease_name", "unknown_disease")
        target_gene = inputs.get("target_gene") or state.workflow_state.get("target_gene", "")
        region = inputs.get("region") or state.workflow_state.get("region", "")
        reference_seq = inputs.get("reference_seq") or state.workflow_state.get("reference_seq", "")
        project_id = inputs.get("project_id") or state.workflow_state.get("project_id", "")
        extract_metadata = bool(inputs.get("extract_metadata") or state.workflow_state.get("extract_metadata"))
        download_fastq = bool(inputs.get("download_fastq") or state.workflow_state.get("download_fastq"))
        reference_dir = Path(__file__).resolve().parent.parent / "crispr"
        hg38_path = reference_dir / "hg38.fa"
        gtf_path = reference_dir / "gencode.v44.annotation.gtf"
        if not hg38_path.is_file():
            raise ValueError(f"Targeted CRISPR reference FASTA not found: {hg38_path}")
        if not gtf_path.is_file():
            raise ValueError(f"Targeted CRISPR reference GTF not found: {gtf_path}")
        if not _can_prepare_fasta_index(hg38_path):
            raise ValueError(
                "Targeted CRISPR reference FASTA is present but cannot be indexed. "
                f"Expected existing index at {hg38_path}.fai or an available 'samtools'/'pysam' runtime dependency."
            )
        output_dir = _get_output_dir("crispr_targeted", state.session_id)

        yield StatusUpdate(
            status_type=StatusType.EXECUTING,
            title="🎯 CRISPR Targeted Analysis Starting",
            message=f"Analyzing targeted editing for **{disease_name}**"
                    + (f" (gene: {target_gene})" if target_gene else "") + "...",
            details=(f"Protospacer: {protospacer}\n"
                     f"Acquisition: {'public download' if project_id else 'local input'}\n"
                     f"Reference dir: {reference_dir}\n"
                     "Running nf-core/crisprseq targeted pipeline"),
            agent_name=agent_info.name,
            progress=0.1
        )

        _ensure_conda_env_on_path()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: run_targeted_pipeline(
                input_dir=input_dir,
                output_dir=str(output_dir),
                protospacer=protospacer,
                target_gene=target_gene,
                region=region,
                reference_seq=reference_seq,
                hg38=str(hg38_path),
                gtf=str(gtf_path),
                project_id=project_id,
                extract_metadata=extract_metadata,
                download_fastq=download_fastq,
                generate_report=True,
                profile="docker" if shutil.which("docker") else "singularity",
            )
        )

        state.workflow_state["crispr_targeted_output_dir"] = str(output_dir)
        _cleanup_nextflow_output(output_dir)

        yield StatusUpdate(
            status_type=StatusType.COMPLETED,
            title="✅ Targeted CRISPR Complete",
            message=f"Targeted editing analysis finished for {disease_name}",
            details=f"Gene: {target_gene or 'N/A'}\nResults saved to: {output_dir}",
            agent_name=agent_info.name,
            progress=1.0,
            output_dir=str(output_dir)
        )

    except Exception as e:
        logger.exception(f"CRISPR Targeted analysis failed: {e}")
        yield StatusUpdate(
            status_type=StatusType.ERROR,
            title="❌ CRISPR Targeted Failed",
            message=f"Error: {str(e)}",
            agent_name=agent_info.name
        )
        raise
