"""
Agent Registry - Defines all available agents, their capabilities, inputs/outputs.

This module provides a centralized registry of all specialized agents that the
supervisor can route to. Each agent has:
- Name and description
- Required inputs (files/parameters)
- Produced outputs
- Keywords for intent matching
- The actual callable function
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of agents available in the system"""
    COHORT_RETRIEVAL = "cohort_retrieval"
    DEG_ANALYSIS = "deg_analysis"
    GENE_PRIORITIZATION = "gene_prioritization"
    PATHWAY_ENRICHMENT = "pathway_enrichment"
    DECONVOLUTION = "deconvolution"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    HARMONIZATION = "harmonization"
    MDP_ANALYSIS = "mdp_analysis"
    PERTURBATION_ANALYSIS = "perturbation_analysis"
    MULTIOMICS_INTEGRATION = "multiomics_integration"
    FASTQ_PROCESSING = "fastq_processing"
    MOLECULAR_REPORT = "molecular_report"
    CRISPR_PERTURB_SEQ = "crispr_perturb_seq"
    CRISPR_SCREENING = "crispr_screening"
    CRISPR_TARGETED = "crispr_targeted"
    CAUSALITY = "causality"


@dataclass
class InputRequirement:
    """Defines a required input for an agent"""
    name: str
    description: str
    file_type: Optional[str] = None  # e.g., "csv", "tsv", "h5ad"
    is_file: bool = True
    required: bool = True
    can_come_from: Optional[str] = None  # Agent that can produce this
    example: Optional[str] = None


@dataclass 
class OutputSpec:
    """Defines an output produced by an agent"""
    name: str
    description: str
    file_type: Optional[str] = None
    state_key: str = ""  # Key used in workflow state


@dataclass
class AgentInfo:
    """Complete information about a specialized agent"""
    agent_type: AgentType
    name: str
    display_name: str
    description: str
    detailed_description: str
    
    # What this agent needs
    required_inputs: List[InputRequirement] = field(default_factory=list)
    optional_inputs: List[InputRequirement] = field(default_factory=list)
    
    # What this agent produces
    outputs: List[OutputSpec] = field(default_factory=list)
    
    # For intent matching
    keywords: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    
    # Execution info
    estimated_time: str = "1-5 minutes"
    depends_on: List[AgentType] = field(default_factory=list)
    
    # Pipeline chain support: keys this agent produces in workflow_state
    produces: List[str] = field(default_factory=list)
    # At least one of these inputs must be available (supports multiple input sources)
    requires_one_of: List[str] = field(default_factory=list)
    
    def get_missing_inputs(self, available: Dict[str, Any]) -> List[InputRequirement]:
        """Check which required inputs are missing"""
        missing = []
        for inp in self.required_inputs:
            if inp.name not in available or not available[inp.name]:
                missing.append(inp)
        return missing
    
    def can_run(self, available: Dict[str, Any]) -> bool:
        """Check if agent has all required inputs"""
        return len(self.get_missing_inputs(available)) == 0


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

COHORT_RETRIEVAL_AGENT = AgentInfo(
    agent_type=AgentType.COHORT_RETRIEVAL,
    name="cohort_retrieval",
    display_name="🔍 Cohort Retrieval Agent",
    description="Searches GEO and ArrayExpress databases for disease-related datasets",
    detailed_description="""
The Cohort Retrieval Agent searches public genomics databases (GEO and ArrayExpress) 
to find relevant RNA-seq and gene expression datasets for a specified disease.

**What it does:**
1. Parses your query to extract disease name, tissue type, and experiment type
2. Searches GEO (Gene Expression Omnibus) for matching datasets
3. Searches ArrayExpress for additional datasets
4. Falls back to ontology-based search if direct search yields no results
5. Downloads and organizes the found datasets

**Best for:**
- Finding public datasets for a disease you want to study
- Discovering available RNA-seq data for a condition
- Starting a new analysis without your own data
""",
    required_inputs=[
        InputRequirement(
            name="disease_name",
            description="The disease or condition to search for (e.g., 'lupus', 'breast cancer')",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="tissue_filter",
            description="Tissue type to filter by (e.g., 'blood', 'PBMC', 'tissue')",
            is_file=False,
            required=False,
            example="blood"
        ),
        InputRequirement(
            name="experiment_filter",
            description="Experiment type (e.g., 'single cell', 'bulk RNA-seq')",
            is_file=False,
            required=False,
            example="bulk RNA-seq"
        )
    ],
    outputs=[
        OutputSpec(
            name="cohort_output_dir",
            description="Directory containing downloaded datasets",
            state_key="cohort_output_dir"
        ),
        OutputSpec(
            name="summary_text",
            description="Human-readable summary of found datasets",
            state_key="cohort_summary_text"
        )
    ],
    keywords=[
        # Primary terms
        "search", "find", "dataset", "GEO", "ArrayExpress", "cohort", "download",
        "public data", "RNA-seq data", "expression data", "studies", "experiments",
        "retrieve", "fetch", "discover", "available datasets",
        # Expanded synonyms
        "look for", "get data", "patient data", "patient samples", "sample data",
        "query database", "search database", "find studies", "locate datasets",
        "public repositories", "gene expression omnibus", "EBI", "NCBI",
        "transcriptomics data", "microarray data", "bulk RNA", "single cell data",
        "download datasets", "fetch samples", "get samples", "collect data",
        "disease datasets", "clinical samples", "available data"
    ],
    example_queries=[
        "Find datasets for lupus disease",
        "Search for breast cancer RNA-seq studies",
        "Get GEO datasets for pancreatic cancer in blood",
        "Find single cell datasets for cervical cancer",
        "What public datasets are available for colon cancer?",
        "Look for patient samples in GEO",
        "Download expression data for multiple sclerosis",
        "Get me some data to analyze for diabetes"
    ],
    estimated_time="2-10 minutes (depends on search results)",
    depends_on=[],
    produces=["cohort_output_dir", "cohort_summary_text"],
    requires_one_of=["disease_name"]  # Only needs disease name
)


DEG_ANALYSIS_AGENT = AgentInfo(
    agent_type=AgentType.DEG_ANALYSIS,
    name="deg_analysis",
    display_name="📊 DEG Analysis Agent",
    description="Performs Differential Expression Gene analysis on count data",
    detailed_description="""
The DEG (Differential Expression Gene) Analysis Agent performs statistical analysis
to identify genes that are significantly up-regulated or down-regulated between
conditions (e.g., disease vs. control).

**What it does:**
1. Loads your count matrix (genes × samples)
2. Extracts metadata (sample conditions)
3. Runs DESeq2 differential expression analysis
4. Maps gene IDs to gene symbols
5. Filters significant DEGs based on p-value and fold change
6. Generates DEG result files

**Best for:**
- Analyzing your own RNA-seq count data
- Identifying differentially expressed genes
- Comparing disease vs. healthy samples
""",
    required_inputs=[
        InputRequirement(
            name="counts_file",
            description="Gene expression count matrix (genes as rows, samples as columns)",
            file_type="csv",
            is_file=True,
            required=True,
            example="GSE12345_counts.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Name of the disease/condition being analyzed",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="metadata_file",
            description="Sample metadata file with condition labels",
            file_type="csv",
            is_file=True,
            required=False,
            example="GSE12345_metadata.csv"
        )
    ],
    outputs=[
        OutputSpec(
            name="deg_base_dir",
            description="Directory containing DEG analysis results",
            state_key="deg_base_dir"
        ),
        OutputSpec(
            name="deg_files",
            description="CSV files with differentially expressed genes",
            file_type="csv",
            state_key="deg_files"
        )
    ],
    keywords=[
        # Primary terms
        "DEG", "differential expression", "DESeq2", "fold change", "p-value",
        "upregulated", "downregulated", "gene expression", "count matrix",
        "statistical analysis", "compare", "disease vs control",
        # Expanded synonyms
        "differentially expressed", "expression analysis", "expression changes",
        "gene changes", "changed genes", "altered genes", "dysregulated genes",
        "up-regulated", "down-regulated", "overexpressed", "underexpressed",
        "increased expression", "decreased expression", "expression differences",
        "case vs control", "treated vs untreated", "compare groups", "group comparison",
        "significance analysis", "statistical test", "limma", "edgeR",
        "log fold change", "logFC", "LFC", "adjusted p-value", "FDR",
        "counts data", "raw counts", "expression matrix", "TPM", "FPKM",
        "analyze my data", "run analysis on", "process my counts"
    ],
    example_queries=[
        "Run DEG analysis on my count data",
        "Find differentially expressed genes in my lupus dataset",
        "Perform differential expression analysis",
        "Analyze gene expression changes between disease and control",
        "Which genes are upregulated in my cancer samples?",
        "Compare treated vs control samples",
        "What genes are altered in my dataset?",
        "Run statistical analysis on my expression data"
    ],
    estimated_time="5-15 minutes",
    depends_on=[],
    produces=["deg_base_dir", "deg_input_file"],
    requires_one_of=["cohort_output_dir", "counts_file"]  # Either cohort data OR direct counts file
)


GENE_PRIORITIZATION_AGENT = AgentInfo(
    agent_type=AgentType.GENE_PRIORITIZATION,
    name="gene_prioritization",
    display_name="🎯 Gene Prioritization Agent",
    description="Filters and prioritizes DEGs based on biological significance",
    detailed_description="""
The Gene Prioritization Agent takes raw DEG results and applies intelligent filtering
to identify the most biologically significant genes for your disease of interest.

**What it does:**
1. Loads your DEG file (differentially expressed genes)
2. Scores genes using disease relevance databases
3. Applies AI/LLM-based analysis for druggability assessment
4. Ranks genes by combined biological significance score
5. Outputs prioritized gene list

**Best for:**
- Narrowing down large DEG lists to actionable targets
- Finding genes most relevant to your specific disease
- Preparing gene lists for pathway analysis
""",
    required_inputs=[
        InputRequirement(
            name="deg_input_file",
            description="CSV file containing differential expression results (DEG file)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="deg_analysis",
            example="lupus_DEGs_filtered.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Name of the disease for relevance scoring",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[],
    outputs=[
        OutputSpec(
            name="prioritized_genes_path",
            description="CSV file with prioritized and scored genes",
            file_type="csv",
            state_key="prioritized_genes_path"
        )
    ],
    keywords=[
        # Primary terms
        "prioritize", "rank", "filter", "top genes", "significant genes",
        "gene list", "disease genes", "target genes", "important genes",
        "score", "GeneCards", "PPI", "network",
        # Expanded synonyms
        "key genes", "relevant genes", "most important", "best genes",
        "therapeutic targets", "drug targets", "druggable", "actionable genes",
        "biomarkers", "potential biomarkers", "candidate genes", "lead genes",
        "disease relevance", "disease association", "clinically relevant",
        "narrow down", "shortlist", "select genes", "gene selection",
        "highest scoring", "top ranked", "best candidates", "most significant",
        "filter DEGs", "prioritize DEGs", "rank DEGs", "score genes",
        "identify key", "find important", "which genes matter",
        "protein interaction", "network analysis", "hub genes"
    ],
    example_queries=[
        "Prioritize the DEGs for lupus",
        "Which genes are most important in my results?",
        "Rank the differentially expressed genes",
        "Filter and score my DEG list",
        "Get the top disease-relevant genes",
        "Find therapeutic targets in my gene list",
        "Which genes should I focus on?",
        "Identify key biomarkers for this disease"
    ],
    estimated_time="3-8 minutes",
    depends_on=[AgentType.DEG_ANALYSIS],
    produces=["prioritized_genes_path"],
    requires_one_of=["deg_base_dir", "deg_input_file"]  # Needs DEG results
)


PATHWAY_ENRICHMENT_AGENT = AgentInfo(
    agent_type=AgentType.PATHWAY_ENRICHMENT,
    name="pathway_enrichment",
    display_name="🛤️ Pathway Enrichment Agent",
    description="Performs pathway and functional enrichment analysis on gene lists",
    detailed_description="""
The Pathway Enrichment Agent identifies biological pathways and functional categories
that are significantly enriched in your gene list.

**What it does:**
1. Takes your prioritized gene list
2. Runs enrichment against multiple databases (KEGG, GO, Reactome)
3. Identifies significantly enriched pathways
4. Removes duplicate/redundant pathways
5. Categorizes pathways by biological function
6. Adds literature-based clinical relevance scores
7. Consolidates results into a ranked pathway report

**Best for:**
- Understanding the biological functions affected in disease
- Identifying druggable pathways
- Finding mechanistic insights from gene lists
""",
    required_inputs=[
        InputRequirement(
            name="prioritized_genes_path",
            description="CSV file with prioritized genes (from gene prioritization)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="gene_prioritization",
            example="lupus_DEGs_prioritized.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease name for clinical relevance scoring",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[],
    outputs=[
        OutputSpec(
            name="pathway_consolidation_path",
            description="CSV file with consolidated pathway results",
            file_type="csv",
            state_key="pathway_consolidation_path"
        )
    ],
    keywords=[
        # Primary terms
        "pathway", "enrichment", "KEGG", "GO", "Reactome", "functional",
        "biological process", "molecular function", "signaling", "metabolism",
        "immune", "cell cycle", "apoptosis",
        # Expanded synonyms
        "pathways affected", "affected pathways", "dysregulated pathways",
        "biological pathways", "cellular pathways", "signaling pathways",
        "gene ontology", "GO terms", "GO enrichment", "KEGG pathways",
        "functional analysis", "functional annotation", "gene function",
        "what processes", "biological function", "cellular function",
        "mechanism", "mechanisms involved", "underlying mechanisms",
        "enriched terms", "overrepresented", "pathway analysis",
        "what pathways", "which pathways", "pathway involvement",
        "signaling cascades", "regulatory pathways", "metabolic pathways",
        "inflammation", "immune response", "cytokine", "interferon",
        "what is happening", "biological meaning", "functional interpretation"
    ],
    example_queries=[
        "Run pathway enrichment on my gene list",
        "What pathways are affected in lupus?",
        "Perform functional enrichment analysis",
        "Find enriched biological processes",
        "Which signaling pathways are dysregulated?",
        "What mechanisms are involved in this disease?",
        "Analyze the biological function of these genes",
        "What is the functional meaning of these results?"
    ],
    estimated_time="5-15 minutes",
    depends_on=[AgentType.GENE_PRIORITIZATION],
    produces=["pathway_consolidation_path"],
    requires_one_of=["prioritized_genes_path"]  # Needs prioritized genes
)


DECONVOLUTION_AGENT = AgentInfo(
    agent_type=AgentType.DECONVOLUTION,
    name="deconvolution",
    display_name="🧬 Deconvolution Agent",
    description="Estimates cell type composition from bulk RNA-seq data",
    detailed_description="""
The Deconvolution Agent estimates the proportion of different cell types present
in your bulk RNA-seq samples using computational deconvolution methods.

**What it does:**
1. Takes your bulk expression data
2. Applies deconvolution algorithm (CIBERSORTx, xCell, or BisQue)
3. Estimates cell type proportions for each sample
4. Generates visualizations of cell composition
5. Compares cell types between conditions

**Supported Methods:**
- **xCell**: Fast, reference-free, good for immune cell types
- **CIBERSORTx**: High accuracy, requires reference signature
- **BisQue**: Uses single-cell reference for higher resolution

**Best for:**
- Understanding immune infiltration in tumors
- Comparing cell composition between disease and control
- Identifying cell types driving disease phenotype
""",
    required_inputs=[
        InputRequirement(
            name="bulk_file",
            description="Bulk RNA-seq expression matrix (TPM or counts)",
            file_type="csv",
            is_file=True,
            required=True,
            example="lupus_expression.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease name for context",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="metadata_file",
            description="Sample metadata with condition labels",
            file_type="csv",
            is_file=True,
            required=False,
            example="lupus_metadata.csv"
        ),
        InputRequirement(
            name="technique",
            description="Deconvolution method: 'xcell', 'cibersortx', or 'bisque'",
            is_file=False,
            required=False,
            example="xcell"
        ),
        InputRequirement(
            name="h5ad_file",
            description="Single-cell reference (required for bisque)",
            file_type="h5ad",
            is_file=True,
            required=False,
            example="reference.h5ad"
        )
    ],
    outputs=[
        OutputSpec(
            name="deconvolution_output_dir",
            description="Directory with deconvolution results and plots",
            state_key="deconvolution_output_dir"
        ),
        OutputSpec(
            name="cibersort_results",
            description="Cell type proportion matrix",
            file_type="csv",
            state_key="cibersort_results"
        )
    ],
    keywords=[
        # Primary terms
        "deconvolution", "cell type", "CIBERSORT", "xCell", "BisQue",
        "immune cells", "cell composition", "infiltration", "proportion",
        "T cells", "B cells", "macrophages", "neutrophils",
        # Expanded synonyms
        "cell types", "cell populations", "cellular composition", "cell fractions",
        "immune infiltration", "immune profile", "immune landscape",
        "tumor microenvironment", "TME", "tissue composition",
        "cell percentage", "cell abundance", "cell proportions",
        "what cells", "which cells", "cell content",
        "monocytes", "dendritic cells", "NK cells", "lymphocytes",
        "immune signature", "immune phenotype", "immunophenotyping",
        "bulk deconvolution", "estimate cells", "infer cell types",
        "single cell reference", "reference based", "marker genes",
        "cell decomposition", "cellular decomposition", "mixture analysis"
    ],
    example_queries=[
        "Run cell type deconvolution on my data",
        "What cell types are present in my samples?",
        "Estimate immune cell composition",
        "Run CIBERSORT analysis",
        "Compare cell types between disease and control",
        "What is the immune profile in these patients?",
        "Analyze the tumor microenvironment",
        "Which immune cells are infiltrating?"
    ],
    estimated_time="5-20 minutes",
    depends_on=[],
    produces=["deconvolution_output_dir"],
    requires_one_of=["bulk_file"]
)


TEMPORAL_ANALYSIS_AGENT = AgentInfo(
    agent_type=AgentType.TEMPORAL_ANALYSIS,
    name="temporal_analysis",
    display_name="⏱️ Temporal Analysis Agent",
    description="Analyzes gene expression dynamics over time or disease progression",
    detailed_description="""
The Temporal Analysis Agent performs time-course or progression analysis of bulk RNA-seq data.

**What it does:**
1. Estimates pseudotime from expression data (PCA or PhenoPath)
2. Fits impulse models to identify temporal gene patterns
3. Runs pathway enrichment along the temporal axis
4. Generates trajectory visualizations and reports

**Best for:**
- Time-course experiments
- Disease progression studies
- Developmental trajectories
- Treatment response over time
""",
    required_inputs=[
        InputRequirement(
            name="counts_file",
            description="Bulk RNA-seq raw counts matrix (genes × samples)",
            file_type="csv",
            is_file=True,
            required=True,
            example="expression_counts.csv"
        ),
        InputRequirement(
            name="metadata_file",
            description="Sample metadata with time/condition information",
            file_type="csv",
            is_file=True,
            required=True,
            example="sample_metadata.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease or condition name",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="treatment_level",
            description="Treatment/condition level for DE analysis",
            is_file=False,
            required=False,
            example="Disease"
        )
    ],
    outputs=[
        OutputSpec(
            name="temporal_output_dir",
            description="Directory with temporal analysis results",
            state_key="temporal_output_dir"
        )
    ],
    keywords=[
        "temporal", "time course", "time-course", "progression", "trajectory",
        "pseudotime", "dynamics", "temporal dynamics", "time series",
        "impulse", "temporal pattern", "disease progression",
        "developmental", "longitudinal", "kinetics", "time point",
        "over time", "temporal expression", "temporal gene"
    ],
    example_queries=[
        "Run temporal analysis on my time-course data",
        "Analyze gene expression dynamics over time",
        "Find temporal patterns in disease progression",
        "Identify genes that change over time"
    ],
    estimated_time="10-30 minutes",
    depends_on=[],
    produces=["temporal_output_dir"],
    requires_one_of=["counts_file"]
)


HARMONIZATION_AGENT = AgentInfo(
    agent_type=AgentType.HARMONIZATION,
    name="harmonization",
    display_name="🔗 Harmonization Agent",
    description="Batch correction and normalization across RNA-seq datasets",
    detailed_description="""
The Harmonization Agent performs batch correction and cross-study normalization.

**What it does:**
1. ComBat batch correction (sva R package)
2. Quantile normalization across samples
3. Multi-dataset integration and merging
4. QC plots before/after correction

**Best for:**
- Combining datasets from different studies
- Removing batch effects
- Cross-platform normalization
- Multi-cohort meta-analysis
""",
    required_inputs=[
        InputRequirement(
            name="counts_file",
            description="Expression counts matrix (genes × samples)",
            file_type="csv",
            is_file=True,
            required=True,
            example="expression_counts.csv"
        ),
        InputRequirement(
            name="metadata_file",
            description="Sample metadata with batch/condition columns",
            file_type="csv",
            is_file=True,
            required=True,
            example="sample_metadata.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease or condition name",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="mode",
            description="Harmonization mode: 'single' or 'local' (multi-dataset)",
            is_file=False,
            required=False,
            example="single"
        ),
        InputRequirement(
            name="data_root",
            description="Root directory for 'local' mode dataset discovery",
            is_file=False,
            required=False,
            example="/data/multi_studies/"
        ),
        InputRequirement(
            name="combine",
            description="Merge harmonized datasets (for 'local' mode)",
            is_file=False,
            required=False,
            example="true"
        )
    ],
    outputs=[
        OutputSpec(
            name="harmonization_output_dir",
            description="Directory with harmonized results",
            state_key="harmonization_output_dir"
        )
    ],
    keywords=[
        "harmonization", "harmonize", "batch correction", "combat", "batch effect",
        "normalize", "normalization", "cross-study", "multi-dataset", "integrate",
        "merge studies", "combine datasets", "meta-analysis", "quantile",
        "batch", "technical variation", "remove batch", "correct batch"
    ],
    example_queries=[
        "Harmonize my multi-study RNA-seq data",
        "Remove batch effects from my expression data",
        "Normalize and combine datasets from different studies",
        "Run batch correction on my counts matrix"
    ],
    estimated_time="5-15 minutes",
    depends_on=[],
    produces=["harmonization_output_dir"],
    requires_one_of=["counts_file"]
)


MDP_ANALYSIS_AGENT = AgentInfo(
    agent_type=AgentType.MDP_ANALYSIS,
    name="mdp_analysis",
    display_name="🌐 MDP Analysis Agent",
    description="Multi-Disease Pathway comparison and cross-disease analysis",
    detailed_description="""
The MDP (Multi-Disease Pathway) Analysis Agent compares pathways across multiple diseases.

**What it does:**
1. Extracts disease-specific pathway activity from expression data
2. Compares pathway enrichment across 2+ diseases
3. Identifies shared and unique molecular mechanisms
4. Uses Neo4j knowledge graph for diseases without data files

**Best for:**
- Comparing pathways between diseases (e.g., lupus vs rheumatoid arthritis)
- Finding common therapeutic targets across conditions
- Cross-disease drug repurposing analysis
- Understanding disease relationships at pathway level
""",
    required_inputs=[],
    optional_inputs=[
        InputRequirement(
            name="counts_file",
            description="Expression counts matrix for one or more diseases",
            file_type="csv",
            is_file=True,
            required=False,
            example="lupus_counts.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease name(s) to analyze",
            is_file=False,
            required=False,
            example="lupus, breast cancer"
        ),
        InputRequirement(
            name="tissue",
            description="Tissue type for the analysis",
            is_file=False,
            required=False,
            example="blood"
        )
    ],
    outputs=[
        OutputSpec(
            name="mdp_output_dir",
            description="Directory with cross-disease pathway comparison results",
            state_key="mdp_output_dir"
        )
    ],
    keywords=[
        "mdp", "multi-disease", "cross-disease", "pathway comparison", "compare diseases",
        "disease comparison", "pathway similarity", "shared pathways", "common mechanisms",
        "multi disease pathway", "compare pathways", "pathway analysis", "cross-condition",
        "disease pathways", "compare lupus", "compare cancer", "multiple diseases"
    ],
    example_queries=[
        "Run MDP for lupus and breast cancer",
        "Compare pathways between Alzheimer's and Parkinson's disease",
        "Multi-disease analysis for vasculitis and lupus",
        "What pathways are shared between asthma and COPD?"
    ],
    estimated_time="5-20 minutes",
    depends_on=[],
    produces=["mdp_output_dir"],
    requires_one_of=[]
)


# =============================================================================
# PERTURBATION ANALYSIS AGENT
# =============================================================================

PERTURBATION_ANALYSIS_AGENT = AgentInfo(
    agent_type=AgentType.PERTURBATION_ANALYSIS,
    name="perturbation_analysis",
    display_name="💊 Perturbation Analysis Agent",
    description="Drug perturbation analysis using DEPMAP cell essentiality and L1000 drug signatures",
    detailed_description="""
The Perturbation Analysis Agent performs drug perturbation analysis by integrating
multiple data sources to identify potential therapeutic targets and drug candidates.

**What it does:**
1. DEPMAP integration: Evaluates cell essentiality scores for prioritized genes
2. L1000 integration: Matches genes to drug perturbation signatures from LINCS
3. Integration: Combines essentiality + drug signatures + pathway context
4. Scoring: Computes perturbation scores, drug-gene associations, pathway-drug links

**Best for:**
- Finding essential and non-essential genes in a disease context
- Identifying drug candidates that reverse disease gene signatures
- Discovering gRNA targets for top essential genes
- Drug repurposing through connectivity scoring
""",
    required_inputs=[
        InputRequirement(
            name="prioritized_genes_path",
            description="CSV file with prioritized genes (from gene prioritization)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="gene_prioritization",
            example="disease_DEGs_prioritized.csv"
        ),
        InputRequirement(
            name="pathway_consolidation_path",
            description="CSV file with consolidated pathway results (from pathway enrichment)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="pathway_enrichment",
            example="disease_Pathways_Consolidated.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease name for analysis context",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="dep_map_mode",
            description="Model selection mode: by_disease, by_lineage, by_ids, by_names, or keyword",
            is_file=False,
            required=False,
            example="by_disease"
        ),
        InputRequirement(
            name="genes_selection",
            description="Gene selection mode: 'all' or 'top'",
            is_file=False,
            required=False,
            example="all"
        ),
        InputRequirement(
            name="l1000_tissue",
            description="Tissue filter for L1000 analysis",
            is_file=False,
            required=False,
            example="breast"
        ),
        InputRequirement(
            name="l1000_drug",
            description="Specific drug to focus on in L1000 analysis",
            is_file=False,
            required=False,
            example="doxorubicin"
        ),
    ],
    outputs=[
        OutputSpec(
            name="perturbation_output_dir",
            description="Directory with perturbation analysis results (DEPMAP + L1000 + Integration)",
            state_key="perturbation_output_dir"
        )
    ],
    keywords=[
        # Primary terms
        "perturbation", "drug perturbation", "DEPMAP", "DepMap", "L1000", "LINCS",
        "cell essentiality", "essential genes", "non-essential genes", "essentiality",
        "drug signatures", "drug response", "drug target", "drug repurposing",
        "connectivity score", "perturbation score", "gRNA", "guide RNA",
        # Expanded synonyms
        "drug discovery", "therapeutic targets", "drug candidates", "drug sensitivity",
        "gene essentiality", "CRISPR screen", "CRISPR dependency", "gene dependency",
        "drug perturbation signatures", "reversal score", "integration score",
        "pharmacogenomics", "drug gene association", "codependency",
        "cell line models", "perturbation analysis", "perturbation pipeline"
    ],
    example_queries=[
        "Run perturbation analysis on my prioritized genes",
        "Find essential genes in my disease data",
        "What gRNAs are best for top essential genes?",
        "Run drug perturbation analysis with DEPMAP and L1000",
        "How many essential and non-essential genes are in the data?",
        "Identify drug candidates that reverse the disease signature",
        "Run the full perturbation pipeline on DEGs data",
        "Analyze drug sensitivity for my prioritized genes"
    ],
    estimated_time="15-45 minutes",
    depends_on=[AgentType.PATHWAY_ENRICHMENT],
    produces=["perturbation_output_dir"],
    requires_one_of=["prioritized_genes_path"]  # Needs at minimum the prioritized genes
)


# =============================================================================
# MULTIOMICS INTEGRATION AGENT
# =============================================================================

# Valid layer names the pipeline accepts
MULTIOMICS_LAYER_NAMES = frozenset({
    "genomics", "transcriptomics", "epigenomics", "proteomics", "metabolomics"
})

MULTIOMICS_INTEGRATION_AGENT = AgentInfo(
    agent_type=AgentType.MULTIOMICS_INTEGRATION,
    name="multiomics_integration",
    display_name="Multi-Omics Integration",
    description=(
        "Integrates multiple omics layers (genomics, transcriptomics, epigenomics, "
        "proteomics, metabolomics) using MOFA+/dimensionality reduction, performs "
        "ML-based biomarker discovery, cross-omics correlation, and literature mining."
    ),
    detailed_description="""
The Multi-Omics Integration Agent combines multiple omics data layers into a
unified analysis using dimensionality reduction and factor analysis.

**What it does:**
1. Ingests omics layer files (CSV with 'feature' column + numeric sample columns)
2. Preprocesses and QC-filters each layer independently
3. Integrates layers via MOFA+ / PCA-based dimensionality reduction
4. Performs ML-based biomarker discovery across integrated features
5. Runs cross-omics correlation analysis
6. Mines PubMed literature for supporting evidence

**Best for:**
- Combining genomics, transcriptomics, proteomics, metabolomics, or epigenomics data
- Discovering cross-omics biomarkers and molecular signatures
- Systems-biology level understanding of disease mechanisms
""",
    required_inputs=[
        InputRequirement(
            name="disease_name",
            description="Disease or condition being studied",
            is_file=False,
            example="lupus"
        ),
    ],
    optional_inputs=[
        InputRequirement(
            name="metadata_path",
            description="Sample metadata CSV with labels (optional)",
            is_file=True,
            example="metadata.csv"
        ),
        InputRequirement(
            name="label_column",
            description="Column name in metadata for sample grouping",
            is_file=False,
            example="condition"
        ),
        InputRequirement(
            name="query_term",
            description="PubMed query term for literature mining",
            is_file=False,
            example="lupus multi-omics biomarkers"
        ),
    ],
    outputs=[
        OutputSpec(
            name="multiomics_output_dir",
            description="Directory with multi-omics integration results (integration, biomarkers, cross-omics, literature)",
            state_key="multiomics_output_dir"
        )
    ],
    keywords=[
        # Primary terms
        "multi-omics", "multiomics", "omics integration", "data integration",
        "genomics", "transcriptomics", "epigenomics", "proteomics", "metabolomics",
        "MOFA", "multi-omics factor analysis", "dimensionality reduction",
        # Analysis types
        "cross-omics", "cross-omics correlation", "biomarker discovery",
        "multi-layer", "integrated biomarkers", "omics layers",
        "ML biomarkers", "machine learning biomarkers",
        # Broader synonyms
        "integrate my omics data", "combine omics layers",
        "multi-modal integration", "pan-omics", "systems biology",
    ],
    example_queries=[
        "Integrate my genomics and transcriptomics data for lupus",
        "Run multi-omics integration on these layer files",
        "Perform cross-omics correlation analysis",
        "Find biomarkers across proteomics and metabolomics layers",
        "MOFA integration of my uploaded omics data",
        "Run the full multiomics pipeline with literature mining",
    ],
    estimated_time="20-60 minutes",
    depends_on=[],  # Independent - no upstream agent dependencies
    produces=["multiomics_output_dir"],
    requires_one_of=["multiomics_layers"]  # At least one omics layer file
)


FASTQ_PROCESSING_AGENT = AgentInfo(
    agent_type=AgentType.FASTQ_PROCESSING,
    name="fastq_processing",
    display_name="🧬 FASTQ Processing Agent",
    description="Processes raw FASTQ sequencing files through QC, trimming, quantification and reporting",
    detailed_description="""
The FASTQ Processing Agent takes raw sequencing reads (.fastq / .fq / .gz) and runs a
fully automated pipeline covering quality control, adapter trimming, transcript-level
quantification and multi-sample reporting.

**Pipeline steps (per sample):**
1. **FastQC** – initial quality assessment of raw reads
2. **Cutadapt** – adapter removal and quality trimming
3. **Trimmomatic** – additional quality filtering (sliding window, minimum length)
4. **Salmon** – transcript-level quantification against a reference index
5. **MultiQC** – aggregate QC report across all samples

After per-sample processing the agent can optionally combine individual Salmon
quant files into a single gene-level count matrix suitable for downstream DEG
analysis.

**What it produces:**
- Per-sample FastQC HTML reports & ZIPs
- Trimmed FASTQ files
- Salmon quant.sf quantification tables
- Combined gene-count matrix (CSV)
- MultiQC HTML summary report
- Detailed run logs

**Best for:**
- Processing raw sequencing data before expression analysis
- Quality control of FASTQ files
- Generating count matrices from raw reads
""",
    required_inputs=[
        InputRequirement(
            name="fastq_input_dir",
            description="Local filesystem path to a directory containing FASTQ files (.fastq, .fq, .fastq.gz, .fq.gz)",
            file_type="directory",
            is_file=False,
            required=True,
            example="/data/my_project/fastq_files/"
        ),
        InputRequirement(
            name="disease_name",
            description="Name of the disease/condition being studied (used for folder naming)",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    outputs=[
        OutputSpec(
            name="fastq_output_dir",
            description="Root directory containing all pipeline outputs",
            state_key="fastq_output_dir"
        ),
    ],
    keywords=[
        "fastq", "sequencing", "raw reads", "quality control", "QC",
        "trimming", "adapter", "salmon", "quantification", "fastqc",
        "trimmomatic", "cutadapt", "multiqc", "raw data", "sequencer",
        "read processing", "ngs", "next generation sequencing",
        "paired end", "single end", "R1", "R2",
        "fastq.gz", "fq.gz", "illumina", "process reads",
    ],
    example_queries=[
        "Process my FASTQ files for lupus",
        "Run quality control on my sequencing reads",
        "Trim and quantify my raw sequencing data",
        "I have raw FASTQ files that need processing",
        "Run the FASTQ pipeline on my sequencing data",
    ],
    estimated_time="10-60 minutes (depends on number of samples and file sizes)",
    depends_on=[],
    produces=["fastq_output_dir"],
    requires_one_of=["fastq_input_dir"],
)


# =============================================================================
# MOLECULAR REPORT AGENT
# =============================================================================

MOLECULAR_REPORT_AGENT = AgentInfo(
    agent_type=AgentType.MOLECULAR_REPORT,
    name="molecular_report",
    display_name="📋 Molecular Report Agent",
    description="Generates a comprehensive molecular analysis report with gene analysis, pathway enrichment, deconvolution, and drug recommendations",
    detailed_description="""
The Molecular Report Agent produces publication-quality molecular analysis reports
by synthesizing prioritized genes, enriched pathways, deconvolution, and optionally
drug recommendations into a structured clinical-grade document.

**What it does:**
1. Loads prioritized genes and pathway consolidation data
2. Generates dynamic disease context via LLM
3. Maps genes and pathways to disease annotations with multi-evidence validation
4. Analyzes deconvolution data (if available)
5. Generates narratives with clinical confirmation
6. Produces a structured DOCX and/or PDF document with cover page

**Best for:**
- Creating comprehensive molecular reports from completed analysis
- Generating patient-specific reports with demographic information
- Producing reports with or without drug recommendation sections
""",
    required_inputs=[
        InputRequirement(
            name="prioritized_genes_path",
            description="CSV file with prioritized genes (from gene prioritization)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="gene_prioritization",
            example="disease_DEGs_prioritized.csv"
        ),
        InputRequirement(
            name="pathway_consolidation_path",
            description="CSV file with consolidated pathway results (from pathway enrichment)",
            file_type="csv",
            is_file=True,
            required=True,
            can_come_from="pathway_enrichment",
            example="disease_Pathways_Consolidated.csv"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease name for report context",
            is_file=False,
            required=True,
            example="lupus"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="xcell_path",
            description="CSV with deconvolution results (xCell/CIBERSORT)",
            file_type="csv",
            is_file=True,
            required=False,
            can_come_from="deconvolution",
            example="CIBERSORT_results.csv"
        ),
        InputRequirement(
            name="patient_info_path",
            description="CSV with patient demographic information",
            file_type="csv",
            is_file=True,
            required=False,
            example="Patient_info.csv"
        ),
    ],
    outputs=[
        OutputSpec(
            name="report_output_dir",
            description="Directory containing generated report files",
            state_key="report_output_dir"
        ),
    ],
    keywords=[
        "molecular report", "full report", "patient report",
        "comprehensive report", "analysis report", "generate report",
        "clinical report", "molecular analysis", "final report",
        "report generation", "structured report",
    ],
    example_queries=[
        "Generate a molecular report for lupus",
        "Create a comprehensive molecular report with drug recommendations",
        "Generate a molecular report from my prioritized genes",
        "Produce a molecular analysis report for my patient",
    ],
    estimated_time="2-10 minutes",
    depends_on=[AgentType.PATHWAY_ENRICHMENT],
    produces=["report_output_dir", "report_docx_path", "report_pdf_path"],
    requires_one_of=["prioritized_genes_path"],
)


# =============================================================================
# CRISPR PERTURB-SEQ AGENT
# =============================================================================

CRISPR_PERTURB_SEQ_AGENT = AgentInfo(
    agent_type=AgentType.CRISPR_PERTURB_SEQ,
    name="crispr_perturb_seq",
    display_name="🧬 CRISPR Perturb-seq Agent",
    description="Runs CRISPR Perturb-seq analysis on 10X scRNA-seq data with perturbation calling, mixscape, DEG signatures, ML models, Bayesian networks, and causal inference",
    detailed_description="""
The CRISPR Perturb-seq Agent processes 10X scRNA-seq matrices from CRISPR
perturbation experiments through 13 stages: ingestion, perturbation calling,
mixscape, post-mixscape analysis, DEG signatures, ML dataset export, model
training (XGBoost/RF), prediction & ranking, QC, causal inference, Bayesian
networks, latent AI, and report generation.

**Best for:**
- Analyzing CRISPR perturbation screens with single-cell readout
- Identifying gene perturbation effects at single-cell resolution
- Building predictive models from Perturb-seq data
""",
    required_inputs=[
        InputRequirement(
            name="crispr_10x_input_dir",
            description="Directory containing 10X scRNA-seq matrix files (barcodes.tsv.gz, genes.tsv.gz/features.tsv.gz, matrix.mtx.gz)",
            file_type="directory",
            is_file=False,
            required=True,
            example="/data/perturb_seq/GSE12345/"
        ),
    ],
    optional_inputs=[
        InputRequirement(
            name="disease_name",
            description="Disease or condition being studied",
            is_file=False,
            required=False,
            example="AML"
        )
    ],
    outputs=[
        OutputSpec(
            name="crispr_perturb_seq_output_dir",
            description="Directory containing all Perturb-seq pipeline outputs",
            state_key="crispr_perturb_seq_output_dir"
        ),
    ],
    keywords=[
        "perturb-seq", "perturb seq", "perturbseq", "crispr perturb",
        "crispr single cell", "crispr scrna", "crispr 10x",
        "perturbation screen single cell", "mixscape",
        "gene perturbation scRNA", "crispr knockout scrna",
    ],
    example_queries=[
        "Run Perturb-seq analysis on my 10X data",
        "Analyze CRISPR perturbation single-cell experiment",
        "Process my Perturb-seq screen",
    ],
    estimated_time="30-120 minutes",
    depends_on=[],
    produces=["crispr_perturb_seq_output_dir"],
    requires_one_of=["crispr_10x_input_dir"],
)


# =============================================================================
# CRISPR SCREENING AGENT
# =============================================================================

CRISPR_SCREENING_AGENT = AgentInfo(
    agent_type=AgentType.CRISPR_SCREENING,
    name="crispr_screening",
    display_name="🔬 CRISPR Screening Agent",
    description="Runs CRISPR genetic screening analysis using nf-core/crisprseq with sgRNA count tables for hit identification via MAGeCK, BAGEL2, and directional scoring",
    detailed_description="""
The CRISPR Screening Agent runs bulk CRISPR genetic screens through
nf-core/crisprseq with Nextflow + Singularity. Supports 6 analysis modes
(MAGeCK MLE/RRA/count, BAGEL2, directional scoring). Takes sgRNA count
tables and contrast definitions as input.

**Best for:**
- Genome-wide CRISPR knockout/activation/inhibition screens
- Identifying essential genes or drug resistance genes
- Hit calling from pooled CRISPR library screens
""",
    required_inputs=[
        InputRequirement(
            name="crispr_screening_input_dir",
            description="Directory containing sgRNA count table (count_table.tsv) and contrast definitions (rra_contrasts.txt)",
            file_type="directory",
            is_file=False,
            required=True,
            example="/data/crispr_screen/"
        ),
    ],
    optional_inputs=[
        InputRequirement(
            name="disease_name",
            description="Disease or condition being studied",
            is_file=False,
            required=False,
            example="GBM"
        ),
        InputRequirement(
            name="modes",
            description="Optional screening analysis modes to run",
            is_file=False,
            required=False,
            example="[3]"
        )
    ],
    outputs=[
        OutputSpec(
            name="crispr_screening_output_dir",
            description="Directory containing screening analysis outputs",
            state_key="crispr_screening_output_dir"
        ),
    ],
    keywords=[
        "crispr screen", "crispr screening", "genetic screen",
        "sgrna", "guide rna", "mageck", "bagel2",
        "knockout screen", "crispr library", "pooled screen",
        "crisprseq", "crispr-seq", "hit calling",
        "essential genes screen", "fitness screen",
    ],
    example_queries=[
        "Run CRISPR screening analysis on my sgRNA counts",
        "Analyze my pooled CRISPR knockout screen",
        "Identify hits from my CRISPR library screen",
    ],
    estimated_time="15-60 minutes",
    depends_on=[],
    produces=["crispr_screening_output_dir"],
    requires_one_of=["crispr_screening_input_dir"],
)


# =============================================================================
# CRISPR TARGETED AGENT
# =============================================================================

CRISPR_TARGETED_AGENT = AgentInfo(
    agent_type=AgentType.CRISPR_TARGETED,
    name="crispr_targeted",
    display_name="🎯 CRISPR Targeted Agent",
    description="Runs targeted CRISPR editing analysis on paired FASTQ files with nf-core/crisprseq for indel quantification, editing efficiency, and off-target detection",
    detailed_description="""
The CRISPR Targeted Agent analyzes targeted CRISPR editing experiments
using nf-core/crisprseq with Nextflow + Singularity. Takes paired-end FASTQ
files plus protospacer and gene target information. Quantifies editing
efficiency, indel profiles, and off-target effects.

**Best for:**
- Validating CRISPR gene editing at specific loci
- Quantifying indel frequencies from amplicon sequencing
- Assessing on-target vs off-target editing efficiency
""",
    required_inputs=[
        InputRequirement(
            name="protospacer",
            description="Protospacer sequence (guide RNA target, e.g. ATCGATCGATCGATCGATCG)",
            is_file=False,
            required=True,
            example="ATCGATCGATCGATCGATCG"
        )
    ],
    optional_inputs=[
        InputRequirement(
            name="crispr_targeted_input_dir",
            description="Directory containing paired FASTQ files for amplicon sequencing",
            file_type="directory",
            is_file=False,
            required=False,
            example="/data/crispr_targeted/fastqs/"
        ),
        InputRequirement(
            name="project_id",
            description="Public sequencing project ID for automatic metadata extraction and FASTQ download",
            is_file=False,
            required=False,
            example="PRJNA1240319"
        ),
        InputRequirement(
            name="target_gene",
            description="Target gene symbol (e.g. BCL11A). One of target_gene, region, or reference_seq should be provided.",
            is_file=False,
            required=False,
            example="BCL11A"
        ),
        InputRequirement(
            name="region",
            description="Target genomic region in chr:start-end format",
            is_file=False,
            required=False,
            example="chr10:100000-101000"
        ),
        InputRequirement(
            name="reference_seq",
            description="Direct amplicon reference sequence",
            is_file=False,
            required=False,
            example="ACGTACGTACGTACGTACGTACGTACGTACGT"
        ),
        InputRequirement(
            name="disease_name",
            description="Disease or condition being studied",
            is_file=False,
            required=False,
            example="SCD"
        ),
    ],
    outputs=[
        OutputSpec(
            name="crispr_targeted_output_dir",
            description="Directory containing targeted CRISPR analysis outputs",
            state_key="crispr_targeted_output_dir"
        ),
    ],
    keywords=[
        "crispr targeted", "targeted editing", "gene editing",
        "indel", "amplicon sequencing", "editing efficiency",
        "protospacer", "guide rna validation", "on-target",
        "off-target", "crispr validation", "knock-in", "knock-out",
        "crispr amplicon", "crispr fastq",
    ],
    example_queries=[
        "Analyze my targeted CRISPR editing experiment",
        "Quantify indels from my CRISPR amplicon sequencing",
        "Check editing efficiency for BCL11A knockout",
    ],
    estimated_time="15-45 minutes",
    depends_on=[],
    produces=["crispr_targeted_output_dir"],
    requires_one_of=["crispr_targeted_input_dir"],
)


CAUSALITY_AGENT = AgentInfo(
    agent_type=AgentType.CAUSALITY,
    name="causality",
    display_name="🔬 Causality Agent",
    description="Runs causal inference analysis integrating upstream pipeline outputs (DEG, pathway, deconvolution, temporal, perturbation) with literature evidence to produce mechanistic causal narratives and evidence matrices",
    detailed_description="""
The Causality Agent synthesises outputs from multiple upstream agents into a
causal inference framework.  It inspects files, applies eligibility gates,
classifies intent (7 categories), runs a literature pipeline, builds an
execution plan, and produces evidence-scored causal narratives.

**Best for:**
- Mechanistic interpretation of multi-omics results
- Causal gene-disease reasoning backed by literature
- Evidence synthesis across DEG, pathway, deconvolution, perturbation outputs
""",
    required_inputs=[],
    optional_inputs=[
        InputRequirement(
            name="disease_name",
            description="Disease or condition being studied",
            is_file=False,
            required=False,
            example="lupus"
        ),
    ],
    outputs=[
        OutputSpec(
            name="causality_output_dir",
            description="Directory containing causality analysis outputs",
            state_key="causality_output_dir"
        ),
    ],
    keywords=[
        "causality", "causal inference", "causal analysis", "mechanistic",
        "evidence synthesis", "causal reasoning", "cause and effect",
        "causal narrative", "evidence matrix", "literature synthesis",
    ],
    example_queries=[
        "What are the causal mechanisms linking DEGs to disease pathways?",
        "Synthesise causal evidence from my analysis results",
        "Run causal inference on the upstream pipeline outputs",
    ],
    estimated_time="5-15 minutes",
    depends_on=[],
    produces=["causality_output_dir"],
    requires_one_of=[],
)


# =============================================================================
# PIPELINE ORDER - Defines the standard analysis chain
# =============================================================================

PIPELINE_ORDER = [
    AgentType.COHORT_RETRIEVAL,
    AgentType.DEG_ANALYSIS,
    AgentType.GENE_PRIORITIZATION,
    AgentType.PATHWAY_ENRICHMENT,
    AgentType.PERTURBATION_ANALYSIS,
    AgentType.MOLECULAR_REPORT,
]

# Maps detected file types to the input keys they can satisfy
# A single file type can satisfy multiple agent input requirements
FILE_TYPE_TO_INPUT_KEY = {
    'raw_counts': ['counts_file', 'bulk_file'],  # DEG analysis OR deconvolution
    'deg_results': ['deg_input_file'],
    'prioritized_genes': ['prioritized_genes_path'],
    'pathway_results': ['pathway_consolidation_path'],
    'multiomics_layer': ['multiomics_layers'],  # Accumulated into dict by supervisor
    'patient_info': ['patient_info_path'],
    'deconvolution_results': ['xcell_path'],
    'crispr_10x_data': ['crispr_10x_input_dir'],
    'crispr_count_table': ['crispr_screening_input_dir'],
    'crispr_fastq_data': ['crispr_targeted_input_dir'],
}


# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENT_REGISTRY: Dict[AgentType, AgentInfo] = {
    AgentType.COHORT_RETRIEVAL: COHORT_RETRIEVAL_AGENT,
    AgentType.DEG_ANALYSIS: DEG_ANALYSIS_AGENT,
    AgentType.GENE_PRIORITIZATION: GENE_PRIORITIZATION_AGENT,
    AgentType.PATHWAY_ENRICHMENT: PATHWAY_ENRICHMENT_AGENT,
    AgentType.DECONVOLUTION: DECONVOLUTION_AGENT,
    AgentType.TEMPORAL_ANALYSIS: TEMPORAL_ANALYSIS_AGENT,
    AgentType.HARMONIZATION: HARMONIZATION_AGENT,
    AgentType.MDP_ANALYSIS: MDP_ANALYSIS_AGENT,
    AgentType.PERTURBATION_ANALYSIS: PERTURBATION_ANALYSIS_AGENT,
    AgentType.MULTIOMICS_INTEGRATION: MULTIOMICS_INTEGRATION_AGENT,
    AgentType.FASTQ_PROCESSING: FASTQ_PROCESSING_AGENT,
    AgentType.MOLECULAR_REPORT: MOLECULAR_REPORT_AGENT,
    AgentType.CRISPR_PERTURB_SEQ: CRISPR_PERTURB_SEQ_AGENT,
    AgentType.CRISPR_SCREENING: CRISPR_SCREENING_AGENT,
    AgentType.CRISPR_TARGETED: CRISPR_TARGETED_AGENT,
    AgentType.CAUSALITY: CAUSALITY_AGENT,
}


def get_agent_by_name(name: str) -> Optional[AgentInfo]:
    """Get agent info by name string"""
    for agent in AGENT_REGISTRY.values():
        if agent.name == name or agent.agent_type.value == name:
            return agent
    return None


def get_agent_capabilities_text() -> str:
    """Generate a text description of all agent capabilities for LLM context"""
    text = "Available Bioinformatics Agents:\n\n"
    
    for agent in AGENT_REGISTRY.values():
        text += f"## {agent.display_name}\n"
        text += f"**Name:** {agent.name}\n"
        text += f"**Description:** {agent.description}\n"
        text += f"**Keywords:** {', '.join(agent.keywords[:10])}\n"
        text += f"**Example queries:**\n"
        for q in agent.example_queries[:3]:
            text += f"  - {q}\n"
        text += f"**Required inputs:**\n"
        for inp in agent.required_inputs:
            text += f"  - {inp.name}: {inp.description}\n"
        text += "\n"
    
    return text
