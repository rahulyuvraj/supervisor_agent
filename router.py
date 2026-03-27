"""
Intent Router - LLM-based query routing with detailed reasoning

This module uses an LLM to:
1. Understand user intent
2. Match to the appropriate agent
3. Extract parameters from the query
4. Provide reasoning for the routing decision
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .llm_provider import llm_complete, _safe_json_parse

from .agent_registry import (
    AGENT_REGISTRY, AgentInfo, AgentType, 
    get_agent_capabilities_text, get_agent_by_name,
    PIPELINE_ORDER
)
from .state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class AgentIntent:
    """Single agent intent detection result"""
    agent_type: AgentType
    agent_name: str
    confidence: float
    reasoning: str
    order: int = 0  # Execution order in pipeline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "order": self.order
        }


@dataclass
class RoutingDecision:
    """Result of intent routing"""
    agent_type: Optional[AgentType]
    agent_name: Optional[str]
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Human-readable explanation
    extracted_params: Dict[str, Any]  # Parameters extracted from query
    missing_inputs: List[str]  # Required inputs that are missing
    is_general_query: bool  # True if not agent-specific
    suggested_response: Optional[str]  # For general queries
    # Multi-agent support
    is_multi_agent: bool = False  # True if multiple agents detected
    agent_pipeline: List[AgentIntent] = None  # Ordered list of agents to execute
    
    def __post_init__(self):
        if self.agent_pipeline is None:
            self.agent_pipeline = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value if self.agent_type else None,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "extracted_params": self.extracted_params,
            "missing_inputs": self.missing_inputs,
            "is_general_query": self.is_general_query,
            "suggested_response": self.suggested_response,
            "is_multi_agent": self.is_multi_agent,
            "agent_pipeline": [a.to_dict() for a in self.agent_pipeline] if self.agent_pipeline else []
        }



ROUTING_SYSTEM_PROMPT = """You are an expert bioinformatics assistant that routes user queries to specialized analysis agents.

## Available Agents:

{agent_capabilities}

## Your Task:
Analyze the user's query and determine:
1. Which agent(s) should handle this request (can be multiple for complex queries)
2. Why you chose each agent (reasoning)
3. The order in which agents should execute (based on data dependencies)
4. What parameters can be extracted from the query

## Response Format (JSON):
{{
    "is_multi_agent": true/false,
    "agents": [
        {{
            "agent_name": "agent_name",
            "confidence": 0.0-1.0,
            "reasoning": "Why this agent is needed",
            "order": 1
        }}
    ],
    "extracted_params": {{
        "disease_name": "primary disease or condition from THIS query, or null if none mentioned",
        "disease_names": ["list of ALL diseases mentioned", "or empty array"],
        "tissue_filter": "extracted tissue or null",
        "experiment_filter": "extracted experiment type or null",
        "technique": "deconvolution technique if mentioned"
    }},
    "is_general_query": true/false,
    "suggested_response": "For general/knowledge queries: provide a comprehensive, expert-level biomedical response covering mechanisms, context, clinical significance, and key findings. Be thorough and substantive — 3+ paragraphs for scientific questions. For agent-routed queries: null",
    "reasoning": "Brief explanation of your overall routing decision"
}}

## Agent Dependencies (STRICT ORDER - respect this exactly):
0. fastq_processing → processes raw sequencing reads (FASTQ/FQ/GZ), produces QC reports and count matrices. Independent - does NOT depend on other pipeline steps. Requires uploaded FASTQ files.
1. cohort_retrieval → provides datasets for any analysis
2. deg_analysis → requires count data, produces DEG list  
3. gene_prioritization → requires DEG list, produces prioritized genes
4. pathway_enrichment → requires ONLY prioritized genes (MUST run AFTER gene_prioritization)
5. deconvolution → requires bulk expression data
6. mdp_analysis → Multi-Disease Pathways analysis, can use counts/DEGs/gene lists + supports MULTIPLE diseases
7. perturbation_analysis → Drug perturbation (DEPMAP + L1000). Requires prioritized genes AND pathway consolidation files. MUST run AFTER pathway_enrichment.
8. multiomics_integration → Multi-omics data integration (genomics, transcriptomics, epigenomics, proteomics, metabolomics). Independent - does NOT depend on other pipeline steps. Requires uploaded omics layer files.
9. molecular_report → Generates a comprehensive molecular analysis report (PDF/DOCX) with gene analysis, pathway enrichment, deconvolution, and drug recommendations. Requires prioritized genes AND pathway consolidation files. MUST run AFTER pathway_enrichment. Optional: deconvolution data, patient info.
   IMPORTANT: ONLY route to molecular_report when user explicitly says "molecular report", "comprehensive report", "patient report", or "full report". Generic requests like "I need a PDF report", "generate a report", or "create a docx" should NOT route here — those are handled by the built-in report renderer.
10. crispr_perturb_seq → CRISPR Perturb-seq analysis on 10X scRNA-seq data. Independent. Requires directory with barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz. Runs 13 stages: perturbation calling, mixscape, DEG signatures, ML models, Bayesian networks, causal inference.
11. crispr_screening → CRISPR genetic screening via nf-core/crisprseq. Independent. Requires directory with sgRNA count tables (count_table.tsv, rra_contrasts.txt). Runs MAGeCK, BAGEL2, directional scoring.
12. crispr_targeted → Targeted CRISPR editing analysis via nf-core/crisprseq. Independent. Requires directory with paired FASTQs + protospacer sequence + target gene. Quantifies indels, editing efficiency, off-target effects.
13. structured_report → Generates a structured evidence report (Markdown/PDF/DOCX) from completed pipeline outputs. Reads DEG tables, pathway enrichment, drug discovery artifacts, scores evidence, detects cross-module conflicts, renders validated sections. MUST run AFTER the pipeline agents that produce outputs. Route here when user asks for a structured report, analysis report, evidence report, pipeline report, or summary report AFTER pipeline agents have completed.
   IMPORTANT: Do NOT confuse with molecular_report (clinical patient report). structured_report is a data-driven evidence summary of pipeline outputs. Route to structured_report when user says "generate a structured report", "summarise results", "evidence report", "create analysis report", or "pipeline summary report".

IMPORTANT: pathway_enrichment can NEVER run directly after deg_analysis. The order MUST be:
deg_analysis → gene_prioritization → pathway_enrichment
For perturbation, the full chain is:
deg_analysis → gene_prioritization → pathway_enrichment → perturbation_analysis

## MDP Multi-Disease Examples:
- "Compare lupus and breast cancer pathways" → mdp_analysis with disease_names=["lupus", "breast cancer"]
- "Run MDP for vasculitis vs neuropathy" → mdp_analysis with disease_names=["vasculitis", "neuropathy"]
- "Analyze pathways for these 3 conditions: lupus, cancer, alzheimer's" → mdp_analysis with disease_names=["lupus", "cancer", "alzheimer's"]

## Multi-Agent Query Examples:
- "Find lupus datasets and analyze differentially expressed genes" → [cohort_retrieval, deg_analysis]
- "What pathways are affected by the top genes?" → [gene_prioritization, pathway_enrichment]
- "Complete analysis from datasets to pathways for breast cancer" → [cohort_retrieval, deg_analysis, gene_prioritization, pathway_enrichment]
- "Analyze cell composition and prioritize immune genes" → [deconvolution, gene_prioritization]
- "Run perturbation analysis on my prioritized genes for lupus" → [perturbation_analysis] (if prioritized genes + pathway files already available)
- "Full pipeline with drug perturbation for lupus" → [deg_analysis, gene_prioritization, pathway_enrichment, perturbation_analysis]
- "Find essential genes and drug targets for my disease" → [perturbation_analysis]
- "Run DEPMAP and L1000 analysis" → [perturbation_analysis]
- "Integrate my genomics and transcriptomics data for lupus" → [multiomics_integration]
- "Run multi-omics integration on my uploaded layer files" → [multiomics_integration]
- "Perform cross-omics biomarker discovery" → [multiomics_integration]
- "Generate a molecular report for lupus" → [molecular_report]
- "Full pipeline with molecular report" → [deg_analysis, gene_prioritization, pathway_enrichment, molecular_report]
- "Generate molecular report with drug recommendations" → [molecular_report] (include therapeutic section)
- "Process my FASTQ files" → [fastq_processing]
- "Run QC and quantification on my raw sequencing reads" → [fastq_processing]
- "Trim and align my uploaded FASTQ data" → [fastq_processing]
- "Run Perturb-seq analysis on my 10X CRISPR data" → [crispr_perturb_seq]
- "Analyze my CRISPR perturbation screen" → [crispr_perturb_seq]
- "Run CRISPR screening on my sgRNA counts" → [crispr_screening]
- "Identify hits from my pooled CRISPR library screen" → [crispr_screening]
- "Analyze my targeted CRISPR editing experiment" → [crispr_targeted]
- "Quantify indels from my CRISPR amplicon sequencing" → [crispr_targeted]
- "Generate a structured report of my analysis" → [structured_report] (if pipeline outputs already available)
- "Summarise results from the pipeline run" → [structured_report]
- "Create an evidence report from my completed analysis" → [structured_report]
- "Full pipeline with structured report for lupus" → [deg_analysis, gene_prioritization, pathway_enrichment, perturbation_analysis, structured_report]

## Guidelines:
- Detect multiple intents within a single query
- Order agents by data dependency
- For MDP: Extract ALL disease names into disease_names array
- If user says "and then", "also", "after that" - likely multi-agent
- Words like "complete analysis", "full pipeline", "end-to-end" suggest multi-agent
- Be specific in reasoning for EACH agent
- Confidence should be high (>0.8) only when intent is very clear

## Topic-Change Detection:
- Extract disease_name from the CURRENT query, not from prior context.
- If the user mentions a different gene (e.g. ABL1 vs ERBB2), pathway, drug, or therapeutic area than the current disease context, they have likely shifted topics. Set disease_name to null or the new disease rather than carrying forward the old one.
- Only re-use the prior disease when the user's query is a direct follow-up (e.g. "run pathway enrichment on those genes", "generate a report for it").
- When the user mentions only a gene or pathway without a disease, do NOT automatically assume the prior disease still applies — set disease_name to null unless the query explicitly references the prior context.

## Context:
- User may have already uploaded files or run previous agents
- Consider the conversation history when routing
- If data from a previous agent is available, skip that agent
"""


class IntentRouter:
    """
    Routes user queries to appropriate agents using LLM-based intent classification.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Pre-build agent capabilities text
        self.agent_capabilities = get_agent_capabilities_text()
        
        logger.info("🧭 IntentRouter initialized")
    
    async def route(
        self,
        user_query: str,
        conversation_state: ConversationState
    ) -> RoutingDecision:
        """
        Route a user query to the appropriate agent.
        
        Args:
            user_query: The user's message
            conversation_state: Current conversation state
            
        Returns:
            RoutingDecision with agent selection and reasoning
        """
        logger.info(f"🔍 Routing query: {user_query[:100]}...")
        
        # Build context from conversation state
        context = self._build_context(conversation_state)
        
        # Build the routing prompt
        system_prompt = ROUTING_SYSTEM_PROMPT.format(
            agent_capabilities=self.agent_capabilities
        )
        
        user_prompt = f"""
## User Query:
{user_query}

## Current Context:
{context}

## Conversation History (last 5 messages):
{conversation_state.get_conversation_summary(5)}

Analyze this query and provide your routing decision as JSON.
"""

        
        try:
            llm_result = await llm_complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            result = _safe_json_parse(llm_result.text)
            
            # Parse the routing decision
            decision = self._parse_routing_result(result, conversation_state)
            
            logger.info(
                f"📍 Routing decision: agent={decision.agent_name}, "
                f"confidence={decision.confidence:.2f}, "
                f"general={decision.is_general_query}"
            )
            logger.info(f"💭 Reasoning: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Routing failed: {e}")
            # Return a fallback decision
            return RoutingDecision(
                agent_type=None,
                agent_name=None,
                confidence=0.0,
                reasoning=f"Failed to route query: {str(e)}",
                extracted_params={},
                missing_inputs=[],
                is_general_query=True,
                suggested_response="I encountered an error understanding your request. Could you please rephrase it?"
            )
    
    def _build_context(self, state: ConversationState) -> str:
        """Build context string from conversation state"""
        parts = []
        
        # Current disease — with topic-change guidance
        if state.current_disease:
            parts.append(
                f"- Current disease being analyzed: {state.current_disease}\n"
                f"  NOTE: If the user's query is about a different disease, gene, "
                f"pathway, drug, or therapeutic area than '{state.current_disease}', "
                f"extract the NEW disease/condition as disease_name. "
                f"Do not carry forward '{state.current_disease}' when the query "
                f"topic has clearly changed."
            )
        
        # Uploaded files
        if state.uploaded_files:
            files_list = ", ".join(state.uploaded_files.keys())
            parts.append(f"- Uploaded files: {files_list}")
        
        # Available workflow outputs
        available = state.get_available_inputs()
        if available:
            available_keys = [k for k in available.keys() if not k.startswith("_")]
            if available_keys:
                parts.append(f"- Available data/results: {', '.join(available_keys[:10])}")
        
        # Recent agent executions
        if state.agent_executions:
            recent = state.agent_executions[-3:]
            agents_run = [f"{e.agent_display_name} ({e.status})" for e in recent]
            parts.append(f"- Recently run agents: {', '.join(agents_run)}")
        
        # Waiting for input
        if state.waiting_for_input and state.required_inputs:
            parts.append(f"- Waiting for user to provide: {', '.join(state.required_inputs)}")
        
        return "\n".join(parts) if parts else "No previous context."
    
    def _parse_routing_result(
        self,
        result: Dict[str, Any],
        state: ConversationState
    ) -> RoutingDecision:
        """Parse LLM routing result into RoutingDecision - supports multi-agent"""
        
        # Check if this is a multi-agent response
        is_multi_agent = result.get("is_multi_agent", False)
        agents_list = result.get("agents", [])
        
        # Build agent pipeline if multi-agent
        agent_pipeline = []
        if is_multi_agent and agents_list:
            # First, parse all agent intents
            parsed_intents = []
            for agent_data in agents_list:
                agent_name = agent_data.get("agent_name")
                agent_info = get_agent_by_name(agent_name)
                if agent_info:
                    parsed_intents.append(AgentIntent(
                        agent_type=agent_info.agent_type,
                        agent_name=agent_name,
                        confidence=agent_data.get("confidence", 0.7),
                        reasoning=agent_data.get("reasoning", ""),
                        order=agent_data.get("order", 0)
                    ))
            
            # CRITICAL: Re-sort agents by PIPELINE_ORDER to enforce correct dependencies
            # This ensures gene_prioritization always runs before pathway_enrichment
            def get_pipeline_order(intent: AgentIntent) -> int:
                """Get canonical order from PIPELINE_ORDER, fallback to large number for non-pipeline agents"""
                try:
                    return PIPELINE_ORDER.index(intent.agent_type)
                except ValueError:
                    return 100  # Non-pipeline agents go last
            
            agent_pipeline = sorted(parsed_intents, key=get_pipeline_order)
            
            # Update order values to match actual execution order
            for i, intent in enumerate(agent_pipeline):
                intent.order = i
            
            logger.info(f"🔗 Multi-agent pipeline detected (sorted by PIPELINE_ORDER): {[a.agent_name for a in agent_pipeline]}")
        
        # For single agent or backward compatibility
        agent_name = result.get("agent_name")
        if not agent_name and agents_list:
            # Use first agent from list
            agent_name = agents_list[0].get("agent_name") if agents_list else None
        
        agent_type = None
        agent_info = None
        
        if agent_name:
            agent_info = get_agent_by_name(agent_name)
            if agent_info:
                agent_type = agent_info.agent_type
        
        # Extract parameters
        extracted_params = result.get("extracted_params", {})
        # Clean up null values
        extracted_params = {k: v for k, v in extracted_params.items() if v is not None}
        
        # Check for missing inputs (only for first agent in pipeline)
        missing_inputs = []
        if agent_info:
            available = state.get_available_inputs()
            available.update(extracted_params)
            missing = agent_info.get_missing_inputs(available)
            missing_inputs = [inp.name for inp in missing]
        
        # Get combined reasoning for multi-agent
        reasoning = result.get("reasoning", "")
        if not reasoning and agents_list:
            reasoning = " → ".join(
                [a.get("reasoning", "") for a in agents_list[:3] if a.get("reasoning")]
            )
        if not reasoning:
            # Fall back to a truncated suggested_response for general queries
            suggested = result.get("suggested_response") or ""
            if suggested:
                reasoning = suggested[:120]
        
        return RoutingDecision(
            agent_type=agent_type,
            agent_name=agent_name,
            confidence=result.get("confidence", 0.5) if not is_multi_agent else max([a.get("confidence", 0.5) for a in agents_list], default=0.5),
            reasoning=reasoning,
            extracted_params=extracted_params,
            missing_inputs=missing_inputs,
            is_general_query=result.get("is_general_query", False),
            suggested_response=result.get("suggested_response"),
            is_multi_agent=is_multi_agent,
            agent_pipeline=agent_pipeline
        )
    
    async def clarify_intent(
        self,
        user_query: str,
        possible_agents: List[AgentInfo]
    ) -> str:
        """Generate a clarification question when intent is ambiguous"""
        
        agent_options = "\n".join([
            f"- **{a.display_name}**: {a.description}"
            for a in possible_agents
        ])
        
        prompt = f"""
The user said: "{user_query}"

This could relate to multiple agents:
{agent_options}

Generate a friendly clarification question to understand which analysis they want.
Keep it conversational and helpful.
"""
        
        llm_result = await llm_complete(
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful bioinformatics assistant.",
            temperature=0.7,
        )
        
        return llm_result.text


class KeywordRouter:
    """
    Simple keyword-based routing as a fallback.
    Useful when LLM is unavailable or for fast initial routing.
    """
    
    def __init__(self):
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self) -> Dict[str, AgentType]:
        """Build keyword to agent type mapping"""
        keyword_map = {}
        for agent_type, agent_info in AGENT_REGISTRY.items():
            for keyword in agent_info.keywords:
                keyword_map[keyword.lower()] = agent_type
        return keyword_map
    
    def route(self, query: str) -> Tuple[Optional[AgentType], float]:
        """
        Route based on keyword matching.
        
        Returns:
            Tuple of (agent_type, confidence)
        """
        query_lower = query.lower()
        
        # Count keyword matches per agent
        matches: Dict[AgentType, int] = {}
        
        for keyword, agent_type in self.keyword_map.items():
            if keyword in query_lower:
                matches[agent_type] = matches.get(agent_type, 0) + 1
        
        if not matches:
            return None, 0.0
        
        # Get agent with most matches
        best_agent = max(matches, key=matches.get)
        total_keywords = len(AGENT_REGISTRY[best_agent].keywords)
        confidence = min(matches[best_agent] / 3, 1.0)  # Normalize
        
        return best_agent, confidence
