"""
intelligence.py — LLM / AI layer for the causality pipeline.

Sections:
    1. Local fallback engines (_LocalIntentEngine, _LocalClarificationEngine,
       _LocalClaimEngine, _LocalBriefEngine, _LocalNarrateEngine,
       _LocalSynthesisEngine, LocalClient)
    2. ClarificationEngine
    3. IntentClassifier
    4. LiteraturePipeline (8 stages)
    5. ResultSynthesiser
    6. StepNarrator

All components receive an ``llm`` object (duck-typed) with a ``complete()``
method matching the demo's ClaudeClient interface.
"""
from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core_agent import (
        ParsedIntent, LitBrief, LitClaim, LiteraturePaper,
        FileAuditResult, FinalResult, ClarificationResult,
    )

log = logging.getLogger("causal_platform")

from .constants import (
    CAUSALITY_MODEL,
    EPMC_BASE,
    LIT_MAX_PAPERS,
    LIT_TIMEOUT,
    LIT_TOP_K,
    MAX_TOKENS_BRIEF,
    MAX_TOKENS_CLAIMS,
    MAX_TOKENS_INTENT,
    MAX_TOKENS_NARRATE,
    MAX_TOKENS_RESULT,
    PUBMED_BASE,
    S2_BASE,
)

_EMAIL = "platform@causal.bio"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Local fallback engines
# ═════════════════════════════════════════════════════════════════════════════

class _LocalIntentEngine:
    """Rule-based intent classifier when no LLM is available."""

    _INTENTS = [
        ("I_01", "Causal Drivers Discovery",
         ["M12", "M13", "M14", "M15", "M_DC"],
         [("causal driver", 5), ("what causes", 4), ("causal mechanism", 4),
          ("upstream regulator", 4), ("find cause", 3), ("identify cause", 3),
          ("causal gene", 3), ("drives", 2), ("discover", 1)]),
        ("I_02", "Directed Causality X->Y",
         ["M12", "M_DC"],
         [("does .+ cause", 6), ("causally affect", 6), ("x.*cause.*y", 5),
          ("is .+ upstream", 5), ("causal effect of", 4), ("does .+ drive", 4),
          ("causal link between", 3), ("affect", 1)]),
        ("I_03", "Intervention / Actionability",
         ["M_IS", "M_PI"],
         [("drug target", 5), ("which target", 5), ("inhibit", 4), ("therapeutic", 4),
          ("intervention", 4), ("actionable", 4), ("treat", 3), ("drug", 3),
          ("clinical", 2), ("suppress", 2)]),
        ("I_04", "Comparative Causality",
         ["M12", "M12b", "M13", "M14", "M15", "M_DC", "DELTA"],
         [("compare", 5), ("difference between", 5), ("responder.*non.responder", 5),
          ("group a.*group b", 5), ("early.*late", 4), ("stratif", 4),
          ("versus", 3), ("across cohort", 3), ("between groups", 3)]),
        ("I_05", "Counterfactual / What-If",
         ["M_IS", "M_DC"],
         [("what if", 6), ("simulate", 6), ("counterfactual", 6), ("what happen", 5),
          ("predict.*effect", 5), ("if we.*inhibit", 5), ("knock.*down.*by", 5),
          ("what would", 4), ("scenario", 3)]),
        ("I_06", "Evidence Inspection / Explain",
         ["M15", "M14"],
         [("why is .+ ranked", 6), ("explain.*evidence", 6), ("what evidence", 5),
          ("evidence for", 5), ("justify", 4), ("evidence support", 4),
          ("which evidence", 4), ("evidence behind", 3), ("explain", 2)]),
        ("I_07", "Standard Association Analysis",
         [],
         [("differentially expressed", 5), ("deg", 4), ("pathway enrichment", 5),
          ("association", 4), ("correlation", 4), ("gsea", 4), ("go enrichment", 4),
          ("kegg", 3), ("top gene", 2)]),
    ]

    _DOMAIN_KEYWORDS = {
        "oncology":     ["cancer", "tumor", "oncogen", "carcinoma", "metastasis", "KRAS", "TP53"],
        "immunology":   ["immune", "autoimmune", "inflammat", "lupus", "SLE", "T cell", "B cell",
                         "cytokine", "interferon", "JAK", "STAT"],
        "neuroscience": ["neuron", "brain", "neurodegenera", "alzheimer", "parkinson",
                         "dopamine", "synaptic", "cortex"],
        "metabolism":   ["metabol", "insulin", "diabetes", "glucose", "lipid", "obesity",
                         "adipose", "AMPK", "mTOR"],
        "aging":        ["aging", "ageing", "senescen", "longevity", "telomer", "FOXO"],
        "cardiology":   ["cardiac", "heart", "cardiomyopathy", "arrhythmia", "myocardial"],
        "development":  ["development", "differentiation", "stem cell", "embryo", "pluripotent"],
        "microbiome":   ["microbiome", "microbiota", "gut bacteria", "16S", "metagenom"],
        "pharmacology": ["drug", "compound", "inhibitor", "agonist", "pharmacol"],
    }

    def classify(self, query: str) -> dict:
        q = query.lower()
        scores: dict[str, float] = {}
        for iid, iname, chain, kws in self._INTENTS:
            score = 0.0
            for pattern, weight in kws:
                if re.search(pattern, q):
                    score += weight
            scores[iid] = score

        best_id = max(scores, key=lambda k: scores[k])
        best_score = scores[best_id]
        confidence = 0.65 + min(best_score / 30.0, 1.0) * 0.33

        defn = next(d for d in self._INTENTS if d[0] == best_id)
        _, iname, chain, _ = defn

        domain, phenotype = self._extract_context(query)
        gene_x, gene_y, intervention, groups, magnitude = self._extract_entities(query)

        needs_clarif = False
        clarif_q = None
        if best_id == "I_02" and not gene_x:
            needs_clarif = True
            clarif_q = "Which gene/protein is X (the potential cause), and what is Y (the outcome)?"
        elif best_id == "I_04" and not groups:
            needs_clarif = True
            clarif_q = "Which two groups should be compared?"
        elif best_id == "I_05" and not intervention:
            needs_clarif = True
            clarif_q = "Which gene/drug should be simulated, and to what level?"

        requires = {
            "expression":        best_id in ("I_01", "I_02", "I_04", "I_07"),
            "phenotype_labels":  best_id in ("I_01", "I_02", "I_04", "I_07"),
            "gwas_eqtl":         best_id in ("I_01", "I_02", "I_03"),
            "perturbation":      best_id in ("I_01", "I_03"),
            "temporal":          best_id in ("I_01", "I_02", "I_04"),
            "prior_network":     best_id in ("I_01", "I_02"),
            "intervention_data": best_id in ("I_03", "I_05"),
        }
        fallbacks = {
            "I_01": "Association-only ranking with explicit disclaimer",
            "I_02": "Graph-only hypothesis with uncertainty flags",
            "I_03": "Observed-only ranking with context caveats",
            "I_04": "Single-run results with stratification note",
            "I_05": "Observed-only results labelled as simulated",
            "I_06": "Partial evidence card from available artifacts",
            "I_07": "DEGs and pathways only",
        }
        parallel = []
        if best_id == "I_04":
            parallel = ["DAGBuilder group A and B run in parallel after shared Bio Prep"]
        elif best_id == "I_01":
            parallel = ["T_05 (temporal) + T_08 (GWAS/MR) run in parallel during Bio/Intervention Prep"]

        return {
            "intent_id":           best_id,
            "intent_name":         iname,
            "confidence":          round(confidence, 3),
            "needs_clarification": needs_clarif,
            "clarifying_question": clarif_q,
            "context": {
                "domain":              domain,
                "phenotype":           phenotype,
                "tissue_or_cell_type": self._extract_tissue(query),
                "organism":            self._extract_organism(query),
                "outcome_variable":    None,
                "cohort_description":  None,
            },
            "entities": {
                "gene_x":       gene_x,
                "gene_y":       gene_y,
                "intervention": intervention,
                "magnitude":    magnitude,
                "groups":       groups,
                "comparison":   groups,
                "pathway":      self._extract_pathway(query),
            },
            "requires":               requires,
            "needs_literature_first": best_id in ("I_01", "I_02", "I_03"),
            "requires_existing_dag":  best_id in ("I_03", "I_05", "I_06"),
            "routing_summary":        self._routing_summary(best_id, iname, domain, phenotype),
            "module_chain":           chain,
            "parallel_blocks":        parallel,
            "fallback":               fallbacks.get(best_id, "Association-only ranking"),
        }

    def _extract_context(self, query: str):
        q = query.lower()
        domain, phenotype = "molecular biology", ""
        for dom, kws in self._DOMAIN_KEYWORDS.items():
            if any(kw.lower() in q for kw in kws):
                domain = dom
                break
        m = re.search(
            r"\b(?:of|in|for|drive|cause)\s+([A-Za-z][A-Za-z0-9 _\-]{2,40}?)"
            r"(?:\s+(?:in|using|with|from|cohort|patient|dataset|study|analysis)|$)",
            query, re.I,
        )
        if m:
            phenotype = m.group(1).strip()
        return domain, phenotype

    @staticmethod
    def _extract_tissue(query: str):
        tissues = ["blood", "pbmc", "kidney", "liver", "lung", "brain", "heart",
                   "muscle", "adipose", "skin", "colon", "breast", "pancreas",
                   "cell line", "k562", "jurkat", "hela", "single.cell"]
        for t in tissues:
            if re.search(t, query, re.I):
                return t.replace(".", " ")
        return None

    @staticmethod
    def _extract_organism(query: str) -> str:
        if re.search(r"\bmouse\b|\bmurine\b|\bMus musculus\b", query, re.I):
            return "mouse"
        if re.search(r"\bzebrafish\b|\bdanio\b", query, re.I):
            return "zebrafish"
        if re.search(r"\byeast\b|\bS\. cerevisiae\b", query, re.I):
            return "yeast"
        return "human"

    @staticmethod
    def _extract_entities(query: str):
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,7}(?:\d+)?)\b', query)
        gene_x = genes[0] if genes else None
        gene_y = genes[1] if len(genes) > 1 else None
        m_int = re.search(
            r'\b(inhibit(?:ion|ing)?|knock(?:down|out)|overexpress(?:ion)?|activat(?:e|ion))\b',
            query, re.I,
        )
        intervention = m_int.group(1) if m_int else gene_x
        m_mag = re.search(r'(\d+)\s*%', query)
        magnitude = f"{m_mag.group(1)}%" if m_mag else None
        m_grp = re.search(
            r'(responders?|non.responders?|control|treated|early|late|group\s*[AB12])',
            query, re.I,
        )
        groups = m_grp.group(1) if m_grp else None
        return gene_x, gene_y, intervention, groups, magnitude

    @staticmethod
    def _extract_pathway(query: str):
        known = ["MAPK", "PI3K", "AKT", "mTOR", "Wnt", "Notch", "Hedgehog", "JAK.STAT",
                 "NF.kB", "TGF.beta", "p53", "RAS", "VEGF", "HIF", "Hippo"]
        for pw in known:
            if re.search(pw, query, re.I):
                return pw
        return None

    @staticmethod
    def _routing_summary(iid: str, iname: str, domain: str, phenotype: str) -> str:
        pheno = phenotype or "the phenotype of interest"
        dom = domain or "molecular biology"
        summaries = {
            "I_01": f"Full causal driver discovery in {dom} — {pheno}: all 4 phases run.",
            "I_02": f"Targeted X->Y causal test in {dom} — {pheno}: bio-prep then do-calculus.",
            "I_03": f"Intervention/drug target prioritisation for {pheno}: reads existing DAG.",
            "I_04": f"Comparative causal analysis for {pheno}: dual DAG build + DELTA.",
            "I_05": f"Counterfactual simulation for {pheno}: reads existing DAG.",
            "I_06": f"Evidence inspection for {pheno}: read-only, no new compute.",
            "I_07": f"Association analysis for {pheno}: DEG + pathway only.",
        }
        return summaries.get(iid, f"{iname}: {pheno}")


class _LocalClarificationEngine:
    _ORGANISMS = ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast",
                  "arabidopsis", "c. elegans", "pig", "non-human primate"]
    _TISSUES   = ["liver", "lung", "brain", "cortex", "heart", "kidney", "colon",
                  "breast", "skin", "blood", "pbmc", "neuron", "hepatocyte",
                  "fibroblast", "macrophage", "t cell", "b cell", "nk cell",
                  "stem cell", "organoid", "tumor", "stroma", "endothel"]

    def generate(self, query: str, file_types: list, intent_name: str) -> dict:
        ql = query.lower()
        questions: list[str] = []
        blocking: list[int] = []
        if not any(o in ql for o in self._ORGANISMS):
            questions.append("1. What organism is your data from? (e.g. human, mouse, rat)")
            blocking.append(len(questions))
        if not any(t in ql for t in self._TISSUES):
            questions.append(
                f"{len(questions)+1}. What tissue or cell type? (e.g. colon epithelium, PBMC)"
            )
        has_meta = any(t in file_types for t in ("metadata", "phenotype"))
        has_expr = "expression" in file_types
        if has_expr and not has_meta:
            questions.append(
                f"{len(questions)+1}. Do you have a metadata file with case/control labels?"
            )
            blocking.append(len(questions))
        if not file_types:
            questions.append(
                f"{len(questions)+1}. Which data files? (a) Expression (b) Metadata "
                "(c) GWAS/MR (d) CRISPR (e) eQTL (f) Temporal"
            )
        if not questions or len(questions) < 2:
            questions.append(
                f"{len(questions)+1}. Primary goal? (a) Identify all causal drivers "
                "(b) Druggable targets (c) Mechanistic pathway (d) Compare groups"
            )
        return {
            "questions": questions[:4],
            "can_proceed_without_answers": len(blocking) == 0,
            "blocking_question_indices": blocking[:4],
        }


class _LocalClaimEngine:
    def extract(self, abstracts_text: str) -> dict:
        return {"claims": []}


class _LocalBriefEngine:
    def build(self, context_str: str) -> dict:
        return {
            "inferred_context":       "molecular biology phenotype",
            "prior_evidence_summary": "Literature search skipped — no LLM available.",
            "causal_vs_associative":  "mixed",
            "recommended_modules":    ["M12", "M13", "M14", "M15", "M_DC"],
            "data_gaps":              ["expression matrix", "metadata with phenotype labels"],
            "supervisor_brief":       "Running with local fallback engines. "
                                      "Causal analysis proceeds with available data.",
        }


class _LocalNarrateEngine:
    _TEMPLATES = {
        "T_00": "Cohort Data Retrieval searches GEO/SRA for a matching dataset. "
                "Retrieved files flow into T_01 normalization.",
        "T_01": "Expression Normalization applies log2-CPM and QC filtering. "
                "The normalized matrix feeds T_02, T_04, and T_05.",
        "T_02": "Differential Expression (DESeq2) identifies up/down-regulated genes. "
                "The DEG table drives T_03 and M12.",
        "T_03": "Pathway Enrichment runs ORA and GSEA across GO/KEGG/Reactome. "
                "Enriched pathways provide structural priors for M12.",
        "T_04": "Cell-Type Deconvolution estimates immune/stromal fractions. "
                "Fractions flow into M12 as immunological context.",
        "T_04b": "Single-Cell Pipeline performs QC, clustering, annotation. "
                 "Outputs pseudobulk expression for M12.",
        "T_05": "Temporal Pipeline fits impulse model + Granger causality. "
                "Granger edges flow into M12 as directional priors.",
        "T_06": "Perturbation Pipeline computes ACE scores from DepMap/CRISPR. "
                "CRISPR scores have weight 0.25 in M15.",
        "T_07": "Prior Knowledge Network assembles SIGNOR/STRING/KEGG edges. "
                "Network edges have weight 0.15 in M15.",
        "T_08": "GWAS/eQTL/MR Preprocessing fetches genetic association evidence. "
                "Genetic evidence has weight 0.30 in M15.",
        "M12": "DAGBuilder runs PC/FCI/GES consensus structure learning. "
               "The consensus DAG is the central downstream artifact.",
        "M13": "DAGValidator tests edge stability via bootstrap. "
               "Produces the validated DAG for M14.",
        "M14": "CentralityCalculator computes kME/betweenness/PageRank. "
               "Ranked targets feed M15 and final synthesis.",
        "M15": "EvidenceAggregator fuses six streams (genetic 0.30, perturbation 0.25, "
               "temporal 0.20, network 0.15, expression 0.05, immuno 0.05).",
        "M_DC": "DoCalculusEngine applies Backdoor Criterion for directionality. "
                "Confirms or refutes causal direction per candidate.",
        "M_IS": "InSilicoSimulator propagates do(X=0) intervention through the DAG. "
                "Outputs guide M_PI target prioritisation.",
        "M_PI": "PharmaInterventionEngine scores targets against DrugBank/ChEMBL/DepMap/LINCS. "
                "Produces a Target Product Profile.",
        "DELTA": "Delta Analysis compares two group-specific DAGs. "
                 "Reveals shared vs context-dependent drivers.",
    }

    def narrate(self, step: dict) -> str:
        sid = step.get("id", "")
        return self._TEMPLATES.get(sid, step.get("desc", f"{sid}: processing step."))


class _LocalSynthesisEngine:
    def synthesise(self, context: str) -> dict:
        return {
            "headline":          "Analysis pipeline executed — modules pending implementation.",
            "analyzed_context":  "molecular biology phenotype of interest",
            "top_findings":      [
                "Expression normalization and QC completed.",
                "Causal modules (M12–M_DC) awaiting implementation.",
            ],
            "tier1_candidates":  [],
            "tier2_candidates":  [],
            "actionable_targets": [],
            "evidence_quality": {
                "streams_present":    ["expression"],
                "streams_missing":    ["genetic", "perturbation", "temporal", "network", "immuno"],
                "overall_confidence": "low",
                "note": "Causal modules not yet implemented.",
            },
            "caveats":             ["Implement M12 DAGBuilder in tool_registry.py"],
            "next_experiments":    ["Implement M12 DAGBuilder"],
            "missing_data_impact": ["All causal evidence streams need implementation"],
        }


class LocalClient:
    """
    Deterministic rule-based client (no LLM).
    Routes by fingerprinting the system prompt.
    """
    def __init__(self):
        self._intent    = _LocalIntentEngine()
        self._clarif    = _LocalClarificationEngine()
        self._claims    = _LocalClaimEngine()
        self._brief     = _LocalBriefEngine()
        self._narrate   = _LocalNarrateEngine()
        self._synthesis = _LocalSynthesisEngine()

    def complete(self, user_msg: str, system_prompt: str,
                 max_tokens: int, as_json: bool = True):
        sys_l = system_prompt.lower()
        if "intake agent" in sys_l:
            return self._clarif.generate(user_msg, [], "")
        if "intent" in sys_l and "i_01" in sys_l:
            return self._intent.classify(user_msg)
        if "claim extractor" in sys_l:
            return self._claims.extract(user_msg)
        if "planning brief" in sys_l:
            return self._brief.build(user_msg)
        if "narrating a live" in sys_l:
            step_id = ""
            m = re.search(r"Step:\s*\S+\s*\((\w+)\)", user_msg)
            if m:
                step_id = m.group(1)
            return self._narrate.narrate({"id": step_id, "desc": user_msg[:120]})
        if "synthesising the final" in sys_l:
            return self._synthesis.synthesise(user_msg)
        return {"raw": user_msg[:200]}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ClarificationEngine
# ═════════════════════════════════════════════════════════════════════════════

_CLARIF_SYSTEM = """\
You are the intake agent for a molecular causal discovery platform.

Given the user's query and the files they have uploaded, generate 2-4 concise
clarifying questions. Focus ONLY on genuinely ambiguous or missing information.

Return ONLY valid JSON (no markdown):
{
  "questions": ["1. ...", "2. ...", ...],
  "can_proceed_without_answers": true/false,
  "blocking_question_indices": [1, 2]
}"""


class ClarificationEngine:
    _ORGANISMS = ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast",
                  "arabidopsis", "c. elegans", "pig"]
    _TISSUES   = ["liver", "lung", "brain", "cortex", "heart", "kidney", "colon",
                  "breast", "skin", "blood", "pbmc", "neuron", "hepatocyte",
                  "fibroblast", "macrophage", "t cell", "b cell", "nk cell",
                  "stem cell", "organoid", "tumor"]

    def __init__(self, llm):
        self._llm   = llm
        self._local = _LocalClarificationEngine()

    def generate_questions(self, query: str, file_types: list,
                           intent_name: str = "") -> list:
        files_str = ", ".join(file_types) if file_types else "none"
        user_msg = (f"Query: {query}\nFiles uploaded: {files_str}\n"
                    f"Detected intent: {intent_name}")
        try:
            resp = self._llm.complete(user_msg, _CLARIF_SYSTEM, 500)
            if isinstance(resp, dict):
                return resp.get("questions", [])
        except Exception as exc:
            log.debug("ClarificationEngine API failed: %s — using local", exc)
        resp = self._local.generate(query, file_types, intent_name)
        return resp.get("questions", [])

    def build_result(self, original_query: str, questions: list,
                     answers: list, skipped: bool = False):
        from .core_agent import ClarificationResult
        if skipped or not any(a.strip() for a in answers):
            return ClarificationResult(
                questions=questions, answers=answers,
                enriched_query=original_query, enriched_context={},
                can_proceed=True, skipped=True,
            )
        qa_lines = [f"{q.strip()} -> {a.strip()}"
                     for q, a in zip(questions, answers) if a.strip()]
        enriched_query = original_query
        if qa_lines:
            enriched_query += "\n[User clarifications: " + "; ".join(qa_lines) + "]"
        ctx = self._extract_context(original_query + " " + " ".join(answers))
        return ClarificationResult(
            questions=questions, answers=answers,
            enriched_query=enriched_query, enriched_context=ctx,
            can_proceed=True, skipped=False,
        )

    @staticmethod
    def _extract_context(text: str) -> dict:
        tl = text.lower()
        ctx: dict = {}
        for org in ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast"]:
            if org in tl:
                ctx["organism"] = org
                break
        for tissue in ["liver", "lung", "brain", "heart", "kidney", "colon",
                       "breast", "skin", "blood", "pbmc", "neuron"]:
            if tissue in tl:
                ctx["tissue_or_cell_type"] = tissue
                break
        if any(w in tl for w in ["drug", "druggable", "therapeutic", "target"]):
            ctx["goal"] = "druggable_targets"
        elif any(w in tl for w in ["mechanism", "pathway"]):
            ctx["goal"] = "mechanistic"
        elif any(w in tl for w in ["compare", "difference", "group", "responder"]):
            ctx["goal"] = "comparative"
        else:
            ctx["goal"] = "full_driver_landscape"
        mag = re.search(r"(\d+)\s*%", text)
        if mag:
            ctx["intervention_magnitude"] = f"{mag.group(1)}%"
        return ctx


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IntentClassifier
# ═════════════════════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """\
You are the Supervisor Agent for a molecular causal discovery platform.

INTENT TYPES:
I_01 = Causal Drivers Discovery
I_02 = Directed Causality X->Y
I_03 = Intervention / Actionability
I_04 = Comparative Causality
I_05 = Counterfactual / What-If
I_06 = Evidence Inspection / Explain
I_07 = Standard Association Analysis

Return ONLY valid JSON:
{
  "intent_id": "I_0X",
  "intent_name": "...",
  "confidence": 0.0-1.0,
  "needs_clarification": true/false,
  "clarifying_question": "...",
  "context": {"domain":"","phenotype":"","tissue_or_cell_type":null,"organism":null,
              "outcome_variable":null,"cohort_description":null},
  "entities": {"gene_x":null,"gene_y":null,"intervention":null,"magnitude":null,
               "groups":null,"comparison":null,"pathway":null},
  "requires": {"expression":false,"phenotype_labels":false,"gwas_eqtl":false,
               "perturbation":false,"temporal":false,"prior_network":false,
               "intervention_data":false},
  "needs_literature_first": true,
  "requires_existing_dag": false,
  "routing_summary": "...",
  "module_chain": [],
  "parallel_blocks": [],
  "fallback": "..."
}"""

_DEFAULT_MODULE_CHAIN = ["M12", "M13", "M14", "M15", "M_DC"]


class IntentClassifier:
    def __init__(self, llm):
        self._llm = llm

    def classify(self, query: str):
        from .core_agent import ParsedIntent, IntentID
        try:
            resp = self._llm.complete(query, _INTENT_SYSTEM, MAX_TOKENS_INTENT)
            return ParsedIntent(
                intent_id=IntentID(resp.get("intent_id", "I_01")),
                intent_name=resp.get("intent_name", "Causal Drivers Discovery"),
                confidence=float(resp.get("confidence", 0.7)),
                needs_clarification=bool(resp.get("needs_clarification", False)),
                clarifying_question=resp.get("clarifying_question"),
                context=resp.get("context", {}),
                entities=resp.get("entities", {}),
                requires=resp.get("requires", {}),
                needs_literature_first=bool(resp.get("needs_literature_first", True)),
                requires_existing_dag=bool(resp.get("requires_existing_dag", False)),
                routing_summary=resp.get("routing_summary", ""),
                module_chain=resp.get("module_chain", []),
                parallel_blocks=resp.get("parallel_blocks", []),
                fallback=resp.get("fallback", "Association-only ranking with disclaimer"),
            )
        except Exception as exc:
            log.warning("Intent classification failed: %s — defaulting to I_01", exc)
            return ParsedIntent(
                intent_id=IntentID.I_01,
                intent_name="Causal Drivers Discovery",
                confidence=0.6,
                needs_clarification=False,
                clarifying_question=None,
                routing_summary="Default causal discovery route.",
                module_chain=_DEFAULT_MODULE_CHAIN,
                fallback="Association-only ranking with disclaimer",
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LiteraturePipeline
# ═════════════════════════════════════════════════════════════════════════════

_SYS_PARSE = """\
You are the query parser for a molecular causal discovery platform.
Return ONLY valid JSON:
{"domain":"","phenotype":"","tissue_or_cell_type":null,"organism":null,
 "key_genes":[],"key_pathways":[],"search_terms":[],"study_type_filter":"causal"}"""

_SYS_CLAIMS = """\
You are a molecular biology claim extractor.
Return ONLY valid JSON:
{"claims":[{"entity_x":"","relation":"","entity_y":"","direction":"",
 "evidence_type":"","strength":"","pmid":"","confidence":0.0,"quote":""}]}"""

_SYS_CONFLICTS = """\
You are a molecular biology evidence analyst.
Return ONLY valid JSON: {"additional_conflicts": []}"""

_SYS_BRIEF = """\
You are the literature planning brief builder.
Return ONLY valid JSON:
{"inferred_context":"","prior_evidence_summary":"","causal_vs_associative":"mixed",
 "recommended_modules":[],"data_gaps":[],"supervisor_brief":""}"""


class LiteraturePipeline:
    def __init__(self, llm):
        self._llm = llm

    def run(self, query: str, intent_name: str, entities: dict):
        from .core_agent import LiteraturePaper, LitClaim, LitBrief

        log.info("[LIT_01] Parsing query...")
        parsed_query = self._parse_query(query, intent_name, entities)

        log.info("[LIT_02] Expanding entities...")
        entity_list = self._expand_entities(parsed_query)

        log.info("[LIT_03] Building search queries...")
        search_queries = self._build_search_queries(parsed_query, entity_list)
        raw_hits: list[dict] = []
        queries_used: list[str] = []
        for q in search_queries:
            pmids = self._search_pubmed(q)
            if pmids:
                queries_used.append(f"PubMed: {q}")
                for pmid in pmids[:8]:
                    raw_hits.append({"pmid": pmid, "source": "pubmed"})
            epmc = self._search_europepmc(q)
            if epmc:
                queries_used.append(f"EPMC: {q}")
                raw_hits.extend(epmc)
        if search_queries:
            s2 = self._search_semantic_scholar(search_queries[0])
            if s2:
                raw_hits.extend(s2)

        log.info("[LIT_04] Deduplicating %d raw hits...", len(raw_hits))
        papers = self._deduplicate_and_rank(raw_hits, LiteraturePaper)

        log.info("[LIT_05] Fetching abstracts for %d papers...", len(papers))
        papers = self._fetch_abstracts(papers)
        papers_with_content = [p for p in papers if p.abstract or p.title]

        log.info("[LIT_06] Extracting claims from %d papers...", len(papers_with_content))
        claims = self._extract_claims(papers_with_content)

        log.info("[LIT_07] Grading evidence...")
        high_conf_edges, conflicts, conflict_rate = self._grade_evidence(claims)

        log.info("[LIT_08] Building planning brief...")
        brief_dict = self._build_planning_brief(
            parsed_query, papers_with_content, claims,
            high_conf_edges, conflicts, conflict_rate,
        )
        brief = LitBrief(
            inferred_context=brief_dict.get("inferred_context", parsed_query.get("domain", "")),
            key_entities=entity_list[:12],
            search_queries_used=queries_used[:6],
            papers_found=len(raw_hits),
            papers_processed=len(papers_with_content),
            claims=[self._dict_to_claim(c, LitClaim) for c in claims],
            high_confidence_edges=high_conf_edges,
            conflicts=conflicts,
            causal_vs_associative=brief_dict.get("causal_vs_associative", "mixed"),
            prior_evidence_summary=brief_dict.get("prior_evidence_summary", ""),
            recommended_modules=brief_dict.get("recommended_modules", []),
            data_gaps=brief_dict.get("data_gaps", []),
            supervisor_brief=brief_dict.get("supervisor_brief", ""),
            conflict_rate=conflict_rate,
        )
        log.info("[LIT] Complete: %d papers, %d claims", brief.papers_processed, len(brief.claims))
        return brief

    def _parse_query(self, query: str, intent_name: str, entities: dict) -> dict:
        try:
            return self._llm.complete(
                f"Query: {query}\nIntent: {intent_name}\nEntities: {entities}",
                _SYS_PARSE, 500,
            )
        except Exception:
            return {"domain": "molecular biology", "key_genes": [],
                    "search_terms": [query[:100]], "phenotype": ""}

    @staticmethod
    def _expand_entities(parsed: dict) -> list:
        terms = (list(parsed.get("key_genes", []))
                 + list(parsed.get("key_pathways", []))
                 + list(parsed.get("search_terms", [])))
        phenotype = parsed.get("phenotype", "")
        if phenotype:
            terms.append(phenotype)
        seen: set[str] = set()
        out: list[str] = []
        for t in terms:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out[:20]

    @staticmethod
    def _build_search_queries(parsed: dict, entities: list) -> list:
        domain    = parsed.get("domain", "")
        phenotype = parsed.get("phenotype", "")
        custom    = parsed.get("search_terms", [])
        queries   = list(custom[:3])
        if phenotype and domain:
            queries.append(
                f"{phenotype}[Title/Abstract] AND {domain}[Title/Abstract] AND (causal OR mechanism)"
            )
        if entities:
            top = " OR ".join(f'"{e}"' for e in entities[:4])
            queries.append(f"({top}) AND causal[Title/Abstract]")
        if phenotype:
            queries.append(
                f"{phenotype}[Title/Abstract] AND (gene expression OR transcriptomics OR GWAS)"
            )
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            if q.strip() and q.strip() not in seen:
                seen.add(q.strip())
                deduped.append(q.strip())
        return deduped[:5]

    @staticmethod
    def _search_pubmed(query: str) -> list:
        import httpx
        try:
            params = {"db": "pubmed", "term": query, "retmax": 15,
                      "sort": "relevance", "retmode": "json",
                      "tool": "CausalPlatform", "email": _EMAIL}
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
                r.raise_for_status()
                return r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            log.debug("PubMed search failed: %s", exc)
            return []

    @staticmethod
    def _search_europepmc(query: str) -> list:
        import httpx
        try:
            params = {"query": query, "format": "json", "pageSize": 10,
                      "resultType": "core", "sort": "RELEVANCE"}
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{EPMC_BASE}/search", params=params)
                r.raise_for_status()
                return [
                    {"pmid": it.get("pmid"), "doi": it.get("doi"),
                     "title": it.get("title", ""), "abstract": it.get("abstractText", ""),
                     "year": it.get("pubYear"), "journal": it.get("journalTitle", ""),
                     "source": "europepmc"}
                    for it in r.json().get("resultList", {}).get("result", [])
                ]
        except Exception as exc:
            log.debug("EuropePMC failed: %s", exc)
            return []

    @staticmethod
    def _search_semantic_scholar(query: str) -> list:
        import httpx
        try:
            params = {"query": query, "limit": 8,
                      "fields": "title,year,abstract,externalIds,venue"}
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{S2_BASE}/paper/search", params=params)
                r.raise_for_status()
                return [
                    {"pmid": it.get("externalIds", {}).get("PubMed"),
                     "doi": it.get("externalIds", {}).get("DOI"),
                     "title": it.get("title", ""), "abstract": it.get("abstract", ""),
                     "year": it.get("year"), "journal": it.get("venue", ""),
                     "source": "semanticscholar"}
                    for it in r.json().get("data", [])
                ]
        except Exception as exc:
            log.debug("Semantic Scholar failed: %s", exc)
            return []

    @staticmethod
    def _deduplicate_and_rank(raw_hits: list, paper_cls) -> list:
        seen_pmids: set[str] = set()
        seen_dois: set[str]  = set()
        seen_titles: set[str] = set()
        unique: list[dict] = []
        for h in raw_hits:
            pmid  = str(h.get("pmid") or "").strip()
            doi   = str(h.get("doi") or "").strip().lower()
            title = str(h.get("title") or "").strip().lower()[:80]
            if pmid and pmid in seen_pmids:
                continue
            if doi and doi in seen_dois:
                continue
            if title and title in seen_titles:
                continue
            if pmid:
                seen_pmids.add(pmid)
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)
            unique.append(h)

        def _score(h: dict) -> float:
            has_abstract = 2.0 if h.get("abstract") else 0.0
            year = int(h.get("year") or 2000)
            causal_bonus = sum(
                1 for kw in ["causal", "gwas", "crispr", "knockdown", "mendelian",
                              "perturbation", "driver", "mechanism"]
                if kw in (h.get("title") or "").lower()
            )
            return has_abstract + (year - 2000) * 0.1 + causal_bonus

        unique.sort(key=_score, reverse=True)
        return [
            paper_cls(pmid=h.get("pmid"), doi=h.get("doi"), title=h.get("title", ""),
                      abstract=h.get("abstract", ""), year=h.get("year"),
                      journal=h.get("journal", ""), source=h.get("source", ""))
            for h in unique[:LIT_MAX_PAPERS]
        ]

    def _fetch_abstracts(self, papers: list) -> list:
        import httpx
        need = [p for p in papers if not p.abstract and p.pmid][:10]
        if not need:
            return papers
        pmids = ",".join(p.pmid for p in need if p.pmid)
        try:
            params = {"db": "pubmed", "id": pmids, "rettype": "abstract",
                      "retmode": "xml", "tool": "CausalPlatform", "email": _EMAIL}
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{PUBMED_BASE}/efetch.fcgi", params=params)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                abstract_map: dict[str, str] = {}
                for article in root.findall(".//PubmedArticle"):
                    pmid_el = article.find(".//PMID")
                    abs_els = article.findall(".//AbstractText")
                    if pmid_el is not None and abs_els:
                        abstract_map[pmid_el.text.strip()] = " ".join(
                            (el.text or "") for el in abs_els if el.text
                        ).strip()
                for p in need:
                    if p.pmid and p.pmid in abstract_map:
                        p.abstract = abstract_map[p.pmid]
        except Exception as exc:
            log.debug("Abstract fetch failed: %s", exc)
        return papers

    def _extract_claims(self, papers: list) -> list:
        with_abstract = [p for p in papers if p.abstract][:LIT_TOP_K]
        if not with_abstract:
            return []
        entries = [
            f"Paper {i+1} (PMID:{p.pmid or 'N/A'}, {p.year or '?'}):\n"
            f"Title: {p.title}\nAbstract: {p.abstract[:800]}"
            for i, p in enumerate(with_abstract)
        ]
        try:
            resp = self._llm.complete("\n\n---\n".join(entries), _SYS_CLAIMS, MAX_TOKENS_CLAIMS)
            return resp.get("claims", []) if isinstance(resp, dict) else []
        except Exception as exc:
            log.debug("Claim extraction failed: %s", exc)
            return []

    def _grade_evidence(self, claims: list) -> tuple:
        if not claims:
            return [], [], 0.0
        high_conf = [
            {"from": c["entity_x"], "to": c["entity_y"],
             "relation": c["relation"],
             "mechanism": f"{c['evidence_type']}: {c.get('quote', '')}",
             "strength": c["strength"], "pmid": c.get("pmid")}
            for c in claims
            if c.get("evidence_type") in ("genetic", "perturbation")
            and c.get("strength") == "strong"
            and float(c.get("confidence", 0)) >= 0.75
        ]
        pair_rels: dict = {}
        for c in claims:
            key = (c["entity_x"].lower(), c["entity_y"].lower())
            pair_rels.setdefault(key, []).append(c["relation"])
        rule_conflicts = [
            f"{x} -> {y}: conflicting activation vs inhibition"
            for (x, y), rels in pair_rels.items()
            if ("activates" in rels or "drives" in rels) and ("inhibits" in rels)
        ]
        llm_conflicts: list = []
        if rule_conflicts and len(claims) > 5:
            claim_summary = "; ".join(
                f"{c['entity_x']} {c['relation']} {c['entity_y']} "
                f"({c['evidence_type']}, {c['strength']})"
                for c in claims[:20]
            )
            try:
                resp = self._llm.complete(claim_summary, _SYS_CONFLICTS, 400)
                if isinstance(resp, dict):
                    llm_conflicts = resp.get("additional_conflicts", [])
            except Exception:
                pass
        all_conflicts = list(set(rule_conflicts + llm_conflicts))[:5]
        conflict_rate = len(all_conflicts) / max(len(claims), 1)
        return high_conf[:10], all_conflicts, conflict_rate

    def _build_planning_brief(self, parsed_query, papers, claims,
                              high_conf_edges, conflicts, conflict_rate) -> dict:
        paper_titles = "; ".join(f"{p.title[:60]} ({p.year})" for p in papers[:8])
        claim_summary = "; ".join(
            f"{c['entity_x']} {c['relation']} {c['entity_y']} "
            f"({c['evidence_type']}, {c['strength']})"
            for c in claims[:15]
        )
        context_str = str({
            k: parsed_query.get(k)
            for k in ["domain", "phenotype", "tissue_or_cell_type", "study_type_filter"]
        })
        try:
            return self._llm.complete(
                f"Context: {context_str}\nPapers ({len(papers)}): {paper_titles}\n"
                f"Claims ({len(claims)}): {claim_summary}\n"
                f"Conflicts: {'; '.join(conflicts) or 'none'}",
                _SYS_BRIEF, MAX_TOKENS_BRIEF,
            ) or {}
        except Exception:
            return {}

    @staticmethod
    def _dict_to_claim(c: dict, claim_cls):
        return claim_cls(
            entity_x=c.get("entity_x", ""),
            relation=c.get("relation", "regulates"),
            entity_y=c.get("entity_y", ""),
            direction=c.get("direction", "unknown"),
            evidence_type=c.get("evidence_type", "association"),
            strength=c.get("strength", "weak"),
            pmid=c.get("pmid"),
            quote=c.get("quote", "")[:120],
            confidence=float(c.get("confidence", 0.5)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ResultSynthesiser
# ═════════════════════════════════════════════════════════════════════════════

_SYNTHESISER_SYSTEM = """\
You are the Supervisor Agent synthesising the final result of a molecular causal analysis.
Adapt to the domain and phenotype analyzed. Be precise and statistical.

Return ONLY valid JSON:
{
  "headline": "...",
  "analyzed_context": "...",
  "top_findings": [],
  "tier1_candidates": [],
  "tier2_candidates": [],
  "actionable_targets": [{"entity":"","action":"","druggability":"","existing_drug":null,"rationale":""}],
  "evidence_quality": {"streams_present":[],"streams_missing":[],"overall_confidence":"","note":""},
  "caveats": [],
  "next_experiments": [],
  "missing_data_impact": []
}"""


class ResultSynthesiser:
    def __init__(self, llm):
        self._llm = llm

    def synthesise(self, query: str, intent, steps_run: list,
                   lit_brief, audits: list,
                   artifact_store: dict | None = None):
        from .core_agent import FinalResult
        artifact_store = artifact_store or {}
        modules_done = [s["id"] for s in steps_run if s.get("status") == "done"]
        modules_pending = [s["id"] for s in steps_run if s.get("status") == "pending"]
        artifacts = [o for s in steps_run if not s.get("skip")
                     for o in s.get("outputs", [])]

        lit_summary = lit_brief.supervisor_brief if lit_brief else "No literature search performed."
        key_entities = lit_brief.key_entities[:8] if lit_brief else []

        real_note = (f"Real outputs: {', '.join(artifact_store.keys())}"
                     if artifact_store
                     else f"Pending: {', '.join(modules_pending)}")

        user_msg = (
            f"Query: {query}\nContext: {json.dumps(intent.context)}\n"
            f"Intent: {intent.intent_name}\nModules done: {', '.join(modules_done)}\n"
            f"Files: {', '.join(a.type_label for a in audits)}\n"
            f"Literature: {lit_summary}\nKey entities: {', '.join(key_entities)}\n"
            f"{real_note}"
        )
        try:
            resp = self._llm.complete(user_msg, _SYNTHESISER_SYSTEM, MAX_TOKENS_RESULT)
            if isinstance(resp, dict) and "raw" not in resp:
                return FinalResult(
                    headline=resp.get("headline", "Analysis complete."),
                    analyzed_context=resp.get("analyzed_context", ""),
                    top_findings=resp.get("top_findings", []),
                    tier1_candidates=resp.get("tier1_candidates", []),
                    tier2_candidates=resp.get("tier2_candidates", []),
                    actionable_targets=resp.get("actionable_targets", []),
                    evidence_quality=resp.get("evidence_quality", {}),
                    caveats=resp.get("caveats", []),
                    next_experiments=resp.get("next_experiments", []),
                    missing_data_impact=resp.get("missing_data_impact", []),
                    modules_run=modules_done,
                    artifacts_produced=list(set(artifacts)),
                )
        except Exception as exc:
            log.warning("ResultSynthesiser failed: %s", exc)
        return FinalResult(
            headline="Analysis complete — review logs.",
            modules_run=modules_done,
            artifacts_produced=list(set(artifacts)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — StepNarrator
# ═════════════════════════════════════════════════════════════════════════════

_NARRATE_SYSTEM = """\
You are the Supervisor Agent narrating a live molecular analysis step.
Write exactly 3 sentences: algorithm, question answered, downstream impact.
Return plain text only."""


class StepNarrator:
    def __init__(self, llm):
        self._llm   = llm
        self._local = _LocalNarrateEngine()

    def narrate(self, step: dict, intent) -> str:
        ctx = intent.context or {}
        user_msg = (
            f"Step: {step['label']} ({step['id']})\n"
            f"Algorithm: {step.get('tool', '')}\n"
            f"Outputs: {', '.join(step.get('outputs', []))}\n"
            f"Domain: {ctx.get('domain', 'molecular biology')}\n"
            f"Phenotype: {ctx.get('phenotype', 'the phenotype of interest')}\n"
            f"Intent: {intent.intent_name}"
        )
        try:
            result = self._llm.complete(user_msg, _NARRATE_SYSTEM, MAX_TOKENS_NARRATE, as_json=False)
            if result and isinstance(result, str) and len(result) > 20:
                return result
        except Exception:
            pass
        return self._local.narrate(step)
