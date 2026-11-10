from typing import Any, Dict, List
from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridEvidence

SYSTEM_PROMPT = """You are an expert Clinical Trials & Pharmacology AI Copilot specializing in cardiology and cardiac electrophysiology.
Your task is to synthesize verified, multi-hop medical answers based EXCLUSIVELY on the provided dual-indexed evidence (Neo4j Knowledge Graph facts and PubMed / ClinicalTrials text chunks).

STRICT GUIDELINES:
1. CITATIONS ARE MANDATORY:
   - For every trial fact, cite the NCT identifier, e.g. [NCT05543278].
   - For every paper or adverse event claim, cite the PubMed ID, e.g. [PMID42492110].
2. STUDY ARM DISAMBIGUATION:
   - Explicitly distinguish between EXPERIMENTAL arms and COMPARATOR / PLACEBO arms. Never claim a placebo was the tested active drug.
3. DEMOGRAPHIC ELIGIBILITY:
   - When discussing age criteria, state the exact minimum and maximum age bounds (e.g., "Patients aged >= 18").
4. MECHANISM OF ACTION & SAFETY:
   - When discussing ion channels or genes (e.g. KCNH2/hERG, SCN5A), explain the pharmacology and link any reported safety events (e.g., QT prolongation, Torsades de Pointes).
5. NO HALLUCINATIONS:
   - If evidence for a specific question is absent, explicitly state that it is not documented in the current knowledge graph or retrieved abstracts.
"""


def format_evidence_context(evidence: HybridEvidence) -> str:
    """Format structured Knowledge Graph paths and unstructured vector chunks into prompt context."""
    sections: List[str] = []

    # 1. Structured Knowledge Graph Facts
    sections.append("### 1. STRUCTURED KNOWLEDGE GRAPH FACTS (NEO4J VERIFIED)")
    if evidence.graph_evidence:
        for idx, g in enumerate(evidence.graph_evidence[:10], 1):
            src = g.get("source", "")
            if src == "GRAPH_GENE_PATHWAY":
                sections.append(
                    f"{idx}. [Trial: {g.get('nct_id')}] '{g.get('title')}'\n"
                    f"   - Study Arm: '{g.get('arm')}' (Type: {g.get('arm_type')})\n"
                    f"   - Administers Drug: '{g.get('drug')}' -> Targets Gene: '{g.get('gene')}'\n"
                    f"   - Demographics: Min Age {g.get('min_age')} yrs, Max Age {g.get('max_age')}, Sex: {g.get('sex')}\n"
                    f"   - Conditions: {', '.join(g.get('conditions', []))}"
                )
            elif src == "GRAPH_DRUG_PATHWAY":
                sections.append(
                    f"{idx}. [Trial: {g.get('nct_id')}] '{g.get('title')}'\n"
                    f"   - Study Arm: '{g.get('arm')}' (Type: {g.get('arm_type')})\n"
                    f"   - Administers Drug: '{g.get('drug')}'\n"
                    f"   - Target Genes: {', '.join(g.get('target_genes', []))}"
                )
            elif src == "GRAPH_SAFETY_LITERATURE":
                sections.append(
                    f"{idx}. [Paper: PMID {g.get('pmid')}] '{g.get('title')}' ({g.get('year')})\n"
                    f"   - Reported Adverse Events: {', '.join(g.get('adverse_events', []))}\n"
                    f"   - Cited Drugs: {', '.join(g.get('drugs', []))}\n"
                    f"   - Studied Genes: {', '.join(g.get('genes', []))}"
                )
    else:
        sections.append("No direct multi-hop graph paths matched the query entities.")

    # 2. Complete Clinical Trial Dossiers (Sibling Expanded Context)
    if getattr(evidence, "sibling_chunks", None):
        sections.append("\n### 2. COMPLETE CLINICAL TRIAL DOSSIERS (REHYDRATED SIBLING CHUNKS)")
        for nct, s_chunks in evidence.sibling_chunks.items():
            sections.append(f"\n==================== TRIAL DOSSIER: [{nct}] ====================")
            for c in s_chunks:
                ctype = c.get("chunk_type", "CHUNK")
                txt = c.get("text", "").strip()
                sections.append(f"[{ctype}]\n{txt}\n")
    else:
        # Fallback to individual vector hits
        sections.append("\n### 2. CLINICAL TRIAL TEXT EVIDENCE (VECTOR CHUNKS)")
        trial_hits = [v for v in evidence.vector_evidence if v.chunk_type != "LITERATURE_ABSTRACT"]
        if trial_hits:
            for idx, v in enumerate(trial_hits[:4], 1):
                sections.append(
                    f"[{idx}] Document: {v.doc_id} ({v.chunk_type}) - Similarity Score: {v.score:.4f}\n"
                    f"{v.text.strip()}\n"
                )
        else:
            sections.append("No specific trial text chunks retrieved.")

    # 3. Unstructured Scientific Literature Evidence
    sections.append("\n### 3. RESEARCH LITERATURE EVIDENCE (PUBMED ABSTRACTS)")
    lit_hits = [v for v in evidence.vector_evidence if v.chunk_type == "LITERATURE_ABSTRACT"]
    if lit_hits:
        for idx, v in enumerate(lit_hits[:3], 1):
            sections.append(
                f"[{idx}] PubMed PMID {v.doc_id} - Similarity Score: {v.score:.4f}\n"
                f"{v.text.strip()}\n"
            )
    else:
        sections.append("No specific literature abstract chunks retrieved.")

    return "\n".join(sections)


def create_user_prompt(query: str, evidence: HybridEvidence) -> str:
    """Build the complete user prompt with question and structured context."""
    context_str = format_evidence_context(evidence)
    return (
        f"USER CLINICAL QUESTION:\n{query}\n\n"
        f"RETRIEVED MULTI-HOP EVIDENCE:\n{context_str}\n\n"
        "SYNTHESIS INSTRUCTION:\n"
        "Provide a comprehensive, medically accurate analysis answering the user's question. "
        "Structure your response with clear headings: \n"
        "1. Executive Summary\n"
        "2. Clinical Trials & Comparative Arms Analysis (with NCT citations)\n"
        "3. Pharmacology & Target Gene Mechanism (e.g., KCNH2/hERG)\n"
        "4. Safety Profile & Adverse Events (with PMID citations)\n"
        "5. Patient Eligibility & Demographic Considerations"
    )
