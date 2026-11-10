from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridEvidence, HybridRetriever
from clinical_trial_rag_copilot.agent.llm_client import BaseLLMClient, get_llm_client
from clinical_trial_rag_copilot.agent.prompt_templates import SYSTEM_PROMPT, create_user_prompt
from clinical_trial_rag_copilot.agent.verifier import GraphFactualityGuardrail, VerificationReport
from clinical_trial_rag_copilot.observability.tracer import PipelineTracer, traceable
from clinical_trial_rag_copilot.utils.logger import get_logger

logger = get_logger("agent.copilot")


class StepLatency(BaseModel):
    """Execution latency for an individual pipeline stage."""
    step_name: str
    duration_ms: float


class CopilotAnswer(BaseModel):
    """Structured response from the Clinical Trials & Pharmacology GraphRAG Copilot."""
    query: str
    answer: str
    guardrail_status: str
    guardrail_banner: str
    trace_id: str = ""
    total_latency_ms: float = 0.0
    latencies: List[StepLatency] = Field(default_factory=list)
    verified_trials: List[str] = Field(default_factory=list)
    verified_interactions: List[Dict[str, str]] = Field(default_factory=list)
    retrieved_ncts: List[str] = Field(default_factory=list)
    retrieved_pmids: List[str] = Field(default_factory=list)


class ClinicalGraphRAGCopilot:
    """End-to-End Production Clinical Trial & Pharmacology GraphRAG Copilot with full observability."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.retriever = HybridRetriever()
        self.llm = llm_client or get_llm_client()
        self.guardrail = GraphFactualityGuardrail()

    @traceable(name="ClinicalGraphRAGCopilot.ask", run_type="chain")
    def ask(self, query: str, top_k: int = 6) -> CopilotAnswer:
        """Process a clinical question through the complete GraphRAG pipeline with telemetry tracing."""
        tracer = PipelineTracer()

        # Step 1: Hybrid Retrieval (Graph + Vector + RRF)
        with tracer.start_span("hybrid_retrieval", {"query": query, "top_k": top_k}) as span:
            evidence = self.retriever.retrieve(query, top_k=top_k)
            span.set_outputs({
                "extracted_entities": evidence.extracted_entities.model_dump(),
                "graph_items_count": len(evidence.graph_evidence),
                "vector_items_count": len(evidence.vector_evidence),
                "fused_ncts": evidence.fused_ncts,
                "fused_pmids": evidence.fused_pmids,
            })

        # Step 2: Prompt Formulation
        with tracer.start_span("prompt_assembly", {"fused_ncts_count": len(evidence.fused_ncts)}) as span:
            user_prompt = create_user_prompt(query, evidence)
            span.set_outputs({"prompt_length_chars": len(user_prompt)})

        # Step 3: LLM Generation
        with tracer.start_span("llm_synthesis", {"provider": type(self.llm).__name__}) as span:
            raw_answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            span.set_outputs({"raw_answer_length": len(raw_answer)})

        # Step 4: Self-RAG Factuality Guardrail
        with tracer.start_span("self_rag_guardrail", {"answer_snippet": raw_answer[:200]}) as span:
            report = self.guardrail.verify(raw_answer)
            span.set_outputs({
                "status": report.status,
                "verified_trials": report.verified_trials,
                "unverified_trials": report.unverified_trials,
                "verified_interactions": report.verified_interactions,
            })

        # Finalize telemetry trace
        trace = tracer.finalize(query=query, guardrail_status=report.status)

        # Enhance NCT & PMID links in markdown for interactive navigation
        enhanced_answer = raw_answer
        for nct in report.verified_trials:
            enhanced_answer = enhanced_answer.replace(
                f"[{nct}]", f"[[{nct}]](https://clinicaltrials.gov/study/{nct})"
            )
        for pmid in report.verified_papers:
            enhanced_answer = enhanced_answer.replace(
                f"[PMID{pmid}]", f"[[PMID:{pmid}]](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
            )
            enhanced_answer = enhanced_answer.replace(
                f"[PMID:{pmid}]", f"[[PMID:{pmid}]](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
            )

        # Append guardrail banner
        final_answer = f"{enhanced_answer}\n\n---\n{report.guardrail_banner}"

        latencies = [
            StepLatency(step_name=s.step_name, duration_ms=s.duration_ms)
            for s in trace.spans
        ]

        return CopilotAnswer(
            query=query,
            answer=final_answer,
            guardrail_status=report.status,
            guardrail_banner=report.guardrail_banner,
            trace_id=trace.trace_id,
            total_latency_ms=trace.total_latency_ms,
            latencies=latencies,
            verified_trials=report.verified_trials,
            verified_interactions=report.verified_interactions,
            retrieved_ncts=evidence.fused_ncts,
            retrieved_pmids=evidence.fused_pmids,
        )


if __name__ == "__main__":
    copilot = ClinicalGraphRAGCopilot()

    test_q = (
        "Which clinical trials evaluate Amiodarone targeting KCNH2 in patients with Atrial Fibrillation, "
        "what are the study arms, and what adverse events are reported?"
    )

    logger.info("Running observable production GraphRAG copilot test...")
    result = copilot.ask(test_q)

    logger.info("Trace ID: %s | Total latency: %.1fms", result.trace_id, result.total_latency_ms)
    for lat in result.latencies:
        logger.info("  - %s: %.1fms", lat.step_name, lat.duration_ms)
    logger.info("Guardrail Status: %s", result.guardrail_status)
    logger.info("Verified Trials: %s", result.verified_trials)
    logger.info("Verified Interactions: %s", result.verified_interactions)
