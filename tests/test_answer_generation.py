"""Offline orchestration checks; these do not measure LLM factual accuracy."""
from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.agent.copilot import ClinicalGraphRAGCopilot
from clinical_trial_rag_copilot.agent.prompt_templates import (
    EMPTY_EVIDENCE_ANSWER, SYSTEM_PROMPT, format_evidence_context,
)
from clinical_trial_rag_copilot.agent.verifier import VerificationReport
from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridEvidence, QueryEntities
from clinical_trial_rag_copilot.retrieval.vector_store import SearchResult


def evidence(**updates):
    fields = dict(query="What is the minimum age?", extracted_entities=QueryEntities(),
                  graph_evidence=[], vector_evidence=[], fused_ncts=[], fused_pmids=[])
    fields.update(updates)
    return HybridEvidence(**fields)


def hit(text="Minimum age: 18 years."):
    return SearchResult(chunk_id="NCT00000001_ELIGIBILITY", doc_id="NCT00000001",
                        chunk_type="ELIGIBILITY", text=text, score=0.8)


@pytest.fixture
def copilot(tmp_path, monkeypatch):
    import clinical_trial_rag_copilot.observability.tracer as telemetry
    monkeypatch.setattr(telemetry, "TRACES_DIR", tmp_path)
    # Avoid initializing external services: exercise ask() with controlled evidence.
    instance = ClinicalGraphRAGCopilot.__new__(ClinicalGraphRAGCopilot)
    instance.retriever = Mock()
    instance.llm = Mock()
    instance.llm.generate.return_value = "Minimum age: 18 years. [NCT00000001]"
    instance.guardrail = Mock()
    instance.guardrail.verify.return_value = VerificationReport(
        is_valid=True, status="PASSED", verified_trials=["NCT00000001"],
        guardrail_banner="Test verification banner",
    )
    return instance


@pytest.mark.parametrize("retrieved", [
    evidence(),
    evidence(fused_ncts=["NCT00000001"], retrieval_warnings=["Search unavailable"]),
    evidence(vector_evidence=[hit("  ")], sibling_chunks={"NCT00000001": [{"text": ""}]}),
    evidence(graph_evidence=[{"source": "UNSUPPORTED"}]),
])
def test_no_source_content_skips_generation_and_verification(copilot, retrieved):
    copilot.retriever.retrieve.return_value = retrieved
    result = copilot.ask(retrieved.query, capture_evaluation=True)
    copilot.llm.generate.assert_not_called()
    copilot.guardrail.verify.assert_not_called()
    assert result.answer.startswith(EMPTY_EVIDENCE_ANSWER)
    assert result.guardrail_status == "INSUFFICIENT_EVIDENCE"
    assert result.verified_trials == []
    assert result.trace_id
    assert result.evaluation_sample["response"] == EMPTY_EVIDENCE_ANSWER
    assert result.evaluation_sample["retrieved_contexts"] == [format_evidence_context(retrieved)]


@pytest.mark.parametrize("retrieved", [
    evidence(vector_evidence=[hit()]),
    evidence(sibling_chunks={"NCT00000001": [{"text": "Minimum age: 18 years."}]}),
    evidence(graph_evidence=[{"source": "GRAPH_TRIAL_CONSTRAINTS", "nct_id": "NCT00000001"}]),
])
def test_source_content_reaches_llm_with_shared_rules(copilot, retrieved):
    copilot.retriever.retrieve.return_value = retrieved
    result = copilot.ask(retrieved.query, capture_evaluation=True)
    system, user = copilot.llm.generate.call_args.args
    assert system == SYSTEM_PROMPT
    assert retrieved.query in user
    assert format_evidence_context(retrieved) in user
    assert result.evaluation_sample["retrieved_contexts"][0] in user
    copilot.guardrail.verify.assert_called_once_with(copilot.llm.generate.return_value)
    assert "https://clinicaltrials.gov/study/NCT00000001" in result.answer


def test_empty_sibling_sections_do_not_hide_vector_evidence(copilot):
    retrieved = evidence(sibling_chunks={"NCT00000001": [{"text": "  "}]},
                         vector_evidence=[hit()])
    copilot.retriever.retrieve.return_value = retrieved
    copilot.ask(retrieved.query)
    assert "Minimum age: 18 years." in copilot.llm.generate.call_args.args[1]
