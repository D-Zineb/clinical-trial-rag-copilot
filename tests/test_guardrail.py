import pytest
from clinical_trial_rag_copilot.agent.verifier import GraphFactualityGuardrail


@pytest.fixture
def guardrail():
    return GraphFactualityGuardrail()


def test_guardrail_extracts_entities(guardrail):
    """Verify entity parser correctly extracts NCTs, PMIDs, drugs, and genes from text."""
    sample_text = (
        "In trial NCT05543278, patients were administered Amiodarone to reduce postoperative arrhythmias. "
        "Amiodarone is known to inhibit KCNH2 potassium channels, as noted in PMID 40159403."
    )

    drugs, genes, ncts, pmids = guardrail.extract_claimed_entities(sample_text)

    assert "NCT05543278" in ncts
    assert "40159403" in pmids
    assert "Amiodarone" in drugs
    assert "KCNH2" in genes


def test_guardrail_resilient_handling(guardrail):
    """Verify that guardrail handles text gracefully without throwing exceptions."""
    text_with_hallucinations = (
        "Trial NCT99999999 tested SuperDrug on gene XYZ123 without adverse effects."
    )

    report = guardrail.verify(text_with_hallucinations)
    assert report is not None
    assert report.status in ["WARNING", "OFFLINE_RESILIENT"]
    assert len(report.guardrail_banner) > 0
