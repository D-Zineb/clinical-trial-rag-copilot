import pytest
from clinical_trial_rag_copilot.agent.verifier import GraphFactualityGuardrail


@pytest.fixture
def guardrail():
    return GraphFactualityGuardrail()


def test_guardrail_extracts_entities(guardrail):
    """Verify entity parser correctly extracts NCTs, PMIDs, drugs, and genes from text."""
    sample_text = (
        "In trial NCT05543278, patients were administered Osimertinib to reduce postoperative arrhythmias. "
        "Osimertinib is known to inhibit EGFR potassium channels, as noted in PMID 40159403."
    )

    drugs, genes, ncts, pmids = guardrail.extract_claimed_entities(sample_text)

    assert "NCT05543278" in ncts
    assert "40159403" in pmids
    assert "Osimertinib" in drugs
    assert "EGFR" in genes


def test_guardrail_resilient_handling(guardrail):
    """Verify that guardrail handles text gracefully without throwing exceptions."""
    text_with_hallucinations = (
        "Trial NCT99999999 tested SuperDrug on gene XYZ123 without adverse effects."
    )

    report = guardrail.verify(text_with_hallucinations)
    assert report is not None
    assert report.status in ["WARNING", "OFFLINE_RESILIENT"]
    assert len(report.guardrail_banner) > 0


class FakeSession:
    def __init__(self, fail=False):
        self.active = False
        self.fail = fail
        self.calls = []

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False

    def run(self, query, parameters):
        assert self.active, 'Query outside session'
        if self.fail:
            raise RuntimeError('connection unavailable')
        self.calls.append(parameters)
        session = self

        class Result:
            def single(self):
                assert session.active, 'Result consumed outside session'
                if 'drug' in parameters:
                    return {'action': 'INHIBITS'}
                return {'c': 0 if parameters.get('nct') == 'NCT99999999' else 1}

        return Result()


def connected_guardrail(session):
    from types import SimpleNamespace
    guard = GraphFactualityGuardrail.__new__(GraphFactualityGuardrail)
    guard.drug_map = {'afatinib': 'Afatinib'}
    guard.gene_map = {'egfr': 'EGFR'}
    guard.db = SimpleNamespace(is_connected=True, driver=SimpleNamespace(session=lambda: session))
    return guard


def test_online_guardrail_checks_every_citation_within_session():
    session = FakeSession()
    report = connected_guardrail(session).verify('NCT05215548 NCT99999999 PMID:40159403 Afatinib EGFR')
    assert report.verified_trials == ['NCT05215548']
    assert report.unverified_trials == ['NCT99999999']
    assert report.verified_papers == ['40159403']
    assert report.verified_interactions == [{'drug': 'Afatinib', 'gene': 'EGFR', 'action': 'INHIBITS'}]
    assert report.status == 'WARNING'
    assert len(session.calls) == 4


def test_online_guardrail_handles_no_citations():
    assert connected_guardrail(FakeSession()).verify('No evidence available.').verified_trials == []


def test_query_failure_cannot_claim_verified_trials():
    report = connected_guardrail(FakeSession(fail=True)).verify('NCT05215548')
    assert report.status == 'WARNING'
    assert not report.is_valid
    assert report.verified_trials == []
    assert 'could not be completed' in report.guardrail_banner
