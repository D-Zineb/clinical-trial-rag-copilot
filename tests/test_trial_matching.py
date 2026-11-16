from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.retrieval.trial_matching import (
    MatchRequest, PatientProfile, assess_criterion, extract_criteria,
    match_candidates, match_trial,
)


def trial(inclusion="Age >= 18", exclusion="ECOG > 1"):
    return {"nct_id": "NCT1", "brief_title": "Synthetic trial",
            "inclusion_criteria": inclusion, "exclusion_criteria": exclusion}


def test_provenance_and_compound_logic_are_preserved():
    source = trial("  EGFR exon 19 deletion OR L858R\nAND prior therapy required.  ")
    criteria = extract_criteria(source)
    c = criteria[0]
    assert source[c.source_field][c.start:c.end] == c.text
    assert "OR" in c.text and "AND" in c.text
    assert len(c.source_hash) == 64
    assert extract_criteria(source)[0].criterion_id == c.criterion_id
    changed = {**source, "inclusion_criteria": "Different criteria"}
    assert extract_criteria(changed)[0].criterion_id != c.criterion_id
    assert assess_criterion(c, PatientProfile(summary="EGFR positive")).assessment == "unknown"


@pytest.mark.parametrize("age,ecog,expected", [
    (30, 1, ["satisfied", "satisfied"]),
    (17, 2, ["violated", "violated"]),
    (None, None, ["unknown", "unknown"]),
])
def test_numeric_rules_and_exclusion_polarity(age, ecog, expected):
    result = match_trial(trial(), PatientProfile(summary="Synthetic profile", age_years=age, ecog=ecog))
    assert [c.assessment for c in result.criteria] == expected
    assert result.assessment_complete is False
    assert result.possible_exclusion == ("violated" in expected)


def test_partial_match_never_drops_exception_or_temporal_logic():
    for text in ["Age >= 18 unless prior transplant", "ECOG <= 1 within 7 days", "No prior osimertinib", "Age >= 18 OR parental consent"]:
        result = match_trial(trial(text, ""), PatientProfile(summary="Patient", age_years=40, ecog=0))
        assert result.criteria[0].assessment == "unknown"


def test_no_criteria_and_missing_sources_are_visible():
    retriever = Mock()
    retriever.retrieve.return_value.fused_ncts = ["NCT1", "NCT1", "NCT_missing"]
    retriever.retrieve.return_value.retrieval_warnings = []
    response = match_candidates(MatchRequest(patient=PatientProfile(summary="Patient")), retriever, [trial("", "")])
    assert len(response.matches) == 1
    assert response.matches[0].missing_information
    assert any("unavailable" in w for w in response.warnings)
    assert response.matches[0].status == "candidate_for_review"


def test_unknown_polarity_is_not_assessed():
    source = {"nct_id": "NCT1", "eligibility_criteria": "Age >= 18"}
    result = match_trial(source, PatientProfile(summary="Patient", age_years=30))
    assert result.criteria[0].assessment == "unknown"
