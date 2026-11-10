import pytest
from clinical_trial_rag_copilot.data.fetch_trials import parse_age_to_years, split_eligibility_criteria
from clinical_trial_rag_copilot.data.run_acquisition import clean_drug_name


def test_clean_drug_name():
    """Verify pharmaceutical drug name cleaning strips formulations, dosages, and noise."""
    assert clean_drug_name("Amiodarone Hydrochloride 200 mg") == "Amiodarone"
    assert clean_drug_name("Metoprolol Succinate Oral Tablet 50mg") == "Metoprolol"
    assert clean_drug_name("Spironolactone 25 mg oral tablet") == "Spironolactone"
    assert clean_drug_name("Vericiguat (Verquvo)") == "Vericiguat"
    assert clean_drug_name("Placebo oral tablet") == "Placebo"


def test_parse_age_to_years():
    """Verify conversion of heterogeneous ClinicalTrials.gov age strings to float years."""
    assert parse_age_to_years("18 Years") == 18.0
    assert parse_age_to_years("65 Years") == 65.0
    assert parse_age_to_years("6 Months") == 0.5
    assert parse_age_to_years("12 Weeks") == round(12 / 52.0, 2)
    assert parse_age_to_years(None) is None
    assert parse_age_to_years("") is None
    assert parse_age_to_years("Not Applicable") is None


def test_split_eligibility_criteria():
    """Verify deterministic splitting of unstructured clinical eligibility criteria."""
    sample_criteria = (
        "Inclusion Criteria:\n"
        "1. Age >= 18 years\n"
        "2. Documented paroxysmal atrial fibrillation\n\n"
        "Exclusion Criteria:\n"
        "1. Severe renal impairment (eGFR < 30)\n"
        "2. Prior cardiac ablation within 30 days\n"
    )

    incl, excl = split_eligibility_criteria(sample_criteria)

    assert incl is not None
    assert "Age >= 18 years" in incl
    assert "paroxysmal atrial fibrillation" in incl

    assert excl is not None
    assert "Severe renal impairment" in excl
    assert "Prior cardiac ablation" in excl


def test_split_criteria_missing_exclusion():
    """Handle eligibility texts that only list inclusion criteria."""
    text_only_incl = "Inclusion Criteria: Patients with symptomatic heart failure."
    incl, excl = split_eligibility_criteria(text_only_incl)
    assert incl is not None
    assert "symptomatic heart failure" in incl
    assert excl is None
