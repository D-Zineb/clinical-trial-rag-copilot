import pytest
from clinical_trial_rag_copilot.data.fetch_trials import parse_age_to_years, split_eligibility_criteria
from clinical_trial_rag_copilot.data.run_acquisition import clean_drug_name
from clinical_trial_rag_copilot.graph.ingest import clean_drug_name as clean_graph_drug_name


@pytest.mark.parametrize("clean_drug_name", [clean_drug_name])
def test_clean_drug_name(clean_drug_name):
    """Verify pharmaceutical drug name cleaning strips formulations, dosages, and noise."""
    assert clean_drug_name("Osimertinib Hydrochloride 200 mg") == "Osimertinib"
    assert clean_drug_name("Erlotinib Succinate Oral Tablet 50mg") == "Erlotinib"
    assert clean_drug_name("Afatinib 25 mg oral tablet") == "Afatinib"
    assert clean_drug_name("Gefitinib (Verquvo)") == "Gefitinib"
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
        "2. Documented paroxysmal non-small cell lung cancer\n\n"
        "Exclusion Criteria:\n"
        "1. Severe renal impairment (eGFR < 30)\n"
        "2. Prior cardiac ablation within 30 days\n"
    )

    incl, excl = split_eligibility_criteria(sample_criteria)

    assert incl is not None
    assert "Age >= 18 years" in incl
    assert "paroxysmal non-small cell lung cancer" in incl

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


def test_graph_drug_name_missing():
    assert clean_graph_drug_name(None) == ""
    assert clean_graph_drug_name("") == ""


@pytest.mark.parametrize("clean_drug_name", [clean_drug_name])
def test_arm_reference_prefix_joins_the_same_drug_as_target_records(clean_drug_name):
    assert clean_drug_name('Drug: Osimertinib') == clean_drug_name('Osimertinib')
    assert clean_drug_name('drug: Erlotinib 150 mg') == 'Erlotinib'
    assert clean_drug_name('Biological: Pembrolizumab') == 'Pembrolizumab'


def test_graph_identity_preserves_ambiguous_source_names():
    assert clean_graph_drug_name('Drug: Erlotinib 150 mg') == 'Erlotinib 150 Mg'
    assert clean_graph_drug_name('Afatinib, Osimertinib, Carboplatin and Pemetrexed') == 'Afatinib, Osimertinib, Carboplatin And Pemetrexed'
    assert clean_graph_drug_name('Drug: Osimertinib') == clean_graph_drug_name('Osimertinib')
