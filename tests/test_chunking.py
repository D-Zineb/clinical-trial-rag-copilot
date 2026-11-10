import pytest
from clinical_trial_rag_copilot.retrieval.chunking import (
    create_trial_chunks,
    create_literature_chunk,
    build_all_chunks,
)


@pytest.fixture
def sample_trial():
    return {
        "nct_id": "NCT05543278",
        "brief_title": "Pragmatic Amiodarone Trial to Reduce Postoperative Atrial Fibrillation",
        "phase": "PHASE3",
        "overall_status": "RECRUITING",
        "sex": "ALL",
        "min_age_years": 18.0,
        "max_age_years": -1.0,
        "minimum_age": "18 Years",
        "maximum_age": None,
        "conditions": ["Atrial Fibrillation", "Cardiac Surgery"],
        "intervention_names": ["Amiodarone"],
        "brief_summary": "Evaluating prophylactic oral amiodarone in patients undergoing surgery.",
        "inclusion_criteria": "Patients undergoing planned cardiac surgery aged 18 or older.",
        "exclusion_criteria": "Preoperative baseline bradycardia under 50 bpm or severe pulmonary disease.",
        "primary_outcome_measures": ["Incidence of postoperative atrial fibrillation"],
    }


@pytest.fixture
def sample_paper():
    return {
        "pmid": "40159403",
        "title": "Practical compendium of antiarrhythmic drugs in clinical cardiology.",
        "year": 2025,
        "abstract": "The European Heart Rhythm Association presents guidelines for antiarrhythmic therapy, focusing on KCNH2 potassium channel inhibition and QT prolongation risks.",
        "mesh_terms": ["Arrhythmias, Cardiac", "Anti-Arrhythmia Agents", "Long QT Syndrome"],
    }


def test_create_trial_chunks(sample_trial):
    """Verify that a trial produces 3 typed chunks: OVERVIEW, INCLUSION, and EXCLUSION."""
    chunks = create_trial_chunks(sample_trial)
    assert len(chunks) == 3

    types = {c.chunk_type for c in chunks}
    assert types == {"TRIAL_OVERVIEW", "TRIAL_INCLUSION", "TRIAL_EXCLUSION"}

    overview = next(c for c in chunks if c.chunk_type == "TRIAL_OVERVIEW")
    assert "Amiodarone" in overview.text
    assert overview.metadata["nct_id"] == "NCT05543278"
    assert overview.metadata["min_age_years"] == 18.0

    inclusion = next(c for c in chunks if c.chunk_type == "TRIAL_INCLUSION")
    assert "INCLUSION CRITERIA" in inclusion.text

    exclusion = next(c for c in chunks if c.chunk_type == "TRIAL_EXCLUSION")
    assert "EXCLUSION CRITERIA" in exclusion.text


def test_create_literature_chunk(sample_paper):
    """Verify conversion of PubMed article into LITERATURE_ABSTRACT chunk."""
    chunk = create_literature_chunk(sample_paper)
    assert chunk is not None
    assert chunk.chunk_type == "LITERATURE_ABSTRACT"
    assert chunk.doc_id == "40159403"
    assert chunk.metadata["year"] == 2025
    assert "KCNH2" in chunk.text


def test_build_all_chunks(sample_trial, sample_paper):
    """Verify batch assembly of mixed clinical and literature collections."""
    all_chunks = build_all_chunks([sample_trial], [sample_paper])
    assert len(all_chunks) == 4
