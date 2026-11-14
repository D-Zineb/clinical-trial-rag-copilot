import pytest
from clinical_trial_rag_copilot.eval.prepare_cases import prepare_cases


def data():
    return ([{"nct_id": "NCT1", "brief_title": "Trial one"},
             {"nct_id": "NCT2", "brief_title": "Trial two"}],
            [{"pmid": "1", "title": "Paper one", "abstract": "a" * 60},
             {"pmid": "2", "title": "Paper two", "abstract": "b" * 60},
             {"pmid": "3", "title": "Unindexable", "abstract": ""}])


def test_deterministic_across_source_order_and_only_indexable_targets():
    trials, papers = data()
    cases, digest = prepare_cases(trials, papers, 10)
    assert (cases, digest) == prepare_cases(trials[::-1], papers[::-1], 10)
    assert {c["target_id"] for c in cases} == {"NCT1", "NCT2", "1", "2"}
    assert all(c["query"] == c["source_title"] for c in cases)
    assert all("smoke" in c["label_type"] for c in cases)


def test_limit_and_corpus_change():
    trials, papers = data()
    cases, digest = prepare_cases(trials, papers, 1)
    assert len(cases) == 2
    trials[0]["brief_title"] = "Changed trial"
    assert prepare_cases(trials, papers, 1)[1] != digest


def test_empty_kind_fails_explicitly():
    with pytest.raises(ValueError, match="No indexable trial"):
        prepare_cases([], data()[1])
