"""Source-join and ranking-metric checks for graph-value experiments."""
import pytest

from clinical_trial_rag_copilot.eval.graph_value import (
    bootstrap_interval, build_cases, flat_matches, ranking_metrics,
    raw_reference, source_tables,
)


def sources():
    trials = [
        {"nct_id": "NCT1", "conditions": ["NSCLC"], "arm_groups": [
            {"intervention_names": ["Drug: A", "Drug: B"]}]},
        {"nct_id": "NCT2", "conditions": ["NSCLC"], "arm_groups": [
            {"intervention_names": ["Drug: A"]}]},
        {"nct_id": "NCT3", "conditions": ["Other"], "arm_groups": [
            {"intervention_names": ["Drug: B"]}]},
    ]
    targets = [{"drug_name": "A", "targets": [{"gene_symbol": "EGFR"}]},
               {"drug_name": "B", "targets": [{"gene_symbol": "ALK"}]}]
    return trials, targets


def test_gene_must_belong_to_requested_drug_not_other_arm_drug():
    trials, drugs = sources()
    rows, targets = source_tables(trials, drugs)
    query = {"drugs": ["A"], "genes": ["ALK"]}
    assert raw_reference(trials, targets, query) == []
    assert not any(flat_matches(row, query) for row in rows.values())
    query = {"conditions": ["NSCLC"], "genes": ["EGFR"]}
    assert raw_reference(trials, targets, query) == ["NCT1", "NCT2"]
    assert sorted(nct for nct, row in rows.items() if flat_matches(row, query)) == ["NCT1", "NCT2"]


def test_source_case_selection_is_stable_before_retrieval():
    trials, drugs = sources()
    first = build_cases(trials, drugs, per_category=2)
    assert first and first == build_cases(list(reversed(trials)), drugs, per_category=2)
    for case in first:
        assert len(case["expected_ids"]) >= 2
        assert case["label_type"].endswith("not_clinical_eligibility")


def test_full_ideal_ranking_penalizes_missing_relevant_documents():
    metrics = ranking_metrics(["NCT1", "NCT1"], ["NCT1", "NCT2"])
    assert metrics["precision10"] == .1
    assert metrics["recall100"] == .5
    assert metrics["ndcg10"] == pytest.approx(1 / (1 + 1 / 1.584962500721156))
    assert ranking_metrics([], [])['ndcg10'] is None
    assert bootstrap_interval([0, 0, 0]) == [0, 0]
