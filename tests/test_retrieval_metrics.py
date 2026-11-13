import math

from clinical_trial_rag_copilot.eval.retrieval_benchmark import known_item_metrics


def test_known_item_metrics_deduplicate_documents():
    result = known_item_metrics(["A", "A", "B"], "B", 2)
    assert result["hit"] == 1
    assert result["target_rank"] == 2
    assert result["reciprocal_rank"] == 0.5
    assert result["ndcg"] == 1 / math.log2(3)


def test_known_item_metrics_respect_cutoff_and_missing_target():
    for ranking in (["A", "B"], []):
        assert known_item_metrics(ranking, "B", 1) == {
            "hit": 0, "reciprocal_rank": 0, "ndcg": 0, "target_rank": None,
        }
