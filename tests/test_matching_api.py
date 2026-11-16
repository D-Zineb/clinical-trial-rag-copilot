from unittest.mock import Mock

from fastapi.testclient import TestClient
from clinical_trial_rag_copilot.api import main


def test_matching_route_returns_source_evidence(monkeypatch):
    retriever = Mock()
    retriever.retrieve.return_value.fused_ncts = ["NCT1"]
    retriever.retrieve.return_value.retrieval_warnings = []
    monkeypatch.setattr("clinical_trial_rag_copilot.retrieval.hybrid_search.HybridRetriever", lambda **kwargs: retriever)
    monkeypatch.setattr(main, "get_vector_store_instance", lambda: Mock())
    monkeypatch.setattr(main, "load_dataset", lambda name: [{
        "nct_id": "NCT1", "brief_title": "Synthetic trial",
        "inclusion_criteria": "Age >= 18", "exclusion_criteria": "ECOG > 1",
    }])
    response = TestClient(main.app).post("/api/v1/match", json={
        "patient": {"summary": "Synthetic NSCLC patient", "age_years": 40, "ecog": 2},
        "top_k": 1,
    })
    assert response.status_code == 200
    result = response.json()["matches"][0]
    assert result["possible_exclusion"] is True
    assert result["assessment_complete"] is False
    assert result["criteria"][0]["source_hash"]
    assert result["criteria"][0]["source_url"].endswith("/NCT1")


def test_invalid_matching_requests_fail_before_loading_data(monkeypatch):
    load = Mock(side_effect=AssertionError("must not load data"))
    monkeypatch.setattr(main, "load_dataset", load)
    client = TestClient(main.app)
    assert client.post("/api/v1/match", json={"patient": {"summary": " "}}).status_code == 400
    assert client.post("/api/v1/match", json={"patient": {"summary": "Patient", "ecog": 6}}).status_code == 422
    assert client.post("/api/v1/match", json={"patient": {"summary": "Patient"}, "top_k": 10, "candidate_k": 5}).status_code == 422
    load.assert_not_called()
