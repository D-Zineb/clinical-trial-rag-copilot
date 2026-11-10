import pytest
from starlette.testclient import TestClient
from clinical_trial_rag_copilot.api.main import app

client = TestClient(app)


def test_api_health_check():
    """Verify health endpoint returns status and index readiness."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "neo4j_connected" in data
    assert "vector_store_ready" in data
    assert "indexed_chunks" in data


def test_api_query_endpoint():
    """Verify end-to-end GraphRAG query endpoint returns answer and telemetry trace."""
    payload = {
        "query": "Which clinical trials evaluate Amiodarone targeting KCNH2 in Atrial Fibrillation?",
        "top_k": 3,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert len(data["answer"]) > 50
    assert len(data["trace_id"]) > 0
    assert data["total_latency_ms"] >= 0
    assert isinstance(data["retrieved_ncts"], list)


def test_api_guardrail_verify_endpoint():
    """Verify standalone guardrail factuality audit endpoint."""
    payload = {
        "text": "Trial NCT05543278 tested Amiodarone on KCNH2 potassium channels."
    }
    response = client.post("/api/v1/guardrail/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "is_valid" in data
    assert "status" in data
    assert "guardrail_banner" in data


def test_api_traces_endpoint():
    """Verify telemetry traces retrieval endpoint."""
    response = client.get("/api/v1/traces?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
