import pytest
from starlette.testclient import TestClient
from clinical_trial_rag_copilot.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api(tmp_path, monkeypatch):
    import clinical_trial_rag_copilot.api.main as api
    import clinical_trial_rag_copilot.retrieval.hybrid_search as hybrid
    import clinical_trial_rag_copilot.observability.tracer as telemetry
    from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk
    from clinical_trial_rag_copilot.retrieval.embeddings import LocalTfIdfEmbeddingProvider
    from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore

    store = LocalVectorStore(store_dir=tmp_path / "index")
    store.build_index([DocumentChunk(
        chunk_id="NCT05543278_OVERVIEW", doc_id="NCT05543278",
        chunk_type="TRIAL_OVERVIEW",
        text="Amiodarone targeting KCNH2 in atrial fibrillation clinical trials.",
        metadata={"nct_id": "NCT05543278"},
    )], provider=LocalTfIdfEmbeddingProvider())
    monkeypatch.setattr(api, "_vector_store", store)
    monkeypatch.setattr(api, "_copilot", None)
    monkeypatch.setattr(api, "_guardrail", None)
    monkeypatch.setattr(hybrid, "get_vector_store", lambda: store)
    traces = tmp_path / "traces"
    traces.mkdir()
    monkeypatch.setattr(telemetry, "TRACES_DIR", traces)

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
    assert "NCT05543278" in data["retrieved_ncts"]


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
