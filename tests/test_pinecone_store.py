import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk
from clinical_trial_rag_copilot.retrieval.vector_store import PineconeVectorStore, get_vector_store


@pytest.fixture
def store():
    client = MagicMock()
    client.list_indexes.return_value.names.return_value = ["clinical-trial-copilot"]
    client.describe_index.return_value = SimpleNamespace(
        status={"ready": True}, metric="cosine", dimension=2, host="test-host")
    provider = MagicMock()
    provider.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    provider.embed_query.return_value = [1.0, 0.0]
    return PineconeVectorStore(client=client, provider=provider)


def test_upsert_search_and_siblings(store):
    chunk = DocumentChunk(chunk_id="NCT1_OVERVIEW", doc_id="NCT1",
                          chunk_type="TRIAL_OVERVIEW", text="cardiology",
                          metadata={"age": 65, "missing": None, "nested": {"a": 1}})
    store.build_index([chunk])
    record = store.index.upsert.call_args.kwargs["vectors"][0]
    assert record["id"] == chunk.chunk_id
    assert record["metadata"]["age"] == 65
    assert "missing" not in record["metadata"]
    match = SimpleNamespace(metadata=record["metadata"], score=0.9)
    store.index.query.return_value = SimpleNamespace(matches=[match])
    hits = store.search("cardiology", chunk_type="TRIAL_OVERVIEW")
    assert hits[0].metadata == chunk.metadata
    assert hits[0].score == 0.9
    assert store.index.query.call_args.kwargs["filter"] == {"chunk_type": {"$eq": "TRIAL_OVERVIEW"}}
    store.index.list.return_value = [[chunk.chunk_id]]
    store.index.fetch.return_value = SimpleNamespace(vectors={chunk.chunk_id: match})
    assert store.get_chunks_by_doc_id("NCT1")[0]["text"] == "cardiology"
    store.index.list.assert_called_once_with(prefix="NCT1_", namespace=store.namespace)
    store.index.delete.assert_not_called()


def test_dimension_mismatch_never_upserts(store):
    store.client.describe_index.return_value.dimension = 3
    chunk = DocumentChunk(chunk_id="C1", doc_id="D1", chunk_type="T", text="text")
    with pytest.raises(ValueError, match="dimension"):
        store.build_index([chunk])
    store.index.upsert.assert_not_called()


def test_namespace_health_and_unavailable(store):
    store.load()
    store.index.describe_index_stats.return_value = SimpleNamespace(
        namespaces={store.namespace: {"vector_count": 5}})
    assert store.is_indexed()
    assert store.indexed_chunks == 5
    store.index.describe_index_stats.side_effect = RuntimeError("offline")
    assert not store.is_indexed()


def test_configuration(monkeypatch):
    monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)
    assert isinstance(get_vector_store(), PineconeVectorStore)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    with pytest.raises(ValueError, match="stable embeddings"):
        PineconeVectorStore()._provider()
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "typo")
    with pytest.raises(ValueError, match="Unknown"):
        get_vector_store()


def test_invalid_inputs(store):
    with pytest.raises(ValueError, match="empty"):
        store.build_index([])
    with pytest.raises(ValueError, match="predicates"):
        store.search("query", filter_fn=lambda m: True)
    assert store.search("query", top_k=0) == []


def test_index_creation(store):
    store.client.list_indexes.return_value.names.return_value = []
    store.build_index([DocumentChunk(chunk_id="NCT1_OVERVIEW", doc_id="NCT1",
                                     chunk_type="T", text="text")])
    kwargs = store.client.create_index.call_args.kwargs
    assert kwargs["dimension"] == 2
    assert kwargs["metric"] == "cosine"


def test_literature_sibling_lookup(store):
    store.load()
    store.index.fetch.return_value = SimpleNamespace(vectors={})
    assert store.get_chunks_by_doc_id("12345") == []
    store.index.fetch.assert_called_once_with(ids=["PMID_12345"], namespace=store.namespace)
