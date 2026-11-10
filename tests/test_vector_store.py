import pytest
import shutil
from pathlib import Path
from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk
from clinical_trial_rag_copilot.retrieval.embeddings import LocalTfIdfEmbeddingProvider
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore


@pytest.fixture
def temp_store_dir(tmp_path):
    store_dir = tmp_path / "test_vector_store"
    yield store_dir
    if store_dir.exists():
        shutil.rmtree(store_dir)


def test_vector_store_build_and_search(temp_store_dir):
    """Verify vector index construction, cosine search, and metadata persistence."""
    chunks = [
        DocumentChunk(
            chunk_id="NCT1_OVERVIEW",
            chunk_type="TRIAL_OVERVIEW",
            doc_id="NCT00000001",
            text="Amiodarone oral therapy for paroxysmal atrial fibrillation in elderly adults.",
            metadata={"nct_id": "NCT00000001", "drug": "Amiodarone", "min_age_years": 65.0},
        ),
        DocumentChunk(
            chunk_id="NCT2_OVERVIEW",
            chunk_type="TRIAL_OVERVIEW",
            doc_id="NCT00000002",
            text="Spironolactone in chronic systolic heart failure with reduced ejection fraction.",
            metadata={"nct_id": "NCT00000002", "drug": "Spironolactone", "min_age_years": 18.0},
        ),
        DocumentChunk(
            chunk_id="PMID1_ABSTRACT",
            chunk_type="LITERATURE_ABSTRACT",
            doc_id="11111111",
            text="KCNH2 potassium channel blockade induces action potential duration prolongation and torsades.",
            metadata={"pmid": "11111111", "gene": "KCNH2"},
        ),
    ]

    store = LocalVectorStore(store_dir=temp_store_dir)
    provider = LocalTfIdfEmbeddingProvider(n_features=64)
    store.build_index(chunks, provider=provider)

    assert store.is_indexed()
    assert store.vectors is not None
    assert store.vectors.shape[0] == 3
    assert store.vectors.shape[1] <= 64

    # Search test 1: Amiodarone query
    hits = store.search("Amiodarone atrial fibrillation", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "NCT00000001"
    assert hits[0].score > 0.3

    # Search test 2: Chunk type filter
    lit_hits = store.search("potassium channel", top_k=2, chunk_type="LITERATURE_ABSTRACT")
    assert len(lit_hits) == 1
    assert lit_hits[0].doc_id == "11111111"

    # Search test 3: Custom predicate filter
    elderly_hits = store.search(
        "cardiac trial",
        top_k=2,
        filter_fn=lambda m: m.get("min_age_years", 0) >= 60.0,
    )
    assert len(elderly_hits) == 1
    assert elderly_hits[0].doc_id == "NCT00000001"


def test_vector_store_persistence(temp_store_dir):
    """Verify that a persisted vector index can be re-loaded cleanly in a new instance."""
    chunks = [
        DocumentChunk(
            chunk_id="C1",
            chunk_type="TRIAL_OVERVIEW",
            doc_id="NCT100",
            text="Vericiguat stimulator of soluble guanylate cyclase.",
            metadata={"nct_id": "NCT100"},
        )
    ]
    store1 = LocalVectorStore(store_dir=temp_store_dir)
    store1.build_index(chunks, provider=LocalTfIdfEmbeddingProvider())

    # Re-instantiate in a new object
    store2 = LocalVectorStore(store_dir=temp_store_dir)
    assert store2.is_indexed()
    store2.load()

    assert len(store2.chunks) == 1
    assert store2.chunks[0]["doc_id"] == "NCT100"
    hits = store2.search("Vericiguat", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "NCT100"
