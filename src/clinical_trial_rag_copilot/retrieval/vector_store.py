import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk, build_all_chunks
from clinical_trial_rag_copilot.retrieval.embeddings import BaseEmbeddingProvider, get_embedding_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "output"
DEFAULT_STORE_DIR = DATA_DIR / "vector_store"


class SearchResult(BaseModel):
    """Result returned by vector similarity search."""
    chunk_id: str
    doc_id: str
    chunk_type: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LocalVectorStore:
    """High-performance, pure-Python persistent vector store with cosine similarity search.
    
    Provides sub-millisecond similarity queries across thousands of clinical chunks
    with zero native C++ compilation dependencies and full metadata filtering.
    """

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.store_dir / "vectors.npy"
        self.metadata_file = self.store_dir / "chunks_metadata.json"
        self.provider_cache_file = self.store_dir / "embedding_provider.pkl"

        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.embedding_provider: Optional[BaseEmbeddingProvider] = None

    def is_indexed(self) -> bool:
        """Check if vector store index exists on disk."""
        return self.vectors_file.exists() and self.metadata_file.exists()

    def build_index(self, chunks: List[DocumentChunk], provider: Optional[BaseEmbeddingProvider] = None):
        """Embed all chunks and construct the normalized vector index."""
        self.embedding_provider = provider or get_embedding_provider()
        texts = [c.text for c in chunks]

        print(f"[VectorStore] Embedding {len(chunks)} chunks...")
        t0 = time.time()
        raw_vectors = self.embedding_provider.embed_documents(texts)
        vec_array = np.array(raw_vectors, dtype=np.float32)

        # L2-normalize vectors for instant dot-product cosine similarity
        norms = np.linalg.norm(vec_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.vectors = vec_array / norms

        self.chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_type": c.chunk_type,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        elapsed = time.time() - t0
        print(f"[VectorStore] Indexing completed in {elapsed:.2f}s (Matrix shape: {self.vectors.shape})")
        self.save()

    def save(self):
        """Persist vector matrix, metadata, and fitted embedding provider to disk."""
        if self.vectors is not None:
            np.save(self.vectors_file, self.vectors)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        if self.embedding_provider is not None:
            try:
                with open(self.provider_cache_file, "wb") as f:
                    pickle.dump(self.embedding_provider, f)
            except Exception as e:
                print(f"[VectorStore] Warning: Could not cache embedding provider object: {e}")
        print(f"[VectorStore] Persisted index to {self.store_dir}")

    def load(self):
        """Load vector matrix and metadata from disk."""
        if not self.is_indexed():
            raise FileNotFoundError(f"Vector store not found in {self.store_dir}. Build it first.")

        print(f"[VectorStore] Loading index from {self.store_dir}...")
        self.vectors = np.load(self.vectors_file)
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if self.provider_cache_file.exists():
            try:
                with open(self.provider_cache_file, "rb") as f:
                    self.embedding_provider = pickle.load(f)
            except Exception:
                self.embedding_provider = get_embedding_provider()
        else:
            self.embedding_provider = get_embedding_provider()

        print(f"[VectorStore] Loaded {len(self.chunks)} chunks (Shape: {self.vectors.shape})")

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all sibling chunks belonging to a document ID (e.g. NCT ID) for full context rehydration."""
        if not self.chunks:
            self.load()
        return [c for c in self.chunks if c.get("doc_id") == doc_id]

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[str] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[SearchResult]:
        """Perform cosine similarity search with optional chunk type and metadata filtering."""
        if self.vectors is None or not self.chunks:
            self.load()

        # Compute query vector
        raw_q = self.embedding_provider.embed_query(query)
        q_vec = np.array(raw_q, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # Compute cosine similarity across all stored vectors via dot product
        scores = np.dot(self.vectors, q_vec)

        # Ranked candidate indices in descending score order
        ranked_indices = np.argsort(scores)[::-1]

        results: List[SearchResult] = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break

            chunk = self.chunks[idx]

            # Optional filter by chunk_type
            if chunk_type and chunk["chunk_type"] != chunk_type:
                continue

            # Optional custom filter
            if filter_fn and not filter_fn(chunk["metadata"]):
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    doc_id=chunk["doc_id"],
                    chunk_type=chunk["chunk_type"],
                    text=chunk["text"],
                    score=float(scores[idx]),
                    metadata=chunk["metadata"],
                )
            )

        return results


def index_all_data():
    """Build and persist the complete vector index from trials and literature datasets."""
    trials_file = DATA_DIR / "cardiology_trials.json"
    pubmed_file = DATA_DIR / "pubmed_cardiology_literature.json"

    if not trials_file.exists() or not pubmed_file.exists():
        raise FileNotFoundError("Missing data files. Run run_acquisition.py first.")

    with open(trials_file, "r", encoding="utf-8") as f:
        trials = json.load(f)
    with open(pubmed_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    chunks = build_all_chunks(trials, papers)
    store = LocalVectorStore()
    store.build_index(chunks)
    return store


if __name__ == "__main__":
    store = index_all_data()

    # Demonstration search
    test_query = "Amiodarone postoperative atrial fibrillation KCNH2"
    print(f"\n[Test Search] Query: '{test_query}'")
    hits = store.search(test_query, top_k=3)
    for h in hits:
        print(f"  [{h.chunk_type}] (Score: {h.score:.4f}) Doc: {h.doc_id}")
        snippet = h.text.replace("\n", " ")[:140]
        print(f"    Snippet: {snippet}...\n")
