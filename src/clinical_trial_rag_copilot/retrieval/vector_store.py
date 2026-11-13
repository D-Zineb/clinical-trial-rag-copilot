from clinical_trial_rag_copilot.data.storage import load_dataset
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
from clinical_trial_rag_copilot.utils.logger import get_logger

logger = get_logger("retrieval.vector_store")

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

        logger.info("Embedding %d chunks...", len(chunks))
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
        logger.info("Indexing completed in %.2fs (matrix shape: %s)", elapsed, self.vectors.shape)
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
                logger.warning("Could not cache embedding provider object: %s", e)
        logger.info("Persisted index to %s", self.store_dir)

    def load(self):
        """Load vector matrix and metadata from disk."""
        if not self.is_indexed():
            raise FileNotFoundError(f"Vector store not found in {self.store_dir}. Build it first.")

        logger.debug("Loading index from %s...", self.store_dir)
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

        logger.info("Loaded %d chunks from disk (matrix shape: %s)", len(self.chunks), self.vectors.shape)

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



class PineconeVectorStore:
    """Serverless cosine search; chunk text and metadata live in Pinecone."""

    def __init__(self, *, client=None, provider=None):
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "clinical-trial-copilot")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "cardiology-v1")
        self.client = client
        self.index = None
        self.embedding_provider = provider

    def _client(self):
        if self.client is None:
            key = os.getenv("PINECONE_API_KEY")
            if not key:
                raise ValueError("Set PINECONE_API_KEY or VECTOR_STORE_PROVIDER=local for offline use.")
            from pinecone import Pinecone
            self.client = Pinecone(api_key=key)
        return self.client

    def _provider(self):
        if self.embedding_provider is None:
            name = os.getenv("EMBEDDING_PROVIDER", "fastembed").strip().lower()
            if name == "local":
                raise ValueError("Pinecone requires stable embeddings. Set EMBEDDING_PROVIDER=fastembed.")
            self.embedding_provider = get_embedding_provider()
        return self.embedding_provider

    def load(self):
        description = self._client().describe_index(self.index_name)
        if not description.status["ready"]:
            raise RuntimeError("Pinecone index is not ready yet.")
        if description.metric != "cosine":
            raise ValueError("Pinecone index must use cosine similarity.")
        self.dimension = description.dimension
        self.index = self._client().Index(host=description.host)

    @property
    def indexed_chunks(self):
        if self.index is None:
            self.load()
        stats = self.index.describe_index_stats()
        return stats.namespaces.get(self.namespace, {}).get("vector_count", 0)

    def is_indexed(self):
        try:
            return self.indexed_chunks > 0
        except Exception as exc:
            logger.warning("Pinecone unavailable: %s", exc)
            return False

    def build_index(self, chunks, provider=None):
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique.")
        if provider is not None:
            self.embedding_provider = provider
        embedding_provider = self._provider()
        from clinical_trial_rag_copilot.retrieval.embeddings import LocalTfIdfEmbeddingProvider
        if isinstance(embedding_provider, LocalTfIdfEmbeddingProvider):
            raise ValueError("Corpus-fitted TF-IDF is not supported with Pinecone.")
        # Embed in bounded batches; never delete the existing namespace.
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            logger.info("Embedding chunks %d-%d of %d...", start + 1, start + len(batch), len(chunks))
            vectors = embedding_provider.embed_documents([c.text for c in batch])
            if len(vectors) != len(batch) or not vectors or not len(vectors[0]):
                raise ValueError("Embedding provider returned an invalid batch.")
            dimension = len(vectors[0])
            if self.index is None:
                logger.info("Connecting to Pinecone index %s...", self.index_name)
                client = self._client()
                if self.index_name not in client.list_indexes().names():
                    from pinecone import ServerlessSpec
                    client.create_index(
                        name=self.index_name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                            region=os.getenv("PINECONE_REGION", "us-east-1")),
                    )
                deadline = time.monotonic() + 120
                while not client.describe_index(self.index_name).status["ready"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Pinecone index creation timed out.")
                    time.sleep(1)
                self.load()
            records = []
            for chunk, values in zip(batch, vectors):
                if len(values) != self.dimension:
                    raise ValueError("Embedding dimension differs from Pinecone index; use a new index.")
                metadata = {
                    **{k: v for k, v in chunk.metadata.items()
                       if isinstance(v, (str, bool, int, float)) or
                       (isinstance(v, list) and all(isinstance(x, str) for x in v))},
                    "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type, "text": chunk.text,
                    "_metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
                }
                if len(json.dumps(metadata).encode("utf-8")) > 39000:
                    raise ValueError(f"Chunk {chunk.chunk_id} exceeds Pinecone metadata size; split it first.")
                records.append({"id": chunk.chunk_id, "values": values, "metadata": metadata})
            logger.info("Uploading batch to namespace %s...", self.namespace)
            self.index.upsert(vectors=records, namespace=self.namespace)
            logger.info("Uploaded %d/%d chunks.", start + len(batch), len(chunks))
        logger.info("Upserted %d chunks to %s/%s; visibility is eventually consistent.",
                    len(chunks), self.index_name, self.namespace)

    @staticmethod
    def _chunk(record):
        metadata = record.metadata
        return dict(chunk_id=metadata["chunk_id"], doc_id=metadata["doc_id"],
                    chunk_type=metadata["chunk_type"], text=metadata["text"],
                    metadata=json.loads(metadata["_metadata_json"]))

    def get_chunks_by_doc_id(self, doc_id):
        if self.index is None:
            self.load()
        # Trial chunks share a prefix; literature uses PMID_<doc_id>.
        if not doc_id.startswith("NCT"):
            response = self.index.fetch(ids=[f"PMID_{doc_id}"], namespace=self.namespace)
            return [self._chunk(record) for record in response.vectors.values()
                    if record.metadata["doc_id"] == doc_id]
        chunks = []
        for ids in self.index.list(prefix=f"{doc_id}_", namespace=self.namespace):
            response = self.index.fetch(ids=ids, namespace=self.namespace)
            for record in response.vectors.values():
                chunk = self._chunk(record)
                if chunk["doc_id"] == doc_id:
                    chunks.append(chunk)
        return sorted(chunks, key=lambda c: c["chunk_id"])

    def search(self, query, top_k=5, chunk_type=None, filter_fn=None):
        if top_k <= 0:
            return []
        if filter_fn is not None:
            raise ValueError("Use Pinecone metadata filters instead of Python predicates.")
        if self.index is None:
            self.load()
        vector = self._provider().embed_query(query)
        if len(vector) != self.dimension:
            raise ValueError("Query embedding dimension differs from the Pinecone index.")
        response = self.index.query(
            vector=vector, top_k=top_k, namespace=self.namespace,
            include_metadata=True,
            **({"filter": {"chunk_type": {"$eq": chunk_type}}} if chunk_type else {}),
        )
        return [SearchResult(**self._chunk(match), score=match.score) for match in response.matches]


def get_vector_store():
    """Pinecone is the default; offline use requires an explicit local selection."""
    backend = os.getenv("VECTOR_STORE_PROVIDER", "pinecone").strip().lower()
    if backend == "pinecone":
        return PineconeVectorStore()
    if backend == "local":
        return LocalVectorStore()
    raise ValueError(f"Unknown VECTOR_STORE_PROVIDER: {backend}")


def index_all_data():
    """Build and persist the complete vector index from trials and literature datasets."""
    logger.info("Starting vector ingestion; loading source datasets from %s", DATA_DIR)
    trials = load_dataset("cardiology_trials.json")
    papers = load_dataset("pubmed_cardiology_literature.json")

    chunks = build_all_chunks(trials, papers)
    logger.info("Prepared %d chunks from %d trials and %d papers.", len(chunks), len(trials), len(papers))
    store = get_vector_store()
    logger.info("Selected vector backend: %s", type(store).__name__)
    store.build_index(chunks)
    return store


if __name__ == "__main__":
    store = index_all_data()

    # Demonstration search
    test_query = "Amiodarone postoperative atrial fibrillation KCNH2"
    logger.info("Test search query: '%s'", test_query)
    hits = store.search(test_query, top_k=3)
    for h in hits:
        snippet = h.text.replace("\n", " ")[:140]
        logger.info("[%s] Score=%.4f | Doc=%s | Snippet: %s...", h.chunk_type, h.score, h.doc_id, snippet)
