"""Repeatable index builds and verification, invoked through vector_store."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from clinical_trial_rag_copilot.data.storage import load_dataset
from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk, build_all_chunks
from clinical_trial_rag_copilot.retrieval.embeddings import (
    BaseEmbeddingProvider, FastEmbedEmbeddingProvider, OpenAIEmbeddingProvider,
    GoogleGeminiEmbeddingProvider, LocalTfIdfEmbeddingProvider, get_embedding_provider,
)
from clinical_trial_rag_copilot.retrieval.vector_store import get_vector_store, PineconeVectorStore


def embedding_identity(provider):
    """Include model and encoding mode; never reuse corpus-fitted TF-IDF vectors."""
    if isinstance(provider, FastEmbedEmbeddingProvider):
        return f'fastembed:{version("fastembed")}:{provider.model.model_name}:passage_embed:v1'
    if isinstance(provider, OpenAIEmbeddingProvider):
        return f'openai:{provider.model}:default-dimensions:v1'
    if isinstance(provider, GoogleGeminiEmbeddingProvider):
        return f'google:{provider.model}:batchEmbedContents:v1'
    return None


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, provider, path, identity):
        self.provider, self.path, self.identity = provider, Path(path), identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS embeddings (identity TEXT, hash TEXT, vector TEXT, PRIMARY KEY(identity, hash))')

    def embed_documents(self, texts):
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        cached = {}
        with sqlite3.connect(self.path) as db:
            for key in set(keys):
                row = db.execute('SELECT vector FROM embeddings WHERE identity=? AND hash=?',
                                 (self.identity, key)).fetchone()
                if row:
                    cached[key] = json.loads(row[0])
            missing = dict((k, t) for k, t in zip(keys, texts) if k not in cached)
            if missing:
                vectors = self.provider.embed_documents(list(missing.values()))
                if len(vectors) != len(missing) or any(
                    not v or not all(math.isfinite(x) for x in v) for v in vectors
                ) or len({len(v) for v in vectors}) != 1:
                    raise ValueError('Invalid embedding batch; cache not updated')
                for key, vector in zip(missing, vectors):
                    cached[key] = vector
                    db.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)',
                               (self.identity, key, json.dumps(vector)))
        return [cached[k] for k in keys]

    def embed_query(self, text):
        return self.provider.embed_query(text)


def validate_chunks(chunks):
    expected = {c.chunk_id: c for c in chunks}
    if not chunks or len(expected) != len(chunks):
        raise ValueError('Corpus must be nonempty with unique chunk IDs')
    for child in chunks:
        parent_id = child.metadata.get('parent_chunk_id')
        if parent_id:
            parent = expected.get(parent_id)
            if (parent is None or parent.doc_id != child.doc_id
                    or parent.chunk_type != child.chunk_type
                    or parent.metadata.get('parent_chunk_id')):
                raise ValueError(f'Invalid parent link for {child.chunk_id}')
    return expected


def verify_index(store, chunks, attempts=30, delay=2):
    """Verify every stored record, then exercise search and parent expansion."""
    expected = validate_chunks(chunks)
    if isinstance(store, PineconeVectorStore):
        if store.index is None:
            store.load()
        for attempt in range(attempts):
            ids = {i for batch in store.index.list(namespace=store.namespace) for i in batch}
            if ids == set(expected):
                break
            if attempt + 1 < attempts:
                time.sleep(delay)
        else:
            raise RuntimeError('Index IDs do not match the corpus; index is not ready')
        actual = {}
        keys = list(expected)
        for start in range(0, len(keys), 100):
            batch = keys[start:start + 100]
            for attempt in range(attempts):
                records = store.index.fetch(ids=batch, namespace=store.namespace).vectors
                if set(records) == set(batch):
                    break
                if attempt + 1 < attempts:
                    time.sleep(delay)
            else:
                raise RuntimeError('Index records not visible; index is not ready')
            actual.update({key: store._chunk(record) for key, record in records.items()})
    else:
        store.load()
        actual = {c['chunk_id']: c for c in store.chunks}
    if set(actual) != set(expected) or any(actual[k] != c.model_dump() for k, c in expected.items()):
        raise RuntimeError('Stored text or metadata differs from corpus; index is not ready')
    hits = store.search('prior chemotherapy exclusion criteria exceptions', top_k=min(20, len(chunks)))
    if not hits or any(h.chunk_id not in expected for h in hits):
        raise RuntimeError('Search verification failed')
    child_hits = [h for h in hits if h.metadata.get('parent_chunk_id')]
    for hit in child_hits:
        parent_id = hit.metadata['parent_chunk_id']
        siblings = store.get_chunks_by_doc_id(hit.doc_id)
        if not any(c == expected[parent_id].model_dump() for c in siblings):
            raise RuntimeError(f'Parent fetch verification failed: {parent_id}')
    return dict(ready=True, records_verified=len(actual), exact_ids_match=True,
                search_hits=len(hits), child_search_hits=len(child_hits),
                parent_links_verified=sum(bool(c.metadata.get('parent_chunk_id')) for c in chunks))


def rebuild_index(run_root=Path('eval/index_rebuild_runs'), cache_path=Path('eval/embedding_cache.sqlite3')):
    chunks = build_all_chunks(load_dataset('oncology_trials.json'), load_dataset('pubmed_oncology_literature.json'))
    validate_chunks(chunks)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    run = Path(run_root) / stamp
    run.mkdir(parents=True, exist_ok=False)
    store = get_vector_store()
    provider = get_embedding_provider()
    identity = embedding_identity(provider)
    manifest = dict(run=str(run), embedding_identity=identity)
    if isinstance(store, PineconeVectorStore):
        if isinstance(provider, LocalTfIdfEmbeddingProvider):
            raise ValueError('Pinecone requires stable embeddings, not TF-IDF')
        manifest.update(index=store.index_name, previous_namespace=store.namespace)
        store.namespace = f'{store.namespace[:60]}-build-{stamp.lower()}'
        manifest['namespace'] = store.namespace
        # UUID namespace plus explicit emptiness check on an existing index.
        if store.index_name in store._client().list_indexes().names():
            store.load()
            if any(store.index.list(namespace=store.namespace)):
                raise RuntimeError('Target namespace is not empty')
    else:
        store = type(store)(run / 'local_store')
        manifest['local_store'] = str(store.store_dir)
    if identity:
        provider = CachedEmbeddingProvider(provider, cache_path, identity)
    corpus = ''.join(c.model_dump_json() + '\n' for c in chunks)
    (run / 'corpus.jsonl').write_text(corpus)
    manifest['corpus_sha256'] = hashlib.sha256(corpus.encode()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Building {len(chunks)} chunks; run: {run}', flush=True)
    store.build_index(chunks, provider=provider)
    report = {**manifest, **verify_index(store, chunks)}
    (run / 'verified.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return store


def verify_run(run):
    """Recheck an existing run without uploading or regenerating embeddings."""
    run = Path(run)
    manifest_file = run / 'manifest.json'
    # Adopt already completed rebuilds from the earlier helper without reupload.
    manifest = json.loads((manifest_file if manifest_file.exists() else run / 'verified.json').read_text())
    corpus = (run / 'corpus.jsonl').read_bytes()
    if hashlib.sha256(corpus).hexdigest() != manifest['corpus_sha256']:
        raise ValueError('Corpus fingerprint differs from the recorded run')
    # JSON strings can contain literal Unicode line separators; only LF marks
    # a JSONL record boundary (str.splitlines would corrupt those strings).
    chunks = [DocumentChunk(**json.loads(line)) for line in corpus.decode().split('\n') if line]
    if 'namespace' in manifest:
        store = PineconeVectorStore()
        store.index_name, store.namespace = manifest['index'], manifest['namespace']
        provider = get_embedding_provider()
        identity = embedding_identity(provider)
        if manifest.get('embedding_identity') and identity != manifest['embedding_identity']:
            raise ValueError('Embedding configuration differs from the indexed model')
        if manifest.get('embedding_model') and (
            not isinstance(provider, FastEmbedEmbeddingProvider)
            or provider.model.model_name != manifest['embedding_model']
        ):
            raise ValueError('Embedding model differs from the indexed model')
        store.embedding_provider = provider
    else:
        from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore
        store = LocalVectorStore(manifest['local_store'])
    # Invalidate an older readiness result while rechecking this run.
    (run / 'verified.json').write_text(json.dumps({**manifest, 'ready': False}, indent=2))
    report = {**manifest, **verify_index(store, chunks)}
    (run / 'verified.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return store
