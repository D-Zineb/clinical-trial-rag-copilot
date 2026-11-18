import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.retrieval.chunking import DocumentChunk
from clinical_trial_rag_copilot.retrieval.embeddings import LocalTfIdfEmbeddingProvider
from clinical_trial_rag_copilot.retrieval.indexing import (
    CachedEmbeddingProvider, validate_chunks, verify_index, rebuild_index, verify_run,
)
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore, PineconeVectorStore


def test_cache_reuses_exact_text_and_separates_models(tmp_path):
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [[float(len(t)), 1.] for t in texts]
    path = tmp_path / 'cache.sqlite3'
    cache = CachedEmbeddingProvider(provider, path, 'model-a:passage:v1')
    assert cache.embed_documents(['one', 'one', 'two']) == [[3., 1.]] * 3
    provider.embed_documents.assert_called_once_with(['one', 'two'])
    cache = CachedEmbeddingProvider(provider, path, 'model-a:passage:v1')
    cache.embed_documents(['two', 'one'])
    assert provider.embed_documents.call_count == 1
    cache.embed_documents(['changed'])
    assert provider.embed_documents.call_count == 2
    CachedEmbeddingProvider(provider, path, 'model-b:passage:v1').embed_documents(['one'])
    assert provider.embed_documents.call_count == 3


def test_invalid_embedding_batch_is_not_cached(tmp_path):
    provider = Mock()
    provider.embed_documents.return_value = []
    cache = CachedEmbeddingProvider(provider, tmp_path / 'cache.db', 'model')
    with pytest.raises(ValueError, match='Invalid embedding'):
        cache.embed_documents(['text'])
    provider.embed_documents.return_value = [[1., 2.]]
    assert cache.embed_documents(['text']) == [[1., 2.]]


def corpus():
    parent = DocumentChunk(chunk_id='NCT1_EXCLUSION', doc_id='NCT1',
                           chunk_type='TRIAL_EXCLUSION', text='Prior chemotherapy excluded\u2028except adjuvant treatment')
    child = DocumentChunk(chunk_id='NCT1_EXCLUSION_CHILD_0000', doc_id='NCT1',
                          chunk_type=parent.chunk_type, text='Prior chemotherapy excluded',
                          metadata={'parent_chunk_id': parent.chunk_id})
    return [parent, child]


def test_rejects_wrong_parent_and_duplicates():
    chunks = corpus()
    with pytest.raises(ValueError, match='unique'):
        validate_chunks(chunks + chunks)
    chunks[0].doc_id = 'NCT2'
    with pytest.raises(ValueError, match='parent'):
        validate_chunks(chunks)


def test_local_verification_checks_text_not_only_counts(tmp_path):
    chunks = corpus()
    store = LocalVectorStore(tmp_path)
    store.build_index(chunks, LocalTfIdfEmbeddingProvider())
    assert verify_index(store, chunks)['parent_links_verified'] == 1
    stored = json.loads(store.metadata_file.read_text())
    stored[0]['text'] = 'wrong text'
    store.metadata_file.write_text(json.dumps(stored))
    with pytest.raises(RuntimeError, match='Stored text'):
        verify_index(store, chunks)


def test_pinecone_eventual_visibility_and_missing_records():
    store = PineconeVectorStore(client=Mock())
    store.index = Mock()
    chunks = corpus()
    records = {c.chunk_id: SimpleNamespace(metadata={
        **c.model_dump(exclude={'metadata'}), '_metadata_json': json.dumps(c.metadata)
    }) for c in chunks}
    store.index.list.side_effect = [[], [list(records)]]
    store.index.fetch.return_value = SimpleNamespace(vectors=records)
    store.search = Mock(return_value=[SimpleNamespace(chunk_id=chunks[0].chunk_id, metadata={})])
    assert verify_index(store, chunks, attempts=2, delay=0)['ready']
    store.index.list.side_effect = None
    store.index.list.return_value = [list(records) + ['stale_id']]
    with pytest.raises(RuntimeError, match='IDs'):
        verify_index(store, chunks, attempts=1)


def test_official_build_and_verify_without_overwriting_existing_store(tmp_path, monkeypatch):
    from clinical_trial_rag_copilot.retrieval import indexing
    old = LocalVectorStore(tmp_path / 'old')
    old.build_index(corpus(), LocalTfIdfEmbeddingProvider())
    before = old.metadata_file.read_bytes()
    monkeypatch.setattr(indexing, 'get_vector_store', lambda: old)
    monkeypatch.setattr(indexing, 'get_embedding_provider', LocalTfIdfEmbeddingProvider)
    monkeypatch.setattr(indexing, 'load_dataset', lambda name: [])
    monkeypatch.setattr(indexing, 'build_all_chunks', lambda *args: corpus())
    new = rebuild_index(tmp_path / 'runs', tmp_path / 'cache.db')
    assert new.store_dir != old.store_dir
    assert old.metadata_file.read_bytes() == before
    run = new.store_dir.parent
    assert json.loads((run / 'verified.json').read_text())['ready']
    verify_run(run)
    (run / 'corpus.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='fingerprint'):
        verify_run(run)


def test_pinecone_build_preserves_active_namespace(tmp_path, monkeypatch):
    from clinical_trial_rag_copilot.retrieval import indexing
    store = PineconeVectorStore(client=Mock())
    original = store.namespace
    store.client.list_indexes.return_value.names.return_value = [store.index_name]
    store.index = Mock()
    store.index.list.return_value = []
    store.load = Mock()
    store.build_index = Mock()
    monkeypatch.setattr(indexing, 'get_vector_store', lambda: store)
    monkeypatch.setattr(indexing, 'get_embedding_provider', Mock)
    monkeypatch.setattr(indexing, 'load_dataset', lambda name: [])
    monkeypatch.setattr(indexing, 'build_all_chunks', lambda *args: corpus())
    monkeypatch.setattr(indexing, 'verify_index', lambda *args: {'ready': True})
    rebuild_index(tmp_path / 'runs', tmp_path / 'cache.db')
    assert store.namespace != original
    store.index.list.assert_called_once_with(namespace=store.namespace)
    store.index.delete.assert_not_called()
    report = json.loads(next((tmp_path / 'runs').glob('*/verified.json')).read_text())
    assert report['previous_namespace'] == original
    assert report['ready']
