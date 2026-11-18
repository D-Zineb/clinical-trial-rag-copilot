from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.agent.prompt_templates import format_evidence_context
from clinical_trial_rag_copilot.retrieval.chunking import (
    create_trial_chunks, split_eligibility_section,
)
from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, QueryEntities
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore, SearchResult
from clinical_trial_rag_copilot.retrieval.embeddings import LocalTfIdfEmbeddingProvider


def test_nested_lists_and_exceptions_remain_attached():
    first = ('1. Prior drug X is excluded, except adjuvant treatment over 12 months ago.\n'
             '   Note: timing is measured before enrollment.\n')
    second = '2. Inadequate washout defined as:\n\n   1. Chemotherapy: 14 days.\n   2. Surgery: 28 days.\n'
    third = '3. Active infection.'
    assert split_eligibility_section(first + second + third, 60) == [first, second, third]


@pytest.mark.parametrize('text', [
    '* Prior treatment including any of the following:\n* Drug X.\n* Drug Y.',
    '* 1\\. First condition.\n\n  2\\. Another condition.',
    'Master protocol exclusions also apply.\n1. Drug X.\n2. Drug Y.',
    '1. First criterion.\n3. Missing number.',
    'Conditions separated; without reliable boundaries; keep intact.',
    '* Drug X excluded.\n* Exception: adjuvant treatment allowed.',
    '* First condition.\n2. Another condition.',
])
def test_ambiguous_structure_keeps_whole_section(text):
    assert split_eligibility_section(text, 10) == [text]


def trial_chunks(nct='NCT00000001'):
    section = '\n'.join(f'{i}. Criterion {i}: ' + 'source detail ' * 35 for i in range(1, 7))
    return create_trial_chunks(dict(nct_id=nct, brief_title='Trial',
                                   exclusion_criteria=section, inclusion_criteria='* None'))


def test_children_reconstruct_source_and_preserve_short_sections():
    chunks = trial_chunks()
    parent = next(c for c in chunks if c.chunk_id.endswith('_EXCLUSION'))
    children = [c for c in chunks if c.metadata.get('parent_chunk_id') == parent.chunk_id]
    source = parent.text.split('EXCLUSION CRITERIA:\n', 1)[1]
    assert len(children) > 1
    assert ''.join(c.text.split('EXCLUSION CRITERIA:\n', 1)[1] for c in children) == source
    for child in children:
        meta = child.metadata
        assert child.text.endswith(source[meta['section_start']:meta['section_end']])
        assert child.doc_id == parent.doc_id
    assert any(c.chunk_id.endswith('_INCLUSION') and c.text.endswith('* None') for c in chunks)


def retriever_with_hits(hits, stored):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.extractor = Mock()
    retriever.extractor.extract.return_value = QueryEntities()
    retriever.query_graph = Mock(return_value=[])
    retriever.vector_store = Mock()
    retriever.vector_store.search.return_value = hits
    retriever.vector_store.get_chunks_by_doc_id.side_effect = lambda doc: stored.get(doc, [])
    return retriever


def test_parent_fetched_beyond_three_trials_and_not_duplicated_in_prompt():
    stored, hits = {}, []
    for i in range(1, 5):
        doc = f'NCT{i:08d}'
        chunks = trial_chunks(doc)
        stored[doc] = [c.model_dump() for c in chunks]
        children = [c for c in chunks if c.metadata.get('parent_chunk_id')]
        # Duplicate hits must not displace other trial sections at top_k=4.
        hits.extend(SearchResult(**c.model_dump(), score=1 / i) for c in children)
    retriever = retriever_with_hits(hits, stored)
    result = retriever.retrieve('criterion', top_k=4)
    assert len(result.vector_evidence) == 4
    assert len({h.doc_id for h in result.vector_evidence}) == 4
    assert len(result.sibling_chunks['NCT00000004']) == 1
    context = format_evidence_context(result)
    for hit in result.vector_evidence:
        parent = next(c for c in stored[hit.doc_id]
                      if c['chunk_id'] == hit.metadata['parent_chunk_id'])
        assert context.count(parent['text']) == 1
        assert all(not c['metadata'].get('parent_chunk_id')
                   for c in result.sibling_chunks[hit.doc_id])
    assert retriever.vector_store.get_chunks_by_doc_id.call_count == 4


@pytest.mark.parametrize('failure', ['missing', 'exception', 'wrong_document'])
def test_failed_parent_fetch_warns_and_retains_available_evidence(failure):
    child = next(c for c in trial_chunks() if c.metadata.get('parent_chunk_id'))
    stored = {}
    if failure == 'wrong_document':
        stored[child.doc_id] = [dict(chunk_id=child.metadata['parent_chunk_id'],
                                   doc_id='NCT99999999', chunk_type=child.chunk_type,
                                   text='wrong trial', metadata={})]
    retriever = retriever_with_hits([SearchResult(**child.model_dump(), score=1)], stored)
    if failure == 'exception':
        retriever.vector_store.get_chunks_by_doc_id.side_effect = RuntimeError('offline')
    result = retriever.retrieve('criterion')
    assert any('Full parent section' in w for w in result.retrieval_warnings)
    context = format_evidence_context(result)
    assert child.text.strip() in context
    assert 'wrong trial' not in context


def test_local_index_round_trip_restores_full_parent(tmp_path):
    chunks = trial_chunks()
    store = LocalVectorStore(tmp_path)
    store.build_index(chunks, provider=LocalTfIdfEmbeddingProvider())
    reloaded = LocalVectorStore(tmp_path)
    reloaded.load()
    child = next(c for c in chunks if c.metadata.get('parent_chunk_id'))
    hits = reloaded.search('criterion source detail', top_k=len(chunks))
    assert any(h.metadata.get('parent_chunk_id') == child.metadata['parent_chunk_id'] for h in hits)
    parents = reloaded.get_chunks_by_doc_id(child.doc_id)
    assert any(c['chunk_id'] == child.metadata['parent_chunk_id'] for c in parents)
