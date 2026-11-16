from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, QueryEntities
from clinical_trial_rag_copilot.retrieval.vector_store import SearchResult
from clinical_trial_rag_copilot.agent.prompt_templates import create_user_prompt


TARGET = 'NCT07821684'
OTHER = 'NCT07818304'


def chunk(doc):
    return dict(chunk_id=doc + '_OVERVIEW', doc_id=doc, chunk_type='TRIAL_OVERVIEW',
                text=doc + ' source text', metadata={})


def retriever(found=None):
    r = HybridRetriever.__new__(HybridRetriever)
    r.extractor = Mock()
    r.extractor.extract.return_value = QueryEntities()
    r.query_graph = Mock(return_value=[])
    r.vector_store = Mock()
    r.vector_store.search.return_value = [SearchResult(**chunk(OTHER), score=0.99)]
    found = {TARGET: [chunk(TARGET)], OTHER: [chunk(OTHER)]} if found is None else found
    r.vector_store.get_chunks_by_doc_id.side_effect = lambda doc: found.get(doc, [])
    return r


def test_exact_id_recovers_ann_miss_and_enters_actual_prompt():
    r = retriever()
    query = 'What are the comparator treatments for nct07821684?'
    evidence = r.retrieve(query, top_k=1)
    assert evidence.fused_ncts == [TARGET]
    assert list(evidence.sibling_chunks) == [TARGET]
    assert evidence.vector_evidence[0].doc_id == TARGET
    prompt = create_user_prompt(query, evidence)
    assert TARGET + ' source text' in prompt
    assert OTHER + ' source text' not in prompt
    assert r.vector_store.get_chunks_by_doc_id.call_count == 1


def test_multiple_exact_ids_preserve_request_order_without_duplicates():
    r = retriever()
    result = r.retrieve(f'Compare {TARGET}, {OTHER}, and {TARGET}', top_k=2)
    assert result.fused_ncts == [TARGET, OTHER]
    assert r.vector_store.get_chunks_by_doc_id.call_count == 2
    result = r.retrieve(f'Compare {TARGET} and {OTHER}', top_k=1)
    assert result.fused_ncts == [TARGET]
    assert any('exceeds top_k' in w for w in result.retrieval_warnings)


def test_unknown_id_is_not_fabricated_and_is_flagged():
    r = retriever(found={OTHER: [chunk(OTHER)]})
    result = r.retrieve(f'What is {TARGET}?', top_k=1)
    assert TARGET not in result.fused_ncts
    assert any('No indexed source text' in w and TARGET in w for w in result.retrieval_warnings)


def test_exact_text_survives_dense_search_failure():
    r = retriever()
    r.vector_store.search.side_effect = RuntimeError('embedding service unavailable')
    result = r.retrieve(f'Describe {TARGET}', top_k=1)
    assert result.fused_ncts == [TARGET]
    assert any('Dense search failed' in w for w in result.retrieval_warnings)
    with pytest.raises(RuntimeError):
        r.retrieve('No identifier here')


def test_malformed_id_does_not_trigger_exact_lookup():
    r = retriever()
    r.retrieve('NCT078216840 and prefixNCT07821684', top_k=1)
    assert all(call.args[0] != TARGET for call in r.vector_store.get_chunks_by_doc_id.call_args_list)
