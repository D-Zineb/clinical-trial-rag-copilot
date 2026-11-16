from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from clinical_trial_rag_copilot.retrieval.hybrid_search import ClinicalEntityExtractor, HybridRetriever, QueryEntities
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore, SearchResult
from clinical_trial_rag_copilot.eval.graph_evaluation import reference_query
from clinical_trial_rag_copilot.eval.judged_retrieval import metrics, evaluate
from clinical_trial_rag_copilot.retrieval.trial_matching import extract_criteria, assess_criterion, PatientProfile


def extractor():
    e = ClinicalEntityExtractor.__new__(ClinicalEntityExtractor)
    e.known_conditions = {'Non-Small Cell Lung Cancer', 'Small Cell Lung Cancer', 'Lung Cancer', 'Cancer'}
    e.known_drugs = {'Osimertinib', 'Erlotinib'}
    e.known_genes = {'EGFR', 'ALK'}
    e.known_aes = set()
    return e


def test_longest_mentions_aliases_and_boolean_scope():
    e = extractor()
    assert e.extract('Non-Small Cell Lung Cancer').conditions == ['Non-Small Cell Lung Cancer']
    assert e.extract('NSCLC').conditions == ['Non-Small Cell Lung Cancer']
    assert e.extract('Osimertinib or Erlotinib').operators == {'drugs': 'any'}
    assert e.extract('EGFR or ALK and EGFR').interpretation_warnings
    assert e.extract('without EGFR').interpretation_warnings
    assert e.extract('treatment').sex is None


def test_explicit_any_contract_and_unbounded_production_candidates():
    e = QueryEntities(genes=['EGFR', 'ALK'], operators={'genes': 'any'})
    r = HybridRetriever.__new__(HybridRetriever)
    r.db = MagicMock(is_connected=True)
    session = r.db.driver.session.return_value.__enter__.return_value
    session.run.return_value = []
    r.query_graph(e, limit=None, strict=True)
    for call in session.run.call_args_list:
        assert 'LIMIT' not in call.args[0]
        assert 'ANY(' in call.args[0]
    reference, _ = reference_query('trial', e.model_dump())
    assert 'ANY(symbol IN $genes' in reference
    assert 'ALL(name IN $drugs' in reference


def test_graph_only_candidate_displaces_dense_and_has_text():
    r = HybridRetriever.__new__(HybridRetriever)
    r.extractor = Mock()
    r.extractor.extract.return_value = QueryEntities(genes=['EGFR'])
    r.query_graph = Mock(return_value=[{'nct_id': 'NCT2'}])
    r.vector_store = Mock()
    r.vector_store.search.return_value = [SearchResult(chunk_id='a', doc_id='NCT1', chunk_type='TRIAL_OVERVIEW', text='other', score=.9)]
    r.vector_store.score_documents.return_value = {'NCT2': .8}
    r.vector_store.get_chunks_by_doc_id.return_value = [dict(chunk_id='b', doc_id='NCT2', chunk_type='TRIAL_OVERVIEW', text='EGFR trial', metadata={})]
    result = r.retrieve('EGFR', top_k=1)
    assert result.fused_ncts == ['NCT2']
    assert result.vector_evidence[0].doc_id == 'NCT2'
    r.query_graph.assert_called_once_with(r.extractor.extract.return_value, limit=None, strict=True)


def test_stored_vector_scoring_uses_best_chunk_and_excludes_other_docs(tmp_path):
    store = LocalVectorStore(tmp_path)
    store.chunks = [{'doc_id': 'A'}, {'doc_id': 'A'}, {'doc_id': 'B'}]
    store.vectors = np.array([[0., 1.], [1., 0.], [1., 0.]])
    store.embedding_provider = Mock()
    store.embedding_provider.embed_query.return_value = [1., 0.]
    assert store.score_documents('q', ['A', 'missing']) == {'A': 1.0}


def test_atomic_bullets_keep_offsets_and_compound_logic_unknown():
    text = '- Age >= 18\n- ECOG <= 1'
    source = {'nct_id': 'NCT1', 'inclusion_criteria': text}
    criteria = extract_criteria(source)
    assert len(criteria) == 2
    for c in criteria:
        assert text[c.start:c.end] == c.text
        assert c.section_text == text
        assert assess_criterion(c, PatientProfile(summary='patient', age_years=30, ecog=0)).assessment == 'satisfied'
    for text in ['- Age >= 18 OR\n- parental consent', '- Age >= 18\n  - unless prior therapy']:
        source['inclusion_criteria'] = text
        assert len(extract_criteria(source)) == 1


def test_judgments_do_not_treat_unknown_as_irrelevant():
    result = metrics(['unknown'], {'A': 3}, exhaustive=False)
    assert result['ndcg'] is None and result['precision'] is None and result['recall'] is None
    result = metrics(['A', 'A', 'B'], {'A': 3, 'B': 1}, exhaustive=True, k=2)
    assert result['ndcg'] == 1 and result['recall'] == 1 and result['precision'] == 1


def test_review_and_topic_leakage_gates():
    data = dict(corpus_id='frozen', split='test', development_topic_ids=['topic'], cases=[
        dict(id='q', topic_id='topic', review_status='reviewed', judgments={'A': 1})], runs={'dense': {'q': ['A']}})
    with pytest.raises(ValueError, match='leakage'):
        evaluate(data)
    data['development_topic_ids'] = []
    assert evaluate(data)['results']['dense']['summary']['ndcg']['mean'] == 1
    data['cases'][0]['review_status'] = 'draft'
    with pytest.raises(ValueError, match='review'):
        evaluate(data)


def test_registry_condition_collisions_do_not_create_extra_constraints():
    e = extractor()
    e.known_conditions |= {'Osimertinib', 'Egfr'}
    result = e.extract('Osimertinib targeting EGFR in Non-Small Cell Lung Cancer')
    assert result.conditions == ['Non-Small Cell Lung Cancer']
    assert result.drugs == ['Osimertinib'] and result.genes == ['EGFR']
    assert e.extract('EGFR or Osimertinib').interpretation_warnings
    assert e.extract('EGFR mutation eligibility').interpretation_warnings


def test_prompt_includes_trial_discovery_and_semantic_limits():
    from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridEvidence
    from clinical_trial_rag_copilot.agent.prompt_templates import format_evidence_context
    evidence = HybridEvidence(query='q', extracted_entities=QueryEntities(),
        graph_evidence=[dict(source='GRAPH_TRIAL_CONSTRAINTS', nct_id='NCT1', title='A', conditions=['Lung Cancer'])],
        vector_evidence=[], fused_ncts=['NCT1'], fused_pmids=[], retrieval_warnings=['Unknown eligibility'])
    context = format_evidence_context(evidence)
    assert 'NCT1' in context and 'Unknown eligibility' in context
    assert 'NEO4J VERIFIED' not in context


def test_pinecone_graph_scores_use_stored_vectors_and_validate_identity():
    from types import SimpleNamespace
    from clinical_trial_rag_copilot.retrieval.vector_store import PineconeVectorStore
    store = PineconeVectorStore(provider=Mock())
    store.embedding_provider.embed_query.return_value = [1., 0.]
    store.index = Mock()
    store.index.list.return_value = [['NCT1_a', 'NCT1_b']]
    store.index.fetch.return_value.vectors = {
        'a': SimpleNamespace(metadata={'doc_id': 'NCT1'}, values=[0., 2.]),
        'b': SimpleNamespace(metadata={'doc_id': 'NCT1'}, values=[2., 0.]),
        'wrong': SimpleNamespace(metadata={'doc_id': 'NCT2'}, values=[2., 0.]),
    }
    assert store.score_documents('q', ['NCT1']) == {'NCT1': 1.0}


def test_no_answer_measure_requires_exhaustive_labels():
    assert metrics([], {'A': 0}, exhaustive=True)['no_answer_correct'] == 1
    assert metrics(['A'], {'A': 0}, exhaustive=True)['no_answer_correct'] == 0
    assert metrics([], {'A': 0})['no_answer_correct'] is None


def test_parenthesized_condition_retains_full_source_identity():
    e = extractor()
    e.known_conditions.add('Non-Small Cell Lung Cancer (Nsclc)')
    assert e.extract('Find Non-Small Cell Lung Cancer (NSCLC).').conditions == ['Non-Small Cell Lung Cancer (Nsclc)']
