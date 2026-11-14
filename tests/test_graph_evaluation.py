from unittest.mock import MagicMock, Mock

import pytest

from clinical_trial_rag_copilot.eval.graph_evaluation import evaluate_case, reference_query, set_metrics
from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, QueryEntities


def test_set_metrics_deduplicate_and_capture_violations():
    scores = set_metrics(['a', 'a', 'b'], ['a', 'c'])
    assert scores['precision'] == scores['recall'] == scores['f1'] == 0.5
    assert scores['constraint_violations'] == ['b']
    assert scores['missed'] == ['c']
    assert set_metrics([], [])['exact_match']
    assert set_metrics([], ['a'])['f1'] == 0
    assert set_metrics(['a'], [])['precision'] == 0


def test_reference_combines_constraints_without_limit():
    query, params = reference_query('trial', {'drugs': ['Amiodarone'], 'genes': ['KCNH2'], 'conditions': ['AF']})
    assert 'LIMIT' not in query
    assert 'TARGETS_CONDITION' in query and 'INHIBITS' in query
    assert params['genes'] == ['KCNH2']
    with pytest.raises(ValueError, match='Demographic'):
        reference_query('trial', {'min_age': 65})


def test_strict_retrieval_distinguishes_errors_from_empty():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.db = Mock(is_connected=False)
    assert retriever.query_graph(QueryEntities()) == []
    with pytest.raises(RuntimeError):
        retriever.query_graph(QueryEntities(), strict=True)
    retriever.db = Mock(is_connected=True)
    retriever.db.driver.session.side_effect = RuntimeError('query failed')
    with pytest.raises(RuntimeError, match='query failed'):
        retriever.query_graph(QueryEntities(genes=['KCNH2']), strict=True)


def test_modes_separate_extraction_misses_and_execution_errors():
    retriever = Mock()
    session = Mock()
    retriever.db.driver.session = MagicMock()
    retriever.db.driver.session.return_value.__enter__.return_value = session
    session.run.return_value = [{'id': 'NCT1'}]
    retriever.extractor.extract.return_value = QueryEntities()
    retriever.query_graph.side_effect = [[{'nct_id': 'NCT1'}], RuntimeError('offline')]
    rows = evaluate_case({'id': 'q', 'kind': 'trial', 'query': 'Amiodarone',
                          'entities': {'drugs': ['Amiodarone']}}, retriever, 100)
    assert rows[0]['metrics']['exact_match']
    assert rows[1]['metrics'] is None
    assert rows[1]['entity_metrics']['drugs']['recall'] == 0
    assert rows[1]['error'] == 'RuntimeError: offline'
