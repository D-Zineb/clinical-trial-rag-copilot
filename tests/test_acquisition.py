"""Oncology acquisition uses the existing storage and serving pipeline."""
from unittest.mock import Mock


def test_trial_query_targets_egfr_lung_cancer(monkeypatch):
    from clinical_trial_rag_copilot.data import fetch_trials
    response = Mock(status_code=200)
    response.json.return_value = {'studies': []}
    get = Mock(return_value=response)
    monkeypatch.setattr(fetch_trials.requests, 'get', get)
    assert fetch_trials.fetch_oncology_trials(target_count=2) == []
    params = get.call_args.kwargs['params']
    assert 'Lung Cancer' in params['query.cond']
    assert 'EGFR' in params['query.term']
    assert params['pageSize'] == 2


def test_pubmed_query_targets_egfr_lung_cancer(monkeypatch):
    from clinical_trial_rag_copilot.data import fetch_pubmed
    response = Mock(status_code=200)
    response.json.return_value = {'esearchresult': {'idlist': []}}
    get = Mock(return_value=response)
    monkeypatch.setattr(fetch_pubmed.requests, 'get', get)
    assert fetch_pubmed.fetch_pubmed_abstracts(max_results=2) == []
    term = get.call_args.kwargs['params']['term']
    assert 'EGFR' in term and 'non-small cell lung cancer' in term
    assert 'Atrial Fibrillation' not in term
