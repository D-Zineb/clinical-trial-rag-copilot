import json
from unittest.mock import MagicMock, patch
import pytest
from clinical_trial_rag_copilot.graph.prepare_rebuild import prepare, apply


def test_preparation_is_offline_and_preserves_source_identity(tmp_path):
    snapshot = tmp_path / 'snapshot'
    data = snapshot / 'datasets'
    data.mkdir(parents=True)
    records = {'oncology_trials.json': [{'nct_id': 'NCT1', 'arm_groups': [{'label': 'a', 'intervention_names': ['Drug: A, B']}]}],
               'drug_targets_pharmacology.json': [{'drug_name': 'A, B', 'targets': [{'gene_symbol': 'EGFR'}]}],
               'pubmed_oncology_literature.json': [{'pmid': '1', 'abstract': 'A, B acts on EGFR.'}]}
    for name, rows in records.items():
        (data / name).write_text(json.dumps(rows))
    output = tmp_path / 'package.json'
    prepare(snapshot, output)
    package = json.loads(output.read_text())
    assert len(package['source_hashes']) == 3
    assert len(package['statements']) == 3
    assert package['statements'][0]['parameters']['batch'][0]['drugs'] == ['A, B']
    assert 'HAS_TARGET' in package['statements'][1]['query']
    with pytest.raises(ValueError, match='overwrite'):
        prepare(snapshot, output)


def test_apply_refuses_populated_target_without_writes(tmp_path):
    package = tmp_path / 'package.json'
    package.write_text(json.dumps({'identity_contract': 'exact-name-v2'}))
    db = MagicMock(is_connected=True)
    session = db.driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {'count': 1}
    with patch('clinical_trial_rag_copilot.graph.connection.get_db', return_value=db):
        with pytest.raises(RuntimeError, match='populated'):
            apply(package, tmp_path / 'receipt.json')
    session.execute_write.assert_not_called()
    assert session.run.call_count == 1
