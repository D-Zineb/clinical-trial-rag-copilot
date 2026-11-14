import json
import subprocess
from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.data.storage import load_dataset


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.delenv("RAG_DATA_S3_URI", raising=False)


def test_local_directory_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    (tmp_path / "cardiology_trials.json").write_text('[{"nct_id": "NCT1"}]')
    assert load_dataset("cardiology_trials.json") == [{"nct_id": "NCT1"}]


def test_s3_uses_explicit_prefix(monkeypatch):
    monkeypatch.setenv("RAG_DATA_S3_URI", "s3://bucket/snapshot/")
    run = Mock(return_value=Mock(stdout=b'[{"pmid": "123"}]'))
    monkeypatch.setattr(subprocess, "run", run)
    assert load_dataset("pubmed_cardiology_literature.json") == [{"pmid": "123"}]
    assert run.call_args.args[0] == ["aws", "s3", "cp",
        "s3://bucket/snapshot/pubmed_cardiology_literature.json", "-", "--only-show-errors"]


def test_s3_failure_never_falls_back(monkeypatch):
    monkeypatch.setenv("RAG_DATA_S3_URI", "s3://bucket/data")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "aws")))
    with pytest.raises(RuntimeError, match="IAM"):
        load_dataset("cardiology_trials.json")


@pytest.mark.parametrize("value", ['{}', '[1]', 'not-json'])
def test_reject_invalid_dataset(value, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    (tmp_path / "cardiology_trials.json").write_text(value)
    with pytest.raises(ValueError):
        load_dataset("cardiology_trials.json")


def test_reject_unknown_filename():
    with pytest.raises(ValueError):
        load_dataset("../.env")


def test_save_s3_streams_json_without_local_copy(monkeypatch, tmp_path):
    from clinical_trial_rag_copilot.data.storage import save_dataset
    monkeypatch.setenv("RAG_DATA_S3_URI", "s3://bucket/data/")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "unused"))
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    rows = [{"title": "?tude"}]
    assert save_dataset("cardiology_trials.json", rows) == "s3://bucket/data/cardiology_trials.json"
    assert run.call_args.args[0][3:5] == ["-", "s3://bucket/data/cardiology_trials.json"]
    assert json.loads(run.call_args.kwargs["input"]) == rows
    assert not (tmp_path / "unused").exists()


def test_save_failure_does_not_create_local_copy(monkeypatch, tmp_path):
    from clinical_trial_rag_copilot.data.storage import save_dataset
    monkeypatch.setenv("RAG_DATA_S3_URI", "s3://bucket/data")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "unused"))
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "aws")))
    with pytest.raises(RuntimeError, match="write permissions"):
        save_dataset("cardiology_trials.json", [])
    assert not (tmp_path / "unused").exists()


def test_save_local_roundtrip(monkeypatch, tmp_path):
    from clinical_trial_rag_copilot.data.storage import save_dataset
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    save_dataset("cardiology_trials.json", [{"nct_id": "NCT1"}])
    assert load_dataset("cardiology_trials.json") == [{"nct_id": "NCT1"}]


def test_acquisition_routes_all_datasets_through_storage(monkeypatch):
    from clinical_trial_rag_copilot.data import run_acquisition as pipeline
    trial = Mock()
    trial.model_dump.return_value = {"nct_id": "NCT1"}
    monkeypatch.setattr(pipeline, "fetch_cardiology_trials", Mock(return_value=[trial]))
    monkeypatch.setattr(pipeline, "fetch_drug_targets", Mock(return_value=[]))
    monkeypatch.setattr(pipeline, "fetch_pubmed_abstracts", Mock(return_value=[{"pmid": "1"}]))
    save = Mock(return_value="s3://bucket/data")
    monkeypatch.setattr(pipeline, "save_dataset", save)
    pipeline.run_full_pipeline()
    assert [call.args[0] for call in save.call_args_list] == [
        "cardiology_trials.json", "drug_targets_pharmacology.json", "pubmed_cardiology_literature.json"]
    assert save.call_args_list[0].args[1] == [{"nct_id": "NCT1"}]
