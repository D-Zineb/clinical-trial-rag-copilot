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
