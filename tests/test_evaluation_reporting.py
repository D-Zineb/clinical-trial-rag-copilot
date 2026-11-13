import copy
import json
import math

import pytest

from clinical_trial_rag_copilot.eval.reporting import compare_runs, write_report
from clinical_trial_rag_copilot.eval.run import main


@pytest.fixture
def run():
    return dict(cases_sha256="cases", corpus_sha256="corpus", embedding_model="model",
                candidate_k=10, graph_connected=False, backend="PineconeVectorStore", index="test",
                namespace="test", per_query=[dict(method="hybrid", case_id="q1", k=5,
                category="trial", query="<script>alert(1)</script>", target_id="B",
                ranking=["A", "B"], hit=1, reciprocal_rank=0.5, ndcg=1 / math.log2(3),
                target_rank=2, latency_ms=100)])


def test_report_gates_exports_and_escaping(run, tmp_path):
    report = write_report(run, tmp_path, min_mrr=0.6, require_graph=True)
    assert report["status"] == "FAIL"
    assert report["summaries"][0]["mrr"] == 0.5
    assert "fallback" in " ".join(report["warnings"])
    assert "<script>" not in (tmp_path / "report.html").read_text()
    for name in ("report.json", "summary.csv", "per_query.csv", "report.md", "report.html"):
        assert (tmp_path / name).is_file()


def test_baseline_rejects_changed_corpus_and_missing_provenance(run):
    changed = copy.deepcopy(run)
    changed["corpus_sha256"] = "different"
    assert not compare_runs(changed, run, 5)["compatible"]
    del changed["corpus_sha256"]
    assert not compare_runs(changed, run, 5)["compatible"]
    assert compare_runs(run, run, 5)["deltas"][0]["mrr_delta"] == 0


def test_report_rejects_fabricated_metrics_and_duplicate_rows(run, tmp_path):
    run["per_query"][0]["reciprocal_rank"] = 1
    with pytest.raises(ValueError, match="disagree"):
        write_report(run, tmp_path)
    run["per_query"][0]["reciprocal_rank"] = 0.5
    run["per_query"].append(run["per_query"][0].copy())
    with pytest.raises(ValueError, match="Duplicate"):
        write_report(run, tmp_path)


def test_cli_replay_exit_status_and_invalid_input(run, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(run))
    args = ["--input", str(source), "--output-root", str(tmp_path / "runs")]
    assert main(args + ["--min-mrr", "0.4"]) == 0
    assert main(args + ["--min-mrr", "0.9"]) == 1
    source.write_text('{}')
    assert main(args) == 2
    assert len(list((tmp_path / "runs").glob("*/failure.json"))) == 1


def test_query_errors_fail_even_with_permissive_thresholds(run, tmp_path):
    run["per_query"][0]["error"] = "Network request failed"
    assert write_report(run, tmp_path)["status"] == "FAIL"


def test_baseline_regression_gate(run, tmp_path):
    baseline = copy.deepcopy(run)
    baseline["per_query"][0].update(ranking=["B"], reciprocal_rank=1, ndcg=1, target_rank=1)
    report = write_report(run, tmp_path, baseline=baseline, max_mrr_drop=0.1)
    assert report["status"] == "FAIL"
    assert report["comparison"]["deltas"][0]["regressed_queries"] == ["q1"]
