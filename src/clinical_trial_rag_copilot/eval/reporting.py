"""Validate, summarize and render known-item evaluation runs without network access."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def validate_run(run):
    rows = run.get("per_query", [])
    if not rows:
        raise ValueError("Evaluation has no query results")
    seen = set()
    for row in rows:
        key = (row["method"], row["case_id"], row["k"])
        if key in seen:
            raise ValueError(f"Duplicate measurement: {key}")
        seen.add(key)
        if row["k"] <= 0 or not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
            raise ValueError("Invalid cutoff or latency")
        if any(not math.isfinite(row[m]) or not 0 <= row[m] <= 1
               for m in ("hit", "reciprocal_rank", "ndcg")):
            raise ValueError("Metrics must be finite and in [0, 1]")
        # Recompute scores from rankings instead of trusting aggregate fields.
        ranking = list(dict.fromkeys(row["ranking"]))[:row["k"]]
        rank = ranking.index(row["target_id"]) + 1 if row["target_id"] in ranking else None
        expected = (int(rank is not None), 1 / rank if rank else 0,
                    1 / math.log2(rank + 1) if rank else 0)
        if any(abs(row[m] - value) > 1e-9 for m, value in
               zip(("hit", "reciprocal_rank", "ndcg"), expected)):
            raise ValueError(f"Metrics disagree with ranking: {key}")
    # Every method/cutoff must cover the same query set.
    groups = defaultdict(set)
    for row in rows:
        groups[(row["method"], row["k"])].add(row["case_id"])
    query_sets = list(groups.values())
    if any(ids != query_sets[0] for ids in query_sets):
        raise ValueError("Incomplete query coverage across methods/cutoffs")


def summarize(run):
    validate_run(run)
    groups = defaultdict(list)
    for row in run["per_query"]:
        groups[(row["method"], row["k"], "all")].append(row)
        groups[(row["method"], row["k"], row["category"])].append(row)
    result = []
    for (method, k, category), rows in sorted(groups.items()):
        result.append(dict(method=method, k=k, category=category, queries=len(rows),
                           hit_rate=float(np.mean([r["hit"] for r in rows])),
                           mrr=float(np.mean([r["reciprocal_rank"] for r in rows])),
                           ndcg=float(np.mean([r["ndcg"] for r in rows])),
                           p50_ms=float(np.percentile([r["latency_ms"] for r in rows], 50)),
                           p95_ms=float(np.percentile([r["latency_ms"] for r in rows], 95))))
    return result


def compare_runs(current, baseline, k):
    validate_run(baseline)
    # Matching questions alone is insufficient for a fair change comparison.
    keys = ("cases_sha256", "corpus_sha256", "candidate_k", "graph_connected", "backend",
            "index", "namespace", "embedding_model")
    mismatches = [key for key in keys if current.get(key) != baseline.get(key)]
    missing = [key for key in ("cases_sha256", "corpus_sha256", "embedding_model")
               if not current.get(key) or not baseline.get(key)]
    if mismatches or missing:
        return {"compatible": False, "reason": f"Changed settings: {mismatches}; missing provenance: {missing}", "deltas": []}
    old = {(r["method"], r["case_id"]): r for r in baseline["per_query"] if r["k"] == k}
    new = {(r["method"], r["case_id"]): r for r in current["per_query"] if r["k"] == k}
    if old.keys() != new.keys():
        return {"compatible": False, "reason": "Different methods or query coverage", "deltas": []}
    deltas = []
    for method in sorted({key[0] for key in new}):
        pairs = [(old[key], new[key]) for key in new if key[0] == method]
        deltas.append(dict(method=method, hit_rate_delta=float(np.mean([b["hit"] - a["hit"] for a, b in pairs])),
                           mrr_delta=float(np.mean([b["reciprocal_rank"] - a["reciprocal_rank"] for a, b in pairs])),
                           improved_queries=[b["case_id"] for a, b in pairs if b["reciprocal_rank"] > a["reciprocal_rank"]],
                           regressed_queries=[b["case_id"] for a, b in pairs if b["reciprocal_rank"] < a["reciprocal_rank"]]))
    return {"compatible": True, "reason": None, "deltas": deltas}


def csv_safe(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    # Avoid spreadsheet formula execution in exported query/document text.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_report(run, output, *, k=5, baseline=None, method="hybrid",
                 min_hit=0.0, min_mrr=0.0, max_p95=None, require_graph=False,
                 max_mrr_drop=None):
    summaries = summarize(run)
    selected = [s for s in summaries if s["k"] == k and s["category"] == "all"]
    target = next((s for s in selected if s["method"] == method), None)
    if target is None:
        raise ValueError(f"No results for method={method}, k={k}")
    comparison = compare_runs(run, baseline, k) if baseline else None
    errors = [r for r in run["per_query"] if r["k"] == k and r.get("error")]
    checks = [dict(name="no_query_errors", passed=not errors, actual=len(errors)),
              dict(name="minimum_hit_rate", passed=target["hit_rate"] >= min_hit, actual=target["hit_rate"], threshold=min_hit),
              dict(name="minimum_mrr", passed=target["mrr"] >= min_mrr, actual=target["mrr"], threshold=min_mrr)]
    if require_graph:
        checks.append(dict(name="graph_connected", passed=run.get("graph_connected") is True))
    if max_p95 is not None:
        checks.append(dict(name="maximum_p95_ms", passed=target["p95_ms"] <= max_p95,
                           actual=target["p95_ms"], threshold=max_p95))
    if baseline:
        checks.append(dict(name="baseline_compatible", passed=comparison["compatible"]))
    if max_mrr_drop is not None:
        delta = next((d for d in (comparison or {}).get("deltas", []) if d["method"] == method), None)
        checks.append(dict(name="maximum_mrr_regression", passed=delta is not None and delta["mrr_delta"] >= -max_mrr_drop,
                           threshold=max_mrr_drop, actual=delta["mrr_delta"] if delta else None))
    warnings = list(run.get("caveats", []))
    if not run.get("graph_connected"):
        warnings.insert(0, "Neo4j unavailable: hybrid results represent fallback behavior, not graph-plus-vector quality.")
    warnings += ["This is a single-target retrieval report, not an answer-quality or clinical-safety certification.",
                 "Precision, exhaustive recall, and Ragas claim scores are not measured by this dataset."]
    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    report = dict(schema_version=1, status=status, gate_method=method, cutoff=k, checks=checks,
                  metadata={key: value for key, value in run.items() if key not in {"per_query", "summaries"}},
                  summaries=summaries, comparison=comparison, warnings=warnings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, records in (("summary.csv", summaries), ("per_query.csv", run["per_query"])):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: csv_safe(value) for key, value in row.items()} for row in records)
    lines = ["# Retrieval evaluation report", "", f"**Gate status: {status}** — method `{method}`, cutoff {k}.", "",
             "Passing means only that the configured thresholds passed. Defaults are permissive; set project thresholds explicitly.", "",
             "## Run", "", f"Source run: {run.get('created_at', 'unknown')}",
             f"Dataset fingerprint: `{run.get('cases_sha256', 'unknown')}`", "",
             f"Graph connected: **{run.get('graph_connected')}**. Candidate budget: {run.get('candidate_k')}. Ranking is within document type.", "",
             f"Git revision: `{run.get('git_revision')}`; dirty working tree: `{run.get('git_dirty')}`.", "",
             f"Index: `{run.get('index')}`; namespace: `{run.get('namespace')}`; embedding model: `{run.get('embedding_model')}`.", "",
             "## Results", "", "| Method | Queries | Hit@K | MRR@K | nDCG@K | p50 ms | p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in selected:
        lines.append(f"| {s['method']} | {s['queries']} | {s['hit_rate']:.1%} | {s['mrr']:.3f} | {s['ndcg']:.3f} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")
    lines += ["", "## By question category", "", "| Method | Category | Queries | Hit@K | MRR@K |", "|---|---|---:|---:|---:|"]
    for s in summaries:
        if s["k"] == k and s["category"] != "all":
            lines.append(f"| {s['method']} | {s['category']} | {s['queries']} | {s['hit_rate']:.1%} | {s['mrr']:.3f} |")
    lines += ["", "## Gates", ""] + [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['name']} ({json.dumps(c)})" for c in checks]
    if comparison:
        lines += ["", "## Baseline comparison", "", json.dumps(comparison, indent=2)]
    lines += ["", "## Queries needing inspection", ""]
    failures = [r for r in run["per_query"] if r["k"] == k and r["method"] == method and r["target_rank"] != 1]
    for row in failures:
        lines += [f"- **{row['case_id']}**: target `{row['target_id']}`, rank {row['target_rank']}. {row['query']}",
                  f"  Returned: {', '.join(row['ranking'][:k])}"]
    if not failures:
        lines.append("Every known target ranked first for the selected method.")
    lines += ["", "## Definitions", "", "Hit@K: known target found within K distinct documents. MRR@K: mean reciprocal target rank. nDCG@K: single-positive discounted rank, not graded clinical relevance. p50/p95: descriptive warm timings from this run.", "",
              "## Limitations", ""] + [f"- {w}" for w in warnings]
    markdown = "\n".join(lines) + "\n"
    (output / "report.md").write_text(markdown, encoding="utf-8")
    # Self-contained HTML, escaped throughout; no remote scripts or data uploads.
    table = "<table><tr><th>Method</th><th>Hit@K</th><th>MRR@K</th><th>nDCG@K</th><th>p95 ms</th></tr>"
    for s in selected:
        table += f"<tr><td>{html.escape(s['method'])}</td><td>{s['hit_rate']:.1%}</td><td>{s['mrr']:.3f}</td><td>{s['ndcg']:.3f}</td><td>{s['p95_ms']:.0f}</td></tr>"
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Retrieval evaluation</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#182638}'
            'table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ccd}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:20px}</style>'
            f'<h1>Retrieval evaluation — {status}</h1>{table}</table><h2>Full report</h2><pre>{html.escape(markdown)}</pre></html>')
    (output / "report.html").write_text(page, encoding="utf-8")
    return report
