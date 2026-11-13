# Evaluation workflow

Implementation lives in `src/clinical_trial_rag_copilot/eval/`. This directory holds
versioned test questions, review annotations and locally generated run artifacts.

## Run retrieval evaluation

From the project root:

```powershell
uv run python -m clinical_trial_rag_copilot.eval.run --require-graph --min-hit 0.9 --min-mrr 0.8
```

The 0.9/0.8 thresholds are example development gates, not clinical acceptance criteria.
The runner evaluates BM25, Pinecone dense retrieval, Neo4j graph retrieval and hybrid
retrieval. It reads existing data; it does not ingest or modify database records.
Source JSON datasets must exist locally and cloud services must be configured.
It runs real retrieval code, without answer-generation or evaluator-LLM calls.

Each execution creates a new `eval/runs/<UTC timestamp>_<ID>/` directory containing:

| Artifact | Purpose |
|---|---|
| `report.html` | Standalone report viewable in a browser |
| `report.md` | Metrics, category breakdowns, gates, missed targets, limitations |
| `report.json` | Structured summary, provenance, gates and comparisons |
| `summary.csv` | Aggregated metrics for all methods, categories and cutoffs |
| `per_query.csv` | Every ranking, target, latency and query error |
| `retrieval_metrics.json` | Raw measurements, dataset/corpus fingerprints, runtime and git metadata |
| `failure.json` | Fatal setup/validation error, only if the run cannot complete |

Exit codes: **0** configured checks passed; **1** a quality/availability check failed;
**2** execution or input validation failed. With no quality thresholds, defaults are
permissive; a default PASS does not certify quality. Individual retrieval failures
are recorded and fail the report gate. Fatal setup failures create a failure artifact.
The existing graph retriever catches some database errors internally, so graph
connectivity is not proof that every Cypher query completed; the report flags this limit.

All metrics are recomputed/validated against raw rankings before reporting. A
malformed, incomplete or contradictory result file fails validation.

## Compare with a baseline

```powershell
uv run python -m clinical_trial_rag_copilot.eval.run --baseline eval/runs/PREVIOUS_RUN/retrieval_metrics.json --max-mrr-drop 0.02 --require-graph
```

Comparison requires matching question/corpus fingerprints, candidate budget, graph
availability, embedding model, backend, index, namespace, methods and query coverage.
It lists improved/regressed queries and changes in Hit@K and MRR@K. An incompatible
baseline fails the comparison check instead of presenting an apples-to-oranges delta.
Older experimental runs missing corpus/model provenance can be rendered, but cannot
serve as strict comparison baselines. Source hashes do not prove that both remote
indexes contain identical data; keep ingestion versions synchronized separately.

## Rebuild a report without database access

```powershell
uv run python -m clinical_trial_rag_copilot.eval.run --input eval/runs/PREVIOUS_RUN/retrieval_metrics.json
```

This uses saved measurements; it does not rerun retrieval. Reports retain the source
measurement timestamp. Run folders are ignored by git; preserve selected reports as
CI artifacts or in experiment storage. Test cases and reviewed labels remain tracked.

## What is measured

- Known-target Hit@1/3/5/10, reciprocal rank and single-positive nDCG.
- Per-category scores, particularly paraphrases versus exact-title controls.
- Warm retrieval latency p50/p95, including context expansion for hybrid.
- Graph connectivity and known query failures.

The default dataset is 52 convenience-sample queries, with one known target
each. Other retrieved documents are unjudged. These scores do not measure exhaustive
clinical recall, reliable precision, graded usefulness, answer correctness, or
faithfulness. Graph results currently have no relevance-based ORDER BY; graph-only
rank scores are diagnostic. Candidate budgets differ between graph and vector;
rankings are evaluated within requested document type. One sequential run is not
a load test or a statistically representative latency study.

`context_pilot.py` captures reference answers and supporting chunks.
`ollama_context_recall.py` separately judges saved contexts with Ragas when Ollama
and Ragas are installed. No Ragas score is fabricated when that setup is absent.
Draft relevance annotations are not treated as approved labels by this workflow.

## Test evaluation infrastructure

```powershell
uv run pytest tests/test_evaluation_reporting.py tests/test_retrieval_metrics.py -v
```

These offline tests verify ranking arithmetic, report exports, HTML escaping,
baseline compatibility, failing gates, and command exit codes. They do not claim
to validate live retrieval quality. CI also uploads the full suite's JUnit results.

## Dataset version 2

52 questions: 22 trial paraphrases, 22 paper paraphrases, and 8 exact-title controls.
The targets cover 44 distinct source documents (22 trials and 22 papers). The original
24 questions are unchanged and preserved in `retrieval_cases_v1_24.jsonl`. New questions
were written from source titles before retrieval. These are known-item labels, not
clinician-reviewed exhaustive relevance judgments. The source records remain a
convenience sample rather than a random or representative clinical sample.
Reports from the 24- and 52-query datasets are not strict compatible baselines.
