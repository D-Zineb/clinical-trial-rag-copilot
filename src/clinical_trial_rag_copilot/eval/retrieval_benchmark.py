"""Read-only known-item retrieval evaluation; no answer-generation calls."""
from clinical_trial_rag_copilot.data.storage import load_dataset
import argparse
import hashlib
import json
import math
import re
import time
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from clinical_trial_rag_copilot.retrieval.chunking import build_all_chunks
from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever
from clinical_trial_rag_copilot.retrieval.vector_store import DATA_DIR, get_vector_store


def unique(values):
    return list(dict.fromkeys(values))


def known_item_metrics(ranking, target, k):
    ranking = unique(ranking)[:k]
    rank = ranking.index(target) + 1 if target in ranking else None
    return {"hit": int(rank is not None), "reciprocal_rank": 1 / rank if rank else 0,
            "ndcg": 1 / math.log2(rank + 1) if rank else 0, "target_rank": rank}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/retrieval_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=["bm25", "dense", "graph", "hybrid"],
                        default=["bm25", "dense", "graph", "hybrid"])
    args = parser.parse_args(argv)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        parser.error("Cases must be nonempty and have unique IDs")
    if any(c.get("kind") not in {"trial", "paper"} for c in cases):
        parser.error("Each case must specify trial or paper kind")
    if args.candidate_k < 10:
        parser.error("candidate-k must be at least 10")
    trials = load_dataset("cardiology_trials.json")
    papers = load_dataset("pubmed_cardiology_literature.json")
    chunks = build_all_chunks(trials, papers)
    corpus_ids = {c.doc_id for c in chunks}
    if any(c["target_id"] not in corpus_ids for c in cases):
        raise ValueError("Some targets are absent from the source corpus.")
    tokenize = lambda text: re.findall(r"\w+", text.lower())
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    store = get_vector_store()
    retriever = HybridRetriever(vector_store=store)
    # Warm model and network outside the measured query loop.
    if set(args.methods).intersection({"dense", "hybrid"}):
        store.load()
        store.search("cardiology", top_k=1)
    rows = []
    for case in cases:
        print(f"Evaluating {case['id']}", flush=True)
        for method in args.methods:
            start = time.perf_counter()
            chunk_ids = []
            error = None
            try:
                if method == "bm25":
                    scores = bm25.get_scores(tokenize(case["query"]))
                    indices = np.argsort(-scores, kind="stable")[:args.candidate_k]
                    doc_ids = [chunks[i].doc_id for i in indices if scores[i] > 0]
                    chunk_ids = [chunks[i].chunk_id for i in indices if scores[i] > 0]
                elif method == "dense":
                    hits = store.search(case["query"], top_k=args.candidate_k)
                    doc_ids = [hit.doc_id for hit in hits]
                    chunk_ids = [hit.chunk_id for hit in hits]
                elif method == "graph":
                    items = retriever.query_graph(retriever.extractor.extract(case["query"]),
                                                  limit=args.candidate_k * 2)
                    doc_ids = [str(item.get("nct_id") or item.get("pmid")) for item in items
                               if item.get("nct_id") or item.get("pmid")]
                else:
                    evidence = retriever.retrieve(case["query"], top_k=args.candidate_k)
                    doc_ids = evidence.fused_ncts if case["kind"] == "trial" else evidence.fused_pmids
                    chunk_ids = [hit.chunk_id for hit in evidence.vector_evidence]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                doc_ids, chunk_ids = [], []
            latency = (time.perf_counter() - start) * 1000
            # Evaluate rankings within the requested document type. Candidate
            # retrieval remains mixed-type, matching current production behavior.
            ranking = unique(d for d in doc_ids if d.startswith("NCT") == (case["kind"] == "trial"))
            for k in (1, 3, 5, 10):
                rows.append({"case_id": case["id"], "category": case["category"],
                             "query": case["query"], "target_id": case["target_id"],
                             "method": method, "k": k, "error": error, **known_item_metrics(ranking, case["target_id"], k),
                             "ranking": ranking, "chunk_ids": chunk_ids, "latency_ms": latency,
                             "duplicate_fraction": 1 - len(set(doc_ids)) / len(doc_ids) if doc_ids else 0})
    summaries = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["k"], "all")].append(row)
        groups[(row["method"], row["k"], row["category"])].append(row)
    for (method, k, category), group in groups.items():
        summaries.append(dict(method=method, k=k, category=category, queries=len(group),
                              hit_rate=float(np.mean([r["hit"] for r in group])),
                              mrr=float(np.mean([r["reciprocal_rank"] for r in group])),
                              ndcg=float(np.mean([r["ndcg"] for r in group])),
                              latency_p50_ms=float(np.percentile([r["latency_ms"] for r in group], 50)),
                              latency_p95_ms=float(np.percentile([r["latency_ms"] for r in group], 95)),
                              duplicate_fraction=float(np.mean([r["duplicate_fraction"] for r in group]))))
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_revision, git_dirty = None, None
    result = dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                  git_revision=git_revision, git_dirty=git_dirty, python_version=sys.version,
                  corpus_sha256=hashlib.sha256(json.dumps([c.model_dump() for c in chunks], sort_keys=True).encode()).hexdigest(),
                  cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  chunks=len(chunks), documents=len(corpus_ids), candidate_k=args.candidate_k,
                  graph_connected=retriever.db.is_connected, backend=type(store).__name__,
                  index=getattr(store, "index_name", None), namespace=getattr(store, "namespace", None),
                  embedding_model=getattr(getattr(store.embedding_provider, "model", None), "model_name", None),
                  caveats=["Default dataset is a convenience sample, not a representative clinical benchmark.",
                           "Single source-backed target per query; other documents unjudged.",
                           "Hit rate is known-item success, not exhaustive clinical recall.",
                           "NDCG assumes one relevant target; no graded clinical labels.",
                           "Rankings are evaluated within requested document type.",
                           "Hybrid has additional graph candidates and sibling fetch costs.",
                           "Graph candidates currently have no explicit relevance ORDER BY; rank metrics are diagnostic.",
                           "Graph query exceptions can be swallowed by the production retriever; graph connectivity alone does not prove query success.",
                           "One sequential warm run; latency is descriptive, not a load test."],
                  summaries=summaries, per_query=rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retrieval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    retriever.db.close()
    for s in summaries:
        if s["category"] == "all":
            print(f"{s['method']:8} @{s['k']:2} hit={s['hit_rate']:.3f} MRR={s['mrr']:.3f} nDCG={s['ndcg']:.3f} p50={s['latency_p50_ms']:.0f}ms")


if __name__ == "__main__":
    main()
