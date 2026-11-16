"""Source-bound ablation of dense search, flat relations and live Neo4j.

This tests explicit source predicates, NOT clinical eligibility or efficacy.
Cases are frozen before retrieval. All arms share document scores and budgets.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    """Benchmark identity: exact case-insensitive name, optional type prefix.

    Deliberately does not import the production dosage/salt normalizer. Different
    identities or ambiguous aliases remain a possible graph/source disagreement.
    """
    return re.sub(r"^(drug|biological)\s*:\s*", "", value.strip(), flags=re.I).casefold()


def source_tables(trials, drugs):
    targets = defaultdict(set)
    for record in drugs:
        for target in record.get("targets", []):
            if target.get("gene_symbol"):
                targets[canonical(record["drug_name"])].add(target["gene_symbol"].strip().upper())
    rows = {}
    for trial in trials:
        names = {
            canonical(name)
            for arm in trial.get("arm_groups", [])
            for name in arm.get("intervention_names", [])
            if re.match(r"^(Drug|Biological)\s*:", name, re.I)
        }
        rows[trial["nct_id"]] = {
            "drugs": names,
            "conditions": {c.strip().casefold() for c in trial.get("conditions", [])},
            "drug_genes": {drug: targets.get(drug, set()) for drug in names},
        }
    return rows, targets


def flat_matches(row, entities):
    """Execute the declared source predicates against flattened trial records."""
    if entities.get('interpretation_warnings'):
        return False
    if not any(entities.get(field) for field in ('drugs', 'genes', 'conditions')):
        return False
    operators = entities.get('operators', {})
    def matches(field, predicates):
        predicates = list(predicates)
        return not predicates or (any(predicates) if operators.get(field) == 'any' else all(predicates))
    requested_drugs = {canonical(d) for d in entities.get('drugs', [])}
    if not matches('drugs', (d in row['drugs'] for d in requested_drugs)):
        return False
    if not matches('conditions', (c.casefold() in row['conditions'] for c in entities.get('conditions', []))):
        return False
    eligible_drugs = row['drugs'] & requested_drugs if requested_drugs else row['drugs']
    return matches('genes', (any(g in row['drug_genes'].get(d, set()) for d in eligible_drugs)
                              for g in entities.get('genes', [])))


def raw_reference(trials, targets, entities):
    """Gold source joins independently enumerate arms, drugs and target rows.

    This is exhaustive under the stated identity rules, but not an independently
    reviewed clinical judgment. It never reads Neo4j or retrieval results.
    """
    output = []
    for trial in trials:
        if any(c.casefold() not in {v.strip().casefold() for v in trial.get("conditions", [])}
               for c in entities.get("conditions", [])):
            continue
        arm_names = [canonical(name) for arm in trial.get("arm_groups", [])
                     for name in arm.get("intervention_names", [])
                     if re.match(r"^(Drug|Biological)\s*:", name, re.I)]
        requested = [canonical(v) for v in entities.get("drugs", [])]
        if any(v not in arm_names for v in requested):
            continue
        if any(not any(gene in targets.get(drug, set()) for drug in (requested or arm_names))
               for gene in entities.get("genes", [])):
            continue
        output.append(trial["nct_id"])
    return sorted(set(output))


def build_cases(trials, drugs, per_category=8):
    """Source-derived development sample; selection never observes retrieval."""
    rows, targets = source_tables(trials, drugs)
    condition_names = {c.strip().casefold(): c.strip().title() for t in trials for c in t.get("conditions", [])}
    options = {}
    for row in rows.values():
        for drug in row["drugs"] & targets.keys():
            for gene in row["drug_genes"][drug]:
                candidates = [("gene", {"genes": [gene]}),
                              ("drug_gene", {"genes": [gene], "drugs": [drug.title()]})]
                for condition in row["conditions"]:
                    candidates += [
                        ("condition_gene", {"genes": [gene], "conditions": [condition_names[condition]]}),
                        ("condition_drug_gene", {"genes": [gene], "drugs": [drug.title()], "conditions": [condition_names[condition]]}),
                    ]
                for category, entities in candidates:
                    key = json.dumps(entities, sort_keys=True)
                    options[(category, key)] = entities
    # Compute support using a flat scan only to stratify source-derived cases.
    groups = defaultdict(list)
    for (category, key), entities in options.items():
        ids = sorted(nct for nct, row in rows.items() if flat_matches(row, entities))
        if 2 <= len(ids) <= 80:
            groups[category].append((hashlib.sha256(key.encode()).hexdigest(), entities))
    cases = []
    for category in sorted(groups):
        for _, entities in sorted(groups[category])[:per_category]:
            expected = raw_reference(trials, targets, entities)
            drugs_text = " with study-arm drug " + " and ".join(entities["drugs"]) if entities.get("drugs") else " with a study-arm drug"
            condition_text = " targeting condition " + " and ".join(entities["conditions"]) if entities.get("conditions") else ""
            query = f"Find trials{drugs_text} having a pharmacology target record for {' and '.join(entities['genes'])}{condition_text}."
            cases.append({"id": f"{category}_{len(cases)+1:02d}", "category": category,
                          "query": query, "entities": entities, "expected_ids": expected,
                          "label_type": "exhaustive_source_predicate_not_clinical_eligibility"})
    return cases


def ranking_metrics(ranking, relevant, k=10):
    relevant = set(relevant)
    ranking = list(dict.fromkeys(ranking))
    top = ranking[:k]
    hits = [int(nct in relevant) for nct in top]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"ndcg10": dcg / ideal if ideal else None,
            "precision10": sum(hits) / k,
            "recall100": len(set(ranking[:100]) & relevant) / len(relevant) if relevant else None}


def bootstrap_interval(deltas):
    rng = np.random.default_rng(20261116)
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return [None, None]
    draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("eval/graph_value_runs"))
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--cases", type=Path, help="Reuse frozen cases without reselection")
    args = parser.parse_args(argv)
    if args.per_category < 1 or args.candidate_k < 100:
        parser.error("per-category must be positive and candidate-k at least 100")
    snapshot = args.snapshot
    out = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    trial_path = snapshot / "datasets/oncology_trials.json"
    target_path = snapshot / "datasets/drug_targets_pharmacology.json"
    trials = json.loads(trial_path.read_text())
    drugs = json.loads(target_path.read_text())
    chunks = [json.loads(line) for line in (snapshot / "corpus.jsonl").read_text().splitlines()]
    vectors = np.load(snapshot / "vectors.npy", allow_pickle=False)
    verified = json.loads((snapshot / "pinecone_verified.json").read_text())
    if digest(snapshot / "vectors.npy") != verified["vectors_sha256"] or len(chunks) != len(vectors):
        raise ValueError("Embedding snapshot integrity mismatch")
    # Verify every row against its text-keyed cache, binding vector order to text.
    cache = {r["hash"]: r["values"] for r in map(json.loads, (snapshot / "embedding_cache.jsonl").read_text().splitlines())}
    for chunk, vector in zip(chunks, vectors):
        key = hashlib.sha256(chunk["text"].encode()).hexdigest()
        if not np.array_equal(vector, np.asarray(cache[key], dtype=np.float32)):
            raise ValueError("Chunk/vector row mismatch")
    from clinical_trial_rag_copilot.retrieval.chunking import build_all_chunks
    papers = json.loads((snapshot / "datasets/pubmed_oncology_literature.json").read_text())
    rebuilt = {c.chunk_id: c.text for c in build_all_chunks(trials, papers)}
    if rebuilt != {c["chunk_id"]: c["text"] for c in chunks}:
        raise ValueError("Source records do not reproduce embedding corpus")
    rows, targets = source_tables(trials, drugs)
    cases = ([json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
             if args.cases else build_cases(trials, drugs, args.per_category))
    if not cases:
        raise ValueError("No qualifying source cases")
    for case in cases:
        if raw_reference(trials, targets, case["entities"]) != case["expected_ids"]:
            raise ValueError("Frozen labels disagree with source snapshot")
    # Freeze labels and protocol before any model loading or database query.
    case_path = out / "cases.jsonl"
    case_path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_sha256": digest(case_path), "trials_sha256": digest(trial_path),
        "targets_sha256": digest(target_path), "vectors_sha256": digest(snapshot / "vectors.npy"),
        "embedding_model": verified["embedding_model"], "candidate_k": args.candidate_k,
        "selection": "sha256-ordered source predicates, support 2..80, up to 8 per category; development only",
        "primary_comparison": "neo4j_oracle minus flat_oracle nDCG@10",
        "secondary_comparison": "neo4j_oracle minus dense nDCG@10",
        "fixed_ranking": "same max cosine chunk score per trial; same candidate budget; no LLM",
        "label_scope": "source-record relations, not clinical truth or patient eligibility",
        "graph_writes": False,
        "runner_sha256": digest(Path(__file__)),
        "retriever_sha256": digest(Path(__file__).parent.parent / "retrieval/hybrid_search.py"),
        "protocol_version": 3,
        "amendment": "Post-fix development evaluation; original labels frozen; production hybrid replay added with mixed-chunk dense candidates and complete graph enumeration.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    from fastembed import TextEmbedding
    from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, ClinicalEntityExtractor, QueryEntities
    from clinical_trial_rag_copilot.graph.connection import get_db
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.db = get_db()
    retriever.extractor = ClinicalEntityExtractor()
    if not retriever.db.is_connected:
        raise RuntimeError("Live graph required; no fallback graph scores")
    model = TextEmbedding(model_name=verified["embedding_model"], threads=2)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore
    store = LocalVectorStore(out / 'replay_store')
    store.chunks, store.vectors = chunks, vectors
    class CachedQueryProvider:
        def embed_query(self, query):
            return query_vector
    store.embedding_provider = CachedQueryProvider()
    retriever.vector_store = store
    output = []
    interpretations = []
    try:
        for case in cases:
            start = time.perf_counter()
            query_vector = np.asarray(next(iter(model.query_embed(case["query"]))), dtype=np.float32)
            query_vector /= max(np.linalg.norm(query_vector), 1e-12)
            chunk_scores = vectors @ query_vector
            scores = defaultdict(lambda: -float("inf"))
            for chunk, score in zip(chunks, chunk_scores):
                if chunk["doc_id"] in rows:
                    scores[chunk["doc_id"]] = max(scores[chunk["doc_id"]], float(score))
            order = sorted(scores, key=lambda nct: (-scores[nct], nct))
            shared_ms = (time.perf_counter() - start) * 1000
            timings = {"dense": 0.0}
            sets = {"dense": set(order)}
            start = time.perf_counter()
            sets["flat_oracle"] = {nct for nct, row in rows.items() if flat_matches(row, case["entities"])}
            timings["flat_oracle"] = (time.perf_counter() - start) * 1000
            if sets["flat_oracle"] != set(case["expected_ids"]):
                raise ValueError("Independent raw reference and flat query disagree")
            start = time.perf_counter()
            extracted = retriever.extractor.extract(case["query"])
            extraction_ms = (time.perf_counter() - start) * 1000
            interpretations.append({"case_id": case["id"], "supplied": case["entities"],
                                    "extracted": extracted.model_dump()})
            start = time.perf_counter()
            sets["flat_extracted"] = {nct for nct, row in rows.items() if flat_matches(row, extracted.model_dump())}
            timings["flat_extracted"] = (time.perf_counter() - start) * 1000 + extraction_ms
            for method in ["neo4j_oracle", "neo4j_extracted"]:
                start = time.perf_counter()
                entities = QueryEntities(**case["entities"]) if method == "neo4j_oracle" else extracted
                graph_items = retriever.query_graph(entities, limit=len(trials)+1, strict=True)
                sets[method] = {item["nct_id"] for item in graph_items if item.get("nct_id")}
                timings[method] = (time.perf_counter() - start) * 1000
                if method == "neo4j_extracted":
                    timings[method] += extraction_ms
            # Diagnose the current append-only fusion at equal unique-trial budget.
            sets["append_only_hybrid"] = set(order[:args.candidate_k]) | sets["neo4j_extracted"]
            timings["append_only_hybrid"] = timings["neo4j_extracted"]
            for method, ids in sets.items():
                if method == "append_only_hybrid":
                    ranking = order[:args.candidate_k] + [nct for nct in order if nct in ids and nct not in order[:args.candidate_k]]
                else:
                    ranking = [nct for nct in order if nct in ids]
                ranking = ranking[:args.candidate_k]
                output.append({"case_id": case["id"], "category": case["category"], "method": method,
                               "ranking": ranking, "expected_count": len(case["expected_ids"]),
                               "source_predicate_disagreements": sorted(ids.symmetric_difference(case["expected_ids"])) if method.startswith("neo4j") else [],
                               "shared_dense_ms": shared_ms, "additional_work_ms": timings[method],
                               **ranking_metrics(ranking, case["expected_ids"])})
            production_start = time.perf_counter()
            production = retriever.retrieve(case['query'], top_k=args.candidate_k, candidate_k=args.candidate_k)
            output.append({'case_id': case['id'], 'category': case['category'], 'method': 'production_hybrid',
                           'ranking': production.fused_ncts, 'expected_count': len(case['expected_ids']),
                           'warnings': production.retrieval_warnings,
                           'shared_dense_ms': shared_ms,
                           'additional_work_ms': (time.perf_counter() - production_start) * 1000,
                           **ranking_metrics(production.fused_ncts, case['expected_ids'])})
            print(case["id"], flush=True)
            (out / "measurements.json").write_text(json.dumps(output, indent=2))
            (out / "query_interpretations.json").write_text(json.dumps(interpretations, indent=2))
    finally:
        retriever.db.close()
    summaries = {}
    for method in sorted({r["method"] for r in output}):
        data = [r for r in output if r["method"] == method]
        summaries[method] = {key: float(np.mean([r[key] for r in data])) for key in ["ndcg10", "precision10", "recall100"]}
        summaries[method]["additional_work_p50_ms"] = float(np.median([r["additional_work_ms"] for r in data]))
    per_method = {method: {r["case_id"]: r for r in output if r["method"] == method} for method in summaries}
    comparisons = {}
    for baseline in ["dense", "flat_oracle"]:
        deltas = [per_method["neo4j_oracle"][c["id"]]["ndcg10"] - per_method[baseline][c["id"]]["ndcg10"] for c in cases]
        comparisons[baseline] = {"mean_ndcg10_delta": float(np.mean(deltas)),
                                 "descriptive_bootstrap_95_interval": bootstrap_interval(deltas)}
    from clinical_trial_rag_copilot.retrieval.trial_matching import PatientProfile, match_trial
    reviews = [match_trial(t, PatientProfile(summary="Synthetic profile; unspecified biomarker and treatment history", age_years=62, ecog=1)) for t in trials]
    criteria = [c for review in reviews for c in review.criteria]
    coverage = {"sections": len(criteria), "numeric_sections_assessed": sum(c.assessment != "unknown" for c in criteria),
                "complete_trial_assessments": sum(r.assessment_complete for r in reviews),
                "clinical_correctness_evaluated": False}
    report = {"scope": manifest["label_scope"], "case_count": len(cases), "summaries": summaries,
              "comparisons": comparisons, "clinical_matching_coverage": coverage,
              "conclusion": "No unique graph benefit established unless Neo4j improves on flat structured control; source-predicate success is not clinical eligibility validation.",
              "limitations": ["Assistant-designed source-derived development cases; no independent clinical review or held-out clinical set.",
                              "Correct structured query entities supplied to oracle arms; extracted arm shows query interpretation errors.",
                              "Flat predicates and labels encode the same declared relation semantics; this is a contract ablation, not learned clinical relevance.",
                              "Same snapshot cosine scores computed locally for all trials; not a Pinecone network or production latency comparison.",
                              "All 769 trial scores and full structured matches are considered before limiting final candidates to 100; work budgets are disclosed, not claimed equal.",
                              "Bootstrap intervals describe this convenience sample; related cases are not independent clinical topics.",
                              "Drug target records do not imply inhibition, efficacy, resistance or patient eligibility."]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Graph value ablation", "", f"Cases: {len(cases)}. Source-predicate development benchmark, not clinical eligibility.", "",
             "| Method | nDCG@10 | P@10 | Recall@100 | Extra p50 ms |", "|---|---:|---:|---:|---:|"]
    for method, values in summaries.items():
        lines.append(f"| {method} | {values['ndcg10']:.3f} | {values['precision10']:.3f} | {values['recall100']:.3f} | {values['additional_work_p50_ms']:.1f} |")
    lines += ["", "## Interpretation", "", report["conclusion"], "", "## Limitations", ""] + ["- " + item for item in report["limitations"]]
    lines += ["", "## Clinical assessment coverage", "", json.dumps(coverage)]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
