"""Create deterministic known-item smoke cases for the current corpus.

These title controls check indexing, not clinical relevance or paraphrase quality.
Existing reviewed benchmarks are never overwritten by default.
"""
import argparse
import hashlib
import json
from pathlib import Path

from clinical_trial_rag_copilot.data.storage import load_dataset
from clinical_trial_rag_copilot.retrieval.chunking import build_all_chunks


def prepare_cases(trials, papers, per_kind=20):
    if per_kind < 1:
        raise ValueError("per_kind must be positive")
    chunks = build_all_chunks(trials, papers)
    ids = {c.doc_id for c in chunks}
    cases = []
    for kind, records, id_key, title_key in (
        ("trial", trials, "nct_id", "brief_title"),
        ("paper", papers, "pmid", "title"),
    ):
        candidates = {}
        for record in records:
            doc_id = str(record.get(id_key) or "")
            title = (record.get(title_key) or "").strip()
            if doc_id in ids and title:
                candidates[doc_id] = title
        # Fixed hash order avoids API ordering dependence and first-page sampling.
        ordered = sorted(candidates, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
        if not ordered:
            raise ValueError(f"No indexable {kind} documents with titles")
        for doc_id in ordered[:per_kind]:
            title = candidates[doc_id]
            cases.append(dict(id=f"{kind}_snapshot_{doc_id}", kind=kind,
                category=f"{kind}_title_control", query=title, target_id=doc_id,
                source_title=title, label_type="generated exact-title smoke control; not reviewed clinical relevance"))
    corpus_hash = hashlib.sha256(json.dumps(
        sorted((c.model_dump() for c in chunks), key=lambda c: c["chunk_id"]),
        sort_keys=True).encode()).hexdigest()
    return cases, corpus_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=20)
    args = parser.parse_args(argv)
    cases, fingerprint = prepare_cases(load_dataset("cardiology_trials.json"),
                                      load_dataset("pubmed_cardiology_literature.json"), args.per_kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects prior labels and baseline definitions.
    with args.output.open("x", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Created {len(cases)} title controls: {args.output}")
    print(f"Canonical corpus SHA256: {fingerprint}")
    print("Diagnostic smoke test only. Cross-corpus scores are not comparable quality measurements.")


if __name__ == "__main__":
    main()
