"""Prepare five source-backed examples and inspect raw/expanded retrieval contexts."""
import json
from pathlib import Path

from clinical_trial_rag_copilot.retrieval.chunking import build_all_chunks
from clinical_trial_rag_copilot.retrieval.vector_store import DATA_DIR, get_vector_store


CASES = [
    dict(id="age_bounds", doc_id="NCT02114528", section="INCLUSION",
         question="For trial NCT02114528 comparing anti-arrhythmic therapy with catheter ablation for AICD shock prevention, what are the lower and upper age limits?",
         reference="The minimum age is 18 years and the maximum age is 85 years.",
         quotes=["18", "85 years old"]),
    dict(id="stroke_exclusion", doc_id="NCT02114528", section="EXCLUSION",
         question="Does the AICD shock prevention trial NCT02114528 exclude recent stroke and NYHA class IV patients? What stroke time window is specified?",
         reference="The trial excludes patients with stroke within the past 90 days and patients with NYHA functional class IV.",
         quotes=["Stroke within the past 90 days", "functional class IV"]),
    dict(id="geriatric_inclusion", doc_id="NCT07551219", section="INCLUSION",
         question="In ondansetron trial NCT07551219 for geriatric urologic surgery, what minimum age and ASA physical status are required?",
         reference="Patients must be aged 65 years or older and have ASA physical status I-III.",
         quotes=["Patients aged 65 years and older", "physical status I-III"]),
    dict(id="bradycardia_exclusion", doc_id="NCT07551219", section="EXCLUSION",
         question="What baseline heart-rate threshold defines the bradycardia exclusion in ondansetron trial NCT07551219?",
         reference="Baseline bradycardia with heart rate below 50 beats per minute is an exclusion criterion.",
         quotes=["Baseline bradycardia", "50 beats per minute"]),
    dict(id="tachycardia_inclusion", doc_id="NCT05239364", section="INCLUSION",
         question="For consecutive atrial tachycardia ablation trial NCT05239364, what prior procedure and minimum tachycardia cycle length are specified?",
         reference="Consecutive atrial tachycardia must follow prior atrial fibrillation ablation or cardiac surgery; the specified cycle length is at least 200 ms.",
         quotes=["prior AF ablation procedure or cardiac surgery", "200ms"]),
]


def main():
    trials = json.loads((DATA_DIR / "cardiology_trials.json").read_text(encoding="utf-8"))
    papers = json.loads((DATA_DIR / "pubmed_cardiology_literature.json").read_text(encoding="utf-8"))
    chunks = {c.chunk_id: c for c in build_all_chunks(trials, papers)}
    out = Path("eval/context_pilot")
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in CASES:
        chunk_id = f"{case['doc_id']}_{case['section']}"
        chunk = chunks[chunk_id]
        if not all(quote in chunk.text for quote in case["quotes"]):
            raise ValueError(f"Reference quotes missing for {case['id']}")
        cases.append({**case, "reference_context_ids": [chunk_id],
                      "reference_contexts": [chunk.text],
                      "source": f"https://clinicaltrials.gov/study/{case['doc_id']}",
                      "review_status": "assistant checked against local source snapshot; awaiting user review"})
    # Reference labels are prepared before querying; retrieval does not choose labels.
    (out / "reference_cases.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    store = get_vector_store()
    samples = []
    for case in cases:
        print(f"Retrieving {case['id']}", flush=True)
        hits = store.search(case["question"], top_k=5)
        # Diagnostic expansion mirrors sibling fetching for three winning trials,
        # but uses vector document order; this is not the full hybrid pipeline.
        top_trials = list(dict.fromkeys(h.doc_id for h in hits if h.doc_id.startswith("NCT")))[:3]
        contexts = {h.chunk_id: h.text for h in hits}
        for doc_id in top_trials:
            for chunk in store.get_chunks_by_doc_id(doc_id):
                contexts[chunk["chunk_id"]] = chunk["text"]
        expected = set(case["reference_context_ids"])
        raw_ids = [h.chunk_id for h in hits]
        samples.append({**case, "user_input": case["question"],
                        "retrieved_context_ids": raw_ids,
                        "retrieved_contexts": [h.text for h in hits],
                        "expanded_context_ids": list(contexts),
                        "expanded_contexts": list(contexts.values()),
                        "raw_reference_chunk_coverage": len(expected.intersection(raw_ids)) / len(expected),
                        "expanded_reference_chunk_coverage": len(expected.intersection(contexts)) / len(expected)})
    (out / "retrieved_samples.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in samples) + "\n", encoding="utf-8")
    lines = ["# Context evaluation: five-example learning pilot", "",
             "Source-backed draft labels checked by the assistant against local JSON snapshots; not clinician-reviewed. The live source pages were not revalidated. All examples are trial-specific lookup questions, not a representative clinical set.", "",
             "| Example | Question | Reference answer | Supporting chunk | Raw top-5 contains chunk | After sibling expansion |",
             "|---|---|---|---|---|---|"]
    for s in samples:
        lines.append(f"| {s['id']} | {s['question']} | {s['reference']} | {s['reference_context_ids'][0]} | {s['raw_reference_chunk_coverage']:.0%} | {s['expanded_reference_chunk_coverage']:.0%} |")
    lines += ["", "These percentages are exact reference-chunk coverage, NOT Ragas LLM-based context recall. Each case has one labeled supporting chunk, so its score is either zero or one. Another chunk might independently support the answer; an ID miss does not prove missing information.", "",
              "Sibling expansion fetches all sections for the first three distinct trials returned by dense retrieval. Its context budget is larger than five chunks. This diagnostic does not invoke Neo4j or claim a full hybrid comparison.", "",
              "Next: review reference answers and source excerpts in reference_cases.jsonl, then use an explicitly configured judge to calculate Ragas ContextRecall on retrieved_contexts and reference. Inspect which reference claims are supported. Do not interpret reference-chunk coverage as claim-level recall.", "",
              "Reproduce: `uv run python -m clinical_trial_rag_copilot.eval.context_pilot`", "",
              "Ragas definition: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/"]
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for s in samples:
        print(s["id"], "raw:", s["raw_reference_chunk_coverage"], "expanded:", s["expanded_reference_chunk_coverage"])


if __name__ == "__main__":
    main()
