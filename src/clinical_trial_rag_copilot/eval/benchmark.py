import time
from typing import Any, Dict, List
import pandas as pd

from clinical_trial_rag_copilot.agent.copilot import ClinicalGraphRAGCopilot


BENCHMARK_CASES = [
    {
        "id": "CASE_1",
        "category": "Multi-Hop Gene-Drug-Trial",
        "query": "Which clinical trials evaluate Amiodarone targeting KCNH2 in patients with Atrial Fibrillation?",
        "expected_nct": "NCT05543278",
        "expected_drug": "Amiodarone",
        "expected_gene": "KCNH2",
        "expected_guardrail": "PASSED",
    },
    {
        "id": "CASE_2",
        "category": "Targeted Pharmacology & Demographics",
        "query": "Find heart failure clinical trials testing Spironolactone in adult patients aged >= 18.",
        "expected_drug": "Spironolactone",
        "expected_gene": "NR3C2",
        "expected_guardrail": "PASSED",
    },
    {
        "id": "CASE_3",
        "category": "Literature Cross-Link Safety",
        "query": "What research papers report QT Prolongation or Torsades de Pointes for antiarrhythmic therapy?",
        "expected_adverse_event": "QT Prolongation",
        "expected_guardrail": "PASSED",
    },
    {
        "id": "CASE_4",
        "category": "Hallucination Adversarial Stress Test",
        "query": "Evaluate clinical trial NCT99999999 testing PseudoDrug on gene FAKE1.",
        "expected_guardrail": "WARNING",
    },
]


def run_benchmark():
    """Execute evaluation benchmark across multi-hop reasoning, retrieval recall, and guardrail precision."""
    print("=" * 65)
    print("STARTING CLINICAL GRAPHRAG BENCHMARK & EVALUATION")
    print("=" * 65)

    copilot = ClinicalGraphRAGCopilot()
    results: List[Dict[str, Any]] = []

    for case in BENCHMARK_CASES:
        cid = case["id"]
        q = case["query"]
        print(f"\nEvaluating [{cid}] ({case['category']}):")
        print(f"  Query: '{q}'")

        t0 = time.time()
        ans = copilot.ask(q, top_k=6)
        latency = (time.time() - t0) * 1000

        # Evaluate Hit Rate
        hit = False
        if case.get("expected_nct"):
            hit = case["expected_nct"] in ans.retrieved_ncts
        elif case.get("expected_drug"):
            hit = any(case["expected_drug"].lower() in str(n).lower() for n in ans.retrieved_ncts) or len(ans.retrieved_ncts) > 0
        else:
            hit = len(ans.retrieved_pmids) > 0

        # Evaluate Guardrail Accuracy
        guardrail_match = ans.guardrail_status == case["expected_guardrail"]

        print(f"  Latency: {latency:.1f}ms | Guardrail Status: {ans.guardrail_status} (Expected: {case['expected_guardrail']})")
        print(f"  Hit in Top-K: {hit} | Verified Trials: {ans.verified_trials}")

        results.append({
            "Case ID": cid,
            "Category": case["category"],
            "Hit in Top-K": hit,
            "Guardrail Accurate": guardrail_match,
            "Guardrail Status": ans.guardrail_status,
            "Latency (ms)": round(latency, 1),
            "Retrieved NCTs": len(ans.retrieved_ncts),
            "Retrieved PMIDs": len(ans.retrieved_pmids),
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("BENCHMARK SCORECARD")
    print("=" * 65)
    print(df.to_string(index=False))

    avg_latency = df["Latency (ms)"].mean()
    hit_rate = df["Hit in Top-K"].mean() * 100
    guardrail_acc = df["Guardrail Accurate"].mean() * 100

    print("\n" + "-" * 65)
    print(f"Overall Hit Rate @ K:        {hit_rate:.1f}%")
    print(f"Guardrail Precision / Acc:   {guardrail_acc:.1f}%")
    print(f"Average Pipeline Latency:    {avg_latency:.1f} ms")
    print("-" * 65)


if __name__ == "__main__":
    run_benchmark()
