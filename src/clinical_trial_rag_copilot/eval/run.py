"""Run retrieval evaluation or render saved measurements as a report bundle."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from clinical_trial_rag_copilot.eval.reporting import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Replay saved retrieval_metrics.json without database calls")
    parser.add_argument("--cases", type=Path, default=Path("eval/retrieval_cases.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/runs"))
    parser.add_argument("--baseline", type=Path, help="Previous run's retrieval_metrics.json")
    parser.add_argument("--methods", nargs="+", choices=["bm25", "dense", "graph", "hybrid"], default=["bm25", "dense", "graph", "hybrid"])
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--k", type=int, choices=[1, 3, 5, 10], default=5)
    parser.add_argument("--gate-method", choices=["bm25", "dense", "graph", "hybrid"], default="hybrid")
    parser.add_argument("--min-hit", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-mrr-drop", type=float)
    parser.add_argument("--require-graph", action="store_true")
    args = parser.parse_args(argv)
    if not (0 <= args.min_hit <= 1 and 0 <= args.min_mrr <= 1):
        parser.error("Minimum quality thresholds must lie in [0, 1]")
    if args.max_mrr_drop is not None and (not args.baseline or not 0 <= args.max_mrr_drop <= 1):
        parser.error("max-mrr-drop requires a baseline and a value in [0, 1]")
    if args.max_p95_ms is not None and args.max_p95_ms <= 0:
        parser.error("max-p95-ms must be positive")
    output = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    try:
        if args.input:
            run = json.loads(args.input.read_text(encoding="utf-8"))
        else:
            from clinical_trial_rag_copilot.eval.retrieval_benchmark import main as benchmark
            benchmark(["--cases", str(args.cases), "--output", str(output),
                       "--candidate-k", str(args.candidate_k), "--methods", *args.methods])
            run = json.loads((output / "retrieval_metrics.json").read_text(encoding="utf-8"))
        (output / "retrieval_metrics.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
        report = write_report(run, output, k=args.k, baseline=baseline, method=args.gate_method,
                              min_hit=args.min_hit, min_mrr=args.min_mrr, max_p95=args.max_p95_ms,
                              require_graph=args.require_graph, max_mrr_drop=args.max_mrr_drop)
        print(f"{report['status']}: {output / 'report.html'}")
        return 0 if report["status"] == "PASS" else 1
    except (Exception, SystemExit) as exc:
        (output / "failure.json").write_text(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                                                        "error": str(exc)}, indent=2), encoding="utf-8")
        print(f"ERROR: {exc}\nFailure artifact: {output / 'failure.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
