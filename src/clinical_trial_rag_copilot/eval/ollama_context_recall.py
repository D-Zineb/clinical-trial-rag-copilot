"""Judge saved retrieval samples with Ragas and a local Ollama model."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


async def evaluate(args):
    # Do not allow a cloud URL to silently replace the requested free local judge.
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("This runner requires a local Ollama URL.")
    endpoint = args.base_url.rstrip("/")
    with urlopen(endpoint + "/api/tags", timeout=5) as response:
        models = json.load(response)["models"]
    model_info = next((m for m in models if m["name"] == args.model), None)
    if model_info is None:
        raise ValueError(f"Model not installed. Run: ollama pull {args.model}")
    # Imported after preflight so missing Ollama produces a clear error first.
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall

    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("No evaluation samples found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Dummy key is required by the compatible client; no OpenAI service is used.
    async with AsyncOpenAI(base_url=endpoint + "/v1", api_key="ollama", timeout=300, max_retries=0) as client:
        llm = llm_factory(args.model, client=client, temperature=0)
        metric = ContextRecall(llm=llm)
        with args.output.open("w", encoding="utf-8") as output:
            for sample in samples:
                print(f"Judging {sample['id']} with {args.model}...", flush=True)
                row = {"id": sample["id"], "model": args.model,
                       "model_digest": model_info.get("digest"),
                       "context_variant": args.contexts,
                       "reference": sample["reference"]}
                try:
                    result = await metric.ascore(
                        user_input=sample["user_input"],
                        retrieved_contexts=sample[args.contexts],
                        reference=sample["reference"],
                    )
                    row.update(score=float(result.value),
                               details=result.model_dump(mode="json") if hasattr(result, "model_dump") else str(result))
                    print(f"  Context recall: {row['score']:.3f}", flush=True)
                except Exception as exc:
                    row.update(score=None, error=str(exc))
                    print(f"  Evaluation failed: {exc}", flush=True)
                output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OLLAMA_EVAL_MODEL", "qwen2.5:3b"))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--samples", type=Path, default=Path("eval/context_pilot/retrieved_samples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/context_pilot/ollama_scores.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="Start with --limit 1 to inspect one judgment.")
    parser.add_argument("--contexts", choices=["retrieved_contexts", "expanded_contexts"], default="retrieved_contexts")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    try:
        asyncio.run(evaluate(args))
    except Exception as exc:
        parser.exit(1, f"Evaluation could not start: {exc}\n")


if __name__ == "__main__":
    main()
