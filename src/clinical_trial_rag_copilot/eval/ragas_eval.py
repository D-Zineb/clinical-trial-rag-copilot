"""Evaluate saved answers with Ragas; dry-run unless --execute is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Set before importing Ragas: evaluation data should not trigger Ragas analytics.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
MODEL = "gpt-5.4-mini-2026-03-17"
INPUT_USD = 0.75 / 1_000_000
OUTPUT_USD = 4.50 / 1_000_000
SCORING_VERSION = "ragas-0.4.4-faithfulness-factual-f1-v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class JudgeBudget:
    limit_usd: float = 1.0
    max_calls: int = 80
    max_input_tokens: int = 24000
    max_output_tokens: int = 2048
    charged_or_reserved_usd: float = 0.0
    calls: list = field(default_factory=list)

    def reserve(self, prompt: str) -> dict:
        # One token per UTF-8 byte plus ample message-framing allowance is
        # deliberately conservative for this single plain-text message.
        input_bound = len(prompt.encode("utf-8")) + 1024
        if input_bound > self.max_input_tokens:
            raise BudgetExceeded("Prompt exceeds the input allowance; no truncation was applied")
        reservation = input_bound * INPUT_USD + self.max_output_tokens * OUTPUT_USD
        if len(self.calls) >= self.max_calls:
            raise BudgetExceeded("Request count limit reached")
        if self.charged_or_reserved_usd + reservation > self.limit_usd:
            raise BudgetExceeded("Insufficient budget for the next request's reserved maximum")
        self.charged_or_reserved_usd += reservation
        record = dict(status="reserved", reserved_usd=reservation, input_token_bound=input_bound)
        self.calls.append(record)
        return record

    def settle(self, record: dict, usage) -> None:
        if usage is None:
            record["status"] = "usage_missing_reservation_retained"
            raise RuntimeError("Provider returned no token usage; stopping conservatively")
        cost = usage.prompt_tokens * INPUT_USD + usage.completion_tokens * OUTPUT_USD
        self.charged_or_reserved_usd += cost - record["reserved_usd"]
        record.update(status="completed", input_tokens=usage.prompt_tokens,
                      output_tokens=usage.completion_tokens, estimated_usd=cost)
        if cost > record["reserved_usd"]:
            raise RuntimeError("Actual usage exceeded the reservation; stopping")


def load_samples(path: Path, limit: int) -> list[dict]:
    rows, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected an object")
        for key in ("id", "user_input", "response", "reference"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Line {number}: {key} must be a nonempty string")
        contexts = row.get("retrieved_contexts")
        if not isinstance(contexts, list) or not contexts or any(
            not isinstance(c, str) or not c.strip() for c in contexts
        ):
            raise ValueError(f"Line {number}: retrieved_contexts must contain evidence text")
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("No samples supplied")
    return rows[:limit]


def make_judge(client, budget: JudgeBudget):
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms import BaseRagasLLM
    from ragas.run_config import RunConfig

    class BudgetedJudge(BaseRagasLLM):
        def is_finished(self, response):
            return all(g.generation_info["finish_reason"] == "stop"
                       for group in response.generations for g in group)

        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            if n != 1:
                raise ValueError("Only one completion per request is allowed")
            text = prompt.to_string()
            reservation = budget.reserve(text)
            # SDK retries are disabled. Every Ragas repair call also comes here.
            result = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": text}],
                max_completion_tokens=budget.max_output_tokens, reasoning_effort="none",
            )
            budget.settle(reservation, result.usage)
            choice = result.choices[0]
            if choice.finish_reason != "stop" or not choice.message.content:
                raise RuntimeError("Judge response was truncated, refused, or empty")
            return LLMResult(generations=[[Generation(text=choice.message.content,
                generation_info={"finish_reason": choice.finish_reason})]])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            # Deliberately serial: reservations cannot race, even when a metric
            # internally uses asyncio.gather. SDK timeout bounds each request.
            return self.generate_text(prompt, n, temperature, stop, callbacks)

    return BudgetedJudge(run_config=RunConfig(max_retries=1, max_workers=1))


def cache_key(row: dict, budget: JudgeBudget) -> str:
    payload = dict(version=SCORING_VERSION, model=MODEL,
                   max_output_tokens=budget.max_output_tokens,
                   sample={k: row[k] for k in ("user_input", "response", "reference", "retrieved_contexts")})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def score_samples(rows, judge, budget, output, cache_path):
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, FactualCorrectness

    metrics = [Faithfulness(llm=judge), FactualCorrectness(llm=judge, mode="f1")]
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    report = dict(status="running", model=MODEL, scoring_version=SCORING_VERSION,
                  selected_samples=len(rows), budget_usd=budget.limit_usd,
                  max_calls=budget.max_calls, max_input_tokens=budget.max_input_tokens,
                  max_output_tokens=budget.max_output_tokens,
                  input_usd_per_million=INPUT_USD * 1_000_000,
                  output_usd_per_million=OUTPUT_USD * 1_000_000, results=[])
    try:
        for row in rows:
            key = cache_key(row, budget)
            saved = cache.get(key)
            if saved is not None and (set(saved) != {m.name for m in metrics} or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in saved.values()
            )):
                saved = None
            result = dict(id=row["id"], cache_hit=saved is not None, scores={})
            report["results"].append(result)
            if saved is not None:
                result["scores"] = saved
            else:
                sample = SingleTurnSample(**{k: row[k] for k in
                    ("user_input", "response", "reference", "retrieved_contexts")})
                for metric in metrics:
                    value = float(await metric.single_turn_ascore(sample))
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Undefined or invalid {metric.name} score")
                    result["scores"][metric.name] = value
                cache[key] = result["scores"].copy()
                write_json(cache_path, cache)
            write_json(output / "report.json", report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "stopped_budget" if isinstance(exc, BudgetExceeded) else "error"
        # Do not persist provider exception bodies, which can contain source text.
        report["error_type"] = type(exc).__name__
        if isinstance(exc, BudgetExceeded):
            report["stop_reason"] = str(exc)
    finally:
        report["charged_or_reserved_usd"] = budget.charged_or_reserved_usd
        report["calls"] = budget.calls
        report["summary"] = {}
        for metric in metrics:
            values = [r["scores"][metric.name] for r in report["results"] if metric.name in r["scores"]]
            report["summary"][metric.name] = dict(scored=len(values),
                mean=sum(values) / len(values) if values else None)
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    import asyncio
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=24000,
                        help="Conservative byte-based prompt allowance; evidence is never truncated")
    parser.add_argument("--output-root", type=Path, default=Path("eval/ragas_runs"))
    parser.add_argument("--execute", action="store_true", help="Enable paid API requests")
    args = parser.parse_args(argv)
    if (args.limit <= 0 or not math.isfinite(args.budget_usd) or args.budget_usd <= 0
            or args.max_calls <= 0 or not 1 <= args.max_output_tokens <= 16384
            or not 1024 <= args.max_input_tokens <= 200000):
        parser.error("Positive limits required; output tokens <= 16384; input allowance 1024–200000")
    rows = load_samples(args.samples, args.limit)
    budget = JudgeBudget(limit_usd=args.budget_usd, max_calls=args.max_calls,
                         max_output_tokens=args.max_output_tokens, max_input_tokens=args.max_input_tokens)
    print(json.dumps(dict(mode="execute" if args.execute else "dry_run", samples=len(rows),
                         model=MODEL, budget_usd=budget.limit_usd, max_calls=budget.max_calls,
                         metrics=["faithfulness", "factual_correctness"]), indent=2))
    if not args.execute:
        print("Validated only. No API calls or scores. Add --execute to evaluate.")
        return 0
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the project .env or environment")
    from openai import OpenAI
    output = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True)
    write_json(output / "samples.json", rows)
    # Pin the endpoint: an unrelated OPENAI_BASE_URL must not receive this key/data.
    with OpenAI(base_url="https://api.openai.com/v1", max_retries=0, timeout=60) as client:
        report = asyncio.run(score_samples(rows, make_judge(client, budget), budget,
                                          output, args.output_root / "cache.json"))
    print(f"{report['status']}: {output / 'report.json'}")
    print(f"Charged or conservatively reserved: ${budget.charged_or_reserved_usd:.4f}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
