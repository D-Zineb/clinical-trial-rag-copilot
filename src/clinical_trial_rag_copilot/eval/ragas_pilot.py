"""Capture a five-case source-checked pilot; generation is opt-in and budgeted."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from clinical_trial_rag_copilot.eval.ragas_eval import MODEL, JudgeBudget, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=Path('eval/cases/ragas_pilot_references.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse existing samples and generation accounting')
    args = parser.parse_args()
    data = args.snapshot.read_bytes()
    trials = {t['nct_id']: t for t in json.loads(data)}
    refs = [json.loads(line) for line in args.references.read_text().splitlines() if line.strip()]
    for row in refs:
        trial = trials[row['nct_id']]
        for evidence in row['support']:
            if evidence['quote'] not in trial[evidence['field']]:
                raise ValueError(f"Reference quote missing: {row['id']} / {evidence['field']}")
        row.update(source_snapshot=str(args.snapshot), source_sha256=hashlib.sha256(data).hexdigest(),
                   review_status='Assistant checked against local snapshot; not clinician-reviewed; live source not revalidated',
                   source_excerpt={e['field']: trial[e['field']] for e in row['support']})
    print(f'Validated {len(refs)} references against exact source excerpts.', flush=True)
    if not args.execute:
        return
    # Do not overwrite captures or pay twice accidentally.
    args.output.mkdir(parents=True, exist_ok=args.resume)
    reference_path = args.output / 'references.json'
    if reference_path.exists() and json.loads(reference_path.read_text()) != refs:
        raise ValueError('Cannot resume with changed references or snapshot')
    write_json(reference_path, refs)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    from openai import OpenAI
    from clinical_trial_rag_copilot.agent.llm_client import BaseLLMClient
    from clinical_trial_rag_copilot.agent.copilot import ClinicalGraphRAGCopilot
    from clinical_trial_rag_copilot.graph.connection import get_db

    budget = JudgeBudget(limit_usd=0.25, max_calls=5, max_input_tokens=100000,
                         max_output_tokens=2048)
    usage_path = args.output / 'generation_usage.json'
    if args.resume and usage_path.exists():
        previous = json.loads(usage_path.read_text())
        budget.charged_or_reserved_usd = previous['charged_or_reserved_usd']
        budget.calls = previous['calls']
    # One earlier generation may have failed downstream; total cost still shares
    # the original allowance. Permit at most six requests across resumed runs.
    budget.max_calls = 6
    client = OpenAI(base_url='https://api.openai.com/v1', max_retries=0, timeout=60)

    class PilotGenerator(BaseLLMClient):
        def generate(self, system_prompt, user_prompt):
            prompt_hash = hashlib.sha256((system_prompt + '\n' + user_prompt).encode()).hexdigest()
            cache_path = args.output / ('generation_' + prompt_hash + '.json')
            if cache_path.exists():
                return json.loads(cache_path.read_text())['response']
            reservation = budget.reserve(system_prompt + '\n' + user_prompt)
            try:
                response = client.chat.completions.create(model=MODEL,
                    messages=[{'role': 'system', 'content': system_prompt},
                              {'role': 'user', 'content': user_prompt}],
                    reasoning_effort='none', max_completion_tokens=budget.max_output_tokens)
                budget.settle(reservation, response.usage)
                choice = response.choices[0]
                if choice.finish_reason != 'stop' or not choice.message.content:
                    raise RuntimeError('Incomplete generation; refusing to score a truncated answer')
                write_json(cache_path, dict(response=choice.message.content, model=MODEL))
                return choice.message.content
            finally:
                write_json(args.output / 'generation_usage.json', dict(model=MODEL,
                    budget_usd=budget.limit_usd, charged_or_reserved_usd=budget.charged_or_reserved_usd,
                    calls=budget.calls))

    try:
        copilot = ClinicalGraphRAGCopilot(llm_client=PilotGenerator())
        write_json(args.output / 'configuration.json', dict(generator=MODEL,
            top_k=3, graph_connected=get_db().is_connected,
            vector_backend=type(copilot.retriever.vector_store).__name__,
            note='Existing retrieval/prompt/guardrail; budgeted generator override. Same generator and judge model.'))
        saved_path = args.output / 'samples.jsonl'
        saved_ids = {json.loads(line)['id'] for line in saved_path.read_text().splitlines()} if saved_path.exists() else set()
        for row in refs:
            if row['id'] in saved_ids:
                continue
            print(f"Capturing {row['id']}", flush=True)
            answer = copilot.ask(row['user_input'], top_k=3, capture_evaluation=True)
            sample = {**row, **answer.evaluation_sample, 'generator_model': MODEL,
                      'trace_id': answer.trace_id, 'guardrail_status': answer.guardrail_status,
                      'retrieved_ncts': answer.retrieved_ncts, 'retrieved_pmids': answer.retrieved_pmids}
            with (args.output / 'samples.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            write_json(args.output / f"{row['id']}_answer.json", answer.model_dump())
            print(f"Saved {row['id']}: {len(sample['response'])} answer characters; "
                  f"{sum(len(c) for c in sample['retrieved_contexts'])} evidence characters", flush=True)
    finally:
        client.close()
        get_db().close()
    print(f'Generation cost/reservation: ${budget.charged_or_reserved_usd:.4f}', flush=True)


if __name__ == '__main__':
    main()
