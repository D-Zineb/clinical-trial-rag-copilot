import asyncio
import json
import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clinical_trial_rag_copilot.eval.ragas_eval import (
    BudgetExceeded, JudgeBudget, cache_key, load_samples, main, make_judge, score_samples,
)


def sample():
    return dict(id='a', user_input='Age?', response='18 years',
                reference='18 years', retrieved_contexts=['Minimum age 18'])


def test_validation_and_dry_run_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    path = tmp_path / 'samples.jsonl'
    path.write_text(json.dumps(sample()))
    assert main(['--samples', str(path)]) == 0
    path.write_text(json.dumps({**sample(), 'retrieved_contexts': []}))
    with pytest.raises(ValueError, match='evidence text'):
        load_samples(path, 20)
    path.write_text(json.dumps(sample()) + '\n' + json.dumps(sample()))
    with pytest.raises(ValueError, match='Duplicate'):
        load_samples(path, 20)


def test_budget_blocks_before_network_and_retains_uncertain_cost():
    if importlib.util.find_spec('ragas') is None:
        pytest.skip('Install the eval extra to exercise Ragas')
    import ragas
    from langchain_core.prompt_values import StringPromptValue
    client = Mock()
    budget = JudgeBudget(limit_usd=0.00001)
    judge = make_judge(client, budget)
    with pytest.raises(BudgetExceeded):
        judge.generate_text(StringPromptValue(text='hello'))
    client.chat.completions.create.assert_not_called()
    budget.limit_usd = 1
    client.chat.completions.create.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        judge.generate_text(StringPromptValue(text='hello'))
    assert budget.charged_or_reserved_usd > 0
    assert len(budget.calls) == 1


def test_success_usage_and_truncation_accounting():
    if importlib.util.find_spec('ragas') is None:
        pytest.skip('Install the eval extra to exercise Ragas')
    import ragas
    from langchain_core.prompt_values import StringPromptValue
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{}'))])
    budget = JudgeBudget(max_calls=1)
    judge = make_judge(client, budget)
    assert judge.generate_text(StringPromptValue(text='hello')).generations[0][0].text == '{}'
    assert budget.charged_or_reserved_usd == pytest.approx(0.0003)
    with pytest.raises(BudgetExceeded):
        judge.generate_text(StringPromptValue(text='hello'))
    assert client.chat.completions.create.call_count == 1
    budget.max_calls = 2
    client.chat.completions.create.return_value.choices[0].finish_reason = 'length'
    with pytest.raises(RuntimeError, match='truncated'):
        judge.generate_text(StringPromptValue(text='hello'))
    assert budget.charged_or_reserved_usd == pytest.approx(0.0006)


def test_prompt_size_and_cache_identity():
    budget = JudgeBudget(max_input_tokens=1025)
    with pytest.raises(BudgetExceeded, match='input allowance'):
        budget.reserve('long')
    row = sample()
    assert cache_key(row, budget) != cache_key({**row, 'reference': '21 years'}, budget)
    assert cache_key(row, budget) != cache_key({**row, 'retrieved_contexts': ['new']}, budget)


def test_actual_ragas_prompts_scoring_and_cache(tmp_path):
    if importlib.util.find_spec('ragas') is None:
        pytest.skip('Install the eval extra to exercise Ragas')
    import ragas
    client = Mock()

    def completion(**kwargs):
        prompt = kwargs['messages'][0]['content']
        # Exercise real Ragas decomposition, NLI prompts and metric arithmetic.
        if '"statements"' in prompt and '"verdict"' not in prompt:
            payload = {'statements': ['Minimum age is 18 years.']}
        elif '"claims"' in prompt and '"verdict"' not in prompt:
            payload = {'claims': ['Minimum age is 18 years.']}
        else:
            payload = {'statements': [{'statement': 'Minimum age is 18 years.',
                                      'reason': 'Explicit in evidence', 'verdict': 1}]}
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            choices=[SimpleNamespace(finish_reason='stop',
                                     message=SimpleNamespace(content=json.dumps(payload)))])

    client.chat.completions.create.side_effect = completion
    budget = JudgeBudget()
    output = tmp_path / 'first'
    cache = tmp_path / 'cache.json'
    report = asyncio.run(score_samples([sample()], make_judge(client, budget), budget, output, cache))
    assert report['status'] == 'complete', report
    assert report['results'][0]['scores'] == {'faithfulness': 1.0, 'factual_correctness': 1.0}
    assert len(budget.calls) >= 4
    client.chat.completions.create.reset_mock()
    second_budget = JudgeBudget()
    report = asyncio.run(score_samples([sample()], make_judge(client, second_budget),
                                      second_budget, tmp_path / 'second', cache))
    client.chat.completions.create.assert_not_called()
    assert report['results'][0]['cache_hit']
    assert report['charged_or_reserved_usd'] == 0


def test_partial_report_on_budget_stop(tmp_path):
    if importlib.util.find_spec('ragas') is None:
        pytest.skip('Install the eval extra to exercise Ragas')
    import ragas
    client = Mock()
    budget = JudgeBudget(limit_usd=0.00001)
    report = asyncio.run(score_samples([sample()], make_judge(client, budget), budget,
                                      tmp_path / 'run', tmp_path / 'cache.json'))
    assert report['status'] == 'stopped_budget'
    assert report['summary']['faithfulness'] == {'scored': 0, 'mean': None}
    client.chat.completions.create.assert_not_called()
    assert (tmp_path / 'run/report.json').exists()
