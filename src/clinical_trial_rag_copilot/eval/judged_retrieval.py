"""Evaluate frozen, independently reviewed graded document rankings offline.

Input JSON: corpus_id, split='test', cases[{id, topic_id, category,
review_status='reviewed', exhaustive, judgments:{doc_id:0..3}}],
runs:{method:{case_id:[doc_id,...]}}. Unjudged top-k results invalidate precision
and nDCG instead of silently becoming negatives. Recall requires exhaustive labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def metrics(ranking, judgments, *, exhaustive=False, k=10, candidate_k=100):
    ranking = list(dict.fromkeys(ranking))
    if k < 1 or candidate_k < k:
        raise ValueError('Require candidate_k >= k >= 1')
    if any(type(g) is not int or g not in range(4) for g in judgments.values()):
        raise ValueError('Relevance grades must be integers 0..3')
    top = ranking[:k]
    unknown = [doc for doc in top if doc not in judgments]
    valid = not unknown
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    ideal = sum((2**g - 1) / math.log2(i + 2)
                for i, g in enumerate(sorted(judgments.values(), reverse=True)[:k]))
    dcg = sum((2**judgments.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(top))
    return {
        'ndcg': dcg / ideal if valid and ideal else None,
        'precision': sum(doc in relevant for doc in top) / k if valid else None,
        'recall': len(set(ranking[:candidate_k]) & relevant) / len(relevant) if exhaustive and relevant else None,
        'judged_fraction': (len(top) - len(unknown)) / len(top) if top else 1.0,
        'unjudged_ids': unknown,
        'no_relevant_labels': not relevant,
        'empty_result': not ranking,
        'no_answer_correct': int(not ranking) if exhaustive and not relevant else None,
    }


def evaluate(data, *, k=10, candidate_k=100):
    if not data.get('corpus_id') or data.get('split') != 'test':
        raise ValueError('A frozen corpus_id and held-out test split are required')
    cases = data['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs')
    for case in cases:
        if case.get('review_status') != 'reviewed' or not case.get('topic_id'):
            raise ValueError('Independent review and topic_id are required for every case')
    overlap = set(data.get('development_topic_ids', [])) & {c['topic_id'] for c in cases}
    if overlap:
        raise ValueError(f'Development/test topic leakage: {sorted(overlap)}')
    results = {}
    for method, runs in data['runs'].items():
        if set(runs) != {c['id'] for c in cases}:
            raise ValueError(f'{method}: query coverage differs')
        rows = [{**metrics(runs[c['id']], c['judgments'], exhaustive=c.get('exhaustive', False),
                           k=k, candidate_k=candidate_k),
                 'id': c['id'], 'topic_id': c['topic_id'], 'category': c.get('category', 'other')}
                for c in cases]
        def aggregate(group):
            return {metric: {'mean': float(np.mean(values)) if values else None, 'valid_queries': len(values)}
                    for metric in ['ndcg', 'precision', 'recall', 'judged_fraction', 'no_answer_correct']
                    for values in [[r[metric] for r in group if r[metric] is not None]]}
        results[method] = dict(per_query=rows, summary=aggregate(rows),
                               categories={cat: aggregate([r for r in rows if r['category'] == cat])
                                           for cat in sorted({r['category'] for r in rows})})
    comparisons = {}
    methods = sorted(results)
    if not methods:
        raise ValueError('At least one retrieval run is required')
    baseline = data.get('baseline', 'dense' if 'dense' in results else methods[0])
    if baseline not in results:
        raise ValueError('Requested baseline is missing')
    for method in [m for m in methods if m != baseline]:
        groups = {}
        for a, b in zip(results[method]['per_query'], results[baseline]['per_query']):
            if a['ndcg'] is not None and b['ndcg'] is not None:
                groups.setdefault(a['topic_id'], []).append(a['ndcg'] - b['ndcg'])
        values = np.asarray([np.mean(v) for v in groups.values()])
        interval = None
        if len(values) >= 2:
            draws = np.random.default_rng(20261116).choice(values, (5000, len(values)), replace=True).mean(axis=1)
            interval = np.quantile(draws, [.025, .975]).tolist()
        comparisons[f'{method} minus {baseline}'] = dict(
            topic_mean_ndcg_delta=float(values.mean()) if len(values) else None,
            topic_bootstrap_95_interval=interval, paired_topics=len(values))
    return dict(corpus_id=data['corpus_id'], k=k, candidate_k=candidate_k, results=results,
                comparisons=comparisons,
                limitations=['Review status is supplied by dataset authors, not independently verified by this tool.',
                             'Incomplete pools can bias nDCG; recall is omitted without exhaustive judgments.',
                             'This evaluates supplied rankings; record index/model hashes and retrieval costs separately.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--candidate-k', type=int, default=100)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    report = evaluate(json.loads(payload), k=args.k, candidate_k=args.candidate_k)
    report['input_sha256'] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
