"""Read-only graph diagnostics using independent, unbounded reference result sets.

Reference sets establish graph-query correctness, not clinical truth or graph completeness.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def set_metrics(actual, expected):
    actual, expected = set(actual), set(expected)
    correct = len(actual & expected)
    precision = correct / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = correct / len(expected) if expected else 1.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                exact_match=actual == expected, retrieved=len(actual), expected=len(expected),
                constraint_violations=sorted(actual - expected), missed=sorted(expected - actual))


def reference_query(kind, entities):
    """Explicit all/any within categories; AND across categories; gene on requested drug.

    Deliberately independent of production branch selection and LIMIT behavior.
    Demographics are rejected until their intended eligibility semantics are defined.
    """
    if any(entities.get(key) is not None for key in ('min_age', 'max_age', 'sex')):
        raise ValueError('Demographic reference semantics are not yet defined')
    if kind == 'trial':
        query = """MATCH (t:Trial)
        WHERE (size($drugs) = 0 OR ALL(name IN $drugs WHERE EXISTS {
            MATCH (t)-[:HAS_ARM]->(:ArmGroup)-[:ADMINISTERS]->(d:Drug)
            WHERE d.name = name
        }))
        AND (size($genes) = 0 OR ALL(symbol IN $genes WHERE EXISTS {
            MATCH (t)-[:HAS_ARM]->(:ArmGroup)-[:ADMINISTERS]->(d:Drug)-[:HAS_TARGET|INHIBITS]->(g:Gene)
            WHERE g.symbol = symbol AND (size($drugs) = 0 OR d.name IN $drugs)
        }))
        AND (size($conditions) = 0 OR ALL(name IN $conditions WHERE EXISTS {
            MATCH (t)-[:TARGETS_CONDITION]->(c:Condition) WHERE c.name = name
        }))
        RETURN DISTINCT t.nct_id AS id"""
        for field, variable in [('drugs', 'name'), ('genes', 'symbol'), ('conditions', 'name')]:
            if entities.get('operators', {}).get(field) == 'any':
                query = query.replace(f'ALL({variable} IN ${field}', f'ANY({variable} IN ${field}')
        if entities.get('adverse_events'):
            raise ValueError('Trial adverse-event reference path is not defined')
    elif kind == 'paper':
        if entities.get('conditions'):
            raise ValueError('Paper condition reference path is not defined')
        clauses = []
        for field, relation, label, prop in (
            ('drugs', 'CITES_DRUG', 'Drug', 'name'),
            ('genes', 'STUDIES_GENE', 'Gene', 'symbol'),
            ('adverse_events', 'REPORTS_ADVERSE_EVENT', 'AdverseEvent', 'name'),
        ):
            quantifier = 'ANY' if entities.get('operators', {}).get(field) == 'any' else 'ALL'
            clauses.append(f'(size(${field}) = 0 OR {quantifier}(value IN ${field} WHERE EXISTS {{ MATCH (p)-[:{relation}]->(n:{label}) WHERE n.{prop} = value }}))')
        query = 'MATCH (p:Paper) WHERE ' + ' AND '.join(clauses) + ' RETURN DISTINCT p.pmid AS id'
    else:
        raise ValueError('kind must be trial or paper')
    params = {key: entities.get(key, []) for key in ('drugs', 'genes', 'conditions', 'adverse_events')}
    if not any(params.values()):
        raise ValueError('At least one reference entity is required')
    return query, params


def evaluate_case(case, retriever, limit):
    from clinical_trial_rag_copilot.retrieval.hybrid_search import QueryEntities
    gold = QueryEntities(**case['entities'])
    query, params = reference_query(case['kind'], gold.model_dump())
    with retriever.db.driver.session() as session:
        reference = sorted({str(r['id']) for r in session.run(query, params) if r['id'] is not None})
    if 'expected_ids' in case:
        expected = sorted(set(case['expected_ids']))
        if expected != reference:
            raise ValueError(f"{case['id']}: reviewed expected_ids differ from graph reference; review snapshot or labels")
    output = []
    for mode in ('verified_entities', 'extracted_entities'):
        start = time.perf_counter()
        error = None
        entities = None
        items = []
        try:
            entities = gold if mode == 'verified_entities' else retriever.extractor.extract(case['query'])
            items = retriever.query_graph(entities, limit=limit, strict=True)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        ids = sorted({str(item['nct_id'] if case['kind'] == 'trial' else item['pmid'])
                      for item in items if item.get('nct_id' if case['kind'] == 'trial' else 'pmid')})
        entity_scores = {}
        if entities is not None:
            for field in ('genes', 'drugs', 'conditions', 'adverse_events'):
                entity_scores[field] = set_metrics(getattr(entities, field), getattr(gold, field))
        output.append(dict(case_id=case['id'], query=case['query'], kind=case['kind'], mode=mode,
                           entities=entities.model_dump() if entities else None, entity_metrics=entity_scores,
                           error=error, latency_ms=(time.perf_counter() - start) * 1000,
                           ids=ids, reference_ids=reference, evidence=items,
                           metrics=set_metrics(ids, reference) if error is None else None))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('eval/graph_cases.jsonl'))
    parser.add_argument('--output-root', type=Path, default=Path('eval/graph_runs'))
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error('limit must be positive')
    output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    db = None
    try:
        cases = [json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not cases or len({c['id'] for c in cases}) != len(cases):
            raise ValueError('Cases must be nonempty with unique IDs')
        for case in cases:
            reference_query(case['kind'], case['entities'])
        from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, ClinicalEntityExtractor
        from clinical_trial_rag_copilot.graph.connection import get_db
        db = get_db()
        if not db.is_connected:
            raise RuntimeError('Graph database is unavailable')
        # Graph-only construction: never initialize vector stores or embedding models.
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.db = db
        retriever.extractor = ClinicalEntityExtractor()
        rows = [row for case in cases for row in evaluate_case(case, retriever, args.limit)]
        summary = {}
        for mode in ('verified_entities', 'extracted_entities'):
            subset = [r for r in rows if r['mode'] == mode]
            valid = [r for r in subset if r['metrics'] is not None]
            summary[mode] = dict(cases=len(subset), execution_errors=len(subset) - len(valid),
                                 empty_reference_cases=sum(not r['reference_ids'] for r in subset),
                                 **{key: sum(r['metrics'][key] for r in valid) / len(valid) if valid else None
                                    for key in ('precision', 'recall', 'f1', 'exact_match')})
        report = dict(created_at=datetime.now(timezone.utc).isoformat(), limit=args.limit,
                      cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                      label_type='graph_snapshot_reference; optional reviewed expected_ids',
                      limitations=['Reference paths are not independently verified clinical truth.',
                                   'Graph completeness and source relationship accuracy are not measured.',
                                   'Explicit diagnostic limits can reduce recall; production scores all graph matches.',
                                   'No relevance ordering: ranking metrics are intentionally omitted.',
                                   'Demographics are unsupported; conditions tested only for trials.'],
                      summary=summary, per_case=rows)
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        print(f'Report: {output / "report.json"}')
        return int(any(r['error'] or not r['metrics']['exact_match'] for r in rows))
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc))), encoding='utf-8')
        print(f'ERROR: {exc}\nFailure artifact: {output / "failure.json"}')
        return 2
    finally:
        if db is not None:
            db.close()


if __name__ == '__main__':
    raise SystemExit(main())
