"""Freeze ingestion statements for review, then apply only to an empty graph.

Preparation uses the selected local snapshot and never connects to Neo4j.
Apply uses NEO4J_* environment credentials and refuses any populated target.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from clinical_trial_rag_copilot.graph import ingest
from clinical_trial_rag_copilot.graph.schema import CONSTRAINTS


class Capture:
    def __init__(self):
        self.statements = []
        self.drugs, self.genes = set(), set()
        self.driver = self

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def single(self):
        return {'count': 0}

    def run(self, query, parameters=None):
        if 'coalesce(t.identity_contract' in query:
            return self
        if query.startswith('MATCH (d:Drug) RETURN'):
            return [{'name': name} for name in sorted(self.drugs)]
        if query.startswith('MATCH (g:Gene) RETURN'):
            return [{'symbol': name} for name in sorted(self.genes)]
        parameters = parameters or {}
        self.statements.append({'query': query, 'parameters': parameters})
        for row in parameters.get('batch', []):
            self.drugs.update(row.get('drugs', []))
            if row.get('drug_name'):
                self.drugs.add(row['drug_name'])
            if row.get('gene_symbol'):
                self.genes.add(row['gene_symbol'])
        return []


def prepare(snapshot, output):
    if output.exists():
        raise ValueError('Refusing to overwrite an existing frozen package')
    datasets = snapshot / 'datasets'
    sources = {}
    def load(name):
        payload = (datasets / name).read_bytes()
        sources[name] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)
    capture = Capture()
    with patch.object(ingest, 'get_db', return_value=capture), patch.object(ingest, 'load_dataset', side_effect=load):
        ingest.ingest_trials()
        ingest.ingest_drug_targets()
        ingest.ingest_pubmed_literature()
    package = {'identity_contract': 'exact-name-v2', 'source_hashes': sources,
               'constraints': CONSTRAINTS, 'statements': capture.statements}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2), encoding='utf-8')
    print(f'Frozen {len(capture.statements)} batches: {output}')


def apply(package_path, receipt):
    from clinical_trial_rag_copilot.graph.connection import get_db
    payload = package_path.read_bytes()
    package = json.loads(payload)
    if package.get('identity_contract') != 'exact-name-v2':
        raise ValueError('Unsupported graph identity contract')
    if receipt.exists():
        raise ValueError('Receipt already exists')
    db = get_db()
    if not db.is_connected:
        raise RuntimeError('Target graph unavailable')
    try:
        with db.driver.session() as session:
            if session.run('MATCH (n) RETURN count(n) AS count').single()['count']:
                raise RuntimeError('Target is populated; configure a separate empty database. No changes made.')
            for query in package['constraints']:
                session.run(query).consume()
            # A single transaction prevents partially written source batches.
            def write(tx):
                for statement in package['statements']:
                    tx.run(statement['query'], statement['parameters']).consume()
            session.execute_write(write)
            counts = [dict(r) for r in session.run('MATCH (n) RETURN labels(n) AS labels, count(*) AS count')]
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({'package_sha256': hashlib.sha256(payload).hexdigest(),
                                       'node_counts': counts, 'status': 'built_pending_source_parity_validation'}, indent=2))
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('apply')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.snapshot, args.output)
    else:
        apply(args.package, args.receipt)


if __name__ == '__main__':
    main()
