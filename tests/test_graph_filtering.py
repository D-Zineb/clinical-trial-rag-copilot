"""Offline query-contract checks; actual Cypher behavior needs a live Neo4j run."""
from unittest.mock import MagicMock

import pytest

from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, QueryEntities


def capture_query(entities):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.db = MagicMock(is_connected=True)
    session = retriever.db.driver.session.return_value.__enter__.return_value
    session.run.return_value = []
    retriever.query_graph(entities)
    return next((call.args for call in session.run.call_args_list
                 if "MATCH (p:Paper)" in call.args[0]), None)


@pytest.mark.parametrize("entities", [
    QueryEntities(genes=["KCNH2"]),
    QueryEntities(drugs=["Amiodarone"]),
    QueryEntities(adverse_events=["QT Prolongation"]),
    QueryEntities(genes=["KCNH2"], drugs=["Amiodarone"],
                  adverse_events=["QT Prolongation"]),
])
def test_paper_predicates_filter_before_optional_expansion(entities):
    query, params = capture_query(entities)
    predicates, expansion = query.split("OPTIONAL MATCH", 1)
    assert "MATCH (p:Paper)-[:REPORTS_ADVERSE_EVENT]" not in predicates
    assert "WHERE" not in expansion
    for parameter, relation, label, alias, prop in [
        ("aes", "REPORTS_ADVERSE_EVENT", "AdverseEvent", "matched_ae", "name"),
        ("drugs", "CITES_DRUG", "Drug", "matched_drug", "name"),
        ("genes", "STUDIES_GENE", "Gene", "matched_gene", "symbol"),
    ]:
        assert f"size(${parameter}) = 0 OR EXISTS" in predicates
        assert f"MATCH (p)-[:{relation}]->({alias}:{label})" in predicates
        assert f"WHERE {alias}.{prop} IN ${parameter}" in predicates
    assert params == {"aes": entities.adverse_events,
                      "drugs": entities.drugs, "genes": entities.genes}


def test_no_entities_do_not_scan_papers():
    assert capture_query(QueryEntities()) is None

