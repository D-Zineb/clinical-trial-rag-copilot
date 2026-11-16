from unittest.mock import Mock

from clinical_trial_rag_copilot.retrieval.hybrid_search import HybridRetriever, QueryEntities
from clinical_trial_rag_copilot.retrieval.vector_store import SearchResult


def retrieve_papers(vector_ids, graph_ids=(), top_k=5):
    # Exercise real fusion without a database connection or embedding model.
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.extractor = Mock()
    retriever.extractor.extract.return_value = QueryEntities()
    retriever.query_graph = Mock(return_value=[{"pmid": p} for p in graph_ids])
    retriever.vector_store = Mock()
    retriever.vector_store.search.return_value = [
        SearchResult(chunk_id=f"chunk_{i}", doc_id=p, chunk_type="LITERATURE_ABSTRACT",
                     text="paper evidence", score=1 / (i + 1))
        for i, p in enumerate(vector_ids)
    ]
    return retriever.retrieve("question", top_k=top_k).fused_pmids


def test_vector_only_preserves_relevance_order_before_cutoff():
    assert retrieve_papers(["900", "100", "500"], top_k=2) == ["900", "100"]


def test_graph_order_does_not_override_dense_relevance():
    assert retrieve_papers(["900", "500"], ["100", "500"]) == ["900", "500", "100"]


def test_duplicate_papers_do_not_receive_extra_votes():
    assert retrieve_papers(["900", "100", "100", "100"]) == ["900", "100"]


def test_graph_only_and_empty_results():
    assert retrieve_papers([], ["900", "100"]) == ["900", "100"]
    assert retrieve_papers([]) == []


def test_duplicate_trial_chunks_do_not_boost_rank():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.extractor = Mock()
    retriever.extractor.extract.return_value = QueryEntities()
    retriever.query_graph = Mock(return_value=[{"nct_id": "NCT3"}, {"nct_id": "NCT2"}])
    retriever.vector_store = Mock()
    retriever.vector_store.get_chunks_by_doc_id.return_value = []
    retriever.vector_store.search.return_value = [
        SearchResult(chunk_id=str(i), doc_id=nct, chunk_type="TRIAL_OVERVIEW", text="text", score=1/(i+1))
        for i, nct in enumerate(["NCT1", "NCT2", "NCT2", "NCT2"])
    ]
    assert retriever.retrieve("query").fused_ncts == ["NCT1", "NCT2", "NCT3"]
