import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from clinical_trial_rag_copilot.graph.connection import get_db
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore, PineconeVectorStore, get_vector_store, SearchResult
from clinical_trial_rag_copilot.utils.logger import get_logger

logger = get_logger("retrieval.hybrid_search")


class QueryEntities(BaseModel):
    """Structured clinical entities extracted from a natural language query."""
    genes: List[str] = Field(default_factory=list)
    drugs: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    adverse_events: List[str] = Field(default_factory=list)
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    sex: Optional[str] = None
    unresolved_terms: List[str] = Field(default_factory=list)


class GraphCandidate(BaseModel):
    """Candidate trial or paper retrieved from Neo4j Knowledge Graph."""
    doc_id: str
    doc_type: str  # "TRIAL" or "PAPER"
    title: str
    details: Dict[str, Any] = Field(default_factory=dict)
    graph_rank: int = 1


class HybridEvidence(BaseModel):
    """Consolidated multi-hop evidence ready for LLM synthesis."""
    query: str
    extracted_entities: QueryEntities
    graph_evidence: List[Dict[str, Any]]
    vector_evidence: List[SearchResult]
    fused_ncts: List[str]
    fused_pmids: List[str]
    sibling_chunks: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    retrieval_warnings: List[str] = Field(default_factory=list)


class ClinicalEntityExtractor:
    """Fast, deterministic entity extractor matching clinical vocabulary from the Knowledge Graph."""

    def __init__(self):
        self.db = get_db()
        self._load_vocabularies()

    def _load_vocabularies(self):
        """Cache known entity names from Neo4j AuraDB for sub-millisecond matching."""
        if not self.db.is_connected:
            self.known_genes = {"EGFR", "ERBB2", "KRAS", "ALK", "BRAF", "MET"}
            self.known_conditions = {"Non-Small Cell Lung Cancer", "Lung Adenocarcinoma", "Lung Cancer"}
            self.known_drugs = {"Osimertinib", "Erlotinib", "Gefitinib", "Afatinib", "Amivantamab", "Pembrolizumab"}
            self.known_aes = {"Pneumonitis", "Diarrhea", "Skin Rash", "Neutropenia"}
            return

        try:
            with self.db.driver.session() as session:
                # Load genes
                res_genes = session.run("MATCH (g:Gene) RETURN g.symbol AS symbol")
                self.known_genes = {r["symbol"].upper() for r in res_genes if r["symbol"]}

                # Load conditions
                res_conds = session.run("MATCH (c:Condition) RETURN c.name AS name")
                self.known_conditions = {r["name"] for r in res_conds if r["name"]}

                # Load drugs
                res_drugs = session.run("MATCH (d:Drug) RETURN d.name AS name")
                self.known_drugs = {r["name"] for r in res_drugs if r["name"]}

                # Load adverse events
                res_aes = session.run("MATCH (a:AdverseEvent) RETURN a.name AS name")
                self.known_aes = {r["name"] for r in res_aes if r["name"]}
        except Exception as e:
            logger.warning("Could not query Neo4j for vocabulary: %s — using static fallback.", e)
            self.known_genes = {"EGFR", "ERBB2", "KRAS", "ALK", "BRAF", "MET"}
            self.known_conditions = {"Non-Small Cell Lung Cancer", "Lung Adenocarcinoma", "Lung Cancer"}
            self.known_drugs = {"Osimertinib", "Erlotinib", "Gefitinib", "Afatinib", "Amivantamab", "Pembrolizumab"}
            self.known_aes = {"Pneumonitis", "Diarrhea", "Skin Rash", "Neutropenia"}

    def extract(self, query: str) -> QueryEntities:
        """Extract clinical entities and demographic constraints from free text."""
        q_lower = query.lower()
        extracted_genes = []
        extracted_drugs = []
        extracted_conditions = []
        extracted_aes = []

        # Gene matching
        for g in self.known_genes:
            if re.search(r"\b" + re.escape(g.lower()) + r"\b", q_lower):
                extracted_genes.append(g)

        # Drug matching
        for d in self.known_drugs:
            if len(d) >= 4 and re.search(r"\b" + re.escape(d.lower()) + r"\b", q_lower):
                extracted_drugs.append(d)

        # Condition matching
        for c in self.known_conditions:
            if len(c) >= 5 and re.search(r"\b" + re.escape(c.lower()) + r"\b", q_lower):
                extracted_conditions.append(c)

        # Adverse event matching
        # Keep recognized safety terms even when ingestion has no such node.
        # Dropping a missing constraint would silently broaden the query.
        safety_terms = self.known_aes | {"Pneumonitis", "QT Prolongation", "Diarrhea", "Skin Rash", "Neutropenia"}
        for ae in safety_terms:
            if re.search(r"\b" + re.escape(ae.lower()) + r"\b", q_lower):
                extracted_aes.append(ae)

        # Age extraction (e.g., ">= 65", "older than 60", "aged 18 to 75")
        min_age = None
        max_age = None
        m_ge = re.search(r"(?:>=|greater than|older than|above|at least)\s*(\d+)", q_lower)
        if m_ge:
            min_age = float(m_ge.group(1))

        m_le = re.search(r"(?:<=|less than|younger than|under|at most)\s*(\d+)", q_lower)
        if m_le:
            max_age = float(m_le.group(1))

        if "elderly" in q_lower or "older adult" in q_lower:
            min_age = min_age or 65.0

        # Sex extraction
        sex = None
        if "female" in q_lower or "women" in q_lower:
            sex = "FEMALE"
        elif "male" in q_lower or "men" in q_lower:
            sex = "MALE"

        return QueryEntities(
            genes=extracted_genes,
            drugs=extracted_drugs,
            conditions=extracted_conditions,
            adverse_events=extracted_aes,
            min_age=min_age,
            max_age=max_age,
            sex=sex,
            unresolved_terms=sorted(ae for ae in extracted_aes if ae not in self.known_aes),
        )


class HybridRetriever:
    """Dual-retriever fusing Neo4j Knowledge Graph relations and Vector Store semantics."""

    def __init__(self, vector_store: Optional[LocalVectorStore | PineconeVectorStore] = None):
        self.db = get_db()
        self.vector_store = vector_store or get_vector_store()
        self.extractor = ClinicalEntityExtractor()

    def query_graph(self, entities: QueryEntities, limit: int = 15, *, strict: bool = False) -> List[Dict[str, Any]]:
        """Execute deterministic multi-hop Cypher queries based on extracted entities."""
        if not self.db.is_connected or self.db.driver is None:
            if strict:
                raise RuntimeError("Graph database is unavailable")
            return []

        graph_results: List[Dict[str, Any]] = []

        try:
            with self.db.driver.session() as session:
                # Conjoin requested categories before selecting unique trials.
                # These are discovery links, not patient eligibility predicates.
                if entities.genes or entities.drugs or entities.conditions:
                    query_trials = """
                    MATCH (t:Trial)
                    WHERE (size($genes) = 0 OR ALL(symbol IN $genes WHERE EXISTS {
                        MATCH (t)-[:HAS_ARM]->(:ArmGroup)-[:ADMINISTERS]->(d:Drug)-[:INHIBITS]->(g:Gene)
                        WHERE g.symbol = symbol AND (size($drugs) = 0 OR d.name IN $drugs)
                    }))
                      AND (size($drugs) = 0 OR ALL(name IN $drugs WHERE EXISTS {
                        MATCH (t)-[:HAS_ARM]->(:ArmGroup)-[:ADMINISTERS]->(d:Drug)
                        WHERE d.name = name
                    }))
                      AND (size($conditions) = 0 OR ALL(name IN $conditions WHERE EXISTS {
                        MATCH (t)-[:TARGETS_CONDITION]->(c:Condition)
                        WHERE c.name = name
                    }))
                    WITH DISTINCT t ORDER BY t.nct_id LIMIT $limit
                    OPTIONAL MATCH (t)-[:TARGETS_CONDITION]->(c:Condition)
                    RETURN t.nct_id AS nct_id, t.brief_title AS title,
                           t.phase AS phase, collect(DISTINCT c.name) AS conditions
                    ORDER BY nct_id
                    """
                    for record in session.run(query_trials, {
                        "genes": entities.genes, "drugs": entities.drugs,
                        "conditions": entities.conditions, "limit": limit,
                    }):
                        graph_results.append({
                            "source": "GRAPH_TRIAL_CONSTRAINTS",
                            "nct_id": record["nct_id"], "title": record["title"],
                            "phase": record["phase"], "conditions": record["conditions"],
                            "matched_constraints": entities.model_dump(),
                            "eligibility_assessed": False,
                        })

                # Query Path 3: Adverse Events & Literature Cross-Links
                if entities.adverse_events or entities.genes or entities.drugs:
                    query_lit = """
                    MATCH (p:Paper)
                    // Require each requested category before optional expansion.
                    WHERE (size($aes) = 0 OR EXISTS {
                        MATCH (p)-[:REPORTS_ADVERSE_EVENT]->(matched_ae:AdverseEvent)
                        WHERE matched_ae.name IN $aes
                    })
                      AND (size($drugs) = 0 OR EXISTS {
                        MATCH (p)-[:CITES_DRUG]->(matched_drug:Drug)
                        WHERE matched_drug.name IN $drugs
                    })
                      AND (size($genes) = 0 OR EXISTS {
                        MATCH (p)-[:STUDIES_GENE]->(matched_gene:Gene)
                        WHERE matched_gene.symbol IN $genes
                    })
                    OPTIONAL MATCH (p)-[:REPORTS_ADVERSE_EVENT]->(ae:AdverseEvent)
                    OPTIONAL MATCH (p)-[:CITES_DRUG]->(d:Drug)
                    OPTIONAL MATCH (p)-[:STUDIES_GENE]->(g:Gene)
                    RETURN p.pmid AS pmid, p.title AS title, p.year AS year,
                           collect(DISTINCT ae.name) AS adverse_events,
                           collect(DISTINCT d.name)[..4] AS cited_drugs,
                           collect(DISTINCT g.symbol)[..4] AS studied_genes
                    ORDER BY pmid
                    LIMIT $limit
                    """
                    res = session.run(
                        query_lit,
                        {
                            "aes": entities.adverse_events,
                            "drugs": entities.drugs,
                            "genes": entities.genes, "limit": limit,
                        },
                    )
                    for record in res:
                        graph_results.append({
                            "source": "GRAPH_SAFETY_LITERATURE",
                            "pmid": record["pmid"],
                            "title": record["title"],
                            "year": record["year"],
                            "adverse_events": record["adverse_events"],
                            "drugs": record["cited_drugs"],
                            "genes": record["studied_genes"],
                        })
        except Exception as e:
            if strict:
                raise
            logger.warning("Graph query failed (%s); falling back to vector evidence.", e)

        return graph_results

    def retrieve(self, query: str, top_k: int = 8) -> HybridEvidence:
        """Retrieve ranked dense evidence with supplemental graph discoveries."""
        # 1. Extract entities
        entities = self.extractor.extract(query)

        warnings = [
            "Graph vocabulary matching is incomplete; unresolved query terms remain in dense search.",
            "Drug-target and publication-mention links do not establish patient eligibility or therapeutic effect.",
        ]
        if entities.min_age is not None or entities.max_age is not None or entities.sex:
            warnings.append("Demographic constraints are not enforced by graph discovery; review source eligibility.")
        if entities.unresolved_terms:
            warnings.append("Recognized terms absent from graph vocabulary: " + ", ".join(entities.unresolved_terms))

        # 2. Query Knowledge Graph
        graph_items = self.query_graph(entities, limit=top_k * 2)

        # 3. Vector Similarity Search
        vector_hits = self.vector_store.search(query, top_k=top_k)

        # Graph rows are an unordered candidate set, not a relevance ranking.
        # Preserve unique dense document order; append graph-only discoveries.
        sorted_ncts = list(dict.fromkeys(
            [h.doc_id for h in vector_hits if h.doc_id.startswith("NCT")]
            + [item["nct_id"] for item in graph_items if item.get("nct_id")]
        ))
        sorted_pmids = list(dict.fromkeys(
            [h.doc_id for h in vector_hits if h.chunk_type == "LITERATURE_ABSTRACT" and h.doc_id]
            + [str(item["pmid"]) for item in graph_items if item.get("pmid")]
        ))

        # Sibling Chunk Expansion: Pull companion chunks (Overview, Inclusion, Exclusion) for top winning trials
        sibling_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for nct in sorted_ncts[:3]:
            sibling_chunks[nct] = self.vector_store.get_chunks_by_doc_id(nct)

        return HybridEvidence(
            query=query,
            extracted_entities=entities,
            graph_evidence=graph_items,
            vector_evidence=vector_hits,
            fused_ncts=sorted_ncts[:top_k],
            fused_pmids=sorted_pmids[:top_k],
            sibling_chunks=sibling_chunks,
            retrieval_warnings=warnings,
        )


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_q = "What clinical trials test Osimertinib targeting EGFR in patients with Non-Small Cell Lung Cancer?"
    logger.info("Test hybrid retrieval query: %s", test_q)
    evidence = retriever.retrieve(test_q, top_k=5)

    logger.info("Extracted entities — Genes: %s | Drugs: %s | Conditions: %s",
                evidence.extracted_entities.genes,
                evidence.extracted_entities.drugs,
                evidence.extracted_entities.conditions)

    logger.info("Graph evidence items: %d", len(evidence.graph_evidence))
    for g in evidence.graph_evidence[:3]:
        logger.info("  [%s] Doc: %s | Title: %s...",
                    g.get("source"), g.get("nct_id") or g.get("pmid"), str(g.get("title", ""))[:60])

    logger.info("Vector evidence items: %d", len(evidence.vector_evidence))
    for v in evidence.vector_evidence[:3]:
        logger.info("  [%s] Score=%.4f | Doc=%s", v.chunk_type, v.score, v.doc_id)

    logger.info("Ranked trial candidates: %s", evidence.fused_ncts)
    logger.info("Ranked paper candidates: %s", evidence.fused_pmids)
