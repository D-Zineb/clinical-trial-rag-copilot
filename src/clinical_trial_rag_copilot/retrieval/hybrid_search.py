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


class ClinicalEntityExtractor:
    """Fast, deterministic entity extractor matching clinical vocabulary from the Knowledge Graph."""

    def __init__(self):
        self.db = get_db()
        self._load_vocabularies()

    def _load_vocabularies(self):
        """Cache known entity names from Neo4j AuraDB for sub-millisecond matching."""
        if not self.db.is_connected:
            self.known_genes = {"KCNH2", "HERG", "SCN5A", "CACNA1C", "ADRB1", "NR3C2"}
            self.known_conditions = {"Atrial Fibrillation", "Heart Failure", "Arrhythmia", "Long QT Syndrome"}
            self.known_drugs = {"Amiodarone", "Metoprolol", "Flecainide", "Sotalol", "Spironolactone", "Vericiguat"}
            self.known_aes = {"QT Prolongation", "Torsades De Pointes", "Cardiotoxicity", "Death, Sudden, Cardiac"}
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
            self.known_genes = {"KCNH2", "HERG", "SCN5A", "CACNA1C", "ADRB1", "NR3C2"}
            self.known_conditions = {"Atrial Fibrillation", "Heart Failure", "Arrhythmia", "Long QT Syndrome"}
            self.known_drugs = {"Amiodarone", "Metoprolol", "Flecainide", "Sotalol", "Spironolactone", "Vericiguat"}
            self.known_aes = {"QT Prolongation", "Torsades De Pointes", "Cardiotoxicity", "Death, Sudden, Cardiac"}

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
        for ae in self.known_aes:
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
        )


class HybridRetriever:
    """Dual-retriever fusing Neo4j Knowledge Graph relations and Vector Store semantics."""

    def __init__(self, vector_store: Optional[LocalVectorStore | PineconeVectorStore] = None):
        self.db = get_db()
        self.vector_store = vector_store or get_vector_store()
        self.extractor = ClinicalEntityExtractor()

    def query_graph(self, entities: QueryEntities, limit: int = 15) -> List[Dict[str, Any]]:
        """Execute deterministic multi-hop Cypher queries based on extracted entities."""
        if not self.db.is_connected or self.db.driver is None:
            return []

        graph_results: List[Dict[str, Any]] = []

        try:
            with self.db.driver.session() as session:
                # Query Path 1: Gene -> Drug -> Arm -> Trial
                if entities.genes:
                    query_gene = """
                    MATCH (t:Trial)-[:HAS_ARM]->(a:ArmGroup)-[:ADMINISTERS]->(d:Drug)-[r:INHIBITS]->(g:Gene)
                    WHERE g.symbol IN $genes
                    OPTIONAL MATCH (t)-[:TARGETS_CONDITION]->(c:Condition)
                    RETURN t.nct_id AS nct_id, t.brief_title AS title, t.phase AS phase,
                           t.min_age_years AS min_age, t.max_age_years AS max_age, t.sex AS sex,
                           a.label AS arm_label, a.type AS arm_type, d.name AS drug_name,
                           g.symbol AS gene_symbol, collect(DISTINCT c.name)[..3] AS conditions
                    LIMIT $limit
                    """
                    res = session.run(query_gene, {"genes": entities.genes, "limit": limit})
                    for record in res:
                        graph_results.append({
                            "source": "GRAPH_GENE_PATHWAY",
                            "nct_id": record["nct_id"],
                            "title": record["title"],
                            "arm": record["arm_label"],
                            "arm_type": record["arm_type"],
                            "drug": record["drug_name"],
                            "gene": record["gene_symbol"],
                            "min_age": record["min_age"],
                            "max_age": record["max_age"],
                            "conditions": record["conditions"],
                        })

                # Query Path 2: Drug -> Trial with Study Arms
                if entities.drugs and not graph_results:
                    query_drug = """
                    MATCH (t:Trial)-[:HAS_ARM]->(a:ArmGroup)-[:ADMINISTERS]->(d:Drug)
                    WHERE d.name IN $drugs
                    OPTIONAL MATCH (d)-[:INHIBITS]->(g:Gene)
                    RETURN t.nct_id AS nct_id, t.brief_title AS title, t.phase AS phase,
                           a.label AS arm_label, a.type AS arm_type, d.name AS drug_name,
                           collect(DISTINCT g.symbol)[..3] AS target_genes
                    LIMIT $limit
                    """
                    res = session.run(query_drug, {"drugs": entities.drugs, "limit": limit})
                    for record in res:
                        graph_results.append({
                            "source": "GRAPH_DRUG_PATHWAY",
                            "nct_id": record["nct_id"],
                            "title": record["title"],
                            "arm": record["arm_label"],
                            "arm_type": record["arm_type"],
                            "drug": record["drug_name"],
                            "target_genes": record["target_genes"],
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
                    LIMIT 8
                    """
                    res = session.run(
                        query_lit,
                        {
                            "aes": entities.adverse_events,
                            "drugs": entities.drugs,
                            "genes": entities.genes,
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
            logger.warning("Graph query failed (%s); falling back to vector evidence.", e)

        return graph_results

    def retrieve(self, query: str, top_k: int = 8) -> HybridEvidence:
        """Perform hybrid retrieval using Reciprocal Rank Fusion across Graph and Vector indices."""
        # 1. Extract entities
        entities = self.extractor.extract(query)

        # 2. Query Knowledge Graph
        graph_items = self.query_graph(entities, limit=top_k * 2)

        # 3. Vector Similarity Search
        vector_hits = self.vector_store.search(query, top_k=top_k)

        # 4. Reciprocal Rank Fusion (RRF) for Trials
        rrf_scores: Dict[str, float] = {}
        k_constant = 60.0

        # Score graph trial candidates
        seen_graph_ncts = set()
        graph_rank = 1
        for item in graph_items:
            nct = item.get("nct_id")
            if nct and nct not in seen_graph_ncts:
                seen_graph_ncts.add(nct)
                rrf_scores[nct] = rrf_scores.get(nct, 0.0) + (1.0 / (k_constant + graph_rank))
                graph_rank += 1

        # Score vector trial candidates
        vec_rank = 1
        for hit in vector_hits:
            doc_id = hit.doc_id
            if doc_id.startswith("NCT"):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + vec_rank))
                vec_rank += 1

        sorted_ncts = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Fuse paper ranks, preserving each source's order rather than sorting IDs.
        # Each paper gets at most one vote per source.
        paper_scores: Dict[str, float] = {}
        paper_lists = [
            [str(item["pmid"]) for item in graph_items if item.get("pmid")],
            [hit.doc_id for hit in vector_hits
             if hit.chunk_type == "LITERATURE_ABSTRACT" and hit.doc_id],
        ]
        for paper_list in paper_lists:
            for rank, pmid in enumerate(dict.fromkeys(paper_list), start=1):
                paper_scores[pmid] = paper_scores.get(pmid, 0.0) + 1.0 / (k_constant + rank)
        sorted_pmids = sorted(paper_scores, key=paper_scores.get, reverse=True)

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
        )


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_q = "What clinical trials test Amiodarone targeting KCNH2 in patients with Atrial Fibrillation?"
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

    logger.info("Fused ranked NCTs (RRF): %s", evidence.fused_ncts)
    logger.info("Fused PMIDs: %s", evidence.fused_pmids)
