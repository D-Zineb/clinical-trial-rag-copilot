import re
from typing import Any, Dict, List, Optional, Set, Literal
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
    operators: Dict[Literal['genes', 'drugs', 'conditions', 'adverse_events'], Literal['all', 'any']] = Field(default_factory=dict)
    interpretation_warnings: List[str] = Field(default_factory=list)


def vocabulary_mentions(query, vocabulary):
    """Resolve overlapping spans deterministically, retaining longest mentions."""
    spans = [(m.start(), m.end(), name) for name in sorted(vocabulary)
             for m in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', query, re.I)]
    selected = []
    for span in sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0], x[2])):
        if not any(span[0] < end and start < span[1] for start, end, _ in selected):
            selected.append(span)
    return sorted(selected)


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

        safety_terms = self.known_aes | {"Pneumonitis", "QT Prolongation", "Diarrhea", "Skin Rash", "Neutropenia"}
        vocabularies = dict(genes=self.known_genes, drugs=self.known_drugs,
                            conditions=self.known_conditions, adverse_events=safety_terms)
        mentions = {field: vocabulary_mentions(query, vocab) for field, vocab in vocabularies.items()}
        # Explicit aliases only; do not infer disease equivalence from substrings.
        aliases = {"NSCLC": "Non-Small Cell Lung Cancer", "SCLC": "Small Cell Lung Cancer"}
        for alias, canonical in aliases.items():
            name = next((v for v in self.known_conditions if v.casefold() == canonical.casefold()), None)
            if name:
                mentions['conditions'].extend((a, b, name) for a, b, _ in vocabulary_mentions(query, {alias}))
        # Registry conditions occasionally contain drug/gene names. A single
        # span must not become multiple conjunctive entity types. Longer disease
        # names win; exact ties prefer specific drug/gene vocabularies.
        priority = {'drugs': 0, 'genes': 1, 'adverse_events': 2, 'conditions': 3}
        selected = []
        candidates = [(a, b, name, field) for field, spans in mentions.items() for a, b, name in spans]
        for candidate in sorted(candidates, key=lambda x: (-(x[1]-x[0]), priority[x[3]], x[0], x[2])):
            a, b, _, _ = candidate
            if not any(a < end and start < b for start, end, _, _ in selected):
                selected.append(candidate)
        mentions = {field: [(a, b, name) for a, b, name, kind in selected if kind == field]
                    for field in mentions}
        extracted = {field: list(dict.fromkeys(name for _, _, name in sorted(spans)))
                     for field, spans in mentions.items()}
        operators = {}
        interpretation_warnings = []
        all_mentions = sorted(selected)
        for left, right in zip(all_mentions, all_mentions[1:]):
            if left[3] != right[3] and re.search(r'\bor\b', query[left[1]:right[0]], re.I):
                interpretation_warnings.append("Cross-category OR requires structured review; graph discovery skipped.")
        if re.search(r'\b(mutation|mutant|variant|resistance|eligible|eligibility)\b', q_lower):
            interpretation_warnings.append("Patient biomarker or eligibility scope is unsupported by discovery links; graph discovery skipped.")
        for field, spans in mentions.items():
            spans = sorted(spans)
            connectors = [query[left[1]:right[0]].strip().lower() for left, right in zip(spans, spans[1:])]
            if connectors and all(c == 'or' for c in connectors):
                operators[field] = 'any'
            elif any(re.search(r'\bor\b', c) for c in connectors):
                interpretation_warnings.append(f"Ambiguous Boolean scope for {field}; graph discovery skipped.")
        if re.search(r'\b(no|not|without|excluding|except)\b', q_lower):
            interpretation_warnings.append("Negation requires structured review; graph discovery skipped.")
        extracted_genes, extracted_drugs = extracted['genes'], extracted['drugs']
        extracted_conditions, extracted_aes = extracted['conditions'], extracted['adverse_events']

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
        if re.search(r"\b(female|women)\b", q_lower):
            sex = "FEMALE"
        elif re.search(r"\b(male|men)\b", q_lower):
            sex = "MALE"

        return QueryEntities(
            operators=operators, interpretation_warnings=interpretation_warnings,
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

    def query_graph(self, entities: QueryEntities, limit: Optional[int] = 15, *, strict: bool = False) -> List[Dict[str, Any]]:
        """Execute deterministic multi-hop Cypher queries based on extracted entities."""
        if not self.db.is_connected or self.db.driver is None:
            if strict:
                raise RuntimeError("Graph database is unavailable")
            return []

        if entities.interpretation_warnings:
            return []
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive or None")
        graph_results: List[Dict[str, Any]] = []

        try:
            with self.db.driver.session() as session:
                # Conjoin requested categories before selecting unique trials.
                # These are discovery links, not patient eligibility predicates.
                if (entities.genes or entities.drugs or entities.conditions) and not entities.adverse_events:
                    query_trials = """
                    MATCH (t:Trial)
                    WHERE (size($genes) = 0 OR ALL(symbol IN $genes WHERE EXISTS {
                        MATCH (t)-[:HAS_ARM]->(:ArmGroup)-[:ADMINISTERS]->(d:Drug)-[:HAS_TARGET|INHIBITS]->(g:Gene)
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
                    for field, variable in [('genes', 'symbol'), ('drugs', 'name'), ('conditions', 'name')]:
                        if entities.operators.get(field) == 'any':
                            query_trials = query_trials.replace(f'ALL({variable} IN ${field}', f'ANY({variable} IN ${field}')
                    if limit is None:
                        query_trials = query_trials.replace(' LIMIT $limit', '')
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
                if (entities.adverse_events or entities.genes or entities.drugs) and not entities.conditions:
                    query_lit = """
                    MATCH (p:Paper)
                    // Require each requested category before optional expansion.
                    WHERE (size($aes) = 0 OR ALL(value IN $aes WHERE EXISTS {
                        MATCH (p)-[:INDEXED_ADVERSE_EVENT|REPORTS_ADVERSE_EVENT]->(matched_ae:AdverseEvent)
                        WHERE matched_ae.name = value
                    }))
                      AND (size($drugs) = 0 OR ALL(value IN $drugs WHERE EXISTS {
                        MATCH (p)-[:MENTIONS_DRUG|CITES_DRUG]->(matched_drug:Drug)
                        WHERE matched_drug.name = value
                    }))
                      AND (size($genes) = 0 OR ALL(value IN $genes WHERE EXISTS {
                        MATCH (p)-[:MENTIONS_GENE|STUDIES_GENE]->(matched_gene:Gene)
                        WHERE matched_gene.symbol = value
                    }))
                    OPTIONAL MATCH (p)-[:INDEXED_ADVERSE_EVENT|REPORTS_ADVERSE_EVENT]->(ae:AdverseEvent)
                    OPTIONAL MATCH (p)-[:MENTIONS_DRUG|CITES_DRUG]->(d:Drug)
                    OPTIONAL MATCH (p)-[:MENTIONS_GENE|STUDIES_GENE]->(g:Gene)
                    RETURN p.pmid AS pmid, p.title AS title, p.year AS year,
                           collect(DISTINCT ae.name) AS adverse_events,
                           collect(DISTINCT d.name)[..4] AS cited_drugs,
                           collect(DISTINCT g.symbol)[..4] AS studied_genes
                    ORDER BY pmid
                    LIMIT $limit
                    """
                    for field, parameter in [('adverse_events', 'aes'), ('drugs', 'drugs'), ('genes', 'genes')]:
                        if entities.operators.get(field) == 'any':
                            query_lit = query_lit.replace(f'ALL(value IN ${parameter}', f'ANY(value IN ${parameter}')
                    if limit is None:
                        query_lit = query_lit.replace('LIMIT $limit', '')
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

    def retrieve(self, query: str, top_k: int = 8, candidate_k: Optional[int] = None) -> HybridEvidence:
        """Retrieve ranked dense evidence with supplemental graph discoveries."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidate_k = max(100, top_k * 4) if candidate_k is None else candidate_k
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
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

        warnings.extend(entities.interpretation_warnings)
        # Identifiers are exact lookup keys, not semantic similarity terms.
        requested_ncts = list(dict.fromkeys(re.findall(r"\bNCT\d{8}\b", query.upper())))
        fetched_chunks = {}
        for nct in requested_ncts:
            try:
                fetched_chunks[nct] = [c for c in self.vector_store.get_chunks_by_doc_id(nct)
                                       if c.get("doc_id") == nct]
                if not fetched_chunks[nct]:
                    warnings.append(f"No indexed source text found for requested trial {nct}.")
            except Exception as exc:
                logger.warning("Exact trial lookup failed for %s: %s", nct, exc)
                warnings.append(f"Exact source lookup failed for requested trial {nct}; review available evidence.")
        exact_ids = [nct for nct in requested_ncts if fetched_chunks.get(nct)]
        if len(exact_ids) > top_k:
            warnings.append("Requested trial count exceeds top_k; only the first requested trials fit the result limit.")
        # Enumerate graph matches before relevance scoring; never truncate by ID.
        try:
            graph_items = self.query_graph(entities, limit=None, strict=True)
        except Exception as exc:
            logger.warning("Graph retrieval failed: %s", exc)
            warnings.append("Graph retrieval failed; using dense candidates.")
            graph_items = []
        try:
            vector_hits = self.vector_store.search(query, top_k=candidate_k)
        except Exception:
            if not exact_ids:
                raise
            vector_hits = []
            warnings.append("Dense search failed; using exact trial text and available graph candidates.")
        scores = {}
        for hit in vector_hits:
            scores[hit.doc_id] = max(scores.get(hit.doc_id, -float('inf')), hit.score)
        graph_ids = {str(item.get('nct_id') or item.get('pmid')) for item in graph_items
                     if item.get('nct_id') or item.get('pmid')}
        if graph_ids:
            # Shared stored embeddings, max chunk cosine per document.
            try:
                graph_scores = self.vector_store.score_documents(query, sorted(graph_ids))
                scores.update(graph_scores)
            except Exception as exc:
                logger.warning("Graph candidate scoring failed: %s", exc)
                warnings.append("Graph candidate scoring failed; retaining available dense scores.")
            missing = graph_ids - scores.keys()
            if missing:
                warnings.append(f"Stored embeddings unavailable for {len(missing)} graph candidates; ranked after scored candidates.")
        all_ids = set(scores) | graph_ids | set(exact_ids)
        exact_order = {doc: rank for rank, doc in enumerate(exact_ids)}
        has_constraints = bool(entities.genes or entities.drugs or entities.conditions or entities.adverse_events)
        # Constraint matches first only for supported interpretation; absent links
        # never delete dense candidates because graph coverage can be incomplete.
        ordered = sorted(all_ids, key=lambda doc: (
            exact_order.get(doc, len(exact_order)),
            doc not in scores,
            -(int(doc in graph_ids) if has_constraints and not entities.interpretation_warnings else 0),
            -scores.get(doc, -float('inf')), doc))
        sorted_ncts = [doc for doc in ordered if doc.startswith('NCT')]
        sorted_pmids = [doc for doc in ordered if not doc.startswith('NCT')]

        # Include source text for graph-only winners in the synthesis context.
        winners = set(sorted_ncts[:top_k] + sorted_pmids[:top_k])
        represented = {hit.doc_id for hit in vector_hits}
        for doc in sorted(winners - represented):
            if doc not in fetched_chunks:
                fetched_chunks[doc] = self.vector_store.get_chunks_by_doc_id(doc)
            for chunk in fetched_chunks[doc]:
                vector_hits.append(SearchResult(**chunk, score=scores.get(doc, 0.0)))

        # Sibling Chunk Expansion: Pull companion chunks (Overview, Inclusion, Exclusion) for top winning trials
        sibling_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for nct in sorted_ncts[:min(3, top_k)]:
            if nct not in fetched_chunks:
                fetched_chunks[nct] = self.vector_store.get_chunks_by_doc_id(nct)
            sibling_chunks[nct] = fetched_chunks[nct]

        return HybridEvidence(
            query=query,
            extracted_entities=entities,
            graph_evidence=sorted([item for item in graph_items if str(item.get("nct_id") or item.get("pmid")) in winners], key=lambda item: ordered.index(str(item.get("nct_id") or item.get("pmid")))),
            vector_evidence=sorted(vector_hits, key=lambda hit: (ordered.index(hit.doc_id), -hit.score))[:top_k],
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
