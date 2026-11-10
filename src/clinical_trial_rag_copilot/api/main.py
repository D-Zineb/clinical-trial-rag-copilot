from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clinical_trial_rag_copilot.agent.copilot import ClinicalGraphRAGCopilot, CopilotAnswer
from clinical_trial_rag_copilot.agent.verifier import GraphFactualityGuardrail, VerificationReport
from clinical_trial_rag_copilot.graph.connection import get_db
from clinical_trial_rag_copilot.observability.tracer import get_recent_traces
from clinical_trial_rag_copilot.retrieval.vector_store import LocalVectorStore, PineconeVectorStore, get_vector_store

app = FastAPI(
    title="Cardiology Clinical Trial & Pharmacology GraphRAG API",
    description=(
        "Production-grade asynchronous REST API for dual-indexed multi-hop "
        "clinical trial and pharmacology question answering with Self-RAG guardrails."
    ),
    version="1.0.0",
)

# Enable CORS for frontend/integration support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_copilot: Optional[ClinicalGraphRAGCopilot] = None
_guardrail: Optional[GraphFactualityGuardrail] = None
_vector_store: Optional[LocalVectorStore | PineconeVectorStore] = None


def get_copilot_instance() -> ClinicalGraphRAGCopilot:
    global _copilot
    if _copilot is None:
        _copilot = ClinicalGraphRAGCopilot()
    return _copilot


def get_guardrail_instance() -> GraphFactualityGuardrail:
    global _guardrail
    if _guardrail is None:
        _guardrail = GraphFactualityGuardrail()
    return _guardrail


def get_vector_store_instance() -> LocalVectorStore | PineconeVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
        if _vector_store.is_indexed():
            _vector_store.load()
    return _vector_store


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Clinician inquiry regarding cardiology trials or pharmacology.",
        json_schema_extra={"example": "Which clinical trials evaluate Amiodarone targeting KCNH2 in Atrial Fibrillation?"},
    )
    top_k: int = Field(default=6, ge=1, le=20, description="Number of top candidates to retrieve.")


class QueryResponse(BaseModel):
    query: str
    answer: str
    guardrail_status: str
    trace_id: str
    total_latency_ms: float
    verified_trials: List[str]
    verified_interactions: List[Dict[str, str]]
    retrieved_ncts: List[str]
    retrieved_pmids: List[str]


class GuardrailVerifyRequest(BaseModel):
    text: str = Field(..., description="Generated text to audit against Knowledge Graph edges and trial IDs.")


class GuardrailVerifyResponse(BaseModel):
    is_valid: bool
    status: str
    verified_trials: List[str]
    unverified_trials: List[str]
    verified_interactions: List[Dict[str, str]]
    guardrail_banner: str


class HealthStatus(BaseModel):
    status: str
    neo4j_connected: bool
    vector_store_ready: bool
    indexed_chunks: int


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthStatus, tags=["System Health"])
def health_check():
    """Verify system connectivity, database health, and index readiness."""
    db = get_db()
    store = get_vector_store_instance()
    is_indexed = store.is_indexed()
    chunk_count = (store.indexed_chunks if isinstance(store, PineconeVectorStore) else len(store.chunks)) if is_indexed else 0

    return HealthStatus(
        status="ok" if (db.is_connected and is_indexed) else "degraded",
        neo4j_connected=db.is_connected,
        vector_store_ready=is_indexed,
        indexed_chunks=chunk_count,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["GraphRAG Inference"])
def ask_question(request: QueryRequest):
    """Execute end-to-end multi-hop GraphRAG query with hybrid retrieval and Self-RAG verification."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    copilot = get_copilot_instance()
    ans: CopilotAnswer = copilot.ask(request.query, top_k=request.top_k)

    return QueryResponse(
        query=ans.query,
        answer=ans.answer,
        guardrail_status=ans.guardrail_status,
        trace_id=ans.trace_id,
        total_latency_ms=ans.total_latency_ms,
        verified_trials=ans.verified_trials,
        verified_interactions=ans.verified_interactions,
        retrieved_ncts=ans.retrieved_ncts,
        retrieved_pmids=ans.retrieved_pmids,
    )


@app.post("/api/v1/guardrail/verify", response_model=GuardrailVerifyResponse, tags=["Guardrail Verification"])
def verify_answer(request: GuardrailVerifyRequest):
    """Audit external or generated clinical text against the live Neo4j Knowledge Graph ground truth."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    guardrail = get_guardrail_instance()
    report: VerificationReport = guardrail.verify(request.text)

    return GuardrailVerifyResponse(
        is_valid=report.is_valid,
        status=report.status,
        verified_trials=report.verified_trials,
        unverified_trials=report.unverified_trials,
        verified_interactions=report.verified_interactions,
        guardrail_banner=report.guardrail_banner,
    )


@app.get("/api/v1/traces", tags=["Observability & Tracing"])
def get_traces(limit: int = 10):
    """Retrieve recent telemetry traces and latency breakdowns."""
    return get_recent_traces(limit=limit)
