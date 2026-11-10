from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Standardized document chunk for vector database storage."""
    chunk_id: str
    chunk_type: str  # "TRIAL_OVERVIEW", "TRIAL_INCLUSION", "TRIAL_EXCLUSION", "LITERATURE_ABSTRACT"
    doc_id: str      # nct_id or pmid
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def create_trial_chunks(trial: Dict[str, Any]) -> List[DocumentChunk]:
    """Transform a clinical trial dictionary into specialized vector chunks."""
    chunks: List[DocumentChunk] = []
    nct_id = trial["nct_id"]
    title = trial.get("brief_title", "")
    phase = trial.get("phase") or (trial.get("phases", ["NA"])[0] if trial.get("phases") else "NA")
    status = trial.get("overall_status", "UNKNOWN")
    sex = trial.get("sex", "ALL")
    min_age_years = trial.get("min_age_years") if trial.get("min_age_years") is not None else -1.0
    max_age_years = trial.get("max_age_years") if trial.get("max_age_years") is not None else -1.0
    conditions_str = ", ".join(trial.get("conditions", []))
    drugs_str = ", ".join(trial.get("intervention_names", []))

    base_meta = {
        "nct_id": nct_id,
        "phase": phase,
        "status": status,
        "sex": sex,
        "min_age_years": float(min_age_years),
        "max_age_years": float(max_age_years),
        "conditions": conditions_str,
        "drugs": drugs_str,
    }

    # -------------------------------------------------------------
    # 1. TRIAL_OVERVIEW Chunk
    # -------------------------------------------------------------
    summary = trial.get("brief_summary", "").strip()
    outcomes = trial.get("primary_outcome_measures", [])
    outcomes_text = "\nPrimary Endpoints:\n" + "\n".join(f"- {o}" for o in outcomes[:4]) if outcomes else ""

    overview_text = (
        f"Trial [{nct_id}]: {title}\n"
        f"Phase: {phase} | Status: {status} | Sex: {sex}\n"
        f"Conditions: {conditions_str}\n"
        f"Interventions: {drugs_str}\n"
        f"Summary: {summary}"
        f"{outcomes_text}"
    ).strip()

    chunks.append(
        DocumentChunk(
            chunk_id=f"{nct_id}_OVERVIEW",
            chunk_type="TRIAL_OVERVIEW",
            doc_id=nct_id,
            text=overview_text,
            metadata={**base_meta, "chunk_type": "TRIAL_OVERVIEW"},
        )
    )

    # -------------------------------------------------------------
    # 2. TRIAL_INCLUSION Chunk
    # -------------------------------------------------------------
    inclusion = trial.get("inclusion_criteria")
    if inclusion and len(inclusion.strip()) > 20:
        age_str = f"Min Age: {trial.get('minimum_age', 'None')}, Max Age: {trial.get('maximum_age', 'None')}"
        incl_text = (
            f"Trial [{nct_id}]: {title}\n"
            f"Tested Interventions: {drugs_str} | Conditions: {conditions_str}\n"
            f"Eligible Sex: {sex} | {age_str}\n"
            f"INCLUSION CRITERIA:\n"
            f"{inclusion.strip()}"
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{nct_id}_INCLUSION",
                chunk_type="TRIAL_INCLUSION",
                doc_id=nct_id,
                text=incl_text,
                metadata={**base_meta, "chunk_type": "TRIAL_INCLUSION"},
            )
        )

    # -------------------------------------------------------------
    # 3. TRIAL_EXCLUSION Chunk
    # -------------------------------------------------------------
    exclusion = trial.get("exclusion_criteria")
    if exclusion and len(exclusion.strip()) > 20:
        excl_text = (
            f"Trial [{nct_id}]: {title}\n"
            f"Tested Interventions: {drugs_str} | Conditions: {conditions_str}\n"
            f"EXCLUSION CRITERIA:\n"
            f"{exclusion.strip()}"
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{nct_id}_EXCLUSION",
                chunk_type="TRIAL_EXCLUSION",
                doc_id=nct_id,
                text=excl_text,
                metadata={**base_meta, "chunk_type": "TRIAL_EXCLUSION"},
            )
        )

    return chunks


def create_literature_chunk(paper: Dict[str, Any]) -> Optional[DocumentChunk]:
    """Transform a PubMed research paper into a scientific literature vector chunk."""
    pmid = str(paper.get("pmid", ""))
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    year = paper.get("year")
    mesh = paper.get("mesh_terms", [])
    mesh_str = ", ".join(mesh[:6]) if mesh else ""

    if not pmid or not abstract or len(abstract) < 50:
        return None

    lit_text = (
        f"Research Paper [PMID: {pmid}] ({year or 'Unknown'}): {title}\n"
        f"Key Medical Topics: {mesh_str}\n"
        f"Abstract:\n"
        f"{abstract}"
    ).strip()

    return DocumentChunk(
        chunk_id=f"PMID_{pmid}",
        chunk_type="LITERATURE_ABSTRACT",
        doc_id=pmid,
        text=lit_text,
        metadata={
            "pmid": pmid,
            "year": int(year) if year and str(year).isdigit() else -1,
            "mesh_terms": mesh_str,
            "chunk_type": "LITERATURE_ABSTRACT",
        },
    )


def build_all_chunks(trials: List[Dict[str, Any]], papers: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """Build all specialized document chunks from trials and literature datasets."""
    all_chunks: List[DocumentChunk] = []

    for t in trials:
        all_chunks.extend(create_trial_chunks(t))

    for p in papers:
        c = create_literature_chunk(p)
        if c:
            all_chunks.append(c)

    return all_chunks
