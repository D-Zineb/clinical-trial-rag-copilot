import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Standardized document chunk for vector database storage."""
    chunk_id: str
    chunk_type: str  # "TRIAL_OVERVIEW", "TRIAL_INCLUSION", "TRIAL_EXCLUSION", "LITERATURE_ABSTRACT"
    doc_id: str      # nct_id or pmid
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def split_eligibility_section(text: str, target_chars: int = 1200) -> List[str]:
    """Group complete top-level criteria; unclear structure stays intact.

    This is a soft character target, not a token limit. Never cut an oversized
    criterion or its indented notes/sublist. Preserve every source character.
    """
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    if len(text) <= target_chars:
        return [text]
    markers = list(re.finditer(r"(?m)^(?:[*+-] |\d+\. )", text))
    if not markers or markers[0].start() != 0:
        return [text]
    # Wrapped/escaped numbering has unreliable indentation in registry exports.
    if re.search(r"(?m)^[*+-] \d+\\?\.", text):
        return [text]
    styles = {m.group().split()[0] if not m.group()[0].isdigit() else 'number'
              for m in markers}
    if len(styles) != 1:
        return [text]
    if 'number' in styles:
        numbers = [int(m.group().split('.')[0]) for m in markers]
        if numbers != list(range(1, len(numbers) + 1)):
            return [text]
    starts = [m.start() for m in markers] + [len(text)]
    items = [text[a:b] for a, b in zip(starts, starts[1:])]
    # Flat bullets after an introduction may actually be children. Also keep
    # explicitly dependent top-level notes with their source section.
    if any(item.rstrip().endswith(':') for item in items[:-1]) or any(
        re.match(r"(?:[*+-]|\d+\.)\s+(?:note|exception|however|unless)\b", item, re.I)
        for item in items[1:]
    ):
        return [text]
    groups = []
    current = ''
    for item in items:
        if current and len(current) + len(item) > target_chars:
            groups.append(current)
            current = ''
        current += item
    if current:
        groups.append(current)
    return groups or [text]


def _add_section_children(chunks: List[DocumentChunk], section: str) -> None:
    """Retain the full parent under its legacy ID and index contextual children."""
    parent = chunks[-1]
    parts = split_eligibility_section(section)
    if len(parts) == 1:
        return
    parent.metadata['section_role'] = 'parent'
    header = parent.text[:-len(section)]
    offset = 0
    for index, part in enumerate(parts):
        chunks.append(DocumentChunk(
            chunk_id=f'{parent.chunk_id}_CHILD_{index:04d}',
            chunk_type=parent.chunk_type, doc_id=parent.doc_id,
            text=header + part,
            metadata={**parent.metadata, 'section_role': 'child',
                      'parent_chunk_id': parent.chunk_id,
                      'section_start': offset, 'section_end': offset + len(part)},
        ))
        offset += len(part)


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
    if inclusion and inclusion.strip():
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
        _add_section_children(chunks, inclusion.strip())

    # -------------------------------------------------------------
    # 3. TRIAL_EXCLUSION Chunk
    # -------------------------------------------------------------
    exclusion = trial.get("exclusion_criteria")
    if exclusion and exclusion.strip():
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
        _add_section_children(chunks, exclusion.strip())

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
