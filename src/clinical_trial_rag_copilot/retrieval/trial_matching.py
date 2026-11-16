"""Source-preserving criterion evidence and conservative patient matching.

Only complete, simple age/ECOG clauses are evaluated automatically. Every other
clause remains unknown, including compound, temporal and variant requirements.
No patient profile is persisted. No trial-level eligibility verdict is produced.
"""
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PARSER_VERSION = "criteria-v1"


class PatientProfile(BaseModel):
    summary: str = Field(min_length=1)
    age_years: float | None = Field(default=None, ge=0, le=130)
    ecog: int | None = Field(default=None, ge=0, le=5)
    # Other reported facts remain in summary; absence never means negative.


class CriterionEvidence(BaseModel):
    criterion_id: str
    trial_id: str
    polarity: Literal["inclusion", "exclusion", "unspecified"]
    source_field: str
    source_hash: str
    start: int
    end: int
    text: str
    parser_version: str = PARSER_VERSION
    review_status: str = "assistant_extracted_unreviewed"
    assessment: Literal["satisfied", "violated", "unknown"] = "unknown"
    explanation: str = "Clause needs structured extraction or reviewer assessment."
    patient_evidence: str | None = None
    source_url: str


class TrialMatch(BaseModel):
    trial_id: str
    title: str
    criteria: list[CriterionEvidence]
    assessment_complete: bool = False
    possible_exclusion: bool = False
    missing_information: list[str] = Field(default_factory=list)
    status: Literal["candidate_for_review"] = "candidate_for_review"


class MatchRequest(BaseModel):
    patient: PatientProfile
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=100, ge=1, le=200)

    @model_validator(mode="after")
    def sufficient_candidates(self):
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self


class MatchResponse(BaseModel):
    mode: str = "criterion_review"
    matches: list[TrialMatch]
    candidate_count: int
    warnings: list[str] = Field(default_factory=list)


def extract_criteria(trial: dict) -> list[CriterionEvidence]:
    """Keep source spans; do not split logical bullets or nested clauses.

    V1 retains each whole section. This deliberately prevents accidental AND/OR
    loss. Atomic splitting requires a reviewed parser; section text is evidence,
    not an automatically validated rule.
    """
    digest = hashlib.sha256(json.dumps(trial, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result = []
    fields = [("inclusion_criteria", "inclusion"), ("exclusion_criteria", "exclusion")]
    if not any(str(trial.get(key) or "").strip() for key, _ in fields):
        fields = [("eligibility_criteria", "unspecified")]
    for key, polarity in fields:
        text = trial.get(key) or ""
        if not isinstance(text, str) or not text.strip():
            continue
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        result.append(CriterionEvidence(
            criterion_id=f"{trial['nct_id']}:{key}:{digest[:16]}",
            trial_id=trial["nct_id"], polarity=polarity, source_field=key,
            source_hash=digest, start=start, end=end, text=text[start:end],
            source_url=f"https://clinicaltrials.gov/study/{trial['nct_id']}",
        ))
    return result


def assess_criterion(criterion: CriterionEvidence, patient: PatientProfile) -> CriterionEvidence:
    """Evaluate only anchored simple numeric clauses; unknown is the default."""
    output = criterion.model_copy(deep=True)
    clause = criterion.text.strip().rstrip(".;")
    match = re.fullmatch(r"(?:age(?:\s+in years)?|ECOG(?:\s+performance status)?)\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)\s*(?:years(?: old)?)?", clause, re.I)
    if not match or criterion.polarity == "unspecified":
        return output
    field = "age_years" if clause.lower().startswith("age") else "ecog"
    value = getattr(patient, field)
    if value is None:
        output.explanation = f"Patient {field} is unknown."
        return output
    op, threshold = match.group(1), float(match.group(2))
    holds = {">=": value >= threshold, "<=": value <= threshold,
             ">": value > threshold, "<": value < threshold, "=": value == threshold}[op]
    satisfied = holds if criterion.polarity == "inclusion" else not holds
    output.assessment = "satisfied" if satisfied else "violated"
    output.patient_evidence = f"{field}={value} (user supplied)"
    output.explanation = f"Explicit numeric comparison; {criterion.polarity} polarity applied."
    return output


def match_trial(trial: dict, patient: PatientProfile) -> TrialMatch:
    criteria = [assess_criterion(c, patient) for c in extract_criteria(trial)]
    unknowns = [c.criterion_id for c in criteria if c.assessment == "unknown"]
    if not criteria:
        unknowns = ["No eligibility criteria available in source record."]
    return TrialMatch(
        trial_id=trial["nct_id"], title=trial.get("brief_title", ""), criteria=criteria,
        # Unreviewed extraction never produces complete eligibility assessment.
        assessment_complete=False,
        possible_exclusion=any(c.assessment == "violated" for c in criteria),
        missing_information=unknowns,
    )


def match_candidates(request: MatchRequest, retriever, trials: list[dict]) -> MatchResponse:
    """Retrieve broadly, then attach complete source eligibility sections.

    Preserve discovery order until a reviewed criterion reranker is qualified.
    Whole sections prevent missing exclusions due to passage truncation.
    """
    evidence = retriever.retrieve(request.patient.summary, top_k=request.candidate_k)
    by_id = {t["nct_id"]: t for t in trials}
    ids = list(dict.fromkeys(evidence.fused_ncts))
    matches = [match_trial(by_id[nct], request.patient) for nct in ids if nct in by_id]
    warnings = [
        "Experimental criterion review: no trial-level eligibility verdict.",
        "Compound, variant and temporal clauses require review; source sections remain intact.",
        "Candidate order follows discovery relevance; criterion reranking is not qualified.",
        "Graph drug-target paths describe interventions, not patient biomarker eligibility.",
    ]
    warnings.extend(evidence.retrieval_warnings)
    missing = [nct for nct in ids if nct not in by_id]
    if missing:
        warnings.append(f"Source records unavailable for {len(missing)} retrieved trial(s).")
    return MatchResponse(matches=matches[:request.top_k], candidate_count=len(matches), warnings=warnings)


def main(argv=None):
    """Export source-preserving eligibility evidence for reviewer annotation."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = args.trials.read_bytes()
    trials = json.loads(payload)
    rows = [c.model_dump() for trial in trials for c in extract_criteria(trial)]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "criteria.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    manifest = {
        "parser_version": PARSER_VERSION,
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "trial_count": len(trials), "section_count": len(rows),
        "trials_without_criteria": sum(not extract_criteria(t) for t in trials),
        "review_status": "unreviewed; sections are not atomic logical criteria",
        "eligibility_verdicts_generated": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
