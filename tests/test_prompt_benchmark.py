from collections import Counter
import json

import pytest

from clinical_trial_rag_copilot.agent.prompt_templates import SYSTEM_PROMPT, has_source_evidence
from clinical_trial_rag_copilot.eval.prompt_benchmark import (
    DIMENSIONS, as_evidence, audit_citation_format, build_cases, summarize, validate_grade,
)


def test_benchmark_has_documented_coverage_and_source_validation(tmp_path):
    snapshot, references = tmp_path / "snapshot.json", tmp_path / "references.jsonl"
    snapshot.write_text(json.dumps([dict(nct_id=f"NCT0000000{i}", inclusion_criteria="Minimum age 18.") for i in range(5)]))
    references.write_text("\n".join(json.dumps(dict(id=f"case_{i}", nct_id=f"NCT0000000{i}",
        user_input=f"What minimum age for NCT0000000{i}?", reference="18 years.",
        support=[dict(field="inclusion_criteria", quote="Minimum age 18.")])) for i in range(5)))
    cases = build_cases(snapshot, references)
    assert len({c["id"] for c in cases}) == 30
    assert Counter(c["category"] for c in cases) == {
        "supported": 5, "partial": 5, "irrelevant": 5, "wrong_trial": 3,
        "injection": 4, "conflict": 4, "missing_patient": 2, "empty": 2,
    }
    assert sum(has_source_evidence(as_evidence(c)) for c in cases) == 28
    role, _, remainder = SYSTEM_PROMPT.partition("\n\n")
    assert "research assistant" in role
    assert "ANSWER THE QUESTION" in remainder
    snapshot.write_text(snapshot.read_text().replace("Minimum age 18.", "Minimum age 21."))
    with pytest.raises(ValueError, match="Source quote mismatch"):
        build_cases(snapshot, references)


def test_judge_cannot_return_truthy_strings():
    grade = dict.fromkeys(DIMENSIONS, True)
    grade["reason"] = "Supported."
    assert validate_grade(grade) == grade
    grade["grounding"] = "false"
    with pytest.raises(ValueError):
        validate_grade(grade)


def test_summary_excludes_fallbacks_and_pairs_only_completed_cases():
    rows = [dict(case_id="a", category="supported", variant="with_role", passed=True,
                 grade=dict.fromkeys(DIMENSIONS, True)),
            dict(case_id="a", category="supported", variant="without_role", passed=False,
                 grade=dict.fromkeys(DIMENSIONS, False)),
            dict(case_id="empty", category="empty", variant="with_role", passed=True,
                 grade=dict.fromkeys(DIMENSIONS, True))]
    result = summarize(rows)
    assert result["variants"]["with_role"]["scored"] == 1
    assert result["paired"]["with_role_only"] == 1


def test_citation_audit_preserves_raw_judge_results():
    row = dict(case_id="a", category="supported", variant="with_role", passed=True,
               generation={"text": "Minimum age 18. [1]"}, grade=dict.fromkeys(DIMENSIONS, True))
    audit = audit_citation_format({"results": [row]})
    assert audit["summary"]["variants"]["with_role"]["passed"] == 0
    assert row["passed"] is True
    row["generation"]["text"] = "Minimum age 18. [NCT00000001]"
    assert audit_citation_format({"results": [row]})["violations"] == []
