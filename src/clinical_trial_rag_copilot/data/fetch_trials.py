import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel, Field

# Base API endpoint for ClinicalTrials.gov v2
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Ensure output directory exists
DATA_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Intervention(BaseModel):
    name: str = ""
    type: str = ""
    description: Optional[str] = None


class Outcome(BaseModel):
    measure: str = ""
    description: Optional[str] = None
    time_frame: Optional[str] = None


class ClinicalTrial(BaseModel):
    nct_id: str
    brief_title: str
    official_title: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    interventions: List[Intervention] = Field(default_factory=list)
    intervention_names: List[str] = Field(default_factory=list)
    intervention_types: List[str] = Field(default_factory=list)
    phases: List[str] = Field(default_factory=list)
    study_type: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    primary_outcomes: List[Outcome] = Field(default_factory=list)
    primary_outcome_measures: List[str] = Field(default_factory=list)
    primary_outcome_timeframes: List[str] = Field(default_factory=list)
    lead_sponsor: Optional[str] = None
    overall_status: Optional[str] = None
    why_stopped: Optional[str] = None
    last_update_date: Optional[str] = None


def split_eligibility_criteria(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Split raw eligibility criteria text into separate inclusion and exclusion sections."""
    if not text:
        return None, None

    import re
    text_lower = text.lower()
    incl_idx = text_lower.find("inclusion criteria")
    excl_idx = text_lower.find("exclusion criteria")

    inclusion = None
    exclusion = None

    if incl_idx != -1 and excl_idx != -1:
        if incl_idx < excl_idx:
            incl_part = text[incl_idx:excl_idx]
            excl_part = text[excl_idx:]
        else:
            excl_part = text[excl_idx:incl_idx]
            incl_part = text[incl_idx:]

        inclusion = re.sub(r"(?i)^(?:key\s+)?inclusion\s+criteria\s*:?\s*", "", incl_part).strip()
        exclusion = re.sub(r"(?i)^(?:key\s+)?exclusion\s+criteria\s*:?\s*", "", excl_part).strip()
    elif incl_idx != -1:
        incl_part = text[incl_idx:]
        inclusion = re.sub(r"(?i)^(?:key\s+)?inclusion\s+criteria\s*:?\s*", "", incl_part).strip()
    elif excl_idx != -1:
        excl_part = text[excl_idx:]
        exclusion = re.sub(r"(?i)^(?:key\s+)?exclusion\s+criteria\s*:?\s*", "", excl_part).strip()
    else:
        inclusion = text.strip()

    return (inclusion if inclusion else None), (exclusion if exclusion else None)


def parse_study(study_data: Dict[str, Any]) -> Optional[ClinicalTrial]:
    """Parse raw study JSON from ClinicalTrials.gov v2 into a clean ClinicalTrial model."""
    protocol = study_data.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId")
    if not nct_id:
        return None

    # Title & summary
    brief_title = id_module.get("briefTitle", "")
    official_title = id_module.get("officialTitle")

    desc_module = protocol.get("descriptionModule", {})
    brief_summary = desc_module.get("briefSummary")
    detailed_description = desc_module.get("detailedDescription")

    # Conditions
    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])

    # Interventions
    arms_module = protocol.get("armsInterventionsModule", {})
    raw_interventions = arms_module.get("interventions", [])
    interventions = [
        Intervention(
            name=interv.get("name", "").strip(),
            type=interv.get("type", "").strip(),
            description=interv.get("description"),
        )
        for interv in raw_interventions
    ]
    intervention_names = [i.name for i in interventions if i.name]
    intervention_types = sorted(list(set(i.type for i in interventions if i.type)))

    # Design & Phases
    design_module = protocol.get("designModule", {})
    phases = design_module.get("phases", [])
    study_type = design_module.get("studyType")

    # Eligibility
    eligibility_module = protocol.get("eligibilityModule", {})
    eligibility_criteria = eligibility_module.get("eligibilityCriteria")
    inclusion_criteria, exclusion_criteria = split_eligibility_criteria(eligibility_criteria)

    # Outcomes
    outcomes_module = protocol.get("outcomesModule", {})
    raw_outcomes = outcomes_module.get("primaryOutcomes", [])
    primary_outcomes = [
        Outcome(
            measure=o.get("measure", "").strip(),
            description=o.get("description"),
            time_frame=o.get("timeFrame"),
        )
        for o in raw_outcomes
    ]
    primary_outcome_measures = [o.measure for o in primary_outcomes if o.measure]
    primary_outcome_timeframes = [o.time_frame.strip() for o in primary_outcomes if o.time_frame]

    # Sponsor
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")

    # Status / Date
    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus")
    why_stopped = status_module.get("whyStopped")
    last_update_date = status_module.get("lastUpdatePostDateStruct", {}).get("date")

    return ClinicalTrial(
        nct_id=nct_id,
        brief_title=brief_title,
        official_title=official_title,
        conditions=conditions,
        interventions=interventions,
        intervention_names=intervention_names,
        intervention_types=intervention_types,
        phases=phases,
        study_type=study_type,
        brief_summary=brief_summary,
        detailed_description=detailed_description,
        eligibility_criteria=eligibility_criteria,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        primary_outcomes=primary_outcomes,
        primary_outcome_measures=primary_outcome_measures,
        primary_outcome_timeframes=primary_outcome_timeframes,
        lead_sponsor=lead_sponsor,
        overall_status=overall_status,
        why_stopped=why_stopped,
        last_update_date=last_update_date,
    )


def fetch_cardiology_trials(
    condition: str = "Arrhythmia OR \"Atrial Fibrillation\" OR \"Heart Failure\"",
    target_count: int = 300,
    page_size: int = 100,
) -> List[ClinicalTrial]:
    """Fetch clinical trials with pagination from ClinicalTrials.gov v2 API."""
    print(f"Fetching trials for condition: {condition} (Target: {target_count})...")
    trials: List[ClinicalTrial] = []
    next_page_token: Optional[str] = None

    while len(trials) < target_count:
        params: Dict[str, Any] = {
            "query.cond": condition,
            "pageSize": min(page_size, target_count - len(trials)),
            "format": "json",
            "countTotal": "true",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(API_BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

        data = response.json()
        studies = data.get("studies", [])
        if not studies:
            print("No more studies returned.")
            break

        for s in studies:
            parsed = parse_study(s)
            if parsed:
                trials.append(parsed)

        print(f"Fetched {len(trials)}/{target_count} trials...")

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            print("Reached the end of available records.")
            break

        time.sleep(0.5)  # Respect API rate limits

    return trials


if __name__ == "__main__":
    trials = fetch_cardiology_trials(target_count=300)
    output_file = DATA_DIR / "cardiology_trials.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([t.model_dump() for t in trials], f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Successfully saved {len(trials)} structured trials to:")
    print(f"       {output_file}")
