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
    phases: List[str] = Field(default_factory=list)
    study_type: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    primary_outcomes: List[Outcome] = Field(default_factory=list)
    lead_sponsor: Optional[str] = None
    overall_status: Optional[str] = None
    why_stopped: Optional[str] = None
    last_update_date: Optional[str] = None


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
            name=interv.get("name", ""),
            type=interv.get("type", ""),
            description=interv.get("description"),
        )
        for interv in raw_interventions
    ]

    # Design & Phases
    design_module = protocol.get("designModule", {})
    phases = design_module.get("phases", [])
    study_type = design_module.get("studyType")

    # Eligibility
    eligibility_module = protocol.get("eligibilityModule", {})
    eligibility_criteria = eligibility_module.get("eligibilityCriteria")

    # Outcomes
    outcomes_module = protocol.get("outcomesModule", {})
    raw_outcomes = outcomes_module.get("primaryOutcomes", [])
    primary_outcomes = [
        Outcome(
            measure=o.get("measure", ""),
            description=o.get("description"),
            time_frame=o.get("timeFrame"),
        )
        for o in raw_outcomes
    ]

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
        phases=phases,
        study_type=study_type,
        brief_summary=brief_summary,
        detailed_description=detailed_description,
        eligibility_criteria=eligibility_criteria,
        primary_outcomes=primary_outcomes,
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
