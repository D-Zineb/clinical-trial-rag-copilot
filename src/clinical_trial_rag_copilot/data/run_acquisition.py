import re
from typing import List

from clinical_trial_rag_copilot.data.fetch_trials import fetch_cardiology_trials
from clinical_trial_rag_copilot.data.fetch_drug_targets import fetch_drug_targets
from clinical_trial_rag_copilot.data.fetch_pubmed import fetch_pubmed_abstracts

from clinical_trial_rag_copilot.data.storage import save_dataset


def clean_drug_name(raw_name: str) -> str:
    """Clean dosage, formulation, and commercial suffixes from drug names."""
    name = re.sub(
        r"(?i)\b(injection|tablet|capsule|oral|infusion|solution|hydrochloride|succinate|tartrate|fumarate|maleate|mesylate|phosphate|sulfate|acetate)\b",
        "",
        raw_name,
    )
    name = re.sub(r"\d+(\.\d+)?\s*(mg|mcg|ml|g)\b", "", name, flags=re.I)
    name = name.split("(")[0].split(",")[0].strip()
    return name.title()


def extract_top_drugs_from_trials(trials_data: List[dict], max_drugs: int = 80) -> List[str]:
    """Extract and prioritize true pharmaceutical drugs tested in trials and study arms."""
    drugs_seen = set()
    ordered_drugs = []

    for t in trials_data:
        # Check interventions list
        for interv in t.get("interventions", []):
            if isinstance(interv, dict) and interv.get("type", "").upper() == "DRUG":
                raw_name = interv.get("name", "")
                cleaned = clean_drug_name(raw_name)
                if len(cleaned) > 2 and not cleaned.lower().startswith("placebo"):
                    if cleaned not in drugs_seen:
                        drugs_seen.add(cleaned)
                        ordered_drugs.append(cleaned)

        # Check arm groups intervention names
        for arm in t.get("arm_groups", []):
            if isinstance(arm, dict):
                for raw_i in arm.get("intervention_names", []):
                    cleaned = clean_drug_name(raw_i)
                    if len(cleaned) > 2 and not cleaned.lower().startswith("placebo"):
                        if cleaned not in drugs_seen:
                            drugs_seen.add(cleaned)
                            ordered_drugs.append(cleaned)

    print(f"Extracted {len(ordered_drugs)} pharmaceutical candidate drugs from clinical trials and study arms.")
    return ordered_drugs[:max_drugs]


def run_full_pipeline(
    trials_target: int = 1000,
    drugs_target: int = 80,
    pubmed_target: int = 800,
    min_date_trials: str = "2024-06-01",
    min_date_pubmed: str = "2024/06/01",
):
    """Orchestrate end-to-end data acquisition for all 3 data sources."""
    print("=" * 60)
    print("STARTING DATA ACQUISITION PIPELINE (POST-CUTOFF: >= June 2024)")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Source 1: Clinical Trials
    # -------------------------------------------------------------
    print(f"\n[STEP 1/3] Fetching Clinical Trials (Target: {trials_target})...")
    trials = fetch_cardiology_trials(target_count=trials_target, min_date=min_date_trials)
    trials_json = [t.model_dump() for t in trials]

    destination = save_dataset("cardiology_trials.json", trials_json)
    print(f"   [OK] Saved {len(trials_json)} records to {destination}")

    # -------------------------------------------------------------
    # 2. Source 2: Pharmacology Targets (ChEMBL / DrugCentral)
    # -------------------------------------------------------------
    print(f"\n[STEP 2/3] Extracting targets for top {drugs_target} drugs...")
    top_drugs = extract_top_drugs_from_trials(trials_json, max_drugs=drugs_target)
    targets_data = fetch_drug_targets(drugs=top_drugs, max_drugs=drugs_target)

    destination = save_dataset("drug_targets_pharmacology.json", targets_data)
    print(f"   [OK] Saved {len(targets_data)} records to {destination}")

    # -------------------------------------------------------------
    # 3. Source 3: PubMed Central Literature
    # -------------------------------------------------------------
    print(f"\n[STEP 3/3] Fetching PubMed literature (Target: {pubmed_target})...")
    pubmed_articles = fetch_pubmed_abstracts(max_results=pubmed_target, min_date=min_date_pubmed)

    destination = save_dataset("pubmed_cardiology_literature.json", pubmed_articles)
    print(f"   [OK] Saved {len(pubmed_articles)} records to {destination}")

    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE: ALL 3 DATASETS SAVED")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
