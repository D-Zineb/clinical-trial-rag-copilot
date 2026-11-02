import json
import re
from pathlib import Path
from typing import List

from clinical_trial_rag_copilot.data.fetch_trials import fetch_cardiology_trials
from clinical_trial_rag_copilot.data.fetch_drug_targets import fetch_drug_targets
from clinical_trial_rag_copilot.data.fetch_pubmed import fetch_pubmed_abstracts

DATA_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_drug_name(raw_name: str) -> str:
    """Clean dosage, formulation, and commercial suffixes from drug names."""
    name = re.sub(r"(?i)\b(injection|tablet|capsule|oral|infusion|solution|hydrochloride|acetate)\b", "", raw_name)
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

    trials_file = DATA_DIR / "cardiology_trials.json"
    with open(trials_file, "w", encoding="utf-8") as f:
        json.dump(trials_json, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved {len(trials_json)} trials to {trials_file.name}")

    # -------------------------------------------------------------
    # 2. Source 2: Pharmacology Targets (ChEMBL / DrugCentral)
    # -------------------------------------------------------------
    print(f"\n[STEP 2/3] Extracting targets for top {drugs_target} drugs...")
    top_drugs = extract_top_drugs_from_trials(trials_json, max_drugs=drugs_target)
    targets_data = fetch_drug_targets(drugs=top_drugs, max_drugs=drugs_target)

    targets_file = DATA_DIR / "drug_targets_pharmacology.json"
    with open(targets_file, "w", encoding="utf-8") as f:
        json.dump(targets_data, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved target profiles for {len(targets_data)} drugs to {targets_file.name}")

    # -------------------------------------------------------------
    # 3. Source 3: PubMed Central Literature
    # -------------------------------------------------------------
    print(f"\n[STEP 3/3] Fetching PubMed literature (Target: {pubmed_target})...")
    pubmed_articles = fetch_pubmed_abstracts(max_results=pubmed_target, min_date=min_date_pubmed)

    pubmed_file = DATA_DIR / "pubmed_cardiology_literature.json"
    with open(pubmed_file, "w", encoding="utf-8") as f:
        json.dump(pubmed_articles, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved {len(pubmed_articles)} research papers to {pubmed_file.name}")

    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE: ALL 3 SOURCES FRESH & SYNCHRONIZED")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
