import json
from pathlib import Path
from typing import Any, Dict, List, Set
import requests

DATA_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_FILE = DATA_DIR / "cardiology_trials.json"
OUTPUT_FILE = DATA_DIR / "drug_targets_pharmacology.json"

# Open pharmacology API (MyChem.info provides harmonized ChEMBL, DrugBank, and DrugCentral data)
MYCHEM_API_URL = "https://mychem.info/v1/query"


def extract_unique_drugs_from_trials() -> List[str]:
    """Extract distinct drug names from our clinical trials dataset."""
    if not TRIALS_FILE.exists():
        print(f"Error: {TRIALS_FILE} not found. Run fetch_trials.py first.")
        return []

    with open(TRIALS_FILE, "r", encoding="utf-8") as f:
        trials = json.load(f)

    drugs: Set[str] = set()
    for t in trials:
        for interv in t.get("interventions", []):
            # Only focus on actual drug / biological interventions
            if interv.get("type", "").upper() in ["DRUG", "BIOLOGICAL", "DIETARY_SUPPLEMENT"]:
                name = interv.get("name", "").strip()
                # Clean up simple noisy strings (e.g., doses like "50mg", "Placebo", etc.)
                if name and not name.lower().startswith("placebo") and len(name) > 2:
                    # Clean up strings like "Amiodarone 200 mg" -> "Amiodarone"
                    clean_name = name.split("(")[0].split(",")[0].strip()
                    drugs.add(clean_name)

    print(f"Found {len(drugs)} unique drug candidate names in clinical trials.")
    return sorted(list(drugs))


def fetch_drug_targets(drugs: List[str], max_drugs: int = 50) -> List[Dict[str, Any]]:
    """Query open pharmacology data to get target genes, mechanisms, and drug classes."""
    # We query top candidate drugs to keep Day 1 fast and robust
    candidate_drugs = drugs[:max_drugs]
    results = []

    print(f"Querying molecular target data for {len(candidate_drugs)} drugs via MyChem / ChEMBL...")

    for drug_name in candidate_drugs:
        try:
            params = {
                "q": drug_name,
                "fields": "drugcentral.bioactivity,drugcentral.drug_use,chembl.pref_name,chembl.drug_mechanisms",
                "size": 1,
            }
            resp = requests.get(MYCHEM_API_URL, params=params, timeout=10)
            if resp.status_code != 200:
                continue

            hits = resp.json().get("hits", [])
            if not hits:
                continue

            hit = hits[0]
            targets = []

            # Extract targets from drugcentral bioactivity
            bioactivity = hit.get("drugcentral", {}).get("bioactivity", [])
            if isinstance(bioactivity, dict):
                bioactivity = [bioactivity]

            for bio in bioactivity:
                target_name = bio.get("target_name")
                gene_symbol = bio.get("gene_symbol")
                action_type = bio.get("action_type") or "target"
                if gene_symbol or target_name:
                    targets.append({
                        "gene_symbol": gene_symbol,
                        "target_name": target_name,
                        "action_type": action_type,
                    })

            # Extract mechanisms from ChEMBL
            chembl_mechanisms = hit.get("chembl", {}).get("drug_mechanisms", [])
            if isinstance(chembl_mechanisms, dict):
                chembl_mechanisms = [chembl_mechanisms]

            for m in chembl_mechanisms:
                mech_target = m.get("target_name")
                action = m.get("mechanism_of_action")
                if mech_target:
                    targets.append({
                        "gene_symbol": None,
                        "target_name": mech_target,
                        "action_type": action or "mechanism",
                    })

            # Deduplicate targets
            seen = set()
            clean_targets = []
            for tgt in targets:
                key = (tgt.get("gene_symbol"), tgt.get("target_name"), tgt.get("action_type"))
                if key not in seen and (tgt.get("gene_symbol") or tgt.get("target_name")):
                    seen.add(key)
                    clean_targets.append(tgt)

            if clean_targets:
                results.append({
                    "drug_name": drug_name,
                    "target_count": len(clean_targets),
                    "targets": clean_targets,
                })
                print(f"  [+] {drug_name}: found {len(clean_targets)} targets/mechanisms.")

        except Exception as e:
            # Continue on individual drug errors
            continue

    return results
