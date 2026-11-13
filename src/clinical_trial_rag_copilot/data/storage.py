"""Read source datasets from a local directory or an explicit S3 prefix."""
import json
import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(__file__).resolve().parent / "output"
DATASETS = {"cardiology_trials.json", "pubmed_cardiology_literature.json",
            "drug_targets_pharmacology.json"}


def load_dataset(name):
    """Load one JSON dataset; S3 failures never silently use local data.

    RAG_DATA_S3_URI selects S3 (AWS CLI required). Otherwise RAG_DATA_DIR
    overrides the existing local output directory. EC2 uses its IAM role.
    """
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    prefix = os.getenv("RAG_DATA_S3_URI", "").strip()
    if prefix:
        if not prefix.startswith("s3://") or not prefix[5:].split("/", 1)[0]:
            raise ValueError("RAG_DATA_S3_URI must be an s3://bucket/prefix URI")
        uri = prefix.rstrip("/") + "/" + name
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
                check=True, capture_output=True, timeout=120,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("S3 dataset loading requires AWS CLI v2 installed") from exc
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot load {uri}; check object and IAM read permissions") from exc
        data = json.loads(result.stdout)
    else:
        path = Path(os.getenv("RAG_DATA_DIR") or DATA_DIR) / name
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"{name} must contain a JSON array of objects")
    return data
