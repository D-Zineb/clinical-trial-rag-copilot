import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DATA_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "pubmed_cardiology_literature.json"

# NCBI E-utilities base URLs (Free, official PubMed API)
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pubmed_abstracts(
    query: str = (
        '(Arrhythmia OR "Atrial Fibrillation" OR "Heart Failure") AND '
        '("KCNH2" OR "hERG" OR "SCN5A" OR "CACNA1C" OR "QT prolongation" OR "Torsades de Pointes" OR "adverse effect" OR proarrhythmia)'
    ),
    max_results: int = 800,
    min_date: Optional[str] = "2024/06/01",
) -> List[Dict[str, Any]]:
    """Fetch PubMed IDs matching query, then download their full abstracts via NCBI API.
    
    If min_date is provided (default: '2024/06/01'), restricts search to papers published
    after that date to ensure zero LLM pre-training data contamination.
    """
    full_query = query
    if min_date:
        full_query = f'{query} AND ("{min_date}"[Date - Publication] : "3000"[Date - Publication])'

    date_msg = f" (Published >= {min_date})" if min_date else ""
    print(f"Searching PubMed{date_msg} (Target: {max_results})...")

    # Step 1: ESearch to get PubMed IDs
    params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "pub_date",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Error searching PubMed: {resp.status_code}")
        return []

    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(id_list)} matching PubMed articles. Fetching abstracts...")

    if not id_list:
        return []

    # Step 2: EFetch in batches of 200 using POST (handles any max_results size without timeouts)
    articles = []
    batch_size = 200

    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i : i + batch_size]
        fetch_data = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "xml",
        }
        try:
            fetch_resp = requests.post(EFETCH_URL, data=fetch_data, timeout=45)
            if fetch_resp.status_code != 200:
                print(f"Error fetching batch {i}: {fetch_resp.status_code}")
                continue

            root = ET.fromstring(fetch_resp.content)
            for article in root.findall(".//PubmedArticle"):
                try:
                    medline = article.find("MedlineCitation")
                    if medline is None:
                        continue

                    pmid_el = medline.find("PMID")
                    pmid = pmid_el.text if pmid_el is not None else ""

                    art_data = medline.find("Article")
                    if art_data is None:
                        continue

                    title_el = art_data.find("ArticleTitle")
                    title = "".join(title_el.itertext()) if title_el is not None else ""

                    # Extract abstract paragraphs
                    abstract_el = art_data.find("Abstract")
                    abstract_text = ""
                    if abstract_el is not None:
                        abstract_parts = [
                            "".join(elem.itertext()).strip()
                            for elem in abstract_el.findall("AbstractText")
                        ]
                        abstract_text = "\n".join(filter(None, abstract_parts))

                    # Publication Year
                    pub_date = art_data.find(".//JournalIssue/PubDate/Year")
                    year = pub_date.text if pub_date is not None else None

                    # MeSH Headings (Medical Subject Headings / Key medical concepts)
                    mesh_headings = []
                    mesh_list = medline.find("MeshHeadingList")
                    if mesh_list is not None:
                        for mesh in mesh_list.findall("MeshHeading"):
                            desc = mesh.find("DescriptorName")
                            if desc is not None and desc.text:
                                mesh_headings.append(desc.text)

                    if abstract_text and len(abstract_text) > 100:
                        articles.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract_text,
                            "year": year,
                            "mesh_terms": mesh_headings[:8],  # Keep top terms
                        })
                except Exception:
                    continue
        except Exception as e:
            print(f"Error processing XML batch: {e}")
            continue

        time.sleep(0.3)  # Respect NCBI rate limits

    return articles
