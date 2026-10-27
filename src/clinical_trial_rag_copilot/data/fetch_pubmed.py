import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List
import requests

DATA_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "pubmed_cardiology_literature.json"

# NCBI E-utilities base URLs (Free, official PubMed API)
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pubmed_abstracts(
    query: str = '(Arrhythmia OR "Atrial Fibrillation") AND ("ion channel" OR SCN5A OR KCNH2 OR hERG OR "adverse effect" OR proarrhythmia)',
    max_results: int = 150,
) -> List[Dict[str, Any]]:
    """Fetch PubMed IDs matching query, then download their full abstracts via NCBI API."""
    print(f"Searching PubMed for: {query}...")

    # Step 1: ESearch to get PubMed IDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Error searching PubMed: {resp.status_code}")
        return []

    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(id_list)} matching PubMed articles. Fetching abstracts...")

    if not id_list:
        return []

    # Step 2: EFetch to get XML details for those PMIDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=30)
    if fetch_resp.status_code != 200:
        print(f"Error fetching article XML: {fetch_resp.status_code}")
        return []

    # Step 3: Parse XML into clean articles
    articles = []
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
        except Exception as e:
            continue

    return articles
