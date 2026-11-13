from clinical_trial_rag_copilot.data.storage import load_dataset
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from clinical_trial_rag_copilot.graph.connection import get_db
from clinical_trial_rag_copilot.graph.schema import init_schema

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "output"


def clean_name(text: str) -> str:
    """Helper to clean whitespace and title-case."""
    return text.strip().title() if text else ""


def clean_drug_name(raw_name: str) -> str:
    """Clean dosage, formulation, and commercial prefixes/suffixes from drug names."""
    if not raw_name:
        return ""
    name = re.sub(
        r"(?i)\b(injection|tablet|capsule|oral|infusion|solution|hydrochloride|succinate|tartrate|fumarate|maleate|mesylate|phosphate|sulfate|acetate)\b",
        "",
        name,
    )
    name = re.sub(r"\d+(\.\d+)?\s*(mg|mcg|ml|g)\b", "", name, flags=re.I)
    name = name.split("(")[0].split(",")[0].strip()
    return name.title()


def ingest_trials(batch_size: int = 250):
    """Batch ingest clinical trials, conditions, sponsors, arm groups, and drug links into Neo4j."""
    trials = load_dataset("cardiology_trials.json")

    db = get_db()
    print(f"\n[1/3] Ingesting {len(trials)} Clinical Trials into Neo4j...")

    # Prepare normalized records for UNWIND batching
    records = []
    for t in trials:
        nct_id = t["nct_id"]
        # Extract drug-type intervention names specifically
        drug_names = []
        for interv in t.get("interventions", []):
            if isinstance(interv, dict) and interv.get("type", "").upper() == "DRUG":
                raw = interv.get("name", "")
                cleaned = clean_drug_name(raw)
                if len(cleaned) > 2 and not cleaned.lower().startswith("placebo"):
                    drug_names.append(cleaned)

        # Parse arm groups and link them to cleaned drug names
        arm_groups_data = []
        for idx, arm in enumerate(t.get("arm_groups", [])):
            if not isinstance(arm, dict):
                continue
            label = arm.get("label", "").strip() or f"Arm_{idx+1}"
            arm_drugs = []
            for raw_i in arm.get("intervention_names", []):
                c_drug = clean_drug_name(raw_i)
                if len(c_drug) > 2 and not c_drug.lower().startswith("placebo"):
                    arm_drugs.append(c_drug)
                    drug_names.append(c_drug)

            arm_groups_data.append({
                "id": f"{nct_id}_{label}",
                "label": label,
                "type": arm.get("type") or "OTHER",
                "description": arm.get("description"),
                "drugs": list(set(arm_drugs)),
            })

        records.append({
            "nct_id": nct_id,
            "brief_title": t.get("brief_title", ""),
            "phase": t.get("phases", ["NA"])[0] if t.get("phases") else "NA",
            "study_type": t.get("study_type", "UNKNOWN"),
            "status": t.get("overall_status", "UNKNOWN"),
            "why_stopped": t.get("why_stopped"),
            "last_update_date": t.get("last_update_date"),
            "lead_sponsor": t.get("lead_sponsor"),
            "conditions": [c.strip().title() for c in t.get("conditions", []) if c],
            "drugs": list(set(drug_names)),
            "sex": t.get("sex", "ALL"),
            "healthy_volunteers": t.get("healthy_volunteers"),
            "minimum_age": t.get("minimum_age"),
            "maximum_age": t.get("maximum_age"),
            "min_age_years": t.get("min_age_years"),
            "max_age_years": t.get("max_age_years"),
            "std_ages": t.get("std_ages", []),
            "arm_groups": arm_groups_data,
        })

    cypher_query = """
    UNWIND $batch AS row
    // 1. Create or Match Trial Node with Demographics
    MERGE (t:Trial {nct_id: row.nct_id})
    SET t.brief_title = row.brief_title,
        t.phase = row.phase,
        t.study_type = row.study_type,
        t.status = row.status,
        t.why_stopped = row.why_stopped,
        t.last_update_date = row.last_update_date,
        t.sex = row.sex,
        t.healthy_volunteers = row.healthy_volunteers,
        t.minimum_age = row.minimum_age,
        t.maximum_age = row.maximum_age,
        t.min_age_years = row.min_age_years,
        t.max_age_years = row.max_age_years,
        t.std_ages = row.std_ages

    // 2. Sponsor Node & Edge
    FOREACH (sp IN CASE WHEN row.lead_sponsor IS NOT NULL THEN [row.lead_sponsor] ELSE [] END |
        MERGE (s:Sponsor {name: sp})
        MERGE (t)-[:SPONSORED_BY]->(s)
    )

    // 3. Condition Nodes & Edges
    FOREACH (cond IN row.conditions |
        MERGE (c:Condition {name: cond})
        MERGE (t)-[:TARGETS_CONDITION]->(c)
    )

    // 4. Drug Nodes & Direct Tests_Drug Edges
    FOREACH (drug IN row.drugs |
        MERGE (d:Drug {name: drug})
        MERGE (t)-[:TESTS_DRUG]->(d)
    )

    // 5. ArmGroup Nodes & Comparative Edges
    FOREACH (arm IN row.arm_groups |
        MERGE (a:ArmGroup {id: arm.id})
        SET a.label = arm.label,
            a.type = arm.type,
            a.description = arm.description
        MERGE (t)-[:HAS_ARM]->(a)
        FOREACH (dname IN arm.drugs |
            MERGE (d:Drug {name: dname})
            MERGE (a)-[:ADMINISTERS]->(d)
        )
    )
    """

    with db.driver.session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            session.run(cypher_query, {"batch": batch})
            print(f"   [OK] Ingested trials batch {i + len(batch)}/{len(records)}")


def ingest_drug_targets(batch_size: int = 100):
    """Ingest pharmacology target genes and mechanisms of action."""
    targets_data = load_dataset("drug_targets_pharmacology.json")

    db = get_db()
    print(f"\n[2/3] Ingesting Drug Targets for {len(targets_data)} Drugs...")

    records = []
    for item in targets_data:
        drug_name = item.get("drug_name", "").strip().title()
        for tgt in item.get("targets", []):
            symbol = tgt.get("gene_symbol")
            if symbol and len(symbol) >= 2:
                records.append({
                    "drug_name": drug_name,
                    "gene_symbol": symbol.strip().upper(),
                    "target_name": tgt.get("target_name"),
                    "target_class": tgt.get("target_class", "Protein"),
                    "action_type": (tgt.get("action_type") or "TARGET").upper(),
                })

    cypher_query = """
    UNWIND $batch AS row
    MERGE (d:Drug {name: row.drug_name})
    MERGE (g:Gene {symbol: row.gene_symbol})
    ON CREATE SET g.name = row.target_name,
                  g.target_class = row.target_class

    MERGE (d)-[r:INHIBITS]->(g)
    SET r.action_type = row.action_type
    """

    with db.driver.session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            session.run(cypher_query, {"batch": batch})
            print(f"   [OK] Ingested targets batch {i + len(batch)}/{len(records)}")


def ingest_pubmed_literature(batch_size: int = 200):
    """Ingest PubMed papers, linking them to adverse events, drugs, and target genes."""
    papers = load_dataset("pubmed_cardiology_literature.json")

    db = get_db()
    print(f"\n[3/3] Ingesting {len(papers)} PubMed Papers and Safety Links...")

    # Query currently existing drugs and genes in the graph to build cross-links
    with db.driver.session() as session:
        existing_drugs = [r["name"] for r in session.run("MATCH (d:Drug) RETURN d.name AS name")]
        existing_genes = [r["symbol"] for r in session.run("MATCH (g:Gene) RETURN g.symbol AS symbol")]

    drug_pattern = re.compile(r"\b(" + "|".join(map(re.escape, existing_drugs)) + r")\b", re.I) if existing_drugs else None
    gene_pattern = re.compile(r"\b(" + "|".join(map(re.escape, existing_genes)) + r")\b", re.I) if existing_genes else None

    # Key cardiac adverse events to track
    adverse_events_whitelist = {
        "Long QT Syndrome", "Torsades de Pointes", "Death, Sudden, Cardiac",
        "Arrhythmias, Cardiac", "Ventricular Fibrillation", "Heart Failure"
    }

    records = []
    for p in papers:
        abstract = p.get("abstract", "")
        # Detect cited drugs and studied genes directly in the scientific abstract
        cited_drugs = list(set(m.group(0).title() for m in drug_pattern.finditer(abstract))) if drug_pattern else []
        studied_genes = list(set(m.group(0).upper() for m in gene_pattern.finditer(abstract))) if gene_pattern else []
        
        # Filter MeSH terms to cardiac conditions & adverse events
        mesh = [m.title() for m in p.get("mesh_terms", []) if any(ae.lower() in m.lower() for ae in adverse_events_whitelist)]

        records.append({
            "pmid": p["pmid"],
            "title": p.get("title", ""),
            "year": int(p["year"]) if p.get("year") and str(p["year"]).isdigit() else None,
            "adverse_events": mesh,
            "cited_drugs": cited_drugs,
            "studied_genes": studied_genes,
        })

    cypher_query = """
    UNWIND $batch AS row
    MERGE (p:Paper {pmid: row.pmid})
    SET p.title = row.title,
        p.year = row.year

    // Link paper to Adverse Events
    FOREACH (ae IN row.adverse_events |
        MERGE (a:AdverseEvent {name: ae})
        MERGE (p)-[:REPORTS_ADVERSE_EVENT]->(a)
    )

    // Link paper to cited drugs
    FOREACH (drug IN row.cited_drugs |
        MERGE (d:Drug {name: drug})
        MERGE (p)-[:CITES_DRUG]->(d)
    )

    // Link paper to studied genes
    FOREACH (gene IN row.studied_genes |
        MERGE (g:Gene {symbol: gene})
        MERGE (p)-[:STUDIES_GENE]->(g)
    )
    """

    with db.driver.session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            session.run(cypher_query, {"batch": batch})
            print(f"   [OK] Ingested papers batch {i + len(batch)}/{len(records)}")


def print_graph_summary():
    """Print high-level statistics of the newly built Knowledge Graph."""
    db = get_db()
    with db.driver.session() as session:
        node_counts = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY count DESC
        """)

        rel_counts = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(*) AS count
        ORDER BY count DESC
        """)

        print("\n" + "=" * 50)
        print("[COMPLETE] KNOWLEDGE GRAPH CONSTRUCTION COMPLETE")
        print("=" * 50)
        print("\n--- NODE INVENTORY ---")
        for r in node_counts:
            print(f"  (:{r['label']}): {r['count']} nodes")

        print("\n--- RELATIONSHIP (EDGE) INVENTORY ---")
        for r in rel_counts:
            print(f"  [:{r['relationship']}]: {r['count']} edges")
        print("=" * 50)


def build_knowledge_graph():
    """Execute end-to-end Knowledge Graph construction."""
    init_schema()
    ingest_trials()
    ingest_drug_targets()
    ingest_pubmed_literature()
    print_graph_summary()


if __name__ == "__main__":
    build_knowledge_graph()
