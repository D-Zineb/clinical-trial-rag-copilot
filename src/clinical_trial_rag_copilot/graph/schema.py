from clinical_trial_rag_copilot.graph.connection import get_db

# Cypher statements to define constraints and indexes
CONSTRAINTS = [
    # 1. Trial Unique Constraint
    """
    CREATE CONSTRAINT trial_nct_id_unique IF NOT EXISTS
    FOR (t:Trial) REQUIRE t.nct_id IS UNIQUE
    """,
    # 2. Drug Unique Constraint
    """
    CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
    FOR (d:Drug) REQUIRE d.name IS UNIQUE
    """,
    # 3. Gene Symbol Unique Constraint
    """
    CREATE CONSTRAINT gene_symbol_unique IF NOT EXISTS
    FOR (g:Gene) REQUIRE g.symbol IS UNIQUE
    """,
    # 4. Condition Name Unique Constraint
    """
    CREATE CONSTRAINT condition_name_unique IF NOT EXISTS
    FOR (c:Condition) REQUIRE c.name IS UNIQUE
    """,
    # 5. Paper PMID Unique Constraint
    """
    CREATE CONSTRAINT paper_pmid_unique IF NOT EXISTS
    FOR (p:Paper) REQUIRE p.pmid IS UNIQUE
    """,
    # 6. Sponsor Name Unique Constraint
    """
    CREATE CONSTRAINT sponsor_name_unique IF NOT EXISTS
    FOR (s:Sponsor) REQUIRE s.name IS UNIQUE
    """,
    # 7. Adverse Event Name Unique Constraint
    """
    CREATE CONSTRAINT adverse_event_name_unique IF NOT EXISTS
    FOR (a:AdverseEvent) REQUIRE a.name IS UNIQUE
    """,
    # 8. ArmGroup Unique Constraint
    """
    CREATE CONSTRAINT arm_group_id_unique IF NOT EXISTS
    FOR (a:ArmGroup) REQUIRE a.id IS UNIQUE
    """
]


def init_schema():
    """Create all required uniqueness constraints and indexes in Neo4j."""
    db = get_db()
    print("Applying Neo4j schema constraints & indexes...")

    with db.driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)

        # Query existing constraints to verify
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record["name"] for record in result]
        print(f"[SUCCESS] Verified {len(constraints)} constraints in Neo4j AuraDB:")
        for name in constraints:
            print(f"   - {name}")


if __name__ == "__main__":
    init_schema()
