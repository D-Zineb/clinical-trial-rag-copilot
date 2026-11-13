"""Manual connectivity check: python -m clinical_trial_rag_copilot.test_neo4j."""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        print("[ERROR] Configure NEO4J_URI and NEO4J_PASSWORD in .env.")
        return 1
    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            driver.verify_connectivity()
            with driver.session() as session:
                record = session.run("RETURN 'Connection Successful!' AS message, datetime() AS current_time").single()
                print(f"[SUCCESS] {record['message']}")
                print(f"Server Timestamp: {record['current_time']}")
    except Exception as exc:
        print(f"[FAILED] Could not connect to Neo4j AuraDB: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
