import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env file
load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

if not uri or not password:
    print("[ERROR] NEO4J_URI or NEO4J_PASSWORD not found in .env file.")
    print("Please create a .env file based on .env.example with your AuraDB credentials.")
    exit(1)

print(f"Connecting to Neo4j AuraDB at: {uri} ...")

try:
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful!' AS message, datetime() AS current_time")
            record = result.single()
            print(f"[SUCCESS] {record['message']}")
            print(f"Server Timestamp: {record['current_time']}")
except Exception as e:
    print(f"[FAILED] Could not connect to Neo4j AuraDB: {e}")
