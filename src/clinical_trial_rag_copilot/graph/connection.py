import os
from typing import Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

# Load .env variables
load_dotenv()


class Neo4jConnection:
    """Singleton Neo4j database driver connection manager."""
    _instance: Optional["Neo4jConnection"] = None
    _driver: Optional[Driver] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
            cls._instance._init_driver()
        return cls._instance

    def _init_driver(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            raise ValueError(
                "Missing Neo4j credentials! Please ensure NEO4J_URI and "
                "NEO4J_PASSWORD are set in your .env file."
            )

        # Initialize the official Neo4j driver
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        # Verify connection immediately
        self._driver.verify_connectivity()
        print(f"[SUCCESS] Connected to Neo4j AuraDB at: {uri}")

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._init_driver()
        return self._driver

    def close(self):
        """Close driver connection cleanly."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("Neo4j connection closed.")

    def execute_query(self, query: str, parameters: Optional[dict] = None):
        """Execute a Cypher query with optional parameters and return list of records."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


def get_db() -> Neo4jConnection:
    """Helper to get the active Neo4j connection instance."""
    return Neo4jConnection()
