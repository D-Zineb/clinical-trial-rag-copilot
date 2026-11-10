import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from clinical_trial_rag_copilot.utils.logger import get_logger

load_dotenv()
logger = get_logger("graph.connection")


class Neo4jConnection:
    """Singleton Neo4j database driver connection manager with production resilience."""
    _instance: Optional["Neo4jConnection"] = None
    _driver: Optional[Driver] = None
    _is_connected: bool = False

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
            logger.warning("Missing NEO4J_URI or NEO4J_PASSWORD in .env. Graph features will be offline.")
            self._is_connected = False
            return

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                connection_timeout=5.0,
                max_connection_lifetime=3600,
            )
            self._driver.verify_connectivity()
            self._is_connected = True
            logger.info("Connected to Neo4j AuraDB at: %s", uri)
        except Exception as e:
            self._is_connected = False
            logger.warning(
                "Could not connect to Neo4j AuraDB (%s): %s. System will operate in resilient fallback mode.",
                uri,
                e,
            )

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._driver is not None

    @property
    def driver(self) -> Optional[Driver]:
        return self._driver

    def close(self):
        """Close driver connection cleanly."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._is_connected = False
            print("Neo4j connection closed.")

    def execute_query(self, query: str, parameters: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query safely. Returns empty list if database is unreachable."""
        if not self.is_connected or self._driver is None:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"[Neo4j Query Error] {e}")
            return []


def get_db() -> Neo4jConnection:
    """Helper to get the active Neo4j connection instance."""
    return Neo4jConnection()
