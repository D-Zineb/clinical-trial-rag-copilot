import importlib
from unittest.mock import patch


def test_connection_check_has_no_import_side_effects():
    """Pytest collection must not connect or exit when credentials are absent."""
    with patch("neo4j.GraphDatabase.driver") as driver, patch("dotenv.load_dotenv") as load:
        module = importlib.import_module("clinical_trial_rag_copilot.test_neo4j")
        importlib.reload(module)
    driver.assert_not_called()
    load.assert_not_called()
