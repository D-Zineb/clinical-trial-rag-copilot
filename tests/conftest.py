"""Run automated tests independently of developer credentials and local datasets."""
import os

# Set before test-module imports; dotenv preserves these explicit values.
for name in ("NEO4J_URI", "NEO4J_PASSWORD", "PINECONE_API_KEY", "OPENAI_API_KEY",
             "GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ[name] = ""
os.environ["VECTOR_STORE_PROVIDER"] = "local"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
