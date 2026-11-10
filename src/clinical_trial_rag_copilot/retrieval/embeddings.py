import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingProvider:
    """Base interface for embedding generators."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Cloud embedding provider using OpenAI API (text-embedding-3-small)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [record.embedding for record in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class GoogleGeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Free tier cloud embedding provider using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-embedding-004"):
        import requests
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={self.api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        requests_payload = [{"model": self.model, "content": {"parts": [{"text": t}]}} for t in texts]
        resp = requests.post(self.url, json={"requests": requests_payload}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [item["values"] for item in data.get("embeddings", [])]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LocalTfIdfEmbeddingProvider(BaseEmbeddingProvider):
    """Pure-Python, zero-dependency local embedding provider using scikit-learn.
    
    Guaranteed to run 100% locally on any machine without requiring C++ compilers
    or external cloud API keys, ideal for offline benchmarking.
    """

    def __init__(self, n_features: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=n_features)
        self.is_fitted = False
        self.n_features = n_features

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix.toarray().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            return self.fit_and_embed(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Local vectorizer must embed documents before embedding queries.")
        matrix = self.vectorizer.transform([text])
        return matrix.toarray().tolist()[0]


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory function to instantiate the active embedding provider.
    
    Priority order:
    1. If EMBEDDING_PROVIDER=openai and OPENAI_API_KEY is present -> OpenAI
    2. If EMBEDDING_PROVIDER=google and GOOGLE_API_KEY is present -> Google Gemini (Free)
    3. Default -> Pure local scikit-learn provider (100% offline & free)
    """
    provider_name = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()

    if provider_name == "openai" and os.getenv("OPENAI_API_KEY"):
        print("[Embedding] Using OpenAI cloud embeddings (text-embedding-3-small)")
        return OpenAIEmbeddingProvider()
    elif provider_name in ["google", "gemini"] and os.getenv("GOOGLE_API_KEY"):
        print("[Embedding] Using Google Gemini embeddings (text-embedding-004)")
        return GoogleGeminiEmbeddingProvider()
    else:
        print("[Embedding] Using local & free pure-Python embedding engine (scikit-learn TF-IDF)")
        return LocalTfIdfEmbeddingProvider()
