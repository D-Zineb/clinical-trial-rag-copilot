import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class BaseLLMClient:
    """Interface for generative language model clients."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client (e.g. gpt-4o-mini, gpt-4o)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


class GoogleGeminiLLMClient(BaseLLMClient):
    """Google Gemini API client (e.g. gemini-1.5-flash) using free tier."""

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        import google.generativeai as genai
        key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"System: {system_prompt}\n\nUser Question & Context:\n{user_prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text or ""


class LocalClinicalSynthesizer(BaseLLMClient):
    """Deterministic local clinical response generator.
    
    Used when no cloud API keys are set, extracting and structuring evidence
    directly from retrieved graph paths and vector chunks with strict citations.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "### Clinical Evidence Synthesis (Local Deterministic Mode)\n\n"
            "Based on the dual-indexed Knowledge Graph and PubMed literature retrieved for this query:\n\n"
            f"{user_prompt[:800]}...\n\n"
            "> [!NOTE]\n"
            "> To generate natural conversational prose using an LLM, configure `OPENAI_API_KEY` or "
            "`GOOGLE_API_KEY` in your `.env` file."
        )


def get_llm_client() -> BaseLLMClient:
    """Factory selecting the appropriate LLM client based on environment variables."""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and len(openai_key.strip()) > 5:
        print("[LLM] Using OpenAI (gpt-4o-mini)")
        return OpenAILLMClient()
    elif google_key and len(google_key.strip()) > 5:
        print("[LLM] Using Google Gemini (gemini-1.5-flash)")
        return GoogleGeminiLLMClient()
    else:
        print("[LLM] No API key detected; using local clinical synthesizer.")
        return LocalClinicalSynthesizer()
