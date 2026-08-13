import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama
from app.config import settings

logger = logging.getLogger(__name__)


PROVIDER_CALL_LEDGER = []


def record_provider_call(record: Dict[str, Any]):
    """Records an LLM provider call entry into the in-memory call ledger."""
    PROVIDER_CALL_LEDGER.append(record)


def clear_provider_call_ledger():
    """Clears the provider call ledger."""
    global PROVIDER_CALL_LEDGER
    PROVIDER_CALL_LEDGER = []


def get_provider_call_ledger() -> list:
    """Returns a copy of the current provider call ledger."""
    return list(PROVIDER_CALL_LEDGER)


def save_provider_call_ledger(output_path: str):
    """Saves the provider call ledger to a JSON file."""
    import json
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(PROVIDER_CALL_LEDGER, f, indent=2)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Sends prompt to the underlying provider.

        Returns a dictionary with:
          - text: str
          - model: str
          - latency: float
          - usage: dict
          - success: bool
          - error: Optional[str]
        """
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider implementation."""

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(provider_name="ollama", model_name=model_name)
        self.base_url = base_url
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)

    def generate(self, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        import uuid
        call_id = f"ollama-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        try:
            res = self.llm.invoke(prompt)
            latency = round(time.time() - start_time, 4)
            text = res.content.strip() if hasattr(res, "content") else str(res).strip()
            usage = getattr(res, "usage_metadata", {}) or {}
            
            result = {
                "request_id": call_id,
                "text": text,
                "model": self.model_name,
                "actual_model": self.model_name,
                "configured_model": self.model_name,
                "latency": latency,
                "usage": usage,
                "success": True,
                "error": None,
            }
            record_provider_call({
                "request_id": call_id,
                "provider": "ollama",
                "configured_model": self.model_name,
                "actual_model": self.model_name,
                "latency_sec": latency,
                "success": True,
                "error": None,
                "usage": usage,
            })
            return result
        except Exception as e:
            latency = round(time.time() - start_time, 4)
            logger.error(f"[OllamaProvider Error] {e}")
            err_str = f"Ollama error: {str(e)}"
            record_provider_call({
                "request_id": call_id,
                "provider": "ollama",
                "configured_model": self.model_name,
                "actual_model": self.model_name,
                "latency_sec": latency,
                "success": False,
                "error": err_str,
                "usage": {},
            })
            return {
                "request_id": call_id,
                "text": "",
                "model": self.model_name,
                "actual_model": self.model_name,
                "configured_model": self.model_name,
                "latency": latency,
                "usage": {},
                "success": False,
                "error": err_str,
            }


def get_provider(
    provider_name: str,
    model_name: str,
    base_url: str = "",
    api_key: str = "",
) -> LLMProvider:
    """Factory function returning the specified LLMProvider instance."""
    p_name = (provider_name or "ollama").lower()
    if p_name == "openrouter":
        from app.rag.openrouter_provider import OpenRouterProvider
        b_url = base_url if (base_url and "openrouter" in base_url.lower()) else settings.openrouter_base_url
        return OpenRouterProvider(api_key=api_key or settings.openrouter_api_key, model_name=model_name, base_url=b_url)
    else:
        b_url = base_url or settings.llm_base_url
        return OllamaProvider(model_name=model_name, base_url=b_url)
