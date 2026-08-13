import os
import time
import logging
import requests
from app.config import settings
from app.rag.providers import LLMProvider, record_provider_call

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """Standalone provider for OpenRouter API integration.

    Communicates with OpenRouter's OpenAI-compatible /chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
    ):
        configured_model = model_name or settings.openrouter_model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        super().__init__(provider_name="openrouter", model_name=configured_model)
        self.api_key = api_key or settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        base = base_url or settings.openrouter_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.base_url = base.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.1) -> dict:
        """Sends a completion request to OpenRouter.

        Returns a dictionary containing response metadata and text content.
        Guarantees that sensitive headers/API keys are NEVER included in error messages
        or return objects.
        """
        if not self.api_key:
            res = {
                "status_code": 401,
                "model": self.model_name,
                "actual_model": self.model_name,
                "configured_model": self.model_name,
                "openrouter_request_id": "N/A",
                "success": False,
                "text": "",
                "content": "",
                "usage": {},
                "latency": 0.0,
                "error": "OPENROUTER_API_KEY is not configured or is empty.",
            }
            record_provider_call({
                "request_id": "N/A",
                "provider": "openrouter",
                "configured_model": self.model_name,
                "actual_model": self.model_name,
                "openrouter_request_id": "N/A",
                "latency_sec": 0.0,
                "success": False,
                "error": res["error"],
                "usage": {},
            })
            return res

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Tutor RAG Backend",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        start_time = time.time()
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=(10, 25))
            latency = round(time.time() - start_time, 4)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                content = ""
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                
                returned_model = data.get("model", self.model_name)
                usage = data.get("usage", {})
                req_id = data.get("id") or response.headers.get("x-request-id") or response.headers.get("cf-ray") or "N/A"

                res = {
                    "status_code": 200,
                    "model": returned_model,
                    "actual_model": returned_model,
                    "configured_model": self.model_name,
                    "openrouter_request_id": req_id,
                    "success": True,
                    "text": content.strip(),
                    "content": content.strip(),
                    "usage": usage,
                    "latency": latency,
                    "error": None,
                }
                record_provider_call({
                    "request_id": req_id,
                    "provider": "openrouter",
                    "configured_model": self.model_name,
                    "actual_model": returned_model,
                    "openrouter_request_id": req_id,
                    "latency_sec": latency,
                    "success": True,
                    "error": None,
                    "usage": usage,
                })
                return res
            else:
                err_text = response.text[:200] if response.text else "No response body"
                req_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "N/A"
                err_str = f"OpenRouter HTTP {response.status_code}: {err_text}"
                res = {
                    "status_code": response.status_code,
                    "model": self.model_name,
                    "actual_model": self.model_name,
                    "configured_model": self.model_name,
                    "openrouter_request_id": req_id,
                    "success": False,
                    "text": "",
                    "content": "",
                    "usage": {},
                    "latency": latency,
                    "error": err_str,
                }
                record_provider_call({
                    "request_id": req_id,
                    "provider": "openrouter",
                    "configured_model": self.model_name,
                    "actual_model": self.model_name,
                    "openrouter_request_id": req_id,
                    "latency_sec": latency,
                    "success": False,
                    "error": err_str,
                    "usage": {},
                })
                return res

        except Exception as e:
            latency = round(time.time() - start_time, 4)
            err_str = f"Request failed: {type(e).__name__} - {str(e)}"
            res = {
                "status_code": 500,
                "model": self.model_name,
                "actual_model": self.model_name,
                "configured_model": self.model_name,
                "openrouter_request_id": "N/A",
                "success": False,
                "text": "",
                "content": "",
                "usage": {},
                "latency": latency,
                "error": err_str,
            }
            record_provider_call({
                "request_id": "N/A",
                "provider": "openrouter",
                "configured_model": self.model_name,
                "actual_model": self.model_name,
                "openrouter_request_id": "N/A",
                "latency_sec": latency,
                "success": False,
                "error": err_str,
                "usage": {},
            })
            return res
