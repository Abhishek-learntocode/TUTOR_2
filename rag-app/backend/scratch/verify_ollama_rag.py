import os
import sys

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app, llm_service
from app.config import settings


def main():
    print("=" * 60)
    print("OLLAMA PRODUCTION RAG PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Verify Active Provider Configuration
    print(f"[*] Configured llm_provider : {settings.llm_provider}")
    print(f"[*] Configured llm_model    : {settings.llm_model}")
    print(f"[*] App LLMService Provider : {llm_service.provider}")
    print(f"[*] App LLMService Model    : {llm_service.model_name}")
    print("-" * 60)

    assert llm_service.provider == "ollama", f"Expected provider 'ollama', got '{llm_service.provider}'"
    assert settings.llm_provider == "ollama", f"Expected settings provider 'ollama', got '{settings.llm_provider}'"

    client = TestClient(app)

    # 2. Health Endpoint Check
    health_resp = client.get("/health")
    print(f"[*] GET /health Status Code : {health_resp.status_code}")
    print(f"[*] GET /health Payload     : {health_resp.json()}")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}
    print("-" * 60)

    # 3. RAG Query Execution
    test_query = "What is virtual memory?"
    print(f"[*] Executing POST /query...")
    print(f"[*] Question: \"{test_query}\"")

    query_resp = client.post("/query", json={"question": test_query})
    print(f"[*] POST /query Status Code : {query_resp.status_code}")
    
    if query_resp.status_code == 200:
        data = query_resp.json()
        answer = data.get("answer", "")
        context = data.get("context", [])
        print(f"[*] Retrieved Context Chunks: {len(context)}")
        print(f"[*] Answer Length           : {len(answer)}")
        print(f"[*] Answer Snippet          : {repr(answer[:150])}...")
        print("-" * 60)
        print("[RESULT] Active RAG pipeline remains powered by Ollama and behaves as expected.")
    else:
        print(f"[ERROR] POST /query failed: {query_resp.text}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
