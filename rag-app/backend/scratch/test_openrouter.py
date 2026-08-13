import os
import sys

from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load .env file from backend root directory
env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from app.config import settings
from app.rag.openrouter_provider import OpenRouterProvider


def main():
    print("=" * 60)
    print("OPENROUTER STANDALONE CONNECTIVITY TEST")
    print("=" * 60)

    # 1. Verify Configuration
    api_key_present = bool(settings.openrouter_api_key and settings.openrouter_api_key.strip())
    model_name = settings.openrouter_model or "openrouter/free"

    print(f"[*] OPENROUTER_API_KEY Configured : {'Yes (Key Present)' if api_key_present else 'No (MISSING)'}")
    print(f"[*] OPENROUTER_MODEL Configured   : {model_name}")
    print(f"[*] OPENROUTER_BASE_URL Configured: {settings.openrouter_base_url}")
    print("-" * 60)

    if not api_key_present:
        print("[ERROR] OPENROUTER_API_KEY is missing from environment/configuration.")
        print("[ERROR] Aborting connectivity test.")
        sys.exit(1)

    # 2. Instantiate Provider
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        model_name=model_name,
        base_url=settings.openrouter_base_url,
    )

    # 3. Send Single Test Request
    prompt = "Explain virtual memory in two sentences."
    print(f"[*] Sending Test Request...")
    print(f"[*] Prompt: \"{prompt}\"")
    print("-" * 60)

    result = provider.generate(prompt=prompt, temperature=0.1)

    # 4. Print Results (Sanitized - No Secrets Exposed)
    status_code = result.get("status_code")
    returned_model = result.get("model")
    success = result.get("success")
    content = result.get("content")
    usage = result.get("usage")
    latency = result.get("latency")
    error = result.get("error")

    print(f"HTTP Status Code     : {status_code}")
    print(f"Selected Model Name  : {returned_model}")
    print(f"Request Success      : {success}")
    print(f"Latency              : {latency:.4f}s")

    if usage:
        print(f"Usage Info           : {usage}")
    else:
        print(f"Usage Info           : N/A")

    print("-" * 60)
    if success:
        print("RESPONSE TEXT:")
        print(content)
    else:
        print("ERROR DETAILS (Sanitized):")
        print(error)

    print("=" * 60)
    if success:
        print("[RESULT] OpenRouter Standalone Connectivity Test PASSED")
    else:
        print("[RESULT] OpenRouter Standalone Connectivity Test FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
