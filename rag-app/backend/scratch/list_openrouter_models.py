import os
import sys
import requests

from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load .env file from backend root directory
env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from app.config import settings


def main():
    print("=" * 100)
    print("OPENROUTER FREE MODELS DISCOVERY")
    print("=" * 100)

    url = "https://openrouter.ai/api/v1/models"
    headers = {}
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch OpenRouter models catalog: HTTP {response.status_code}")
            sys.exit(1)

        data = response.json()
        models = data.get("data", [])

        free_models = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)

            # Check if model is free (zero cost or has :free suffix)
            if prompt_price == 0.0 and completion_price == 0.0 or mid.endswith(":free"):
                context_length = m.get("context_length", "N/A")
                name = m.get("name", mid)
                architecture = m.get("architecture", {})
                modality = architecture.get("modality", "text->text") if isinstance(architecture, dict) else "text"
                
                free_models.append({
                    "id": mid,
                    "name": name,
                    "context_length": context_length,
                    "input_pricing": prompt_price,
                    "output_pricing": completion_price,
                    "modality": modality,
                })

        print(f"Total Models Returned by OpenRouter API : {len(models)}")
        print(f"Total Free / Zero-Cost Models Identified : {len(free_models)}")
        print("-" * 100)

        fmt = "{:<45} | {:<25} | {:<12} | {:<12} | {:<12}"
        print(fmt.format("MODEL ID", "CONTEXT LENGTH", "INPUT PRICE", "OUTPUT PRICE", "MODALITY"))
        print("-" * 100)

        for fm in sorted(free_models, key=lambda x: x["id"]):
            print(fmt.format(
                fm["id"][:45],
                str(fm["context_length"]),
                f"${fm['input_pricing']:.4f}",
                f"${fm['output_pricing']:.4f}",
                str(fm["modality"])[:12]
            ))

        print("=" * 100)
        print("[SUMMARY] Model discovery completed successfully.")
        print("=" * 100)

    except Exception as e:
        print(f"[ERROR] Failed to list models: {type(e).__name__} - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
