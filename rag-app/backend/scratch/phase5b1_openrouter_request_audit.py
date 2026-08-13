import os
import sys
import time
import json
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from app.config import settings
from app.rag.openrouter_provider import OpenRouterProvider
from app.rag.providers import clear_provider_call_ledger, save_provider_call_ledger, get_provider_call_ledger


def main():
    print("=" * 80)
    print("PHASE 5B.1A — OPENROUTER DIRECT REQUEST AUDIT (3 REQUESTS)")
    print("=" * 80)

    clear_provider_call_ledger()

    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        model_name=settings.openrouter_model or "google/gemma-4-31b-it:free",
        base_url=settings.openrouter_base_url,
    )

    prompts = [
        "Explain virtual memory in one sentence.",
        "Explain paging in one sentence.",
        "Compare paging and virtual memory in three sentences.",
    ]

    audit_records = []

    for idx, p in enumerate(prompts, start=1):
        print(f"\n--- Sending Request #{idx} ---")
        print(f"Prompt: \"{p}\"")
        t0 = time.time()
        res = provider.generate(p, temperature=0.1)
        t1 = time.time()

        rec = {
            "request_number": idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "prompt": p,
            "configured_model": provider.model_name,
            "actual_model": res.get("actual_model", "N/A"),
            "status_code": res.get("status_code"),
            "openrouter_request_id": res.get("openrouter_request_id", "N/A"),
            "latency_sec": res.get("latency", round(t1 - t0, 4)),
            "usage": res.get("usage", {}),
            "response_text": res.get("text", ""),
            "success": res.get("success", False),
            "error": res.get("error"),
        }
        audit_records.append(rec)

        print(f"  HTTP Status Code      : {rec['status_code']}")
        print(f"  OpenRouter Request ID : {rec['openrouter_request_id']}")
        print(f"  Actual Model Returned : {rec['actual_model']}")
        print(f"  Latency               : {rec['latency_sec']}s")
        print(f"  Response Snippet      : {repr(rec['response_text'][:100])}...")

    summary = {
        "audit_phase": "5B.1A_DIRECT_OPENROUTER_REQUESTS",
        "expected_requests": 3,
        "actual_requests_sent": len(audit_records),
        "successful_requests": sum(1 for r in audit_records if r["success"]),
        "failed_requests": sum(1 for r in audit_records if not r["success"]),
        "all_successful": sum(1 for r in audit_records if r["success"]) == 3,
        "records": audit_records,
    }

    out_dir = os.path.join(backend_dir, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase5b1_openrouter_direct_request_audit.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ledger_file = os.path.join(out_dir, "phase5b1_provider_call_ledger.json")
    save_provider_call_ledger(ledger_file)

    print("\n==================================================================")
    print(f"DIRECT AUDIT COMPLETE: {summary['successful_requests']}/3 Requests Successful")
    print(f"Saved direct audit to: {out_file}")
    print(f"Saved provider call ledger to: {ledger_file}")
    print("==================================================================")


if __name__ == "__main__":
    main()
