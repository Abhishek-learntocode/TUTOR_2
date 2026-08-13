import os
import sys
import time
import json
import re
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from fastapi.testclient import TestClient
from app.main import app
from app.rag.providers import clear_provider_call_ledger, get_provider_call_ledger, save_provider_call_ledger

client = TestClient(app)

print("=" * 80)
print("PHASE 5B.1A — HYBRID ROUTING PRODUCTION AUDIT (7 MANDATORY QUERIES)")
print("=" * 80)

clear_provider_call_ledger()

mandatory_queries = [
    {
        "test_id": "TEST_1_SIMPLE",
        "query": "What is virtual memory?",
        "expected_routing": "simple",
        "expected_provider": "ollama",
    },
    {
        "test_id": "TEST_2_SIMPLE",
        "query": "Define paging.",
        "expected_routing": "simple",
        "expected_provider": "ollama",
    },
    {
        "test_id": "TEST_3_SIMPLE",
        "query": "According to OS_Notes.txt, what is virtual memory?",
        "expected_routing": "simple",
        "expected_provider": "ollama",
    },
    {
        "test_id": "TEST_4_COMPLEX",
        "query": "Explain paging and compare it with virtual memory.",
        "expected_routing": "complex",
        "expected_provider": "openrouter",
    },
    {
        "test_id": "TEST_5_COMPLEX",
        "query": "Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.",
        "expected_routing": "complex",
        "expected_provider": "openrouter",
    },
    {
        "test_id": "TEST_6_COMPLEX",
        "query": "Using the retrieved material, explain how paging relates to virtual memory and compare the two concepts.",
        "expected_routing": "complex",
        "expected_provider": "openrouter",
    },
    {
        "test_id": "TEST_7_REFUSAL",
        "query": "According to OS_Notes.txt, explain paging.",
        "expected_routing": "simple", # document-scoped query for OS_Notes.txt which has no paging info
        "expected_provider": "ollama",
        "expected_refusal": True,
    },
]

audit_results = []

for item in mandatory_queries:
    tid = item["test_id"]
    qtext = item["query"]
    exp_route = item["expected_routing"]
    exp_prov = item["expected_provider"]
    exp_refusal = item.get("expected_refusal", False)

    print(f"\n------------------------------------------------------------------")
    print(f"RUNNING [{tid}]: '{qtext}'")
    print(f"------------------------------------------------------------------")

    t0 = time.time()
    resp = client.post("/query", json={"question": qtext})
    t1 = time.time()
    lat = round(t1 - t0, 4)

    status_code = resp.status_code
    resp_data = resp.json() if status_code == 200 else {}
    answer = resp_data.get("answer", "")
    context = resp_data.get("context", [])

    is_refusal = "I cannot find the answer in the provided context." in answer
    refusal_pass = is_refusal if exp_refusal else True

    # Retrieve ledger calls for this single POST /query invocation
    current_calls = get_provider_call_ledger()
    # Filter calls that occurred during this query
    last_call = current_calls[-1] if current_calls else {}
    actual_provider = last_call.get("provider", "unknown")
    actual_model = last_call.get("actual_model", "unknown")
    op_req_id = last_call.get("openrouter_request_id", "N/A")

    route_pass = actual_provider.lower() == exp_prov.lower()

    print(f"  HTTP Status Code      : {status_code}")
    print(f"  Total Wall Latency    : {lat}s")
    print(f"  Expected Provider     : {exp_prov} ({exp_route})")
    print(f"  Actual Answer Provider: {actual_provider} / {actual_model}")
    print(f"  OpenRouter Request ID : {op_req_id}")
    print(f"  Context Chunks        : {len(context)}")
    print(f"  Answer Snippet        : {repr(answer[:100])}...")
    print(f"  Routing Match         : {route_pass}")
    print(f"  Refusal Check         : {refusal_pass}")

    rec = {
        "test_id": tid,
        "query": qtext,
        "expected_routing": exp_route,
        "expected_provider": exp_prov,
        "actual_provider": actual_provider,
        "actual_model": actual_model,
        "openrouter_request_id": op_req_id,
        "http_status": status_code,
        "latency_sec": lat,
        "context_count": len(context),
        "answer": answer,
        "is_refusal": is_refusal,
        "refusal_pass": refusal_pass,
        "routing_pass": route_pass,
    }
    audit_results.append(rec)

# Calculate call counts
ollama_answer_calls = sum(1 for r in audit_results if r["actual_provider"] == "ollama")
openrouter_answer_calls = sum(1 for r in audit_results if r["actual_provider"] == "openrouter")

print(f"\n==================================================================")
print(f"HYBRID ROUTING PRODUCTION AUDIT SUMMARY")
print(f"Total Production Queries Executed : {len(audit_results)}")
print(f"Expected Ollama Answer Calls      : 4")
print(f"Actual Ollama Answer Calls        : {ollama_answer_calls}")
print(f"Expected OpenRouter Answer Calls  : 3")
print(f"Actual OpenRouter Answer Calls    : {openrouter_answer_calls}")
print(f"Call Count Match                  : {ollama_answer_calls == 4 and openrouter_answer_calls == 3}")
print(f"==================================================================")

out_dir = os.path.join(backend_dir, "evaluation", "results")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "phase5b1_hybrid_routing_audit.json")

summary_data = {
    "expected_ollama_answer_calls": 4,
    "actual_ollama_answer_calls": ollama_answer_calls,
    "expected_openrouter_answer_calls": 3,
    "actual_openrouter_answer_calls": openrouter_answer_calls,
    "routing_counts_match": ollama_answer_calls == 4 and openrouter_answer_calls == 3,
    "all_tests_passed": all(r["routing_pass"] and r["refusal_pass"] for r in audit_results),
    "tests": audit_results,
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

save_provider_call_ledger(os.path.join(out_dir, "phase5b1_provider_call_ledger.json"))
print(f"Saved hybrid routing audit to: {out_file}")
