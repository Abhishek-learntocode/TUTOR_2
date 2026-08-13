import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

sanity_queries = [
    ("SANITY1", "What is virtual memory?"),
    ("SANITY2", "How does paging work?"),
    ("SANITY3", "According to sample_routing_doc.txt, explain paging."),
    ("SANITY4", "What is Round Robin scheduling?"),
    ("SANITY5", "What is ACID?")
]

print("=== STEP 8: DIRECT RETRIEVAL SANITY TESTS ===")

results = []

for tag, q in sanity_queries:
    print(f"\n[{tag}] Query: '{q}'")
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": q}, timeout=180)
    t1 = time.time()
    wall_sec = round(t1 - t0, 4)

    if resp.status_code == 200:
        data = resp.json()
        ans = data.get("answer", "")
        ctx = data.get("context", [])
        is_ref = "I cannot find the answer" in ans
        sources = list(set([repr(c)[:60] for c in ctx]))
        rec = {
            "tag": tag,
            "query": q,
            "status_code": resp.status_code,
            "http_wall_latency_sec": wall_sec,
            "context_count": len(ctx),
            "sources_sample": sources,
            "answer_snippet": repr(ans[:180]),
            "is_refusal": is_ref
        }
        print(f"  Status       : {resp.status_code}")
        print(f"  Wall Latency : {wall_sec}s")
        print(f"  Context Count: {len(ctx)}")
        print(f"  Refusal      : {is_ref}")
        print(f"  Answer       : {repr(ans[:150])}...")
    else:
        rec = {"tag": tag, "query": q, "status_code": resp.status_code, "error": resp.text[:200]}
        print(f"  Status       : {resp.status_code} ERROR")

    results.append(rec)
    time.sleep(1.0)

with open("scratch/sanity_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSanity tests complete. Results saved to scratch/sanity_test_results.json.")
