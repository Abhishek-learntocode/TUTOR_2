import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

requests_data = [
    ("REQ1", "What is virtual memory?"),
    ("REQ2", "How does paging work, and what problem does it solve?"),
    ("REQ3", "According to OS_Notes.txt, explain paging.")
]

results = []

print("=== EXECUTING 3 SEQUENTIAL FULL-RAG REQUESTS AGAINST BACKEND SERVER ===")

for tag, question in requests_data:
    print(f"\n[{tag}] Question: '{question}'")
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": question}, timeout=180)
    t1 = time.time()
    wall_sec = round(t1 - t0, 4)

    if resp.status_code == 200:
        body = resp.json()
        ans = body.get("answer", "")
        ctx = body.get("context", [])
        rec = {
            "tag": tag,
            "question": question,
            "status_code": resp.status_code,
            "http_wall_latency_sec": wall_sec,
            "answer_length": len(ans),
            "answer_snippet": ans[:200],
            "full_answer": ans,
            "context_count": len(ctx),
            "context_snippets": [c[:100] for c in ctx]
        }
        print(f"  Status       : {resp.status_code}")
        print(f"  Wall Latency : {wall_sec}s")
        print(f"  Context Count: {len(ctx)}")
        print(f"  Answer Snippet: {repr(ans[:150])}...")
    else:
        rec = {
            "tag": tag,
            "question": question,
            "status_code": resp.status_code,
            "http_wall_latency_sec": wall_sec,
            "error": resp.text[:300]
        }
        print(f"  Status       : {resp.status_code} (ERROR)")
        print(f"  Wall Latency : {wall_sec}s")

    results.append(rec)
    print("Waiting 2 seconds before next request...")
    time.sleep(2)

with open("scratch/real_rag_http_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nFinished executing requests. Results saved to scratch/real_rag_http_results.json.")
