import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
dataset_path = "evaluation/datasets/rag_baseline_v1.jsonl"

with open(dataset_path, "r", encoding="utf-8") as f:
    queries = [json.loads(line) for line in f if line.strip()]

print(f"=== RUNNING 35-QUERY BASELINE EVALUATION AGAINST {BASE_URL} ===")
print(f"Total queries: {len(queries)}. Pacing: 0.5s between requests.\n")

results = []

for i, q in enumerate(queries):
    qid = q["id"]
    cat = q["category"]
    qtext = q["query"]

    print(f"[{i+1}/{len(queries)}] Query ID: {qid} | Cat: {cat}")
    print(f"    Query: '{qtext}'")

    t0 = time.time()
    try:
        resp = requests.post(f"{BASE_URL}/query", json={"question": qtext}, timeout=180)
        t1 = time.time()
        wall_sec = round(t1 - t0, 4)

        if resp.status_code == 200:
            data = resp.json()
            ans = data.get("answer", "")
            ctx = data.get("context", [])
            is_refusal = "I cannot find the answer" in ans

            # Evaluate quality
            if cat == "missing_information":
                quality = "CORRECT_REFUSAL" if is_refusal else "UNSUPPORTED_HALLUCINATION"
            elif cat == "ambiguous":
                quality = "CORRECT_REFUSAL" if is_refusal else "ANSWERED"
            elif is_refusal:
                quality = "REFUSAL_DUE_TO_MISSING_CONTEXT"
            else:
                quality = "GROUNDED_ANSWER"

            rec = {
                "id": qid,
                "category": cat,
                "query": qtext,
                "status_code": resp.status_code,
                "http_wall_latency_sec": wall_sec,
                "context_count": len(ctx),
                "answer_length": len(ans),
                "answer_snippet": ans[:150],
                "is_refusal": is_refusal,
                "quality_eval": quality
            }
            print(f"    Status: 200 | Latency: {wall_sec}s | Context: {len(ctx)} | Refusal: {is_refusal} | Quality: {quality}")
        else:
            rec = {
                "id": qid,
                "category": cat,
                "query": qtext,
                "status_code": resp.status_code,
                "http_wall_latency_sec": wall_sec,
                "error": resp.text[:200],
                "quality_eval": "HTTP_ERROR"
            }
            print(f"    Status: {resp.status_code} ERROR | Latency: {wall_sec}s")
    except Exception as e:
        rec = {
            "id": qid,
            "category": cat,
            "query": qtext,
            "status_code": 0,
            "error": str(e),
            "quality_eval": "EXCEPTION"
        }
        print(f"    EXCEPTION: {e}")

    results.append(rec)
    time.sleep(0.5)

os.makedirs("evaluation/results", exist_ok=True)
with open("evaluation/results/rag_baseline_v1_run.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n=== EVALUATION RUN SUMMARY ===")
quality_counts = {}
for r in results:
    q = r.get("quality_eval", "UNKNOWN")
    quality_counts[q] = quality_counts.get(q, 0) + 1

for k, v in quality_counts.items():
    print(f"  - {k}: {v}")

print("Results saved to evaluation/results/rag_baseline_v1_run.json.")
