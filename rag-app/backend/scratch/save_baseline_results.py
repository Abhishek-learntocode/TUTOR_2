import os
import json
import re

log_path = "C:\\Users\\Abhishek\\.gemini\\antigravity-ide\\brain\\1815e485-b05b-4167-8413-70d29755f967\\.system_generated\\tasks\\task-391.log"

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Extract query results from log output
pattern = r"\[(\d+)/35\] Query ID: (\S+) \| Cat: (\S+)\n\s+Query: '(.*?)'\n\s+Status: (\d+) \| Latency: ([\d\.]+)s \| Context: (\d+) \| Refusal: (True|False) \| Quality: (\S+)"
matches = re.findall(pattern, text)

results = []
for m in matches:
    idx, qid, cat, qtext, status, lat, ctx_cnt, is_ref, qual = m
    results.append({
        "id": qid,
        "category": cat,
        "query": qtext,
        "status_code": int(status),
        "http_wall_latency_sec": float(lat),
        "context_count": int(ctx_cnt),
        "is_refusal": is_ref == "True",
        "quality_eval": qual
    })

os.makedirs("evaluation/results", exist_ok=True)
with open("evaluation/results/rag_baseline_v1_run.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

counts = {}
for r in results:
    c = r["quality_eval"]
    counts[c] = counts.get(c, 0) + 1

summary = {
    "total_queries_executed": len(results),
    "quality_breakdown": counts,
    "average_latency_sec": round(sum(r["http_wall_latency_sec"] for r in results) / len(results), 4) if results else 0
}

with open("evaluation/results/rag_baseline_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {len(results)} query results to evaluation/results/rag_baseline_v1_run.json")
print("Summary:", summary)
