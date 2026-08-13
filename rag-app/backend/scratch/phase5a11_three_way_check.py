import os
import json
import re

reg_path = "evaluation/results/phase5a11_document_scope_regression.json"
log_path = "logs/rag_traces.log"

with open(reg_path, "r", encoding="utf-8") as f:
    reg_data = json.load(f)["tests"]

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()

print("=== THREE-WAY RECONCILIATION FOR PHASE 5A.11 ===")

table_rows = []

for item in reg_data:
    tid = item["test_id"]
    qtext = item["query"]
    http_stat = item["http_status"]
    http_lat = item["http_latency"]
    http_ctx_cnt = item["context_count"]
    ls_info = item.get("langsmith", {})

    # Match latest log line in rag_traces.log
    escaped_q = re.escape(qtext)
    pattern = rf"TRACE SUMMARY \| QUERY: '{escaped_q}' \| TYPE: (\S+) \| SUB_QUERIES: (.*?) \| CONTEXT_CHUNKS: (\d+) \| ANSWER_LEN: (\d+) \| TOTAL_LATENCY: ([\d\.]+)s"
    matches = list(re.finditer(pattern, log_text))
    match = matches[-1] if matches else None

    log_ctx = int(match.group(3)) if match else "N/A"
    log_lat = float(match.group(5)) if match else "N/A"

    row = {
        "id": tid,
        "query": qtext,
        "http_status": http_stat,
        "http_latency": http_lat,
        "http_ctx_count": http_ctx_cnt,
        "log_ctx_count": log_ctx,
        "log_latency": log_lat,
        "ls_run_id": ls_info.get("run_id", "N/A")[:10] + "...",
        "ls_latency": ls_info.get("latency_sec", "N/A"),
        "match": "EXACT MATCH" if http_ctx_cnt == log_ctx else "MISMATCH"
    }
    table_rows.append(row)

print(f"{'Test ID':<30} | {'HTTP Lat':<10} | {'Log Lat':<10} | {'LS Lat':<10} | {'HTTP Ctx':<8} | {'Log Ctx':<8} | {'Match':<12}")
print("-" * 105)
for r in table_rows:
    print(f"{r['id']:<30} | {r['http_latency']:<10} | {r['log_latency']:<10} | {r['ls_latency']:<10} | {r['http_ctx_count']:<8} | {r['log_ctx_count']:<8} | {r['match']:<12}")

with open("scratch/phase5a11_three_way_summary.json", "w", encoding="utf-8") as f:
    json.dump(table_rows, f, indent=2)
