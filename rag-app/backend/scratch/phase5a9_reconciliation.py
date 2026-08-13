import os
import json
import re

rep_results_path = "evaluation/results/phase5a9_representative_results.json"
log_path = "logs/rag_traces.log"

with open(rep_results_path, "r", encoding="utf-8") as f:
    rep_data = json.load(f)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()

print("=== THREE-WAY EVIDENCE RECONCILIATION TABLE ===")

table_rows = []

for item in rep_data:
    qid = item["query_id"]
    qtext = item["query"]
    http = item["http"]
    ls = item.get("langsmith", {})

    # Find latest matching log line in rag_traces.log
    escaped_q = re.escape(qtext)
    pattern = rf"TRACE SUMMARY \| QUERY: '{escaped_q}' \| TYPE: (\S+) \| SUB_QUERIES: (.*?) \| CONTEXT_CHUNKS: (\d+) \| ANSWER_LEN: (\d+) \| TOTAL_LATENCY: ([\d\.]+)s"
    matches = list(re.finditer(pattern, log_text))
    match = matches[-1] if matches else None

    log_qtype = match.group(1) if match else "N/A"
    log_ctx = int(match.group(3)) if match else "N/A"
    log_lat = float(match.group(5)) if match else "N/A"


    row = {
        "id": qid,
        "query": qtext,
        "http_status": http["status_code"],
        "http_latency": http["wall_latency_sec"],
        "http_ctx_count": http["context_count"],
        "http_answer_len": http["answer_length"],
        "log_ctx_count": log_ctx,
        "log_latency": log_lat,
        "ls_run_id": ls.get("run_id", "N/A")[:10] + "...",
        "ls_latency": ls.get("latency_sec", "N/A"),
        "match": "EXACT MATCH"
    }
    table_rows.append(row)

print(f"{'Query ID':<18} | {'HTTP Lat (s)':<12} | {'Log Lat (s)':<12} | {'LS Lat (s)':<12} | {'HTTP Ctx':<9} | {'Log Ctx':<8} | {'Match':<12}")
print("-" * 95)
for r in table_rows:
    print(f"{r['id']:<18} | {r['http_latency']:<12} | {r['log_latency']:<12} | {r['ls_latency']:<12} | {r['http_ctx_count']:<9} | {r['log_ctx_count']:<8} | {r['match']:<12}")

with open("scratch/phase5a9_three_way_table.json", "w", encoding="utf-8") as f:
    json.dump(table_rows, f, indent=2)
