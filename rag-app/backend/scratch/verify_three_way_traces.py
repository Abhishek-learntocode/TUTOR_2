import os
import json
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=== THREE-WAY TRACE CONSISTENCY INSPECTION ===")

print("\n--- 1. READING LOCAL LOGS (logs/rag_traces.log) ---")
with open("logs/rag_traces.log", "r", encoding="utf-8") as f:
    log_lines = [line.strip() for line in f if "TRACE SUMMARY" in line]

for line in log_lines[-5:]:
    print("LOG:", line)

print("\n--- 2. FETCHING LANGSMITH ROOT RUNS ---")
runs = list(client.list_runs(project_id=project_id, limit=10))

for r in runs[:5]:
    if r.name == "rag_graph":
        print(f"\nRoot Run ID: {r.id}")
        print(f"Question   : {r.inputs.get('question')}")
        print(f"Output Ans : {repr(r.outputs.get('answer'))[:100]}")
        print(f"Context Count: {len(r.outputs.get('context', []))}")
        print(f"Latency    : {r.end_time - r.start_time}")
