import time
import requests
import json
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"
BASE_URL = "http://127.0.0.1:8000"

print("=== INVESTIGATING LANGSMITH ASYNCHRONOUS TRACE CORRELATION ===")

qtext = "What is virtual memory?"
t0 = time.time()
r = requests.post(f"{BASE_URL}/query", json={"question": qtext}, timeout=120)
t1 = time.time()
print(f"HTTP Status: {r.status_code} | Wall Latency: {t1-t0:.4f}s | Ans Len: {len(r.json().get('answer'))}")

# Poll LangSmith API for up to 10 seconds to observe status transition
print("\nPolling LangSmith API every 1 second to observe run completion...")
completed_run = None

for attempt in range(1, 11):
    time.sleep(1)
    runs = list(ls_client.list_runs(project_id=ls_project_id, is_root=True, limit=5))
    matching = None
    for run in runs:
        if run.inputs.get("question") == qtext or run.inputs.get("query") == qtext:
            matching = run
            break

    if matching:
        has_outputs = matching.outputs is not None
        has_endtime = matching.end_time is not None
        latency = round(matching.end_time.timestamp() - matching.start_time.timestamp(), 4) if matching.end_time else 0
        print(f"Attempt #{attempt}: Run ID={matching.id} | has_outputs={has_outputs} | has_endtime={has_endtime} | Latency={latency}s")
        if has_outputs and has_endtime:
            completed_run = matching
            print("  -> Run is now FULLY COMPLETED in LangSmith!")
            break
    else:
        print(f"Attempt #{attempt}: No matching run found yet.")

if completed_run:
    print("\nCompleted Run Outputs:", completed_run.outputs)
