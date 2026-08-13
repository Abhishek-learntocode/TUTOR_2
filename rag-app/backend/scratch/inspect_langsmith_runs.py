import os
import json
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=== FETCHING RECENT LANGSMITH RUNS FROM PROJECT 'AI tutor' ===")
runs = list(client.list_runs(project_id=project_id, limit=10))

print(f"Total runs fetched: {len(runs)}\n")
for i, run in enumerate(runs[:6]):
    print(f"--- RUN #{i+1} ---")
    print(f"Name      : {run.name}")
    print(f"Run ID    : {run.id}")
    print(f"Run Type  : {run.run_type}")
    print(f"Status    : {run.status}")
    print(f"Start Time: {run.start_time}")
    print(f"End Time  : {run.end_time}")
    print(f"Latency   : {getattr(run, 'latency', 'N/A')}s")
    print(f"Inputs    : {run.inputs}")
    print(f"Outputs   : {run.outputs}")
    print(f"Metadata  : {run.extra.get('metadata') if run.extra else None}")
    print(f"Tags      : {run.tags}")
    print(f"Error     : {run.error}")
    print()
