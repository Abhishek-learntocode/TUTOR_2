import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

reqs = [
    ("REQ1", "What is virtual memory?"),
    ("REQ2", "How does paging work, and what problem does it solve?"),
    ("REQ3", "According to OS_Notes.txt, explain paging.")
]

print("=== EXECUTING 3 REAL HTTP REQUESTS AGAINST BACKEND SERVER ===")
for tag, q in reqs:
    print(f"\n--- {tag}: '{q}' ---")
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": q}, timeout=120)
    t1 = time.time()
    wall_sec = round(t1 - t0, 4)
    print(f"HTTP Status  : {resp.status_code}")
    print(f"Wall Latency : {wall_sec}s")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Answer       : {repr(data.get('answer'))}")
        print(f"Context Count: {len(data.get('context', []))}")
        for i, chunk in enumerate(data.get('context', [])):
            print(f"  Chunk {i+1} : {repr(chunk[:100])}...")
    else:
        print(f"Error Text   : {resp.text[:200]}")
