import os
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
doc_dir = "data/documents"
files = sorted(os.listdir(doc_dir))

print(f"=== STEP 5: REBUILD / REFRESH INDEX VIA OFFICIAL /documents/upload ENDPOINT ===")
print(f"Uploading all {len(files)} documents from '{doc_dir}'...\n")

upload_results = []

for f in files:
    fp = os.path.join(doc_dir, f)
    if os.path.isfile(fp):
        # Determine doc_type based on filename heuristics or default to BOOK / EXAM_PAPER
        if "exam" in f.lower():
            doc_type = "EXAM_PAPER"
        elif "routing" in f.lower() or "notes" in f.lower():
            doc_type = "LECTURE_SLIDE"
        else:
            doc_type = "BOOK"

        print(f"Uploading '{f}' as [{doc_type}]...")
        with open(fp, "rb") as file_bytes:
            upload_files = {"file": (f, file_bytes, "text/plain")}
            upload_data = {"doc_type": doc_type}
            resp = requests.post(f"{BASE_URL}/documents/upload", files=upload_files, data=upload_data, timeout=60)

        if resp.status_code == 200:
            res_data = resp.json()
            upload_results.append(res_data)
            print(f"  -> SUCCESS: {res_data.get('message')}")
        else:
            err = {"filename": f, "status_code": resp.status_code, "error": resp.text[:200]}
            upload_results.append(err)
            print(f"  -> ERROR {resp.status_code}: {resp.text[:200]}")

with open("scratch/ingestion_upload_results.json", "w", encoding="utf-8") as f:
    json.dump(upload_results, f, indent=2)

print("\nIngestion upload complete. Saved results to scratch/ingestion_upload_results.json.")
