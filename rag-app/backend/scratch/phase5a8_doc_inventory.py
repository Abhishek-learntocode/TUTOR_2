import os
import hashlib
import json

doc_dir = "data/documents"
files = sorted(os.listdir(doc_dir))

inventory = []
for f in files:
    fp = os.path.join(doc_dir, f)
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        with open(fp, "rb") as file_bytes:
            content_bytes = file_bytes.read()
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        text = content_bytes.decode("utf-8", errors="ignore")
        lines = len(text.splitlines())
        words = len(text.split())
        inventory.append({
            "filename": f,
            "size_bytes": size,
            "sha256": sha256,
            "lines_count": lines,
            "words_count": words
        })

print("=== STEP 2: SOURCE DOCUMENT INVENTORY ===")
for d in inventory:
    print(f"File: {d['filename']:<25} | Size: {d['size_bytes']:>5} bytes | Lines: {d['lines_count']:>2} | Words: {d['words_count']:>3} | SHA256: {d['sha256'][:10]}...")

with open("scratch/phase5a8_doc_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2)
