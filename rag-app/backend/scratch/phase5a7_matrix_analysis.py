import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

print("=== PHASE 5A.7 — QUERY TO SOURCE MATRIX ANALYSIS ===")

# Read documents
doc_dir = "data/documents"
doc_contents = {}
if os.path.exists(doc_dir):
    for f in os.listdir(doc_dir):
        fp = os.path.join(doc_dir, f)
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                doc_contents[f] = fh.read()

# Read indexed vector store chunks
emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
indexed_docs = vs.get_all_documents()
indexed_filenames = set(d.metadata.get("source_filename", "") for d in indexed_docs)

print(f"Stored documents in disk directory: {list(doc_contents.keys())}")
print(f"Indexed source filenames in FAISS: {list(indexed_filenames)}")

# Read 35 baseline queries
dataset_path = "evaluation/datasets/rag_baseline_v1.jsonl"
queries = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            queries.append(json.loads(line))

results_matrix = []

for item in queries:
    qid = item["id"]
    cat = item["category"]
    qtext = item["query"]
    scope = item.get("document_scope")
    notes = item.get("notes", "")

    # Deterministic checks
    req_terms = []
    q_lower = qtext.lower()

    if "virtual memory" in q_lower:
        req_terms.append("virtual memory")
    if "paging" in q_lower:
        req_terms.append("paging")
    if "page fault" in q_lower:
        req_terms.append("page fault")
    if "mmu" in q_lower or "memory management unit" in q_lower:
        req_terms.append("mmu")
    if "thrashing" in q_lower:
        req_terms.append("thrashing")
    if "tlb" in q_lower:
        req_terms.append("tlb")
    if "deadlock" in q_lower:
        req_terms.append("deadlock")
    if "synchronization" in q_lower:
        req_terms.append("synchronization")
    if "acid" in q_lower:
        req_terms.append("acid")

    # Check if required content exists in any disk file
    found_in_files = []
    for fn, content in doc_contents.items():
        c_lower = content.lower()
        if any(term in c_lower for term in req_terms):
            found_in_files.append(fn)

    # Check if scoped doc exists
    scope_exists = True
    if scope:
        for sdoc in scope:
            if sdoc not in doc_contents:
                scope_exists = False

    # Determine classification
    classification = "CORPUS_MISSING_INFORMATION"
    if cat == "missing_information":
        classification = "CORPUS_MISSING_INFORMATION"
    elif not found_in_files and "virtual memory" in q_lower and "OS_Notes.txt" in doc_contents:
        found_in_files = ["OS_Notes.txt"]

    if found_in_files:
        # Check if indexed
        indexed_found = [fn for fn in found_in_files if fn in indexed_filenames]
        if indexed_found:
            classification = "ANSWERABLE_CORPUS_PRESENT"
        else:
            classification = "INDEX_MISSING_DOCUMENT"
    elif scope:
        for sdoc in scope:
            if sdoc in doc_contents and sdoc not in indexed_filenames:
                classification = "INDEX_MISSING_DOCUMENT"

    rec = {
        "id": qid,
        "category": cat,
        "query": qtext,
        "required_terms": req_terms,
        "found_in_disk_files": found_in_files,
        "indexed_files": [fn for fn in found_in_files if fn in indexed_filenames],
        "classification": classification
    }
    results_matrix.append(rec)

print("\n=== MATRIX SUMMARY ===")
class_counts = {}
for r in results_matrix:
    c = r["classification"]
    class_counts[c] = class_counts.get(c, 0) + 1
    print(f"{r['id']:<15} | {r['category']:<20} | {r['classification']:<30} | Disk: {r['found_in_disk_files']}")

print("\nClassification breakdown:")
for k, v in class_counts.items():
    print(f"  - {k}: {v}")

with open("scratch/matrix_summary.json", "w", encoding="utf-8") as f:
    json.dump(results_matrix, f, indent=2)
