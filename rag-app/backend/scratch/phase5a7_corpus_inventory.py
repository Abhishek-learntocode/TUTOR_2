import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

print("=== PHASE 5A.7 — STEP 1: CORPUS & EVALUATION INVENTORY ===")

# 1. Inspect data/documents/
doc_dir = "data/documents"
doc_files = []
if os.path.exists(doc_dir):
    for f in sorted(os.listdir(doc_dir)):
        fp = os.path.join(doc_dir, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            with open(fp, "r", encoding="utf-8", errors="ignore") as file_handle:
                content = file_handle.read()
            doc_files.append({
                "filename": f,
                "size_bytes": size,
                "lines_count": len(content.splitlines()),
                "chars_count": len(content),
                "snippet": repr(content[:150])
            })

print(f"\n1. Source Documents in '{doc_dir}': {len(doc_files)} files")
for d in doc_files:
    print(f"   - {d['filename']} ({d['size_bytes']} bytes, {d['lines_count']} lines): {d['snippet']}")

# 2. Inspect Vector Store
emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
all_docs = vs.get_all_documents()

print(f"\n2. Vector Store Store Path: {settings.vector_store_path}")
print(f"   - Total Chunks Indexed in FAISS: {len(all_docs)}")

store_docs_by_filename = {}
for doc in all_docs:
    fn = doc.metadata.get("source_filename", "UNKNOWN")
    store_docs_by_filename.setdefault(fn, []).append(doc)

print("   - Chunks breakdown per source_filename in VectorStore:")
for fn, chunks in store_docs_by_filename.items():
    print(f"     * '{fn}': {len(chunks)} chunk(s)")

# 3. Inspect Dataset
dataset_path = "evaluation/datasets/rag_baseline_v1.jsonl"
queries = []
if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

print(f"\n3. Baseline Dataset '{dataset_path}': {len(queries)} queries")
for q in queries[:5]:
    print(f"   - ID: {q.get('query_id') or q.get('id')} | Category: {q.get('category')} | Q: '{q.get('question')}'")

inventory = {
    "source_documents": doc_files,
    "vector_store": {
        "store_path": settings.vector_store_path,
        "total_chunks": len(all_docs),
        "chunks_per_file": {fn: len(c) for fn, c in store_docs_by_filename.items()}
    },
    "dataset_total_queries": len(queries),
    "sample_queries": queries[:5]
}

with open("scratch/inventory_summary.json", "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2)

print("\nInventory complete. Saved summary to scratch/inventory_summary.json.")
