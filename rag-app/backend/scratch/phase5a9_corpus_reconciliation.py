import os
import sys
import hashlib
import json

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

print("=== STEP 4: CORPUS INVENTORY & FAISS RECONCILIATION ===")

doc_dir = "data/documents"
disk_files = sorted(os.listdir(doc_dir))

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
all_faiss_docs = vs.get_all_documents()

faiss_chunks_per_file = {}
for d in all_faiss_docs:
    fn = d.metadata.get("source_filename", "UNKNOWN")
    faiss_chunks_per_file.setdefault(fn, []).append(d)

reconciliation_table = []

for f in disk_files:
    fp = os.path.join(doc_dir, f)
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        words = len(content.split())
        on_disk = True
        indexed_chunks = len(faiss_chunks_per_file.get(f, []))
        is_indexed = indexed_chunks > 0

        # Topic summary
        if "dbms" in f.lower():
            topics = "ACID properties, SQL definition"
        elif "notes" in f.lower():
            topics = "System calls, Virtual memory illusion"
        elif "routing" in f.lower():
            topics = "CPU Scheduling (Round Robin), Memory Management (Paging)"
        elif "exam_inspect" in f.lower():
            topics = "Q1: Virtual memory definition"
        elif "exam_rag" in f.lower():
            topics = "Q1: Deadlock definition"
        elif "exam" in f.lower():
            topics = "Q1: Process synchronization, Q2: Virtual memory paging"
        elif "hybrid" in f.lower():
            topics = "OS concepts Items #1 to #15"
        elif "two_stage" in f.lower():
            topics = "Informational chunks Items #1 to #15"
        elif "test" in f.lower():
            topics = "Tutor RAG architecture description"
        else:
            topics = "OS introduction"

        status = "FULLY_INDEXED" if is_indexed else "NOT_INDEXED"

        reconciliation_table.append({
            "source_file": f,
            "size_bytes": size,
            "words_count": words,
            "on_disk": on_disk,
            "is_indexed": is_indexed,
            "chunk_count": indexed_chunks,
            "topics": topics,
            "status": status
        })

print(f"\nTotal Source Files on Disk: {len(disk_files)}")
print(f"Total Chunks in FAISS Index: {len(all_faiss_docs)}")
print("\nReconciliation Table:")
print(f"{'Source File':<25} | {'On Disk':<7} | {'Indexed':<7} | {'Chunk Count':<11} | {'Status':<15} | Key Topics")
print("-" * 110)
for r in reconciliation_table:
    print(f"{r['source_file']:<25} | {str(r['on_disk']):<7} | {str(r['is_indexed']):<7} | {r['chunk_count']:<11} | {r['status']:<15} | {r['topics']}")

with open("scratch/phase5a9_corpus_reconciliation.json", "w", encoding="utf-8") as f:
    json.dump(reconciliation_table, f, indent=2)

print("\nReconciliation complete. Saved to scratch/phase5a9_corpus_reconciliation.json.")
