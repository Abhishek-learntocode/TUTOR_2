import os
import sys

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever

print("=== STEP 6 & 7: VERIFY VECTOR STORE & BM25 ===")

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
all_docs = vs.get_all_documents()

print(f"Total FAISS chunks indexed: {len(all_docs)}")

chunks_by_file = {}
for d in all_docs:
    fn = d.metadata.get("source_filename", "UNKNOWN")
    chunks_by_file.setdefault(fn, []).append(d)

print(f"Unique source filenames in FAISS: {len(chunks_by_file)}")
for fn, c in sorted(chunks_by_file.items()):
    print(f"  - '{fn}': {len(c)} chunk(s)")

# Verify BM25
bm25 = BM25Retriever()
bm25.rebuild(all_docs)
paging_docs = bm25.retrieve("paging memory management", top_k=5)

print(f"\nBM25 Retrieval for 'paging memory management': {len(paging_docs)} results")
for i, d in enumerate(paging_docs):
    print(f"  Result #{i+1}: Source={d.metadata.get('source_filename')} | Text={repr(d.page_content[:100])}")
