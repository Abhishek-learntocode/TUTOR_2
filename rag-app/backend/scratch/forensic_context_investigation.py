import os
import sys

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever

print("=== PHASE 5A.6 — FORENSIC CONTEXT INVESTIGATION ===")

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)

vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
all_docs = vs.get_all_documents()

print(f"Total documents in vector store: {len(all_docs)}")

os_notes_chunks = [d for d in all_docs if d.metadata.get("source_filename") == "OS_Notes.txt"]
print(f"\n--- OS_Notes.txt Chunks Count: {len(os_notes_chunks)} ---")
for i, d in enumerate(os_notes_chunks):
    print(f"\n[OS_Notes.txt Chunk #{i+1}] Metadata: {d.metadata}")
    print(f"Full Content:\n{repr(d.page_content)}")

bm25 = BM25Retriever()
reranker = Reranker(model_name=settings.reranker_model)
retriever = Retriever(
    vector_store=vs,
    bm25_retriever=bm25,
    reranker=reranker,
    top_k_candidates=settings.top_k_candidates,
    top_k_final=settings.top_k_final,
)

queries = [
    ("REQ1", "What is virtual memory?"),
    ("REQ2", "According to OS_Notes.txt, explain paging."),
    ("REQ3", "How does paging work, and what problem does it solve?")
]

print("\n=== STEP-BY-STEP RETRIEVAL & CONTEXT TRACE ===")

for tag, q in queries:
    print(f"\n==================================================")
    print(f"QUERY [{tag}]: '{q}'")
    print(f"==================================================")

    # 1. Filename candidates
    query_lower = q.lower()
    explicit = []
    for doc in all_docs:
        fn = doc.metadata.get("source_filename", "").lower()
        no_ext = os.path.splitext(fn)[0]
        if fn and (fn in query_lower or (len(no_ext) >= 3 and no_ext in query_lower)):
            explicit.append(doc)
    print(f"1. Filename matching candidates count: {len(explicit)}")

    # 2. Semantic
    semantic = vs.similarity_search(q, k=15)
    print(f"2. Semantic candidates count: {len(semantic)}")

    # 3. Lexical
    bm25.rebuild(all_docs)
    lexical = bm25.retrieve(q, top_k=15)
    print(f"3. Lexical candidates count: {len(lexical)}")

    # 4. Merge & Deduplicate
    seen = set()
    merged = []
    for doc in explicit + semantic + lexical:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    print(f"4. Merged candidates count: {len(merged)}")

    # 5. Rerank
    reranked = reranker.rerank(q, merged, top_k=4)
    print(f"5. Reranked candidates count: {len(reranked)}")

    print("\n--- FINAL CONTEXT PASSED TO LLM ---")
    for i, chunk_text in enumerate(reranked):
        print(f"\nFinal Context Chunk #{i+1}:")
        print(f"{repr(chunk_text)}")
