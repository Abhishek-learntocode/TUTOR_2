import os
import sys

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

print("=== DIAGNOSING VECTOR STORE & EMBEDDING SERVICE ===")
print("Embedding Provider:", settings.embedding_provider)
print("Embedding Model   :", settings.embedding_model)

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)

vec_dim = len(emb_svc.embed_query("test"))
print("Current Embedding Vector Dim:", vec_dim)

vs = VectorStore(embeddings=emb_svc, store_path=settings.vector_store_path)
print("Loaded VectorStore store is None?:", vs.store is None)

if vs.store is not None:
    print("FAISS Index Dimension:", vs.store.index.d)
    print("FAISS ntotal         :", vs.store.index.ntotal)
    docs = vs.get_all_documents()
    print("Total stored documents count:", len(docs))
    if docs:
        print("Sample doc metadata:", docs[0].metadata)
