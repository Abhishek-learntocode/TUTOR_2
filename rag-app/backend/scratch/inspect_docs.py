import os
import sys

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)

vs = VectorStore(embeddings=emb_svc, store_path=settings.vector_store_path)
all_docs = vs.get_all_documents()
print(f"Total FAISS docs: {len(all_docs)}")

filenames = set(d.metadata.get("source_filename", "UNKNOWN") for d in all_docs)
print("Stored source filenames:", filenames)

print("\n--- Testing similarity search directly ---")
results = vs.similarity_search("What is virtual memory?", k=4)
print(f"Direct similarity search returned {len(results)} docs:")
for i, d in enumerate(results):
    print(f"Result #{i+1}: Source={d.metadata.get('source_filename')} | Text snippet={repr(d.page_content[:100])}")
