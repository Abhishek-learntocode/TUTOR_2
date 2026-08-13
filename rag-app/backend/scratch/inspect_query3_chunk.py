import os
import sys

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.llm import LLMService

emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
bm25 = BM25Retriever()
reranker = Reranker(model_name=settings.reranker_model)
retriever = Retriever(
    vector_store=vs,
    bm25_retriever=bm25,
    reranker=reranker,
    top_k_candidates=settings.top_k_candidates,
    top_k_final=settings.top_k_final,
)
llm = LLMService(
    provider=settings.llm_provider,
    model_name=settings.llm_model,
    base_url=settings.llm_base_url,
    api_key=settings.openai_api_key,
)

q3 = "How does paging work, and what problem does it solve?"
print(f"=== TESTING QUERY 3: '{q3}' ===")
chunks = retriever.retrieve(q3)
print(f"Retrieved Chunks Count: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\nChunk #{i+1}:\n{repr(c)}")

ans = llm.generate(q3, chunks)
print(f"\nLLM Output Answer: {repr(ans)}")
