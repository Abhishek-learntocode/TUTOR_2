import os
import sys
from fastapi import FastAPI

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.llm import LLMService
from app.rag.query_analyzer import QueryAnalyzer
from app.graph.nodes import RAGNodes
from app.graph.workflow import RAGGraph
from app.api.routes import router
import logging


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_traces.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rag_tracer")

app = FastAPI(title="Simple RAG API")


# 1. Embedding Service (bge-m3)
embedding_service = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)

# 2. Vector Store (FAISS)
vector_store = VectorStore(
    embeddings=embedding_service.embeddings,
    store_path=settings.vector_store_path,
)



# 3. Lexical BM25 & Reranker
bm25_retriever = BM25Retriever()
reranker = Reranker(model_name=settings.reranker_model)

# 4. Hybrid Retriever
retriever = Retriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
    top_k_candidates=settings.top_k_candidates,
    top_k_final=settings.top_k_final,
)

from app.rag.providers import get_provider

# 5. Role 1: Query Analyzer LLM Provider
qa_provider = get_provider(
    provider_name=settings.query_analyzer_provider,
    model_name=settings.query_analyzer_model,
    base_url=settings.llm_base_url,
    api_key=settings.openrouter_api_key,
)
query_analyzer = QueryAnalyzer(provider=qa_provider)

# 6. Role 2: Hybrid Answer Generator Providers
simple_provider = get_provider(
    provider_name=settings.hybrid_simple_provider,
    model_name=settings.hybrid_simple_model,
    base_url=settings.llm_base_url,
    api_key=settings.openrouter_api_key,
)
simple_llm = LLMService(provider=simple_provider)

complex_provider = get_provider(
    provider_name=settings.hybrid_complex_provider,
    model_name=settings.hybrid_complex_model,
    base_url=settings.llm_base_url,
    api_key=settings.openrouter_api_key,
)
complex_llm = LLMService(provider=complex_provider)

llm_service = simple_llm

# 7. LangGraph RAG Workflow with Dynamic Hybrid Router
nodes = RAGNodes(
    retriever=retriever,
    llm=llm_service,
    query_analyzer=query_analyzer,
    simple_llm=simple_llm,
    complex_llm=complex_llm,
)
rag_graph = RAGGraph(nodes=nodes)
rag_graph.compile()

# Attach instances to app.state
app.state.vector_store = vector_store
app.state.rag_graph = rag_graph

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
