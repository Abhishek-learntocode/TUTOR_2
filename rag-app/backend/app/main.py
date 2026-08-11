import os
import sys
from fastapi import FastAPI

# Add both 'app' directory and its parent ('backend') directory to sys.path
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.llm import LLMService
from app.graph.nodes import RAGNodes
from app.graph.workflow import RAGGraph
from app.api.routes import router

app = FastAPI(title="Simple RAG API")

# 1. Embedding Service
embedding_service = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)

# 2. Vector Store
vector_store = VectorStore(
    embeddings=embedding_service.embeddings,
    store_path=settings.vector_store_path,
)

# 3. Retriever
retriever = Retriever(vector_store=vector_store, top_k=settings.top_k)

# 4. LLM Service
llm_service = LLMService(
    provider=settings.llm_provider,
    model_name=settings.llm_model,
    base_url=settings.llm_base_url,
    api_key=settings.openai_api_key,
)

# 5. LangGraph RAG Workflow
nodes = RAGNodes(retriever=retriever, llm=llm_service)
rag_graph = RAGGraph(nodes=nodes)
rag_graph.compile()

# Attach instances to app.state
app.state.vector_store = vector_store
app.state.rag_graph = rag_graph

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
