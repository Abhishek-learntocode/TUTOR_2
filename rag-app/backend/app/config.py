from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""

    # Generic LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""

    # Role-Specific LLM Configuration
    query_analyzer_provider: str = "ollama"
    query_analyzer_model: str = "qwen2.5:1.5b"

    answer_generator_provider: str = "ollama"
    answer_generator_model: str = "qwen2.5:1.5b"

    # Dynamic Hybrid Routing Configuration
    hybrid_routing_enabled: bool = True
    hybrid_simple_provider: str = "ollama"
    hybrid_simple_model: str = "qwen2.5:1.5b"
    hybrid_complex_provider: str = "openrouter"
    hybrid_complex_model: str = "openrouter/free"

    # OpenRouter Configuration
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Embedding Configuration
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"

    # Reranker Configuration
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Vector Store Configuration
    vector_store_path: str = "data/vector_store"

    # Document Chunking Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Two-Stage Retrieval Configuration
    top_k_candidates: int = 15
    top_k_final: int = 4

    # Backend API Configuration
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000

    # LangSmith Observability Configuration
    langsmith_tracing: bool = False
    langsmith_project: str = "tutor-rag-backend"
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_hide_inputs: bool = False
    langsmith_hide_outputs: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def setup_langsmith_env(cfg: Settings):
    import os

    tracing_requested = (
        cfg.langsmith_tracing
        or os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    if tracing_requested:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if cfg.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project
            os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
        if cfg.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
            os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        if cfg.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = cfg.langsmith_endpoint
            os.environ["LANGCHAIN_ENDPOINT"] = cfg.langsmith_endpoint
        if cfg.langsmith_hide_inputs:
            os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
        if cfg.langsmith_hide_outputs:
            os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"


setup_langsmith_env(settings)

