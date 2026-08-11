from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""

    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""

    # Embedding Configuration
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"

    # Vector Store Configuration
    vector_store_path: str = "data/vector_store"

    # Document Chunking Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval Configuration
    top_k: int = 4

    # Backend API Configuration
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
