from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingService:
    """Creates embeddings for documents and queries."""

    def __init__(self, provider: str = "ollama", model_name: str = "nomic-embed-text:latest", base_url: str = "http://localhost:11434"):
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)

    def embed_documents(self, texts: list[str]):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
