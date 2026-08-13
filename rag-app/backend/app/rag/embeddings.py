from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingService:
    """Creates embeddings for documents and queries with query embedding caching."""

    def __init__(self, provider: str = "ollama", model_name: str = "nomic-embed-text:latest", base_url: str = "http://localhost:11434"):
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)

        self._query_cache: dict[str, list[float]] = {}
        self.last_cache_hit: bool = False

    def embed_documents(self, texts: list[str]):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str):
        if text in self._query_cache:
            self.last_cache_hit = True
            return self._query_cache[text]
        self.last_cache_hit = False
        vec = self.embeddings.embed_query(text)
        self._query_cache[text] = vec
        return vec


