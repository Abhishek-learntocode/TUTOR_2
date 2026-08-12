from collections import OrderedDict
import threading
import time

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingService(Embeddings):
    """Embedding service with query latency measurement and a small LRU cache."""

    def __init__(self, provider: str = "ollama", model_name: str = "nomic-embed-text:latest", base_url: str = "http://localhost:11434", query_cache_size: int = 256):
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        self.query_cache_size = max(0, int(query_cache_size))
        self._query_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self.last_query_embedding_latency = 0.0
        self.last_query_embedding_cache_hit = False
        self.last_document_embedding_latency = 0.0

    def __call__(self, text: str):
        return self.embed_query(text)

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join(str(text).strip().split())

    def embed_documents(self, texts: list[str]):
        start = time.perf_counter()
        result = self.embeddings.embed_documents(texts)
        self.last_document_embedding_latency = round(time.perf_counter() - start, 4)
        return result

    def embed_query(self, text: str):
        query = self._normalize_query(text)
        if not query:
            raise ValueError("Cannot embed an empty query.")
        if self.query_cache_size > 0:
            with self._cache_lock:
                cached = self._query_cache.get(query)
                if cached is not None:
                    self._query_cache.move_to_end(query)
                    self.last_query_embedding_latency = 0.0
                    self.last_query_embedding_cache_hit = True
                    return cached
        start = time.perf_counter()
        embedding = self.embeddings.embed_query(query)
        self.last_query_embedding_latency = round(time.perf_counter() - start, 4)
        self.last_query_embedding_cache_hit = False
        if self.query_cache_size > 0:
            with self._cache_lock:
                self._query_cache[query] = embedding
                self._query_cache.move_to_end(query)
                while len(self._query_cache) > self.query_cache_size:
                    self._query_cache.popitem(last=False)
        return embedding

    def clear_query_cache(self):
        with self._cache_lock:
            self._query_cache.clear()

    def get_query_cache_size(self) -> int:
        with self._cache_lock:
            return len(self._query_cache)