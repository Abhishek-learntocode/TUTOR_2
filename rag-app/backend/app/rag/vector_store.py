import os
import time
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.models.canonical import DocumentChunk


class VectorStore:
    """Stores vectors and performs similarity search using FAISS."""

    def __init__(self, embeddings, store_path: str = "data/vector_store"):
        self.embeddings = embeddings
        self.store_path = store_path
        self.store = None
        self.last_retrieval_metrics = {}
        self._load_if_exists()

    def _load_if_exists(self):
        index_file = os.path.join(self.store_path, "index.faiss")
        pkl_file = os.path.join(self.store_path, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                loaded_store = FAISS.load_local(
                    folder_path=self.store_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # One dimension check is still required, but EmbeddingService
                # caches this query so it does not create repeated model work.
                dummy_dim = len(self.embeddings.embed_query("test"))
                if loaded_store.index.d == dummy_dim:
                    self.store = loaded_store
                else:
                    print(
                        f"[VectorStore Warning] Index dimension mismatch (loaded {loaded_store.index.d} vs model {dummy_dim}). Resetting vector store index for new embedding model."
                    )
                    self.store = None
            except Exception as e:
                print(f"[VectorStore Warning] Failed to load vector store: {e}")
                self.store = None

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        if self.store is None:
            self.store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.store.add_documents(documents)
        os.makedirs(self.store_path, exist_ok=True)
        self.store.save_local(self.store_path)
        return len(documents)

    def _chunk_to_document(self, chunk: DocumentChunk) -> Document:
        """Convert DocumentChunk to LangChain Document while explicitly preserving chunk identity metadata."""
        metadata = dict(chunk.metadata or {})
        metadata["chunk_id"] = chunk.chunk_id
        metadata["document_id"] = chunk.document_id
        return Document(
            page_content=chunk.content,
            metadata=metadata,
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        docs = [self._chunk_to_document(chunk) for chunk in chunks]
        return self.add_documents(docs)

    def get_all_documents(self) -> List[Document]:
        if self.store and hasattr(self.store, "docstore") and hasattr(self.store.docstore, "_dict"):
            return list(self.store.docstore._dict.values())
        return []

    def _record_metrics(self, total, embedding=None, vector_search=None, postprocessing=None):
        self.last_retrieval_metrics = {
            "semantic_retrieval": round(total, 4),
            "embedding": None if embedding is None else round(embedding, 4),
            "vector_search": None if vector_search is None else round(vector_search, 4),
            "semantic_postprocessing": None if postprocessing is None else round(postprocessing, 4),
            "embedding_cache_hit": getattr(self.embeddings, "last_query_embedding_cache_hit", None),
        }

    def similarity_search(self, query: str, k: int = 4):
        if not self.store:
            self._record_metrics(0.0, 0.0, 0.0, 0.0)
            return []
        start = time.perf_counter()
        before_embedding = getattr(self.embeddings, "last_query_embedding_latency", None)
        search_start = time.perf_counter()
        result = self.store.similarity_search(query, k=k)
        total = time.perf_counter() - start
        after_embedding = getattr(self.embeddings, "last_query_embedding_latency", None)
        embedding_time = after_embedding if after_embedding is not None else before_embedding
        vector_time = max(0.0, total - (embedding_time or 0.0))
        self._record_metrics(total, embedding_time, vector_time, 0.0)
        return result

    def similarity_search_scoped(self, query: str, document_ids: list[str], k: int = 15) -> List[Document]:
        if not self.store or not document_ids:
            self._record_metrics(0.0, 0.0, 0.0, 0.0)
            return []
        start = time.perf_counter()
        query_embedding = self.embeddings.embed_query(query)
        embedding_time = getattr(self.embeddings, "last_query_embedding_latency", None)
        doc_set = set(document_ids)

        if len(document_ids) == 1:
            target_id = document_ids[0]
            cands_with_score = self.store.similarity_search_with_score_by_vector(query_embedding, k=k, filter={"document_id": target_id})
            filtered = [doc for doc, score in cands_with_score]
            if not filtered:
                cands_with_score = self.store.similarity_search_with_score_by_vector(query_embedding, k=k, filter={"source_filename": target_id})
                filtered = [doc for doc, score in cands_with_score]
        else:
            total_docs = len(self.store.docstore._dict) if hasattr(self.store, "docstore") and hasattr(self.store.docstore, "_dict") else k * 20
            raw_cands = self.store.similarity_search_with_score_by_vector(query_embedding, k=total_docs)
            filtered = [
                doc for doc, score in raw_cands
                if doc.metadata.get("document_id") in doc_set or doc.metadata.get("source_filename") in doc_set
            ][:k]

        total = time.perf_counter() - start
        vector_search_time = max(0.0, total - (embedding_time or 0.0))
        self._record_metrics(total, embedding_time, vector_search_time, 0.0)
        return filtered