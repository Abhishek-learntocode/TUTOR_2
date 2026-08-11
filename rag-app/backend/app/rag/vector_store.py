import os
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
                dummy_dim = len(self.embeddings.embed_query("test"))
                if loaded_store.index.d == dummy_dim:
                    self.store = loaded_store
                else:
                    print(
                        f"[VectorStore Warning] Index dimension mismatch (loaded {loaded_store.index.d} vs model {dummy_dim}). "
                        "Resetting vector store index for new embedding model."
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

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        docs = [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        return self.add_documents(docs)

    def get_all_documents(self) -> List[Document]:
        """Extracts all stored Document objects from FAISS docstore."""
        if self.store and hasattr(self.store, "docstore") and hasattr(self.store.docstore, "_dict"):
            return list(self.store.docstore._dict.values())
        return []

    def similarity_search(self, query: str, k: int = 4):
        return self.store.similarity_search(query, k=k) if self.store else []
