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
        if os.path.exists(os.path.join(store_path, "index.faiss")):
            self.store = FAISS.load_local(
                folder_path=store_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

    def add_documents(self, documents: List[Document]) -> int:
        """Adds LangChain Document objects to vector store and persists index."""
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
        """Converts DocumentChunks to LangChain Documents and indexes them."""
        docs = [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        return self.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4):
        """Performs similarity search and returns top-k matching Document objects."""
        return self.store.similarity_search(query, k=k) if self.store else []
