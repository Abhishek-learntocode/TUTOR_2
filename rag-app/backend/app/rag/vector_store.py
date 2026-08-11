import os
from langchain_community.vectorstores import FAISS


class VectorStore:
    """Stores vectors and performs similarity search."""

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

    def add_documents(self, documents):
        if not documents:
            return 0
        if self.store is None:
            self.store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.store.add_documents(documents)

        os.makedirs(self.store_path, exist_ok=True)
        self.store.save_local(self.store_path)
        return len(documents)

    def similarity_search(self, query: str, k: int = 4):
        return self.store.similarity_search(query, k=k) if self.store else []
