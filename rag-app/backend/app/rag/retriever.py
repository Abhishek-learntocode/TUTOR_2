class Retriever:
    """Retrieves relevant context from the vector store."""

    def __init__(self, vector_store, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[str]:
        docs = self.vector_store.similarity_search(query, k=self.top_k)
        return [doc.page_content for doc in docs]
