from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:
    """Lexical BM25 retriever built over indexed Document objects."""

    def __init__(self, documents: list[Document] = None):
        self.documents = documents or []
        self.bm25 = None
        if self.documents:
            self.rebuild(self.documents)

    def rebuild(self, documents: list[Document]):
        """Rebuilds BM25 index from all current document chunks."""
        self.documents = documents
        if not documents:
            self.bm25 = None
            return

        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 15) -> list[Document]:
        """Retrieves top-k matching documents using BM25 lexical search."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = sorted(
            zip(scores, self.documents),
            key=lambda x: x[0],
            reverse=True,
        )

        # Keep matching documents with non-zero BM25 relevance score
        matched_docs = [doc for score, doc in scored_docs if score > 0]
        return matched_docs[:top_k]

    def retrieve_scoped(self, query: str, document_ids: list[str], top_k: int = 15) -> list[Document]:
        cands = self.retrieve(query, top_k=top_k * 2)
        doc_set = set(document_ids)
        filtered = [
            d for d in cands
            if d.metadata.get("document_id") in doc_set or d.metadata.get("source_filename") in doc_set
        ]
        return filtered[:top_k]

