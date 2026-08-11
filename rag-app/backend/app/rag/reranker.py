from sentence_transformers import CrossEncoder


class Reranker:
    """Lightweight CrossEncoder Reranker with lazy model loading."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[Reranker Log] Loading CrossEncoder model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list, top_k: int = 4) -> list:
        if not documents:
            return []

        pairs = [
            (query, doc.page_content if hasattr(doc, "page_content") else str(doc))
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        scored_docs = []
        for score, doc in zip(scores, documents):
            score_val = float(score)
            if hasattr(doc, "metadata"):
                doc.metadata["reranker_score"] = round(score_val, 4)
            scored_docs.append((score_val, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        final_docs = [doc for _, doc in scored_docs[:top_k]]
        print(f"[Reranker Log] retrieved candidates = {len(documents)}, reranked candidates = {len(final_docs)}")

        return final_docs
