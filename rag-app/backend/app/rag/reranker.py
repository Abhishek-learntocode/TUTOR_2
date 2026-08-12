import time

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker for retrieved RAG candidates."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Prefer NVIDIA GPU when available.
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"[Reranker Log] Loading CrossEncoder model "
            f"'{self.model_name}' on {self.device}..."
        )

        start = time.perf_counter()

        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

        load_time = time.perf_counter() - start

        print(
            f"[Reranker Log] Model loaded in "
            f"{load_time:.2f}s"
        )

        # Warm up the model once so CUDA/model initialization
        # is not included in the first real query latency.
        self._warmup()

        print(
            f"[Reranker Log] Ready. "
            f"device={self.device}, "
            f"batch_size={self.batch_size}, "
            f"max_length={self.max_length}"
        )

    def _warmup(self):
        """Run one small inference to initialize the model/device."""

        try:
            start = time.perf_counter()

            self.model.predict(
                [("warmup query", "warmup document")],
                batch_size=1,
                show_progress_bar=False,
            )

            elapsed = time.perf_counter() - start

            print(
                f"[Reranker Log] Warmup completed in "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:
            # Warmup failure should not prevent the application
            # from starting. The real query will surface the error.
            print(
                f"[Reranker Warning] Warmup failed: {exc}"
            )

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 4,
    ) -> list:
        """Rerank documents against the query and return top-k."""

        if not documents:
            return []

        if not query or not query.strip():
            return documents[:top_k]

        # CrossEncoder expects:
        # [(query, document_text), ...]
        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        start = time.perf_counter()

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            print(
                f"[Reranker Error] Failed to rerank "
                f"{len(documents)} candidates: {exc}"
            )

            # Safe fallback: preserve retrieval results.
            return documents[:top_k]

        elapsed = time.perf_counter() - start

        # Convert numpy/tensor values to normal Python floats.
        scored_documents = []

        for document, score in zip(documents, scores):
            score = float(score)

            # Store the score in metadata so the rest of the
            # pipeline can use it for observability/citations.
            document.metadata["reranker_score"] = score

            scored_documents.append(
                (document, score)
            )

        # Highest relevance score first.
        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = [
            document
            for document, _ in scored_documents[:top_k]
        ]

        print(
            f"[Reranker Log] "
            f"retrieved candidates = {len(documents)}, "
            f"reranked candidates = {len(results)}, "
            f"inference = {elapsed:.4f}s"
        )

        return results