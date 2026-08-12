import time

from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker


class Retriever:
    """
    Hybrid retrieval pipeline:

        Semantic Search
              +
            BM25
              ↓
            Merge
              ↓
          Reranker
              ↓
           Top-K
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        top_k_candidates: int = 15,
        top_k_final: int = 4,
        min_rerank_score: float = -2.0,
    ):
        self.vector_store = vector_store

        self.bm25_retriever = (
            bm25_retriever
            or BM25Retriever()
        )

        self.reranker = reranker

        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.min_rerank_score = min_rerank_score

    def retrieve_candidates(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> tuple:
        """Run semantic retrieval, BM25, and merge WITHOUT reranking."""

        metrics = {}

        # --------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------

        start = time.perf_counter()

        if document_ids is not None:
            if not document_ids:
                semantic_candidates = []
                metrics["embedding"] = 0.0
                metrics["vector_search"] = 0.0
                metrics["semantic_postprocessing"] = 0.0
                metrics["embedding_cache_hit"] = False
            else:
                semantic_candidates = (
                    self.vector_store.similarity_search_scoped(
                        query,
                        document_ids=document_ids,
                        k=self.top_k_candidates,
                    )
                )
                vector_metrics = getattr(
                    self.vector_store,
                    "last_retrieval_metrics",
                    {},
                ) or {}
                metrics["embedding"] = vector_metrics.get("embedding")
                metrics["vector_search"] = vector_metrics.get("vector_search")
                metrics["semantic_postprocessing"] = vector_metrics.get(
                    "semantic_postprocessing"
                )
                metrics["embedding_cache_hit"] = vector_metrics.get(
                    "embedding_cache_hit"
                )
        else:
            semantic_candidates = (
                self.vector_store.similarity_search(
                    query,
                    k=self.top_k_candidates,
                )
            )
            vector_metrics = getattr(
                self.vector_store,
                "last_retrieval_metrics",
                {},
            ) or {}
            metrics["embedding"] = vector_metrics.get("embedding")
            metrics["vector_search"] = vector_metrics.get("vector_search")
            metrics["semantic_postprocessing"] = vector_metrics.get(
                "semantic_postprocessing"
            )
            metrics["embedding_cache_hit"] = vector_metrics.get(
                "embedding_cache_hit"
            )

        metrics["semantic_retrieval"] = round(
            time.perf_counter() - start,
            4,
        )

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

        start = time.perf_counter()

        if document_ids is not None:
            if not document_ids:
                bm25_candidates = []
            else:
                bm25_candidates = (
                    self.bm25_retriever.retrieve_scoped(
                        query,
                        document_ids=document_ids,
                        top_k=self.top_k_candidates,
                    )
                )
        else:
            bm25_candidates = (
                self.bm25_retriever.retrieve(
                    query,
                    top_k=self.top_k_candidates,
                )
            )


        metrics["bm25_retrieval"] = round(
            time.perf_counter() - start,
            4,
        )

        # --------------------------------------------------
        # Merge
        # --------------------------------------------------

        merge_start = time.perf_counter()

        merged_candidates = self.merge_documents(
            semantic_candidates,
            bm25_candidates,
        )

        metrics["merge"] = round(
            time.perf_counter() - merge_start,
            4,
        )

        metrics["reranking"] = 0.0
        metrics["reranker_inference"] = 0.0

        total_candidates = (
            len(semantic_candidates)
            + len(bm25_candidates)
        )

        counts = {
            "semantic_count": len(semantic_candidates),
            "bm25_count": len(bm25_candidates),
            "merged_count": len(merged_candidates),
            "duplicates_removed": (
                total_candidates - len(merged_candidates)
            ),
            "reranked_count": 0,
        }

        return (
            merged_candidates,
            metrics,
            counts,
            semantic_candidates,
            bm25_candidates,
        )

    def retrieve_documents(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> tuple:
        """Run semantic retrieval, BM25, merge, and reranking (single-hop convenience method)."""
        (
            merged_candidates,
            metrics,
            counts,
            semantic_candidates,
            bm25_candidates,
        ) = self.retrieve_candidates(query, document_ids=document_ids)

        if not merged_candidates:
            metrics["reranking"] = 0.0
            metrics["reranker_inference"] = 0.0
            metrics["max_rerank_score"] = -10.0
            counts["reranked_count"] = 0

            return (
                [],
                metrics,
                counts,
                semantic_candidates,
                bm25_candidates,
                merged_candidates,
            )

        final_docs, rerank_time, max_score = self.rerank_documents(
            query,
            merged_candidates,
            top_k=self.top_k_final,
        )

        metrics["reranking"] = rerank_time
        metrics["reranker_inference"] = rerank_time
        metrics["max_rerank_score"] = max_score

        counts["reranked_count"] = len(final_docs)

        return (
            final_docs,
            metrics,
            counts,
            semantic_candidates,
            bm25_candidates,
            merged_candidates,
        )

    @staticmethod
    def merge_documents(
        *document_lists,
    ):
        """
        Merge semantic and BM25 results.

        Primary identity:

            document_id + chunk_id

        This prevents identical text from different documents
        from being incorrectly treated as duplicates.
        """

        merged = []
        seen = set()

        for documents in document_lists:
            for document in documents:

                metadata = (
                    getattr(
                        document,
                        "metadata",
                        {},
                    )
                    or {}
                )

                document_id = metadata.get(
                    "document_id"
                )

                chunk_id = metadata.get(
                    "chunk_id"
                )

                if document_id and chunk_id:
                    key = (
                        str(document_id),
                        str(chunk_id),
                    )
                else:
                    # Legacy fallback.
                    key = (
                        metadata.get(
                            "source_filename"
                        ),
                        metadata.get(
                            "page_number"
                        ),
                        document.page_content.strip(),
                    )

                if key in seen:
                    continue

                seen.add(key)
                merged.append(document)

        return merged

    def rerank_documents(
        self,
        query: str,
        documents: list,
        top_k: int | None = None,
    ):
        """Rerank candidates and return top-k."""

        if not documents:
            return [], 0.0, -10.0

        top_k = (
            top_k
            or self.top_k_final
        )

        start = time.perf_counter()

        if self.reranker:
            final_docs = self.reranker.rerank(
                query,
                documents,
                top_k=top_k,
            )
        else:
            final_docs = documents[:top_k]

        elapsed = round(
            time.perf_counter() - start,
            4,
        )

        max_score = 0.0

        if final_docs:
            max_score = float(
                final_docs[0]
                .metadata
                .get(
                    "reranker_score",
                    0.0,
                )
            )

        return (
            final_docs,
            elapsed,
            round(max_score, 4),
        )

    def retrieve(
        self,
        query: str,
    ) -> list[str]:
        """Simple retrieval interface returning text."""

        documents, _, _, _, _, _ = (
            self.retrieve_documents(query)
        )

        return [
            document.page_content
            for document in documents
        ]