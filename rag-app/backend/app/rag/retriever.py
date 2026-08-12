import os
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from langchain_core.documents import Document


from app.config import settings


class Retriever:
    """Hybrid Retriever combining BGE-M3 Semantic Vector Search + BM25 Lexical Search + Filename Matching + Reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever = None,
        reranker: Reranker = None,
        top_k_candidates: int = 15,
        top_k_final: int = 4,
        min_rerank_score: float = None,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.min_rerank_score = min_rerank_score if min_rerank_score is not None else getattr(settings, "min_rerank_score", -2.0)


    def retrieve_documents(self, query: str, document_ids: list[str] = None) -> tuple[list[Document], dict[str, float]]:
        import time

        metrics = {}
        all_docs = self.vector_store.get_all_documents()

        if document_ids:
            doc_id_set = set(document_ids)
            all_docs = [
                doc for doc in all_docs
                if doc.metadata.get("document_id") in doc_id_set or doc.metadata.get("source_filename") in doc_id_set
            ]

        if not all_docs:
            print(f"[Hybrid Retriever Log] No documents matched filter: {document_ids}")
            return [], {"semantic_retrieval": 0.0, "bm25_retrieval": 0.0, "reranking": 0.0}

        # 1. Direct Document Filename Matching
        query_lower = query.lower()
        explicit_filename_candidates = []

        for doc in all_docs:
            filename = doc.metadata.get("source_filename", "").lower()
            name_without_ext = os.path.splitext(filename)[0]
            if filename and (filename in query_lower or (len(name_without_ext) >= 3 and name_without_ext in query_lower)):
                explicit_filename_candidates.append(doc)

        # 2. Semantic Retrieval (BGE-M3 Vector Search - Top 15)
        t_sem_start = time.perf_counter()
        semantic_candidates = self.vector_store.similarity_search(query, k=self.top_k_candidates)
        if document_ids:
            doc_id_set = set(document_ids)
            semantic_candidates = [
                doc for doc in semantic_candidates
                if doc.metadata.get("document_id") in doc_id_set or doc.metadata.get("source_filename") in doc_id_set
            ]
        metrics["semantic_retrieval"] = round(time.perf_counter() - t_sem_start, 4)

        # 3. Lexical Retrieval (BM25 Search - Top 15)
        t_bm25_start = time.perf_counter()
        if document_ids:
            lexical_candidates = self.bm25_retriever.retrieve_scoped(query, document_ids, top_k=self.top_k_candidates)
        else:
            lexical_candidates = self.bm25_retriever.retrieve(query, top_k=self.top_k_candidates)
        metrics["bm25_retrieval"] = round(time.perf_counter() - t_bm25_start, 4)

        print(f"[Hybrid Retriever Log] Scoped Document IDs          = {document_ids}")
        print(f"[Hybrid Retriever Log] Explicit filename candidates = {len(explicit_filename_candidates)}")
        print(f"[Hybrid Retriever Log] Semantic candidates          = {len(semantic_candidates)}")
        print(f"[Hybrid Retriever Log] BM25 candidates              = {len(lexical_candidates)}")

        # 4. Merge Candidates & Deduplicate using chunk content
        seen_contents = set()
        merged_candidates: list[Document] = []

        for doc in explicit_filename_candidates + semantic_candidates + lexical_candidates:
            content_key = doc.page_content.strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged_candidates.append(doc)

        print(f"[Hybrid Retriever Log] Merged candidates            = {len(merged_candidates)} (deduplicated)")

        if not merged_candidates:
            print("[Hybrid Retriever Log] Reranked candidates = 0")
            metrics["reranking"] = 0.0
            return [], metrics

        # 5. CrossEncoder Reranker
        t_rerank_start = time.perf_counter()
        if self.reranker:
            final_docs = self.reranker.rerank(query, merged_candidates, top_k=self.top_k_final)
        else:
            final_docs = merged_candidates[:self.top_k_final]
        metrics["reranking"] = round(time.perf_counter() - t_rerank_start, 4)

        counts = {
            "semantic_count": len(semantic_candidates),
            "bm25_count": len(lexical_candidates),
            "merged_count": len(merged_candidates),
            "duplicates_removed": len(explicit_filename_candidates + semantic_candidates + lexical_candidates) - len(merged_candidates),
            "reranked_count": len(final_docs),
        }

        return final_docs, metrics, counts, semantic_candidates, lexical_candidates, merged_candidates

    def retrieve(self, query: str, document_ids: list[str] = None) -> list[str]:
        res = self.retrieve_documents(query, document_ids)
        docs = res[0]
        return [doc.page_content for doc in docs]



