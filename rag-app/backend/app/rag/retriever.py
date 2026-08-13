import os
from langsmith import traceable, get_current_run_tree
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from langchain_core.documents import Document


class Retriever:
    """Hybrid Retriever combining BGE-M3 Semantic Vector Search + BM25 Lexical Search + Filename Matching + Reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever = None,
        reranker: Reranker = None,
        top_k_candidates: int = 15,
        top_k_final: int = 4,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final

    @traceable(name="retrieval", run_type="retriever")
    def retrieve(self, query: str) -> list[str]:
        all_docs = self.vector_store.get_all_documents()
        self.bm25_retriever.rebuild(all_docs)

        # 1. Direct Document Filename Matching (e.g. "summarize OS_Notes.txt" or "Summarize DBMS_Book")
        query_lower = query.lower()
        explicit_filename_candidates = []

        for doc in all_docs:
            filename = doc.metadata.get("source_filename", "").lower()
            name_without_ext = os.path.splitext(filename)[0]
            if filename and (filename in query_lower or (len(name_without_ext) >= 3 and name_without_ext in query_lower)):
                explicit_filename_candidates.append(doc)

        # 2. Semantic Retrieval (BGE-M3 Vector Search - Top 15)
        semantic_candidates = self.vector_store.similarity_search(query, k=self.top_k_candidates)

        # 3. Lexical Retrieval (BM25 Search - Top 15)
        lexical_candidates = self.bm25_retriever.retrieve(query, top_k=self.top_k_candidates)

        print(f"[Hybrid Retriever Log] Explicit filename candidates = {len(explicit_filename_candidates)}")
        print(f"[Hybrid Retriever Log] Semantic candidates          = {len(semantic_candidates)}")
        print(f"[Hybrid Retriever Log] BM25 candidates              = {len(lexical_candidates)}")

        # Document Scoping Enforcement:
        # If explicit filename candidates exist, restrict semantic and lexical candidates
        # exclusively to chunks belonging to those explicit documents.
        allowed_filenames = set(
            doc.metadata.get("source_filename", "").lower()
            for doc in explicit_filename_candidates
            if doc.metadata.get("source_filename")
        )

        if allowed_filenames:
            semantic_candidates = [
                doc for doc in semantic_candidates
                if doc.metadata.get("source_filename", "").lower() in allowed_filenames
            ]
            lexical_candidates = [
                doc for doc in lexical_candidates
                if doc.metadata.get("source_filename", "").lower() in allowed_filenames
            ]

        # 4. Merge Candidates & Deduplicate using chunk content
        seen_contents = set()
        merged_candidates: list[Document] = []

        for doc in explicit_filename_candidates + semantic_candidates + lexical_candidates:

            content_key = doc.page_content.strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged_candidates.append(doc)

        print(f"[Hybrid Retriever Log] Merged candidates            = {len(merged_candidates)} (deduplicated)")

        doc_ref = (
            explicit_filename_candidates[0].metadata.get("source_filename")
            if explicit_filename_candidates
            else None
        )
        resolved_doc_ids = list(
            set(
                doc.metadata.get("source_filename")
                for doc in merged_candidates
                if doc.metadata.get("source_filename")
            )
        )
        cache_hit = getattr(
            getattr(self.vector_store, "embeddings", None), "last_cache_hit", False
        )
        dups_removed = (
            len(explicit_filename_candidates) + len(semantic_candidates) + len(lexical_candidates)
        ) - len(merged_candidates)

        try:
            rt = get_current_run_tree()
            if rt:
                rt.metadata["semantic_count"] = len(semantic_candidates)
                rt.metadata["bm25_count"] = len(lexical_candidates)
                rt.metadata["merged_count"] = len(merged_candidates)
                rt.metadata["duplicates_removed"] = dups_removed
                rt.metadata["document_reference"] = doc_ref
                rt.metadata["resolved_document_ids"] = resolved_doc_ids
                rt.metadata["embedding_cache_hit"] = cache_hit
        except Exception:
            pass

        if not merged_candidates:
            print("[Hybrid Retriever Log] Reranked candidates = 0")
            return []

        # 5. CrossEncoder Reranker (BAAI/bge-reranker-v2-m3 -> Top 4)
        if self.reranker:
            final_docs = self.reranker.rerank(query, merged_candidates, top_k=self.top_k_final)
        else:
            final_docs = merged_candidates[:self.top_k_final]

        print(f"[Hybrid Retriever Log] Reranked candidates          = {len(final_docs)}")

        return [doc.page_content for doc in final_docs]

