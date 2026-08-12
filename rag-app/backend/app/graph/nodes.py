import time
import re
from app.models.state import RAGState
from app.models.responses import SourceCitation
from app.rag.document_resolver import DocumentResolver
from app.utils.tracer import RAGTracer


# ==========================================================
# LATENCY METRICS
# ==========================================================

LATENCY_KEYS = {
    "query_analysis",
    "document_resolution",
    "semantic_retrieval",
    "bm25_retrieval",
    "reranking",
    "llm_generation",
}
def clean_sub_query(query: str) -> str:
    """Clean punctuation/whitespace from generated sub-queries."""

    query = query.strip()

    # Remove punctuation immediately before ? or .
    query = re.sub(r"\s+([?.!])", r"\1", query)
    query = re.sub(r"[,;:]+\s*([?.!])", r"\1", query)

    # Remove trailing comma/semicolon/colon.
    query = re.sub(r"[,;:]+$", "", query).strip()

    # Ensure normal spacing.
    query = re.sub(r"\s+", " ", query)

    return query

def calculate_total_latency(
    metrics: dict[str, float],
) -> float:
    """
    Calculate total latency using only actual timing metrics.

    Counts, scores, and other observability values are
    intentionally excluded.
    """

    return round(
        sum(
            metrics.get(key, 0.0)
            for key in LATENCY_KEYS
        ),
        4,
    )


def make_sources_from_docs(
    docs: list,
) -> list[SourceCitation]:
    """Create unique source citations from retrieved documents."""

    sources = []
    seen = set()

    for idx, document in enumerate(docs):
        metadata = (
            getattr(
                document,
                "metadata",
                {},
            )
            or {}
        )

        document_id = (
            metadata.get("document_id")
            or metadata.get("source_filename")
            or "unknown_doc"
        )

        filename = metadata.get(
            "source_filename",
            document_id,
        )

        chunk_id = metadata.get(
            "chunk_id",
            f"chunk_{idx}",
        )

        page = metadata.get(
            "page_number"
        )

        section = metadata.get(
            "section"
        )

        key = (
            str(document_id),
            str(chunk_id),
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            SourceCitation(
                document_id=str(document_id),
                source_filename=str(filename),
                page_number=(
                    int(page)
                    if page is not None
                    else None
                ),
                section=(
                    str(section)
                    if section is not None
                    else None
                ),
                chunk_id=str(chunk_id),
            )
        )

    return sources


class RAGNodes:
    """Operations executed by the LangGraph workflow."""

    def __init__(
        self,
        retriever,
        llm,
        query_analyzer=None,
        document_resolver=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.query_analyzer = query_analyzer
        self.document_resolver = (
            document_resolver
            or DocumentResolver()
        )
    def retrieve_multi(
        self,
        state: RAGState,
    ) -> dict:
        """
        Execute retrieval independently for each sub-query,
        merge unique chunks across hops, and perform one final
        reranking pass.
        """

        raw_sub_queries = (
            state.sub_queries
            or [state.question]
        )

        sub_queries = [
            clean_sub_query(query)
            for query in raw_sub_queries
            if query and query.strip()
        ]

        if not sub_queries:
            sub_queries = [clean_sub_query(state.question)]

        document_ids = (
            state.resolved_document_ids
            if state.resolved_document_ids
            else None
        )

        merged_documents = []
        seen_chunks = set()

        accumulated_metrics = {
            "semantic_retrieval": 0.0,
            "bm25_retrieval": 0.0,
            "reranking": 0.0,
        }

        accumulated_counts = {
            "semantic_count": 0,
            "bm25_count": 0,
            "merged_count": 0,
            "reranked_count": 0,
            "duplicates_removed": 0,
        }

        successful_queries = []
        missing_queries = []
        hops_trace = []

        for hop_number, sub_query in enumerate(
            sub_queries,
            start=1,
        ):
            (
                sub_docs,
                hop_metrics,
                hop_counts,
                semantic_candidates,
                bm25_candidates,
                merged_candidates,
            ) = self.retriever.retrieve_documents(
                sub_query,
                document_ids=document_ids,
            )

            accumulated_metrics["semantic_retrieval"] += (
                hop_metrics.get(
                    "semantic_retrieval",
                    0.0,
                )
            )

            accumulated_metrics["bm25_retrieval"] += (
                hop_metrics.get(
                    "bm25_retrieval",
                    0.0,
                )
            )

            accumulated_metrics["reranking"] += (
                hop_metrics.get(
                    "reranking",
                    0.0,
                )
            )

            accumulated_counts["semantic_count"] += (
                hop_counts.get(
                    "semantic_count",
                    0,
                )
            )

            accumulated_counts["bm25_count"] += (
                hop_counts.get(
                    "bm25_count",
                    0,
                )
            )

            accumulated_counts["merged_count"] += (
                hop_counts.get(
                    "merged_count",
                    0,
                )
            )

            accumulated_counts["duplicates_removed"] += (
                hop_counts.get(
                    "duplicates_removed",
                    0,
                )
            )

            hops_trace.append(
                (
                    sub_query,
                    [
                        document.page_content
                        for document in sub_docs
                    ],
                )
            )

            RAGTracer.log_semantic_retrieval(
                state.trace_id,
                sub_query,
                semantic_candidates,
            )

            RAGTracer.log_bm25_retrieval(
                state.trace_id,
                sub_query,
                bm25_candidates,
            )

            RAGTracer.log_retrieval_merge(
                state.trace_id,
                hop_counts.get(
                    "semantic_count",
                    0,
                ),
                hop_counts.get(
                    "bm25_count",
                    0,
                ),
                hop_counts.get(
                    "merged_count",
                    0,
                ),
                hop_counts.get(
                    "duplicates_removed",
                    0,
                ),
            )

            if not sub_docs:
                missing_queries.append(sub_query)
                continue

            successful_queries.append(sub_query)

            for document in sub_docs:
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
                    chunk_key = (
                        str(document_id),
                        str(chunk_id),
                    )
                else:
                    chunk_key = (
                        metadata.get(
                            "source_filename"
                        ),
                        metadata.get(
                            "page_number"
                        ),
                        document.page_content.strip(),
                    )

                if chunk_key in seen_chunks:
                    continue

                seen_chunks.add(chunk_key)
                merged_documents.append(document)

        (
            final_docs,
            final_rerank_time,
            final_max_score,
        ) = self.retriever.rerank_documents(
            state.question,
            merged_documents,
            top_k=self.retriever.top_k_final,
        )

        accumulated_metrics["reranking"] += (
            final_rerank_time
        )

        accumulated_counts["reranked_count"] = len(
            final_docs
        )

        context_chunks = [
            document.page_content
            for document in final_docs
        ]

        sources = make_sources_from_docs(final_docs)

        metrics = dict(state.metrics)

        metrics["semantic_retrieval"] = round(
            accumulated_metrics["semantic_retrieval"],
            4,
        )

        metrics["bm25_retrieval"] = round(
            accumulated_metrics["bm25_retrieval"],
            4,
        )

        metrics["reranking"] = round(
            accumulated_metrics["reranking"],
            4,
        )

        metrics["semantic_count"] = (
            accumulated_counts["semantic_count"]
        )

        metrics["bm25_count"] = (
            accumulated_counts["bm25_count"]
        )

        metrics["merged_count"] = len(
            merged_documents
        )

        metrics["reranked_count"] = (
            accumulated_counts["reranked_count"]
        )

        metrics["duplicates_removed"] = (
            accumulated_counts["duplicates_removed"]
        )

        metrics["max_rerank_score"] = final_max_score

        RAGTracer.log_multi_hop(
            state.trace_id,
            sub_queries,
            hops_trace,
        )

        RAGTracer.log_reranking(
            state.trace_id,
            len(merged_documents),
            final_docs,
        )

        if missing_queries and successful_queries:
            answer = (
                f"I found information regarding "
                f"'{successful_queries[0]}', but couldn't "
                f"find enough information regarding "
                f"'{missing_queries[0]}' to reliably answer."
            )

            return {
                "context": context_chunks,
                "sources": sources,
                "answer": answer,
                "metrics": metrics,
            }

        if not context_chunks:
            answer = (
                "I couldn't find sufficient relevant "
                "information across the required "
                "sub-queries."
            )

            return {
                "context": [],
                "sources": [],
                "answer": answer,
                "metrics": metrics,
            }

        return {
            "context": context_chunks,
            "sources": sources,
            "metrics": metrics,
        }

    # ======================================================
    # QUERY ANALYSIS
    # ======================================================
    
    def analyze_query(
        self,
        state: RAGState,
    ) -> dict:

        trace_id = (
            state.trace_id
            or RAGTracer.generate_trace_id()
        )

        metrics = (
            dict(state.metrics)
            if state.metrics
            else {}
        )

        if (
            not state.question
            or not state.question.strip()
        ):
            RAGTracer.log_error(
                trace_id,
                "Query Validation",
                "Empty or whitespace query provided",
                action="Route to invalid_query_prompt",
            )

            return {
                "trace_id": trace_id,
                "operation": "invalid_query",
                "intent": "question",
                "query_type": "single_hop",
                "sub_queries": [],
                "document_references": [],
                "requires_document_resolution": False,
                "resolved_document_ids": [],
                "ambiguous_candidates": [],
                "metrics": metrics,
            }

        operation = "normal_qa"
        intent = "question"
        query_type = "single_hop"

        sub_queries = [
            state.question
        ]

        document_references = []
        requires_resolution = False
        resolved_ids = []
        ambiguous = []

        t0 = time.perf_counter()

        if self.query_analyzer:
            analysis = self.query_analyzer.analyze(
                state.question,
                chat_history=state.chat_history,
            )

            operation = analysis.operation
            intent = analysis.intent
            query_type = analysis.query_type
            sub_queries = analysis.sub_queries
            document_references = (
                analysis.document_references
            )

            requires_resolution = (
                analysis.requires_document_resolution
                or operation
                in [
                    "summarize",
                    "compare",
                    "document_qa",
                ]
            )

        metrics["query_analysis"] = round(
            time.perf_counter() - t0,
            4,
        )

        vague_refs = [
            "that",
            "that book",
            "that document",
            "previous answer",
            "the notes",
        ]

        is_vague = any(
            ref in state.question.lower()
            for ref in vague_refs
        )

        if (
            requires_resolution
            and is_vague
            and not state.chat_history
            and not document_references
        ):
            operation = "missing_reference"

        RAGTracer.log_query_analysis(
            trace_id=trace_id,
            query=state.question,
            intent=intent,
            query_type=query_type,
            doc_refs=document_references,
            sub_queries=sub_queries,
            operation=operation,
        )

        # --------------------------------------------------
        # Document resolution
        # --------------------------------------------------

        t0 = time.perf_counter()

        if (
            requires_resolution
            and operation != "missing_reference"
            and self.document_resolver
            and self.retriever
            and self.retriever.vector_store
        ):
            catalog = (
                self.document_resolver.get_catalog(
                    self.retriever.vector_store
                )
            )

            resolved_ids, ambiguous = (
                self.document_resolver.resolve(
                    query=state.question,
                    catalog=catalog,
                    intent=intent,
                    chat_history=state.chat_history,
                )
            )

            reference_label = (
                document_references[0]
                if document_references
                else state.question
            )

            if ambiguous:
                RAGTracer.log_document_resolution(
                    trace_id=trace_id,
                    reference=reference_label,
                    method="metadata / keyword match",
                    matched_doc="",
                    doc_id="",
                    candidates_count=len(catalog),
                    candidates_list=[
                        c["source_filename"]
                        for c in catalog
                    ],
                    status="AMBIGUOUS",
                    ambiguous_list=ambiguous,
                )

            elif resolved_ids:
                RAGTracer.log_document_resolution(
                    trace_id=trace_id,
                    reference=reference_label,
                    method="partial filename / metadata match",
                    matched_doc=resolved_ids[0],
                    doc_id=resolved_ids[0],
                    candidates_count=len(catalog),
                    candidates_list=[
                        c["source_filename"]
                        for c in catalog
                    ],
                    status="SUCCESS",
                )

        metrics["document_resolution"] = round(
            time.perf_counter() - t0,
            4,
        )

        return {
            "trace_id": trace_id,
            "operation": operation,
            "intent": intent,
            "query_type": query_type,
            "sub_queries": sub_queries,
            "document_references": document_references,
            "requires_document_resolution": requires_resolution,
            "resolved_document_ids": resolved_ids,
            "ambiguous_candidates": ambiguous,
            "metrics": metrics,
        }

    # ======================================================
    # ROUTING
    # ======================================================

    def route_query(
        self,
        state: RAGState,
    ) -> str:
        return self.route_operation(state)

    def route_operation(
        self,
        state: RAGState,
    ) -> str:

        if state.operation == "invalid_query":
            return "invalid_query_prompt"

        if state.operation == "missing_reference":
            return "missing_history_reference_prompt"

        if (
            state.operation == "conversational"
            or state.query_type == "conversational"
        ):
            return "direct_answer"

        if state.ambiguous_candidates:
            return "ambiguity_clarification"

        if state.operation == "summarize":
            if not state.resolved_document_ids:
                return "doc_not_found_prompt"

            return "summarize_document"

        if state.operation == "compare":
            if not state.resolved_document_ids:
                return "doc_not_found_prompt"

            if len(
                state.resolved_document_ids
            ) == 1:
                return "single_doc_compare_prompt"

            return "compare_documents"

        if state.operation == "document_qa":
            if not state.resolved_document_ids:
                return "doc_not_found_prompt"

            if state.query_type == "multi_hop":
                return "retrieve_multi"

            return "retrieve_single"

        if state.query_type == "multi_hop":
            return "retrieve_multi"

        return "retrieve_single"

    # ======================================================
    # PROMPT / ERROR NODES
    # ======================================================

    def invalid_query_prompt(
        self,
        state: RAGState,
    ) -> dict:

        return {
            "context": [],
            "sources": [],
            "answer": (
                "Please enter a valid non-empty query."
            ),
            "metrics": state.metrics,
        }

    def missing_history_reference_prompt(
        self,
        state: RAGState,
    ) -> dict:

        answer = (
            "I'm not sure which document you mean. "
            "Please specify the document."
        )

        RAGTracer.log_error(
            state.trace_id,
            "Document Resolution",
            "Unresolved conversational reference without history",
            action="Request user clarification",
        )

        return {
            "context": [],
            "sources": [],
            "answer": answer,
            "metrics": state.metrics,
        }

    def direct_answer(
        self,
        state: RAGState,
    ) -> dict:

        t0 = time.perf_counter()

        answer = (
            self.llm.generate_conversational(
                state.question,
                chat_history=state.chat_history,
            )
        )

        metrics = dict(state.metrics)

        metrics["llm_generation"] = round(
            time.perf_counter() - t0,
            4,
        )

        metrics["total_latency"] = (
            calculate_total_latency(metrics)
        )

        return {
            "context": [],
            "sources": [],
            "answer": answer,
            "metrics": metrics,
        }

    def doc_not_found_prompt(
        self,
        state: RAGState,
    ) -> dict:

        reference = (
            f"'{state.document_references[0]}'"
            if state.document_references
            else "the specified document"
        )

        answer = (
            f"I could not find {reference} "
            "in the uploaded documents. "
            "Please verify the filename or upload the file."
        )

        RAGTracer.log_error(
            trace_id=state.trace_id,
            stage="Document Resolution",
            problem=(
                f"Could not find document "
                f"for reference {reference}"
            ),
            action="Prompt user to verify filename or upload file",
        )

        return {
            "context": [],
            "sources": [],
            "answer": answer,
            "metrics": state.metrics,
        }

    def single_doc_compare_prompt(
        self,
        state: RAGState,
    ) -> dict:

        document_id = (
            state.resolved_document_ids[0]
        )

        answer = (
            f"I resolved '{document_id}'. "
            "Which second document would you like "
            "me to compare it with?"
        )

        return {
            "context": [],
            "sources": [],
            "answer": answer,
            "metrics": state.metrics,
        }

    def ambiguity_clarification(
        self,
        state: RAGState,
    ) -> dict:

        candidates = "\n".join(
            f"{idx + 1}. {candidate}"
            for idx, candidate
            in enumerate(
                state.ambiguous_candidates
            )
        )

        answer = (
            "I found multiple matching documents:\n"
            f"{candidates}\n\n"
            "Which document would you like me to use?"
        )

        return {
            "context": [],
            "sources": [],
            "answer": answer,
            "metrics": state.metrics,
        }

    # ======================================================
    # SUMMARIZATION
    # ======================================================

    def summarize_document(
        self,
        state: RAGState,
    ) -> dict:

        document_id = (
            state.resolved_document_ids[0]
        )

        all_documents = (
            self.retriever.vector_store
            .get_all_documents()
        )

        target_documents = [
            document
            for document in all_documents
            if (
                document.metadata.get(
                    "document_id"
                ) == document_id
                or document.metadata.get(
                    "source_filename"
                ) == document_id
            )
        ]

        chunks = [
            document.page_content
            for document in target_documents
        ]

        if not chunks:
            answer = (
                f"The document '{document_id}' "
                "appears to be empty or contains "
                "no extractable text."
            )

            return {
                "context": [],
                "sources": [],
                "answer": answer,
                "metrics": state.metrics,
            }

        t0 = time.perf_counter()

        summary = self.llm.generate_summary(
            document_id,
            chunks,
        )

        generation_time = round(
            time.perf_counter() - t0,
            4,
        )

        sources = make_sources_from_docs(
            target_documents[:4]
        )

        metrics = dict(state.metrics)

        metrics["llm_generation"] = (
            generation_time
        )

        metrics["total_latency"] = (
            calculate_total_latency(metrics)
        )

        return {
            "context": chunks[:4],
            "sources": sources,
            "answer": summary,
            "metrics": metrics,
        }

    # ======================================================
    # DOCUMENT COMPARISON
    # ======================================================

    def compare_documents(
        self,
        state: RAGState,
    ) -> dict:

        document_a = (
            state.resolved_document_ids[0]
        )

        document_b = (
            state.resolved_document_ids[1]
        )

        result_a = (
            self.retriever.retrieve_documents(
                state.question,
                document_ids=[document_a],
            )
        )

        result_b = (
            self.retriever.retrieve_documents(
                state.question,
                document_ids=[document_b],
            )
        )

        docs_a, metrics_a, counts_a, _, _, _ = (
            result_a
        )

        docs_b, metrics_b, counts_b, _, _, _ = (
            result_b
        )

        context_a = [
            document.page_content
            for document in docs_a
        ]

        context_b = [
            document.page_content
            for document in docs_b
        ]

        if not context_a or not context_b:

            missing = (
                document_a
                if not context_a
                else document_b
            )

            found = (
                document_b
                if not context_a
                else document_a
            )

            answer = (
                f"I found details for '{found}', "
                f"but couldn't find sufficient context "
                f"for '{missing}' to reliably compare."
            )

            return {
                "context": [],
                "sources": [],
                "answer": answer,
                "metrics": state.metrics,
            }

        t0 = time.perf_counter()

        answer = self.llm.generate_comparison(
            document_a,
            context_a,
            document_b,
            context_b,
            state.question,
        )

        generation_time = round(
            time.perf_counter() - t0,
            4,
        )

        sources = make_sources_from_docs(
            docs_a + docs_b
        )

        metrics = dict(state.metrics)

        metrics["semantic_retrieval"] = round(
            metrics_a.get(
                "semantic_retrieval",
                0,
            )
            + metrics_b.get(
                "semantic_retrieval",
                0,
            ),
            4,
        )

        metrics["bm25_retrieval"] = round(
            metrics_a.get(
                "bm25_retrieval",
                0,
            )
            + metrics_b.get(
                "bm25_retrieval",
                0,
            ),
            4,
        )

        metrics["reranking"] = round(
            metrics_a.get(
                "reranking",
                0,
            )
            + metrics_b.get(
                "reranking",
                0,
            ),
            4,
        )

        metrics["llm_generation"] = (
            generation_time
        )

        metrics["total_latency"] = (
            calculate_total_latency(metrics)
        )

        return {
            "context": (
                context_a + context_b
            ),
            "sources": sources,
            "answer": answer,
            "metrics": metrics,
        }

    # ======================================================
    # SINGLE-HOP RETRIEVAL
    # ======================================================

    def retrieve_single(
        self,
        state: RAGState,
    ) -> dict:

        search_query = (
            state.sub_queries[0]
            if state.sub_queries
            else state.question
        )

        document_ids = (
            state.resolved_document_ids
            if state.resolved_document_ids
            else None
        )

        (
            retrieved_docs,
            retrieval_metrics,
            counts,
            semantic_candidates,
            bm25_candidates,
            merged_candidates,
        ) = self.retriever.retrieve_documents(
            search_query,
            document_ids=document_ids,
        )

        context_chunks = [
            document.page_content
            for document in retrieved_docs
        ]

        sources = make_sources_from_docs(
            retrieved_docs
        )

        # Start from existing metrics.
        metrics = dict(state.metrics)

        # Add retrieval metrics.
        metrics.update(
            retrieval_metrics
        )

        # Add retrieval counts.
        metrics["semantic_count"] = (
            counts["semantic_count"]
        )

        metrics["bm25_count"] = (
            counts["bm25_count"]
        )

        metrics["merged_count"] = (
            counts["merged_count"]
        )

        metrics["reranked_count"] = (
            counts["reranked_count"]
        )

        # Logging.
        RAGTracer.log_semantic_retrieval(
            state.trace_id,
            search_query,
            semantic_candidates,
        )

        RAGTracer.log_bm25_retrieval(
            state.trace_id,
            search_query,
            bm25_candidates,
        )

        RAGTracer.log_retrieval_merge(
            state.trace_id,
            counts["semantic_count"],
            counts["bm25_count"],
            counts["merged_count"],
            counts["duplicates_removed"],
        )

        RAGTracer.log_reranking(
            state.trace_id,
            counts["merged_count"],
            retrieved_docs,
        )

        # No retrieval result.
        if not context_chunks:

            if document_ids:
                answer = (
                    f"The requested document "
                    f"'{document_ids[0]}' did not contain "
                    "enough relevant information "
                    "for this question."
                )
            else:
                answer = (
                    "I couldn't find relevant information "
                    "in the uploaded documents."
                )

            return {
                "context": [],
                "sources": [],
                "answer": answer,
                "metrics": metrics,
            }

        # Relevance threshold.
        max_score = retrieval_metrics.get(
            "max_rerank_score",
            0.0,
        )

        min_threshold = getattr(
            self.retriever,
            "min_rerank_score",
            -2.0,
        )

        if max_score < min_threshold:

            answer = (
                "I found some documents, but the evidence "
                "is not sufficiently relevant to answer "
                "your question confidently."
            )

            return {
                "context": [],
                "sources": [],
                "answer": answer,
                "metrics": metrics,
            }

        return {
            "context": context_chunks,
            "sources": sources,
            "metrics": metrics,
        }

    # ======================================================
    # MULTI-HOP RETRIEVAL
    # ======================================================
    
    

    # ======================================================
    # GENERATION
    # ======================================================

    def generate(
        self,
        state: RAGState,
    ) -> dict:

        if state.answer:
            return {
                "answer": state.answer
            }

        t0 = time.perf_counter()

        try:
            answer = self.llm.generate(
                state.question,
                state.context,
                chat_history=state.chat_history,
            )

        except Exception as exc:

            RAGTracer.log_error(
                state.trace_id,
                "LLM Generation",
                f"Model generation error: {exc}",
                action="Return clean system error message",
            )

            return {
                "answer": (
                    "I'm sorry, but the language model "
                    "service is currently unavailable "
                    "or failed to respond. Please try again later."
                ),
                "metrics": state.metrics,
            }

        generation_time = round(
            time.perf_counter() - t0,
            4,
        )

        metrics = dict(
            state.metrics
        )

        metrics["llm_generation"] = (
            generation_time
        )

        # IMPORTANT:
        # Only latency fields are included in total latency.
        metrics["total_latency"] = (
            calculate_total_latency(metrics)
        )

        total_chars = sum(
            len(chunk)
            for chunk in state.context
        )

        RAGTracer.log_generation_context(
            state.trace_id,
            len(state.context),
            state.sources,
            total_chars,
        )

        RAGTracer.log_generation(
            state.trace_id,
            "qwen2.5:1.5b",
            state.operation,
            len(state.context),
            answer,
            state.sources,
        )

        document_label = ", ".join(
            state.resolved_document_ids
        )

        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=document_label,
            sem_count=int(
                state.metrics.get(
                    "semantic_count",
                    0,
                )
            ),
            bm25_count=int(
                state.metrics.get(
                    "bm25_count",
                    0,
                )
            ),
            merged_count=int(
                state.metrics.get(
                    "merged_count",
                    0,
                )
            ),
            reranked_count=len(
                state.context
            ),
            model="qwen2.5:1.5b",
            answer_status=(
                "Generated answer successfully"
            ),
            metrics=metrics,
            status="SUCCESS",
            full_answer=answer,
        )

        return {
            "answer": answer,
            "metrics": metrics,
        }
