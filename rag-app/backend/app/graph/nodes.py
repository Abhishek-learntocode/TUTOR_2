import time
from app.models.state import RAGState
from app.models.responses import SourceCitation
from app.rag.document_resolver import DocumentResolver
from app.utils.tracer import RAGTracer


def make_sources_from_docs(docs: list) -> list[SourceCitation]:
    sources = []
    seen = set()
    for idx, d in enumerate(docs):
        meta = getattr(d, "metadata", {}) or {}
        doc_id = meta.get("document_id") or meta.get("source_filename") or "unknown_doc"
        filename = meta.get("source_filename", doc_id)
        chunk_id = meta.get("chunk_id", f"chunk_{idx}")
        page = meta.get("page_number")
        section = meta.get("section")

        key = (doc_id, chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceCitation(
                    document_id=str(doc_id),
                    source_filename=str(filename),
                    page_number=int(page) if page is not None else None,
                    section=str(section) if section is not None else None,
                    chunk_id=str(chunk_id),
                )
            )
    return sources


class RAGNodes:
    """Operations executed by the LangGraph workflow."""

    def __init__(self, retriever, llm, query_analyzer=None, document_resolver=None):
        self.retriever = retriever
        self.llm = llm
        self.query_analyzer = query_analyzer
        self.document_resolver = document_resolver or DocumentResolver()

    def analyze_query(self, state: RAGState) -> dict:
        trace_id = state.trace_id or RAGTracer.generate_trace_id()
        metrics = dict(state.metrics) if state.metrics else {}

        if not state.question or not state.question.strip():
            RAGTracer.log_error(trace_id, "Query Validation", "Empty or whitespace query provided", action="Route to invalid_query_prompt")
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
        q_type = "single_hop"
        sub_queries = [state.question]
        doc_refs = []
        req_res = False
        resolved_ids = []
        ambiguous = []

        t_qa_start = time.perf_counter()
        if self.query_analyzer:
            analysis = self.query_analyzer.analyze(state.question, chat_history=state.chat_history)
            operation = analysis.operation
            intent = analysis.intent
            q_type = analysis.query_type
            sub_queries = analysis.sub_queries
            doc_refs = analysis.document_references
            req_res = analysis.requires_document_resolution or (operation in ["summarize", "compare", "document_qa"])
        metrics["query_analysis"] = round(time.perf_counter() - t_qa_start, 4)

        # Handle vague conversational references when history is missing/empty
        vague_refs = ["that", "that book", "that document", "previous answer", "the notes"]
        is_vague = any(vr in state.question.lower() for vr in vague_refs)
        if req_res and is_vague and not state.chat_history and not doc_refs:
            operation = "missing_reference"

        RAGTracer.log_query_analysis(
            trace_id=trace_id,
            query=state.question,
            intent=intent,
            query_type=q_type,
            doc_refs=doc_refs,
            sub_queries=sub_queries,
            operation=operation,
        )

        t_res_start = time.perf_counter()
        if req_res and operation != "missing_reference" and self.document_resolver and self.retriever and self.retriever.vector_store:
            catalog = self.document_resolver.get_catalog(self.retriever.vector_store)
            resolved_ids, ambiguous = self.document_resolver.resolve(
                query=state.question,
                catalog=catalog,
                intent=intent,
                chat_history=state.chat_history,
            )
            ref_label = doc_refs[0] if doc_refs else state.question
            if ambiguous:
                RAGTracer.log_document_resolution(
                    trace_id=trace_id,
                    reference=ref_label,
                    method="metadata / keyword match",
                    matched_doc="",
                    doc_id="",
                    candidates_count=len(catalog),
                    candidates_list=[c["source_filename"] for c in catalog],
                    status="AMBIGUOUS",
                    ambiguous_list=ambiguous,
                )
            elif resolved_ids:
                RAGTracer.log_document_resolution(
                    trace_id=trace_id,
                    reference=ref_label,
                    method="partial filename / metadata match",
                    matched_doc=resolved_ids[0],
                    doc_id=resolved_ids[0],
                    candidates_count=len(catalog),
                    candidates_list=[c["source_filename"] for c in catalog],
                    status="SUCCESS",
                )
        metrics["document_resolution"] = round(time.perf_counter() - t_res_start, 4)

        return {
            "trace_id": trace_id,
            "operation": operation,
            "intent": intent,
            "query_type": q_type,
            "sub_queries": sub_queries,
            "document_references": doc_refs,
            "requires_document_resolution": req_res,
            "resolved_document_ids": resolved_ids,
            "ambiguous_candidates": ambiguous,
            "metrics": metrics,
        }

    def route_query(self, state: RAGState) -> str:
        return self.route_operation(state)

    def route_operation(self, state: RAGState) -> str:
        if state.operation == "invalid_query":
            return "invalid_query_prompt"

        if state.operation == "missing_reference":
            return "missing_history_reference_prompt"

        if state.operation == "conversational" or state.query_type == "conversational":
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
            if len(state.resolved_document_ids) == 1:
                return "single_doc_compare_prompt"
            return "compare_documents"

        if state.operation == "document_qa":
            if not state.resolved_document_ids:
                return "doc_not_found_prompt"
            return "retrieve_multi" if state.query_type == "multi_hop" else "retrieve_single"

        return "retrieve_multi" if state.query_type == "multi_hop" else "retrieve_single"

    def invalid_query_prompt(self, state: RAGState) -> dict:
        ans = "Please enter a valid non-empty query."
        return {"context": [], "sources": [], "answer": ans, "metrics": state.metrics}

    def missing_history_reference_prompt(self, state: RAGState) -> dict:
        ans = "I'm not sure which document you mean. Please specify the document."
        RAGTracer.log_error(state.trace_id, "Document Resolution", "Unresolved conversational reference without history", action="Request user clarification")
        return {"context": [], "sources": [], "answer": ans, "metrics": state.metrics}

    def direct_answer(self, state: RAGState) -> dict:
        t0 = time.perf_counter()
        ans = self.llm.generate_conversational(state.question, chat_history=state.chat_history)
        metrics = dict(state.metrics)
        metrics["llm_generation"] = round(time.perf_counter() - t0, 4)
        metrics["total_latency"] = round(sum(metrics.values()), 4)

        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type="conversational",
            document="",
            sem_count=0,
            bm25_count=0,
            merged_count=0,
            reranked_count=0,
            model="qwen2.5:1.5b",
            answer_status="Generated direct conversational answer",
            metrics=metrics,
            status="SUCCESS",
        )
        return {"context": [], "sources": [], "answer": ans, "metrics": metrics}

    def doc_not_found_prompt(self, state: RAGState) -> dict:
        ref_str = f"'{state.document_references[0]}'" if state.document_references else "the specified document"
        ans = f"I could not find {ref_str} in the uploaded documents. Please verify the filename or upload the file."
        RAGTracer.log_error(
            trace_id=state.trace_id,
            stage="Document Resolution",
            problem=f"Could not find document for reference {ref_str}",
            action="Prompt user to verify filename or upload file",
        )
        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=ref_str,
            sem_count=0,
            bm25_count=0,
            merged_count=0,
            reranked_count=0,
            model="qwen2.5:1.5b",
            answer_status="Document not found prompt returned",
            metrics=state.metrics,
            status="FAILED_NOT_FOUND",
        )
        return {"context": [], "sources": [], "answer": ans}

    def single_doc_compare_prompt(self, state: RAGState) -> dict:
        doc_id = state.resolved_document_ids[0]
        ans = f"I resolved '{doc_id}'. Which second document would you like me to compare it with?"
        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=doc_id,
            sem_count=0,
            bm25_count=0,
            merged_count=0,
            reranked_count=0,
            model="qwen2.5:1.5b",
            answer_status="Prompted for second comparison document",
            metrics=state.metrics,
            status="PARTIAL_RESOLVED",
        )
        return {"context": [], "sources": [], "answer": ans}

    def ambiguity_clarification(self, state: RAGState) -> dict:
        cands_str = "\n".join([f"{idx+1}. {c}" for idx, c in enumerate(state.ambiguous_candidates)])
        ans = f"I found multiple matching documents:\n{cands_str}\n\nWhich document would you like me to use?"
        RAGTracer.log_error(
            trace_id=state.trace_id,
            stage="Document Resolution",
            problem=f"Multiple documents matched reference '{state.document_references}'",
            candidates=state.ambiguous_candidates,
            action="Request user clarification",
        )
        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document="AMBIGUOUS",
            sem_count=0,
            bm25_count=0,
            merged_count=0,
            reranked_count=0,
            model="qwen2.5:1.5b",
            answer_status="Ambiguity clarification requested",
            metrics=state.metrics,
            status="AMBIGUOUS",
        )
        return {"context": [], "sources": [], "answer": ans}

    def summarize_document(self, state: RAGState) -> dict:
        doc_id = state.resolved_document_ids[0]
        all_docs = self.retriever.vector_store.get_all_documents()
        target_docs = [
            d for d in all_docs
            if d.metadata.get("document_id") == doc_id or d.metadata.get("source_filename") == doc_id
        ]
        chunks = [d.page_content for d in target_docs]

        if not chunks:
            ans = f"The document '{doc_id}' appears to be empty or contains no extractable text."
            RAGTracer.log_error(state.trace_id, "Summarization", f"Document {doc_id} contains 0 chunks", action="Return empty document explanation")
            return {"context": [], "sources": [], "answer": ans, "metrics": state.metrics}

        t_gen_start = time.perf_counter()
        summary = self.llm.generate_summary(doc_id, chunks)
        gen_time = round(time.perf_counter() - t_gen_start, 4)

        sources = make_sources_from_docs(target_docs[:4])
        metrics = dict(state.metrics)
        metrics["llm_generation"] = gen_time
        metrics["total_latency"] = round(sum(metrics.values()), 4)

        total_chars = sum(len(c) for c in chunks[:4])
        RAGTracer.log_generation_context(state.trace_id, len(chunks[:4]), sources, total_chars)
        RAGTracer.log_generation(state.trace_id, "qwen2.5:1.5b", "summarize", len(chunks[:4]), summary, sources)

        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=doc_id,
            sem_count=0,
            bm25_count=0,
            merged_count=len(chunks),
            reranked_count=len(chunks[:4]),
            model="qwen2.5:1.5b",
            answer_status="Document summary generated",
            metrics=metrics,
            status="SUCCESS",
        )

        return {"context": chunks[:4], "sources": sources, "answer": summary, "metrics": metrics}

    def compare_documents(self, state: RAGState) -> dict:
        doc_a = state.resolved_document_ids[0]
        doc_b = state.resolved_document_ids[1]
        res_a = self.retriever.retrieve_documents(state.question, document_ids=[doc_a])
        res_b = self.retriever.retrieve_documents(state.question, document_ids=[doc_b])

        docs_a, m_a, c_a, sem_a, bm_a, mrg_a = res_a
        docs_b, m_b, c_b, sem_b, bm_b, mrg_b = res_b

        context_a = [d.page_content for d in docs_a]
        context_b = [d.page_content for d in docs_b]

        if not context_a or not context_b:
            missing_doc = doc_a if not context_a else doc_b
            found_doc = doc_b if not context_a else doc_a
            ans = f"I found details for '{found_doc}', but couldn't find sufficient context for '{missing_doc}' to reliably compare."
            RAGTracer.log_error(state.trace_id, "Comparison", f"Missing context for {missing_doc}", action="Return partial comparison error")
            return {"context": [], "sources": [], "answer": ans, "metrics": state.metrics}

        t_gen_start = time.perf_counter()
        comp_ans = self.llm.generate_comparison(doc_a, context_a, doc_b, context_b, state.question)
        gen_time = round(time.perf_counter() - t_gen_start, 4)

        sources = make_sources_from_docs(docs_a + docs_b)
        metrics = dict(state.metrics)
        metrics["semantic_retrieval"] = round(m_a.get("semantic_retrieval", 0) + m_b.get("semantic_retrieval", 0), 4)
        metrics["bm25_retrieval"] = round(m_a.get("bm25_retrieval", 0) + m_b.get("bm25_retrieval", 0), 4)
        metrics["reranking"] = round(m_a.get("reranking", 0) + m_b.get("reranking", 0), 4)
        metrics["llm_generation"] = gen_time
        metrics["total_latency"] = round(sum(metrics.values()), 4)

        total_chars = sum(len(c) for c in context_a + context_b)
        RAGTracer.log_generation_context(state.trace_id, len(context_a + context_b), sources, total_chars)
        RAGTracer.log_generation(state.trace_id, "qwen2.5:1.5b", "compare", len(context_a + context_b), comp_ans, sources)

        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=f"{doc_a}, {doc_b}",
            sem_count=c_a["semantic_count"] + c_b["semantic_count"],
            bm25_count=c_a["bm25_count"] + c_b["bm25_count"],
            merged_count=c_a["merged_count"] + c_b["merged_count"],
            reranked_count=len(docs_a + docs_b),
            model="qwen2.5:1.5b",
            answer_status="Comparison generated",
            metrics=metrics,
            status="SUCCESS",
        )

        return {"context": context_a + context_b, "sources": sources, "answer": comp_ans, "metrics": metrics}

    def retrieve_single(self, state: RAGState) -> dict:
        search_query = state.sub_queries[0] if state.sub_queries else state.question
        doc_ids = state.resolved_document_ids if state.resolved_document_ids else None
        retrieved_docs, r_metrics, counts, sem_cands, bm_cands, mrg_cands = self.retriever.retrieve_documents(search_query, document_ids=doc_ids)

        context_chunks = [d.page_content for d in retrieved_docs]
        sources = make_sources_from_docs(retrieved_docs)

        metrics = dict(state.metrics)
        metrics.update(r_metrics)

        RAGTracer.log_semantic_retrieval(state.trace_id, search_query, sem_cands)
        RAGTracer.log_bm25_retrieval(state.trace_id, search_query, bm_cands)
        RAGTracer.log_retrieval_merge(state.trace_id, counts["semantic_count"], counts["bm25_count"], counts["merged_count"], counts["duplicates_removed"])
        RAGTracer.log_reranking(state.trace_id, counts["merged_count"], retrieved_docs)

        if not context_chunks:
            if doc_ids:
                ans = f"The requested document '{doc_ids[0]}' did not contain enough relevant information for this question."
            else:
                ans = "I couldn't find relevant information in the uploaded documents."
            RAGTracer.log_error(state.trace_id, "Retrieval", "Zero matching candidate chunks found", action="Return insufficient retrieval response")
            return {"context": [], "sources": [], "answer": ans, "metrics": metrics}

        max_score = r_metrics.get("max_rerank_score", 0.0)
        min_thresh = getattr(self.retriever, "min_rerank_score", -2.0)
        if max_score < min_thresh:
            ans = "I found some documents, but the evidence is not sufficiently relevant to answer your question confidently."
            RAGTracer.log_error(state.trace_id, "Reranking Quality", f"Max reranker score ({max_score:.4f}) below threshold ({min_thresh:.4f})", action="Return low relevance response")
            return {"context": [], "sources": [], "answer": ans, "metrics": metrics}

        state.metrics.update(metrics)
        return {"context": context_chunks, "sources": sources, "metrics": metrics}

    def retrieve_multi(self, state: RAGState) -> dict:
        sub_queries = state.sub_queries or [state.question]
        doc_ids = state.resolved_document_ids if state.resolved_document_ids else None
        merged_docs = []
        seen_chunks = set()
        accumulated_metrics = {"semantic_retrieval": 0.0, "bm25_retrieval": 0.0, "reranking": 0.0}

        successful_sqs = []
        missing_sqs = []
        hops_trace = []

        for sq in sub_queries:
            sub_res = self.retriever.retrieve_documents(sq, document_ids=doc_ids)
            sub_docs, m, c, sem_cands, bm_cands, mrg_cands = sub_res
            for k in accumulated_metrics:
                accumulated_metrics[k] += m.get(k, 0.0)
            hops_trace.append((sq, [d.page_content for d in sub_docs]))

            if sub_docs:
                successful_sqs.append(sq)
                for doc in sub_docs:
                    content_key = doc.page_content.strip()
                    if content_key not in seen_chunks:
                        seen_chunks.add(content_key)
                        merged_docs.append(doc)
            else:
                missing_sqs.append(sq)

        final_docs = merged_docs[:6]
        context_chunks = [d.page_content for d in final_docs]
        sources = make_sources_from_docs(final_docs)

        metrics = dict(state.metrics)
        for k in accumulated_metrics:
            metrics[k] = round(accumulated_metrics[k], 4)

        RAGTracer.log_multi_hop(state.trace_id, sub_queries, hops_trace)
        RAGTracer.log_reranking(state.trace_id, len(merged_docs), final_docs)

        if missing_sqs and successful_sqs:
            ans = f"I found information regarding '{successful_sqs[0]}', but couldn't find enough information regarding '{missing_sqs[0]}' to reliably answer."
            RAGTracer.log_error(state.trace_id, "Multi-Hop Retrieval", f"Partial hop failure: missing evidence for '{missing_sqs[0]}'", action="Return partial multi-hop evidence message")
            return {"context": context_chunks, "sources": sources, "answer": ans, "metrics": metrics}

        if not context_chunks:
            ans = "I couldn't find sufficient relevant information across the required sub-queries."
            RAGTracer.log_error(state.trace_id, "Multi-Hop Retrieval", "All sub-queries returned zero candidates", action="Return insufficient evidence message")
            return {"context": [], "sources": [], "answer": ans, "metrics": metrics}

        state.metrics.update(metrics)
        return {"context": context_chunks, "sources": sources, "metrics": metrics}

    def generate(self, state: RAGState) -> dict:
        if state.answer:
            return {"answer": state.answer}

        t_gen_start = time.perf_counter()
        try:
            raw_ans = self.llm.generate(state.question, state.context, chat_history=state.chat_history)
        except Exception as e:
            RAGTracer.log_error(state.trace_id, "LLM Generation", f"Model generation error: {e}", action="Return clean system error message")
            ans = "I'm sorry, but the language model service is currently unavailable or failed to respond. Please try again later."
            return {"answer": ans, "metrics": state.metrics}

        gen_time = round(time.perf_counter() - t_gen_start, 4)

        metrics = dict(state.metrics)
        metrics["llm_generation"] = gen_time
        metrics["total_latency"] = round(sum(metrics.values()), 4)

        total_chars = sum(len(c) for c in state.context)
        RAGTracer.log_generation_context(state.trace_id, len(state.context), state.sources, total_chars)
        RAGTracer.log_generation(state.trace_id, "qwen2.5:1.5b", state.operation, len(state.context), raw_ans, state.sources)

        doc_label = ", ".join(state.resolved_document_ids) if state.resolved_document_ids else ""
        RAGTracer.log_trace_summary(
            trace_id=state.trace_id,
            query=state.question,
            intent=state.intent,
            query_type=state.query_type,
            document=doc_label,
            sem_count=15,
            bm25_count=15,
            merged_count=len(state.context),
            reranked_count=len(state.context),
            model="qwen2.5:1.5b",
            answer_status="Generated answer successfully",
            metrics=metrics,
            status="SUCCESS",
        )

        return {"answer": raw_ans, "metrics": metrics}






