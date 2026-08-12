import os
import uuid
import logging

os.makedirs("logs", exist_ok=True)
LOG_FILE_PATH = os.path.join("logs", "rag_traces.log")

# Setup logger with stdout and file handler
logger = logging.getLogger("RAGTracer")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Console Handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    f_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)


def emit(msg: str):
    print(msg)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")


class RAGTracer:
    """Human-readable observability and tracing system for RAG queries and ingestion."""

    @staticmethod
    def generate_trace_id() -> str:
        return uuid.uuid4().hex[:6]

    @staticmethod
    def log_query(trace_id: str, query: str):
        msg = f"\nTRACE [{trace_id}]\n\nUSER QUERY\n\"{query}\"\n"
        emit(msg)

    @staticmethod
    def log_query_analysis(
        trace_id: str,
        query: str,
        intent: str,
        query_type: str,
        doc_refs: list[str],
        sub_queries: list[str],
        operation: str = "normal_qa",
    ):
        ref_str = ", ".join([f'"{r}"' for r in doc_refs]) if doc_refs else "None"
        sub_str = str(sub_queries) if sub_queries else "[]"
        msg = (
            f"[QUERY ANALYSIS] trace_id={trace_id}\n"
            f"Original query:\n\"{query}\"\n\n"
            f"Operation:\n{operation}\n\n"
            f"Intent:\n{intent}\n\n"
            f"Query type:\n{query_type}\n\n"
            f"Document reference:\n{ref_str}\n\n"
            f"Sub-queries:\n{sub_str}\n"
        )
        emit(msg)

    @staticmethod
    def log_document_resolution(
        trace_id: str,
        reference: str,
        method: str,
        matched_doc: str,
        doc_id: str,
        candidates_count: int,
        candidates_list: list[str],
        status: str = "SUCCESS",
        ambiguous_list: list[str] = None,
    ):
        if status == "AMBIGUOUS" and ambiguous_list:
            cands_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(ambiguous_list)])
            msg = (
                f"[DOCUMENT RESOLUTION] trace_id={trace_id}\n\n"
                f"Status:\nAMBIGUOUS\n\n"
                f"Candidates:\n{cands_str}\n\n"
                f"Action:\nRequest clarification\n"
            )
        else:
            msg = (
                f"[DOCUMENT RESOLUTION] trace_id={trace_id}\n\n"
                f"Reference:\n\"{reference}\"\n\n"
                f"Resolution method:\n{method}\n\n"
                f"Matched document:\n{matched_doc}\n\n"
                f"document_id:\n{doc_id}\n\n"
                f"Candidates considered:\n{candidates_count}\n\n"
                f"Selected:\n{matched_doc}\n"
            )
        emit(msg)

    @staticmethod
    def log_semantic_retrieval(trace_id: str, query: str, candidates: list):
        top_str = "\n".join([f"{i+1}. {c.metadata.get('chunk_id', 'chunk_'+str(i))} ({c.metadata.get('source_filename', 'doc')})" for i, c in enumerate(candidates[:5])])
        msg = (
            f"[SEMANTIC RETRIEVAL] trace_id={trace_id}\n\n"
            f"Query:\n\"{query}\"\n\n"
            f"Candidates:\n{len(candidates)}\n\n"
            f"Top candidates:\n{top_str}\n"
        )
        emit(msg)

    @staticmethod
    def log_semantic_performance(
        trace_id: str,
        query: str,
        total: float,
        embedding=None,
        vector_search=None,
        postprocessing=None,
        cache_hit=None,
    ):
        def fmt(value):
            return "not instrumented" if value is None else f"{float(value):.4f}s"

        cache_value = "unknown" if cache_hit is None else str(bool(cache_hit)).lower()
        msg = (
            f"[SEMANTIC PERFORMANCE] trace_id={trace_id}\n\n"
            f"Query:\n\"{query}\"\n\n"
            f"Total:\n{fmt(total)}\n\n"
            f"Embedding:\n{fmt(embedding)}\n\n"
            f"Vector search:\n{fmt(vector_search)}\n\n"
            f"Post-processing:\n{fmt(postprocessing)}\n\n"
            f"Embedding cache hit:\n{cache_value}\n"
        )
        emit(msg)

    @staticmethod
    def log_bm25_retrieval(trace_id: str, query: str, candidates: list):
        top_str = "\n".join([f"{i+1}. {c.metadata.get('chunk_id', 'chunk_'+str(i))} ({c.metadata.get('source_filename', 'doc')})" for i, c in enumerate(candidates[:5])])
        msg = (
            f"[BM25 RETRIEVAL] trace_id={trace_id}\n\n"
            f"Candidates:\n{len(candidates)}\n\n"
            f"Top candidates:\n{top_str}\n"
        )
        emit(msg)

    @staticmethod
    def log_retrieval_merge(trace_id: str, sem_count: int, bm25_count: int, unique_count: int, dup_count: int):
        msg = (
            f"[RETRIEVAL MERGE] trace_id={trace_id}\n\n"
            f"Semantic candidates:\n{sem_count}\n\n"
            f"BM25 candidates:\n{bm25_count}\n\n"
            f"Unique candidates:\n{unique_count}\n\n"
            f"Duplicates removed:\n{dup_count}\n"
        )
        emit(msg)

    @staticmethod
    def log_reranking(trace_id: str, input_count: int, reranked_docs: list):
        results_str = "\n".join([
            f"{i+1}. {d.metadata.get('chunk_id', 'chunk_'+str(i))} ({d.metadata.get('source_filename', 'doc')})"
            for i, d in enumerate(reranked_docs)
        ])
        msg = (
            f"[RERANKING] trace_id={trace_id}\n\n"
            f"Candidates:\n{input_count}\n\n"
            f"Top results:\n{results_str}\n"
        )
        emit(msg)

    @staticmethod
    def log_multi_hop(trace_id: str, sub_queries: list[str], hops: list):
        def fmt(value):
            return "n/a" if value is None else f"{float(value):.4f}s"

        sub_str = "\n".join([f"{i+1}. \"{sq}\"" for i, sq in enumerate(sub_queries)])
        msg = (
            f"[QUERY ROUTING] trace_id={trace_id}\n\n"
            f"Query type:\nMULTI_HOP\n\n"
            f"Sub-queries:\n{sub_str}\n\n"
            f"Retrieval operations:\n{len(sub_queries)}\n"
        )
        emit(msg)

        for idx, item in enumerate(hops, 1):
            if len(item) == 3:
                sq, chunks, h_metrics = item
                c_hit = "unknown" if h_metrics.get("embedding_cache_hit") is None else str(bool(h_metrics.get("embedding_cache_hit"))).lower()
                hop_msg = (
                    f"[HOP {idx}] trace_id={trace_id}\n"
                    f"Query:\n\"{sq}\"\n\n"
                    f"Retrieved:\n{len(chunks)} chunks\n\n"
                    f"Hop Latency Breakdown:\n"
                    f"  Semantic: {fmt(h_metrics.get('semantic_retrieval'))}\n"
                    f"    Embedding: {fmt(h_metrics.get('embedding'))}\n"
                    f"    Vector search: {fmt(h_metrics.get('vector_search'))}\n"
                    f"    Post-processing: {fmt(h_metrics.get('semantic_postprocessing'))}\n"
                    f"    Cache hit: {c_hit}\n"
                    f"  BM25: {fmt(h_metrics.get('bm25_retrieval'))}\n"
                    f"  Merge: {fmt(h_metrics.get('merge'))}\n"
                    f"  Reranking: {fmt(h_metrics.get('reranking'))}\n"
                )
            else:
                sq, chunks = item[0], item[1]
                hop_msg = (
                    f"[HOP {idx}] trace_id={trace_id}\n"
                    f"Query:\n\"{sq}\"\n\n"
                    f"Retrieved:\n{len(chunks)} chunks\n"
                )
            emit(hop_msg)

    @staticmethod
    def log_generation_context(trace_id: str, chunks, sources: list, total_chars: int, truncated: bool = False):
        chunks_count = len(chunks) if isinstance(chunks, list) else int(chunks)
        sources_str = "\n".join([
            f"{i+1}. {s.source_filename} — Page {s.page_number or 'N/A'}"
            for i, s in enumerate(sources)
        ]) if sources else "None"
        msg = (
            f"[GENERATION CONTEXT] trace_id={trace_id}\n\n"
            f"Chunks provided:\n{chunks_count}\n\n"
            f"Sources:\n{sources_str}\n\n"
            f"Context characters:\n{total_chars}\n\n"
            f"Context truncated:\n{str(truncated).lower()}\n"
        )
        emit(msg)

    @staticmethod
    def log_generation(trace_id: str, model: str, prompt_type: str, chunks_count: int, answer: str, sources: list):
        sources_used = ", ".join([s.chunk_id for s in sources]) if sources else "None"
        msg = (
            f"[GENERATION] trace_id={trace_id}\n\n"
            f"Model:\n{model}\n\n"
            f"Prompt type:\n{prompt_type}\n\n"
            f"Context chunks:\n{chunks_count}\n\n"
            f"Answer generated:\n\"{answer}\"\n\n"
            f"Sources used:\n{sources_used}\n"
        )
        emit(msg)

    @staticmethod
    def log_trace_summary(
        trace_id: str,
        query: str,
        intent: str,
        query_type: str,
        document: str,
        sem_count: int,
        bm25_count: int,
        merged_count: int,
        reranked_count: int,
        model: str,
        answer_status: str,
        metrics: dict[str, float],
        status: str = "SUCCESS",
        full_answer: str = None,
    ):
        m = metrics or {}
        q_analysis = m.get("query_analysis", 0.0)
        doc_res = m.get("document_resolution", 0.0)
        sem_ret = m.get("semantic_retrieval", 0.0)
        bm25_ret = m.get("bm25_retrieval", 0.0)
        rerank = m.get("reranking", 0.0)
        gen = m.get("llm_generation", 0.0)
        total = m.get("total_latency", sum([q_analysis, doc_res, sem_ret, bm25_ret, rerank, gen]))

        embedding = m.get("embedding")
        vector_search = m.get("vector_search")
        semantic_postprocessing = m.get("semantic_postprocessing")
        merge = m.get("merge")
        reranker_inference = m.get("reranker_inference")

        def fmt(value):
            return "n/a" if value is None else f"{float(value):.4f}s"

        ans_display = full_answer if full_answer else answer_status

        msg = (
            f"==================================================\n"
            f"TRACE SUMMARY [{trace_id}]\n"
            f"==================================================\n\n"
            f"User Query:\n{query}\n\n"
            f"Intent:\n{intent}\n\n"
            f"Query type:\n{query_type}\n\n"
            f"Document:\n{document or 'None'}\n\n"
            f"Retrieval:\n"
            f"  Semantic: {sem_count}\n"
            f"  BM25: {bm25_count}\n"
            f"  Merged: {merged_count}\n"
            f"  Reranked: {reranked_count}\n\n"
            f"LLM:\n{model}\n\n"
            f"Generated Answer:\n{ans_display}\n\n"
            f"Latency:\n"
            f"  Query analysis: {q_analysis:.4f}s\n"
            f"  Document resolution: {doc_res:.4f}s\n"
            f"  Semantic retrieval: {sem_ret:.4f}s\n"
            f"    Embedding: {fmt(embedding)}\n"
            f"    Vector search: {fmt(vector_search)}\n"
            f"    Post-processing: {fmt(semantic_postprocessing)}\n"
            f"  BM25 retrieval: {bm25_ret:.4f}s\n"
            f"  Merge: {fmt(merge)}\n"
            f"  Reranking: {rerank:.4f}s\n"
            f"    Reranker inference: {fmt(reranker_inference)}\n"
            f"  Generation: {gen:.4f}s\n"
            f"  Total: {total:.4f}s\n\n"
            f"Status:\n{status}\n"
            f"==================================================\n"
        )
        emit(msg)


    @staticmethod
    def log_ingestion(
        trace_id: str,
        filename: str,
        doc_type: str,
        parser: str,
        pages_count: int,
        chunks_count: int,
        vector_count: int,
        bm25_count: int,
        status: str = "SUCCESS",
        warnings: list = None,
    ):
        warn_str = f"\nWarnings:\n{warnings}" if warnings else ""
        msg = (
            f"[INGESTION] trace_id={trace_id}\n\n"
            f"File:\n{filename}\n\n"
            f"Type:\n{doc_type}\n\n"
            f"Parser:\n{parser}\n\n"
            f"Pages:\n{pages_count}\n\n"
            f"Chunks created:\n{chunks_count}\n\n"
            f"Embeddings generated:\n{chunks_count}\n\n"
            f"Vector records:\n{vector_count}\n\n"
            f"BM25 records:\n{bm25_count}\n\n"
            f"Status:\n{status}{warn_str}\n"
        )
        emit(msg)

    @staticmethod
    def log_error(trace_id: str, stage: str, problem: str, candidates: list = None, action: str = None):
        cands_str = f"\nCandidates:\n" + "\n".join(candidates) if candidates else ""
        action_str = f"\nAction:\n{action}" if action else ""
        msg = (
            f"[ERROR] trace_id={trace_id}\n\n"
            f"Stage:\n{stage}\n\n"
            f"Problem:\n{problem}{cands_str}{action_str}\n"
        )
        emit(msg)