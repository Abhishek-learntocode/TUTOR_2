from langsmith import get_current_run_tree
from app.models.state import RAGState


class RAGNodes:
    """Operations executed by the LangGraph workflow."""

    def __init__(self, retriever, llm, query_analyzer=None, simple_llm=None, complex_llm=None):
        self.retriever = retriever
        self.llm = llm
        self.query_analyzer = query_analyzer
        self.simple_llm = simple_llm or llm
        self.complex_llm = complex_llm or llm

    def analyze_query(self, state: RAGState) -> dict:
        if self.query_analyzer:
            analysis = self.query_analyzer.analyze(state.question)
            return {
                "query_type": analysis.query_type,
                "sub_queries": analysis.sub_queries,
            }
        return {"query_type": "single_hop", "sub_queries": [state.question]}

    def route_query(self, state: RAGState) -> str:
        return "retrieve_multi" if state.query_type == "multi_hop" else "retrieve_single"

    def retrieve_single(self, state: RAGState) -> dict:
        context_chunks = self.retriever.retrieve(state.question)
        print(f"[Query Routing Log] query_type: single_hop")
        print(f"[Query Routing Log] sub_queries: 1")
        print(f"[Query Routing Log] retrieval_operations: 1")
        print(f"[Query Routing Log] final_context_chunks: {len(context_chunks)}")
        try:
            rt = get_current_run_tree()
            if rt:
                rt.metadata["dispatch_mode"] = "single"
                rt.metadata["concurrent_operations"] = 1
        except Exception:
            pass
        return {"context": context_chunks}

    def retrieve_multi(self, state: RAGState) -> dict:
        sub_queries = state.sub_queries or [state.question]
        retrieved_chunks = []
        seen_chunks = set()

        for sq in sub_queries:
            sub_results = self.retriever.retrieve(sq)
            for chunk in sub_results:
                content_key = chunk.strip()
                if content_key not in seen_chunks:
                    seen_chunks.add(content_key)
                    retrieved_chunks.append(chunk)

        final_context = retrieved_chunks[:6]
        print(f"[Query Routing Log] query_type: multi_hop")
        print(f"[Query Routing Log] sub_queries: {len(sub_queries)} -> {sub_queries}")
        print(f"[Query Routing Log] retrieval_operations: {len(sub_queries)}")
        print(f"[Query Routing Log] final_context_chunks: {len(final_context)}")
        try:
            rt = get_current_run_tree()
            if rt:
                rt.metadata["dispatch_mode"] = "sequential"
                rt.metadata["concurrent_operations"] = len(sub_queries)
        except Exception:
            pass
        return {"context": final_context}

    def generate(self, state: RAGState) -> dict:
        q_lower = state.question.lower()
        is_complex = (
            state.query_type == "multi_hop"
            or "compare" in q_lower
            or "relates to" in q_lower
            or "versus" in q_lower
            or " vs " in q_lower
        )

        routing_decision = "complex" if is_complex else "simple"
        active_llm = self.complex_llm if is_complex else self.simple_llm

        print(f"[Hybrid Router Log] Query: '{state.question[:50]}...' -> Routing Decision: {routing_decision} -> Provider: {getattr(active_llm, 'provider_name', 'default')} / {getattr(active_llm, 'model_name', 'default')}")

        answer = active_llm.generate(state.question, state.context)

        try:
            rt = get_current_run_tree()
            if rt:
                rt.metadata["routing_decision"] = routing_decision
                rt.metadata["answer_provider"] = getattr(active_llm, "provider_name", "unknown")
                rt.metadata["answer_model"] = getattr(active_llm, "model_name", "unknown")
        except Exception:
            pass

        return {"answer": answer}

