from app.models.state import RAGState


class RAGNodes:
    """Operations executed by the LangGraph workflow."""

    def __init__(self, retriever, llm, query_analyzer=None):
        self.retriever = retriever
        self.llm = llm
        self.query_analyzer = query_analyzer

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
        return {"context": final_context}

    def generate(self, state: RAGState) -> dict:
        return {"answer": self.llm.generate(state.question, state.context)}
