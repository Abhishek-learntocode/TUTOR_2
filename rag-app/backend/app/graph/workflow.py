from langgraph.graph import StateGraph, START, END
from app.models.state import RAGState
from app.graph.nodes import RAGNodes


class RAGGraph:
    """Builds and compiles the RAG LangGraph with single-hop, multi-hop, and error handling conditional routing."""

    def __init__(self, nodes: RAGNodes):
        self.nodes = nodes
        self.compiled_graph = None

    def build(self) -> StateGraph:
        graph = StateGraph(RAGState)

        graph.add_node("analyze_query", self.nodes.analyze_query)
        graph.add_node("invalid_query_prompt", self.nodes.invalid_query_prompt)
        graph.add_node("missing_history_reference_prompt", self.nodes.missing_history_reference_prompt)
        graph.add_node("direct_answer", self.nodes.direct_answer)
        graph.add_node("ambiguity_clarification", self.nodes.ambiguity_clarification)
        graph.add_node("doc_not_found_prompt", self.nodes.doc_not_found_prompt)
        graph.add_node("single_doc_compare_prompt", self.nodes.single_doc_compare_prompt)
        graph.add_node("summarize_document", self.nodes.summarize_document)
        graph.add_node("compare_documents", self.nodes.compare_documents)
        graph.add_node("retrieve_single", self.nodes.retrieve_single)
        graph.add_node("retrieve_multi", self.nodes.retrieve_multi)
        graph.add_node("generate", self.nodes.generate)

        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self.nodes.route_operation,
            {
                "invalid_query_prompt": "invalid_query_prompt",
                "missing_history_reference_prompt": "missing_history_reference_prompt",
                "direct_answer": "direct_answer",
                "ambiguity_clarification": "ambiguity_clarification",
                "doc_not_found_prompt": "doc_not_found_prompt",
                "single_doc_compare_prompt": "single_doc_compare_prompt",
                "summarize_document": "summarize_document",
                "compare_documents": "compare_documents",
                "retrieve_single": "retrieve_single",
                "retrieve_multi": "retrieve_multi",
            },
        )
        graph.add_edge("invalid_query_prompt", END)
        graph.add_edge("missing_history_reference_prompt", END)
        graph.add_edge("direct_answer", END)
        graph.add_edge("ambiguity_clarification", END)
        graph.add_edge("doc_not_found_prompt", END)
        graph.add_edge("single_doc_compare_prompt", END)
        graph.add_edge("summarize_document", END)
        graph.add_edge("compare_documents", END)
        graph.add_edge("retrieve_single", "generate")
        graph.add_edge("retrieve_multi", "generate")
        graph.add_edge("generate", END)
        return graph

    def compile(self):
        self.compiled_graph = self.build().compile()
        return self.compiled_graph

    def invoke(self, state: RAGState) -> RAGState:
        if self.compiled_graph is None:
            self.compile()
        res = self.compiled_graph.invoke(state.model_dump())
        return RAGState(**res)
