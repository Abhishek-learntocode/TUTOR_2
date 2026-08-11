from langgraph.graph import StateGraph, START, END
from app.models.state import RAGState
from app.graph.nodes import RAGNodes


class RAGGraph:
    """Builds and compiles the RAG LangGraph with single-hop and multi-hop conditional routing."""

    def __init__(self, nodes: RAGNodes):
        self.nodes = nodes
        self.compiled_graph = None

    def build(self) -> StateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("analyze_query", self.nodes.analyze_query)
        graph.add_node("retrieve_single", self.nodes.retrieve_single)
        graph.add_node("retrieve_multi", self.nodes.retrieve_multi)
        graph.add_node("generate", self.nodes.generate)

        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self.nodes.route_query,
            {
                "retrieve_single": "retrieve_single",
                "retrieve_multi": "retrieve_multi",
            },
        )
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
