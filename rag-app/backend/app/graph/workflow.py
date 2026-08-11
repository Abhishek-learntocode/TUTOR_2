from langgraph.graph import StateGraph, START, END
from app.models.state import RAGState
from app.graph.nodes import RAGNodes


class RAGGraph:
    """Builds and compiles the RAG LangGraph."""

    def __init__(self, nodes: RAGNodes):
        self.nodes = nodes
        self.compiled_graph = None

    def build(self) -> StateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self.nodes.retrieve)
        graph.add_node("generate", self.nodes.generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
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
