from app.models.state import RAGState


class RAGNodes:
    """Operations executed by the LangGraph."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, state: RAGState) -> dict:
        return {"context": self.retriever.retrieve(state.question)}

    def generate(self, state: RAGState) -> dict:
        return {"answer": self.llm.generate(state.question, state.context)}
