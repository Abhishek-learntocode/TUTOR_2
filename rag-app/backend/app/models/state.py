from pydantic import BaseModel, Field


class RAGState(BaseModel):
    """RAG State object passed through LangGraph workflow."""

    question: str
    context: list[str] = Field(default_factory=list)
    answer: str = ""
