from typing import Literal
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    query_type: Literal["single_hop", "multi_hop"] = "single_hop"
    sub_queries: list[str] = Field(default_factory=list)


class RAGState(BaseModel):
    """RAG State object passed through LangGraph workflow."""

    question: str
    query_type: Literal["single_hop", "multi_hop"] = "single_hop"
    sub_queries: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    answer: str = ""
