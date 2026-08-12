from typing import Literal

from pydantic import BaseModel, Field

from app.models.canonical import DocumentChunk
from app.models.responses import SourceCitation


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryAnalysis(BaseModel):
    operation: Literal[
        "normal_qa",
        "document_qa",
        "summarize",
        "compare",
        "conversational",
        "invalid_query",
        "missing_reference",
    ] = "normal_qa"

    intent: Literal[
        "question",
        "summarization",
        "document_lookup",
        "comparison",
    ] = "question"

    query_type: Literal[
        "single_hop",
        "multi_hop",
        "conversational",
    ] = "single_hop"

    sub_queries: list[str] = Field(default_factory=list)
    document_references: list[str] = Field(default_factory=list)
    resolved_query: str | None = None
    requires_document_resolution: bool = False

    resolved_document_ids: list[str] = Field(default_factory=list)
    ambiguous_candidates: list[str] = Field(default_factory=list)


class RAGState(BaseModel):
    """State passed through the LangGraph RAG workflow."""

    trace_id: str = ""
    question: str

    chat_history: list[Message] = Field(default_factory=list)

    operation: Literal[
        "normal_qa",
        "document_qa",
        "summarize",
        "compare",
        "conversational",
        "invalid_query",
        "missing_reference",
    ] = "normal_qa"

    intent: Literal[
        "question",
        "summarization",
        "document_lookup",
        "comparison",
    ] = "question"

    query_type: Literal[
        "single_hop",
        "multi_hop",
        "conversational",
    ] = "single_hop"

    sub_queries: list[str] = Field(default_factory=list)
    document_references: list[str] = Field(default_factory=list)

    requires_document_resolution: bool = False

    resolved_document_ids: list[str] = Field(default_factory=list)
    ambiguous_candidates: list[str] = Field(default_factory=list)

    # Keep structured retrieval results available for citations/debugging.
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)

    # Text actually passed to the generation model.
    context: list[str] = Field(default_factory=list)

    sources: list[SourceCitation] = Field(default_factory=list)

    metrics: dict[str, float] = Field(default_factory=dict)

    answer: str = ""