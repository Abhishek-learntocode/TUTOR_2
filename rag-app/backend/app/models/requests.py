from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for document query."""

    question: str = Field(..., min_length=1)
