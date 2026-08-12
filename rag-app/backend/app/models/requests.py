from pydantic import BaseModel, Field
from app.models.state import Message


class QueryRequest(BaseModel):
    """Request model for document query."""

    question: str = Field(..., min_length=1)
    chat_history: list[Message] = Field(default_factory=list)

