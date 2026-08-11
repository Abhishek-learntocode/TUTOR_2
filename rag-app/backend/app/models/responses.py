from pydantic import BaseModel


class QueryResponse(BaseModel):
    """Response model for query endpoints."""

    answer: str
    context: list[str]


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoints."""

    filename: str
    chunks_created: int
    message: str
