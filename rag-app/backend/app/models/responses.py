from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    document_id: str
    source_filename: str
    page_number: int | None = None
    section: str | None = None
    chunk_id: str


class QueryResponse(BaseModel):
    trace_id: str = ""
    answer: str
    context: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    filename: str
    doc_type: str
    chunks_created: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: str | None = None
