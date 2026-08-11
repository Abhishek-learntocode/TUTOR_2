from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer: str
    context: list[str]


class DocumentUploadResponse(BaseModel):
    filename: str
    doc_type: str
    chunks_created: int
    message: str
