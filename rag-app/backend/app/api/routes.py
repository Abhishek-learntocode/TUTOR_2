import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, DocumentUploadResponse
from app.models.state import RAGState
from app.rag.document_loader import DocumentLoader
from app.rag.document_splitter import DocumentSplitter
from app.config import settings

router = APIRouter()
loader = DocumentLoader()
splitter = DocumentSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    os.makedirs("data/documents", exist_ok=True)
    file_path = os.path.join("data/documents", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        docs = loader.load(file_path)
        chunks = splitter.split(docs)
        count = request.app.state.vector_store.add_documents(chunks)
        return DocumentUploadResponse(
            filename=file.filename,
            chunks_created=count,
            message=f"Successfully uploaded '{file.filename}' and indexed {count} chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest):
    try:
        state = RAGState(question=body.question)
        final_state = request.app.state.rag_graph.invoke(state)
        return QueryResponse(answer=final_state.answer, context=final_state.context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
