import os
import shutil
import traceback
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
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
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form("BOOK"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    os.makedirs("data/documents", exist_ok=True)
    file_path = os.path.join("data/documents", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        canonical_doc = loader.load(file_path, source_filename=file.filename, doc_type=doc_type)
        chunks = splitter.split(canonical_doc)
        count = request.app.state.vector_store.add_chunks(chunks)

        return DocumentUploadResponse(
            filename=file.filename,
            doc_type=canonical_doc.document_type,
            chunks_created=count,
            message=f"Uploaded '{file.filename}' as [{canonical_doc.document_type}] ({count} chunks).",
        )
    except Exception as e:
        print("[Upload Error Traceback]:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


import time
import logging

logger = logging.getLogger("rag_tracer")


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest):
    t0 = time.time()
    try:
        state = RAGState(question=body.question)
        final_state = request.app.state.rag_graph.invoke(state)
        elapsed = time.time() - t0
        logger.info(
            f"TRACE SUMMARY | QUERY: '{body.question}' | TYPE: {final_state.query_type} | "
            f"SUB_QUERIES: {final_state.sub_queries} | CONTEXT_CHUNKS: {len(final_state.context)} | "
            f"ANSWER_LEN: {len(final_state.answer)} | TOTAL_LATENCY: {elapsed:.4f}s"
        )
        return QueryResponse(answer=final_state.answer, context=final_state.context)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"TRACE ERROR | QUERY: '{body.question}' | ERROR: {e} | TOTAL_LATENCY: {elapsed:.4f}s")
        print("[Query Error Traceback]:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

