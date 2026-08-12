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

from app.utils.tracer import RAGTracer

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
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="Invalid request: No filename provided.")

    allowed_exts = [".pdf", ".txt", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=422, detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_exts}")

    trace_id = RAGTracer.generate_trace_id()
    os.makedirs("data/documents", exist_ok=True)
    file_path = os.path.join("data/documents", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        canonical_doc = loader.load(file_path, source_filename=file.filename, doc_type=doc_type)
        chunks = splitter.split(canonical_doc)
        if not chunks:
            raise ValueError(f"Document '{file.filename}' contains no extractable text or chunks.")

        count = request.app.state.vector_store.add_chunks(chunks)

        if hasattr(request.app.state, "bm25_retriever") and request.app.state.bm25_retriever:
            request.app.state.bm25_retriever.rebuild(request.app.state.vector_store.get_all_documents())

        all_doc_count = len(request.app.state.vector_store.get_all_documents())
        pages_cnt = len(canonical_doc.raw_pages) if hasattr(canonical_doc, "raw_pages") else 1
        parser_name = "PyMuPDF / DocumentLoader" if file.filename.endswith(".pdf") else "Text / DocumentLoader"

        RAGTracer.log_ingestion(
            trace_id=trace_id,
            filename=file.filename,
            doc_type=canonical_doc.document_type,
            parser=parser_name,
            pages_count=pages_cnt,
            chunks_count=count,
            vector_count=count,
            bm25_count=all_doc_count,
            status="SUCCESS",
        )

        return DocumentUploadResponse(
            filename=file.filename,
            doc_type=canonical_doc.document_type,
            chunks_created=count,
            message=f"Uploaded '{file.filename}' as [{canonical_doc.document_type}] ({count} chunks).",
        )
    except Exception as e:
        RAGTracer.log_error(trace_id=trace_id, stage="INGESTION", problem=str(e), action="Failed document upload parsing")
        print("[Upload Error Traceback]:")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Document parsing error for '{file.filename}': {str(e)}")


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Invalid request: Query cannot be empty or whitespace only.")

    trace_id = RAGTracer.generate_trace_id()
    RAGTracer.log_query(trace_id, body.question)
    try:
        state = RAGState(trace_id=trace_id, question=body.question, chat_history=body.chat_history)
        final_state = request.app.state.rag_graph.invoke(state)
        return QueryResponse(
            trace_id=final_state.trace_id or trace_id,
            answer=final_state.answer,
            context=final_state.context,
            sources=final_state.sources,
            metrics=final_state.metrics,
        )
    except Exception as e:
        RAGTracer.log_error(trace_id=trace_id, stage="QUERY_EXECUTION", problem=str(e), action="Failed query pipeline execution")
        print("[Query Error Traceback]:")
        traceback.print_exc()
        err_msg = str(e).lower()
        if "connect" in err_msg or "connection" in err_msg or "timeout" in err_msg:
            raise HTTPException(status_code=503, detail="Language model service is unavailable or timed out.")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request.")




