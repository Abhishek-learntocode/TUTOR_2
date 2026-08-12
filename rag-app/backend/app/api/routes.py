import os
import shutil
import traceback

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

from app.config import settings
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, DocumentUploadResponse
from app.models.state import RAGState
from app.rag.document_loader import DocumentLoader
from app.rag.document_splitter import DocumentSplitter


router = APIRouter()

loader = DocumentLoader()
splitter = DocumentSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form("BOOK"),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    os.makedirs("data/documents", exist_ok=True)

    file_path = os.path.join(
        "data/documents",
        file.filename,
    )

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        canonical_doc = loader.load(
            file_path,
            source_filename=file.filename,
            doc_type=doc_type,
        )

        chunks = splitter.split(canonical_doc)

        count = request.app.state.vector_store.add_chunks(chunks)
        all_documents = request.app.state.vector_store.get_all_documents()

        request.app.state.bm25_retriever.rebuild(
            all_documents
        )

        print(
            f"[BM25] Rebuilt index with "
            f"{len(all_documents)} documents."
        )

        return DocumentUploadResponse(
            filename=file.filename,
            doc_type=canonical_doc.document_type,
            chunks_created=count,
            message=(
                f"Uploaded '{file.filename}' "
                f"as [{canonical_doc.document_type}] "
                f"({count} chunks)."
            ),
        )

    except Exception as exc:
        print("[Upload Error Traceback]")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: Request,
    body: QueryRequest,
):
    try:
        state = RAGState(
            question=body.question,
            chat_history=body.chat_history,
        )

        final_state = request.app.state.rag_graph.invoke(state)

        return QueryResponse(
            trace_id=final_state.trace_id,
            answer=final_state.answer,
            context=final_state.context,
            sources=final_state.sources,
            metrics=final_state.metrics,
        )

    except Exception as exc:
        print("[Query Error Traceback]")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        ) from exc