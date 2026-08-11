import re
import uuid
from app.models.canonical import CanonicalDocument, DocumentChunk

QUESTION_RE = re.compile(r"(?i)(?:^|\n)\s*(?:Q\.?\s*(\d+[a-z]?)|Question\s+(\d+[a-z]?)|(\d+)\.)\s*")
MARKS_RE = re.compile(r"(?i)\[\s*(\d+)\s*Marks?\s*\]|\(\s*(\d+)\s*Marks?\s*\)")


class DocumentSplitter:
    """Splits CanonicalDocument into type-aware DocumentChunks with document header metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, doc: CanonicalDocument) -> list[DocumentChunk]:
        if doc.document_type == "exam_paper":
            return self._split_exam(doc)
        return self._split_book(doc)

    def _format_chunk_content(self, doc: CanonicalDocument, body: str) -> str:
        header = f"[Document: {doc.source_filename} | Type: {doc.document_type}]"
        return f"{header}\n\n{body}"

    def _split_book(self, doc: CanonicalDocument) -> list[DocumentChunk]:
        chunks = []
        blocks = [b.strip() for b in doc.content.split("\n\n") if b.strip()]
        curr_buf, curr_size, curr_page, curr_sec = [], 0, 1, "General"

        for b in blocks:
            page_match = re.search(r"<!-- Page (\d+) -->", b)
            if page_match:
                curr_page = int(page_match.group(1))
                b = re.sub(r"<!-- Page \d+ -->", "", b).strip()
                if not b:
                    continue

            if b.startswith("#"):
                curr_sec = b.lstrip("#").strip().split("\n")[0]

            if curr_buf and (curr_size + len(b) > self.chunk_size):
                body_text = "\n\n".join(curr_buf)
                full_text = self._format_chunk_content(doc, body_text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        content=full_text,
                        metadata={
                            "document_id": doc.document_id,
                            "document_type": doc.document_type,
                            "source_filename": doc.source_filename,
                            "page_number": curr_page,
                            "section": curr_sec,
                        },
                    )
                )
                curr_buf, curr_size = [], 0

            curr_buf.append(b)
            curr_size += len(b)

        if curr_buf:
            body_text = "\n\n".join(curr_buf)
            full_text = self._format_chunk_content(doc, body_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    content=full_text,
                    metadata={
                        "document_id": doc.document_id,
                        "document_type": doc.document_type,
                        "source_filename": doc.source_filename,
                        "page_number": curr_page,
                        "section": curr_sec,
                    },
                )
            )
        return chunks

    def _split_exam(self, doc: CanonicalDocument) -> list[DocumentChunk]:
        chunks = []
        matches = list(QUESTION_RE.finditer(doc.content))

        if not matches:
            blocks = [b.strip() for b in doc.content.split("\n\n") if b.strip()]
            for b in blocks:
                full_text = self._format_chunk_content(doc, b)
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        content=full_text,
                        metadata={
                            "document_id": doc.document_id,
                            "document_type": doc.document_type,
                            "source_filename": doc.source_filename,
                            "question_number": "Unknown",
                            "marks": None,
                        },
                    )
                )
            return chunks

        for i in range(len(matches)):
            start_pos = matches[i].start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(doc.content)
            q_text = re.sub(r"<!-- Page \d+ -->", "", doc.content[start_pos:end_pos]).strip()
            if not q_text:
                continue

            q_num = next((g for g in matches[i].groups() if g is not None), str(i + 1))
            marks_match = MARKS_RE.search(q_text)
            marks_val = int(marks_match.group(1) or marks_match.group(2)) if marks_match else None
            full_text = self._format_chunk_content(doc, q_text)

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    content=full_text,
                    metadata={
                        "document_id": doc.document_id,
                        "document_type": doc.document_type,
                        "source_filename": doc.source_filename,
                        "question_number": f"Q{q_num}",
                        "marks": marks_val,
                    },
                )
            )
        return chunks
