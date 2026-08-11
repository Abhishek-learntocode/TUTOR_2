import os
import fitz  # PyMuPDF
from app.models.canonical import CanonicalDocument


class DocumentLoader:
    """Loads PDF and text files into CanonicalDocument format."""

    def load(self, file_path: str, source_filename: str, doc_type: str = "book") -> CanonicalDocument:
        ext = os.path.splitext(file_path)[1].lower()
        pages = []

        if ext == ".pdf":
            pdf = fitz.open(file_path)
            for page_num in range(len(pdf)):
                text = pdf[page_num].get_text("text").strip()
                if text:
                    pages.append(f"<!-- Page {page_num + 1} -->\n{text}")
            pdf.close()
            full_text = "\n\n".join(pages)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read().strip()

        clean_type = "exam_paper" if "EXAM" in str(doc_type).upper() else "book"

        return CanonicalDocument(
            document_id=os.path.basename(file_path),
            document_type=clean_type,
            source_filename=source_filename,
            content=full_text,
        )
