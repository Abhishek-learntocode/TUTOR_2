import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models.canonical import DocumentChunk


class VectorStore:
    """Stores vectors and performs semantic search using FAISS."""

    def __init__(
        self,
        embeddings,
        store_path: str = "data/vector_store",
    ):
        self.embeddings = embeddings
        self.store_path = store_path
        self.store = None

        self._load_if_exists()

    def _load_if_exists(self):
        """Load an existing FAISS index when available."""

        index_file = os.path.join(
            self.store_path,
            "index.faiss",
        )

        pkl_file = os.path.join(
            self.store_path,
            "index.pkl",
        )

        if not (
            os.path.exists(index_file)
            and os.path.exists(pkl_file)
        ):
            return

        try:
            loaded_store = FAISS.load_local(
                folder_path=self.store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Verify embedding dimension.
            dummy_dim = len(
                self.embeddings.embed_query("test")
            )

            if loaded_store.index.d == dummy_dim:
                self.store = loaded_store

                print(
                    f"[VectorStore] Loaded existing index "
                    f"with {len(self.get_all_documents())} documents."
                )
            else:
                print(
                    "[VectorStore Warning] "
                    f"Index dimension mismatch: "
                    f"loaded={loaded_store.index.d}, "
                    f"model={dummy_dim}. "
                    "Ignoring existing index."
                )

        except Exception as exc:
            print(
                f"[VectorStore Warning] "
                f"Failed to load existing index: {exc}"
            )

            self.store = None

    def _chunk_to_document(
        self,
        chunk: DocumentChunk,
    ) -> Document:
        """
        Convert DocumentChunk to LangChain Document.

        chunk_id and document_id are explicitly copied because
        they are fields on DocumentChunk, not necessarily inside
        chunk.metadata.
        """

        metadata = dict(chunk.metadata)

        metadata["chunk_id"] = chunk.chunk_id
        metadata["document_id"] = chunk.document_id

        return Document(
            page_content=chunk.content,
            metadata=metadata,
        )

    def add_documents(
        self,
        documents: List[Document],
    ) -> int:
        """Add LangChain Documents to FAISS."""

        if not documents:
            return 0

        if self.store is None:
            self.store = FAISS.from_documents(
                documents,
                self.embeddings,
            )
        else:
            self.store.add_documents(
                documents,
            )

        os.makedirs(
            self.store_path,
            exist_ok=True,
        )

        self.store.save_local(
            self.store_path,
        )

        return len(documents)

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> int:
        """Add DocumentChunks while preserving chunk identity."""

        if not chunks:
            return 0

        documents = [
            self._chunk_to_document(chunk)
            for chunk in chunks
        ]

        return self.add_documents(documents)

    def get_all_documents(self) -> List[Document]:
        """Return all documents stored in FAISS."""

        if (
            self.store
            and hasattr(self.store, "docstore")
            and hasattr(self.store.docstore, "_dict")
        ):
            return list(
                self.store.docstore._dict.values()
            )

        return []

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """Perform global semantic search."""

        if not self.store:
            return []

        return self.store.similarity_search(
            query,
            k=k,
        )

    def similarity_search_scoped(
        self,
        query: str,
        document_ids: list[str],
        k: int = 15,
    ) -> List[Document]:
        """Perform semantic search restricted to documents."""

        if not self.store:
            return []

        if not document_ids:
            return self.similarity_search(
                query,
                k=k,
            )

        document_set = set(
            str(doc_id)
            for doc_id in document_ids
        )

        # Retrieve a larger candidate pool before filtering.
        candidates = self.store.similarity_search(
            query,
            k=k * 2,
        )

        filtered = []

        for document in candidates:
            metadata = document.metadata or {}

            document_id = metadata.get(
                "document_id"
            )

            source_filename = metadata.get(
                "source_filename"
            )

            if (
                str(document_id) in document_set
                or str(source_filename) in document_set
            ):
                filtered.append(document)

        return filtered[:k]