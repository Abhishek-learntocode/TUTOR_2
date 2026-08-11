from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """Splits documents into retrievable chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents):
        return self.splitter.split_documents(documents) if documents else []
