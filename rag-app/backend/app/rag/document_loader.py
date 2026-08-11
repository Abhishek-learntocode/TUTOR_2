from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentLoader:
    """Loads source documents into LangChain Document objects."""

    def load(self, file_path: str):
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path).load()
        return TextLoader(file_path, encoding="utf-8").load()
