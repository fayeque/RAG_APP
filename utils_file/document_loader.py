from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_path: str) -> list:
    """Loads documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()