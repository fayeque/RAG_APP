from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Chunks the loaded documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)