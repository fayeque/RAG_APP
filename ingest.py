from dotenv import load_dotenv
from utils_file.document_loader import load_documents
from utils_file.document_chunker import chunk_documents
from utils_file.embedding_creator import create_embeddings
from utils_file.oracle_vs_ingestor import ingest_documents_into_oracle_vs
from db_connections.db_connection import get_oracle_connection

load_dotenv()

def run_ingestion():
    print("Ingesting documents...")

    # Load and chunk
    docs = load_documents("pdf_documents/ELCM.pdf")
    final_documents = chunk_documents(docs)

    # Embeddings
    embeddings = create_embeddings()

    # DB connection
    connection = get_oracle_connection()

    # Ingest
    ingest_documents_into_oracle_vs(connection, final_documents, embeddings)

    connection.close()
    print("Ingestion completed.")

if __name__ == "__main__":
    run_ingestion()
