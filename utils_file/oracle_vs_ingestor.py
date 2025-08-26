from db_connections.db_connection import get_oracle_connection
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy

def ingest_documents_into_oracle_vs(connection, final_documents, embeddings, table_name: str = "RAG_DOCUMENTS"):
    """Ingests documents into Oracle Vector Store."""
    try:
        vs = OracleVS(
            embedding_function=embeddings,
            client=connection,
            table_name=table_name,
            distance_strategy=DistanceStrategy.DOT_PRODUCT
        )
        vs.add_documents(final_documents)
    except ValueError:
        # Table does not exist, create a new one
        vs = OracleVS.from_documents(
            final_documents,
            embeddings,
            client=connection,
            table_name=table_name,
            distance_strategy=DistanceStrategy.DOT_PRODUCT
        )
    connection.commit()
    return vs

def retrieve_from_oracle_vs(connection, embeddings, table_name: str = "RAG_DOCUMENTS"):
    """Retrieves from Oracle Vector Store."""
    vs = OracleVS(
        embedding_function=embeddings,
        client=connection,
        table_name=table_name,
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )
    return vs