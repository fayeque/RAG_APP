import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from db_connection import get_oracle_connection
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Set Hugging Face token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")


def load_documents(file_path: str) -> list:
    """Loads documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Chunks the loaded documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def create_embeddings(model_name: str = "./local_model", device: str = "cpu") -> HuggingFaceEmbeddings:
    """Creates Hugging Face embeddings."""
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})


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


def create_retrieval_chain(vs, llm, template: str):
    """Creates a retrieval chain."""
    prompt = PromptTemplate.from_template(template)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


def main():
    print("Ingesting...")

    # Load documents
    docs = load_documents('ELCM.pdf')

    # Chunk documents
    final_documents = chunk_documents(docs)

    # Create embeddings
    embeddings = create_embeddings()

    # Connect to Oracle database
    connection = get_oracle_connection()

    # Ingest documents into Oracle VS
    ingest_documents_into_oracle_vs(connection, final_documents, embeddings)

    # Retrieve from Oracle VS
    vs = retrieve_from_oracle_vs(connection, embeddings)

    # Create LLM
    llm = Ollama(model="gemma3:1b")

    # Create retrieval chain
    template = """Answer the question based only on the following context:
    {context} Question: {question} """
    chain = create_retrieval_chain(vs, llm, template)

    # Ask a question
    user_question = "What is sub facility?"
    response = chain.invoke(user_question)

    print("User question was ->", user_question)
    print("LLM response is ->", response)

    # Close the connection
    connection.close()

    print("Finish")


if __name__ == "__main__":
    main()
