import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sympy.codegen import Print
from db_connection import get_oracle_connection
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

if __name__ == "__main__":
    print("Ingesting...")
    ##loading
    loader = PyPDFLoader('ELCM.pdf')
    docs = loader.load()

    ##Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs)
    # print(final_documents[10])
    # print("--------------------")
    # print(final_documents[11])

    ##Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="./local_model",  # Path to your downloaded model directory
        model_kwargs={"device": "cpu"}  # "cuda" for GPU
    )



    ##connect with oracle vector 23ai db

    connection = get_oracle_connection()
    ##uncomment it when you try to store embed
    # knowledge_base = OracleVS.from_documents(
    #     final_documents,
    #     embeddings,
    #     client=connection,
    #     table_name="RAG_DOCUMENTS",
    #     distance_strategy=DistanceStrategy.DOT_PRODUCT
    # )
    #
    # connection.commit()
    # connection.close()


    ## RETRIEVAL

    # vs = OracleVS(
    #     embedding_function=embeddings,
    #     client=connection,
    #     table_name="RAG_DOCUMENTS",
    #     distance_strategy=DistanceStrategy.DOT_PRODUCT
    # )
    #
    # query = "What is revolving facility?"
    # results = vs.similarity_search(query, k=5)
    #
    # for i, doc in enumerate(results, start=1):
    #     print(f"--- Result {i} ---")
    #     print(doc.page_content)
    #     print(doc.metadata)


    ##RETRIEVE AND SEND TO LLM FOR RESPONSE
    # Set up the template for the questions and context, and instantiate the database retriever object
    # Ollama LLaMA3
    llm = Ollama(model="gemma3:1b")
    template = """Answer the question based only on the following context:
    {context} Question: {question} """
    prompt = PromptTemplate.from_template(template)

    # Retrieval Step 2 - Create retriever without ingesting documents again.
    vs = OracleVS(
        embedding_function=embeddings,
        client=connection,
        table_name="RAG_DOCUMENTS",
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )

    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5}
    )


    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    user_question = ("What is revolving facility?")

    response = chain.invoke(user_question)

    print("User question was ->", user_question)
    print("LLM response is ->", response)

    print("finish")
