import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from db_connection import get_oracle_connection
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy

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

    vs = OracleVS(
        embedding_function=embeddings,
        client=connection,
        table_name="RAG_DOCUMENTS",
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )

    query = "What does ELCM module do in Oracle Flexcube?"
    results = vs.similarity_search(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print(doc.metadata)


    ##RETRIEVE AND SEND TO LLM FOR RESPONSE
    # Set up the template for the questions and context, and instantiate the database retriever object
    # template = """Answer the question based only on the following context:
    # {context} Question: {question} """
    # prompt = PromptTemplate.from_template(template)
    #
    # # Retrieval Step 2 - Create retriever without ingesting documents again.
    # vs = OracleVS(
    #     embedding_function=embeddings,
    #     client=connection,
    #     table_name="RAG_DOCUMENTS",
    #     distance_strategy=DistanceStrategy.DOT_PRODUCT
    # )
    #
    # retriever = vs.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={'k': 3}
    # )
    #
    # chain = (
    #         {"context": retriever, "question": RunnablePassthrough()}
    #         | prompt
    #         | llm
    #         | StrOutputParser()
    # )
    #
    # user_question = ("Tell us about Module 4 of AI Foundations Certification course.")
    #
    # response = chain.invoke(user_question)
    #
    # print("User question was ->", user_question)
    # print("LLM response is ->", response)

    print("finish")
