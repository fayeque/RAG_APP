import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from utils_file.oracle_vs_ingestor import retrieve_from_oracle_vs
from utils_file.retrieval_chain_creator import create_retrieval_chain
from utils_file.embedding_creator import create_embeddings
from utils_file.llm_creator import create_llm
from db_connections.db_connection import get_oracle_connection
import uvicorn

load_dotenv()

app = FastAPI(title="RAG Oracle Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:8080"] for APEX
    allow_credentials=True,
    allow_methods=["*"],        # allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

@app.middleware("http")
async def add_custom_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"  # Allow private network requests
    return response

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():

        # --- Recreate pem and config files from environment variables ---
    pem_content = os.environ.get("OCI_KEY_CONTENT")
    if pem_content:
        Path("oci_api_key.pem").write_text(pem_content)
    
    # Build your config file from other environment variables
    oci_config = f"""[DEFAULT]
    user={os.environ.get('OCI_USER_OCID')}
    fingerprint={os.environ.get('OCI_FINGERPRINT')}
    key_file=oci_api_key.pem
    tenancy={os.environ.get('OCI_TENANCY_OCID')}
    region={os.environ.get('OCI_REGION')}
    """
    Path("oci_config").write_text(oci_config)
    # ---------------------------------------------------------------

    global vs, chain, connection
    print("Starting API...")

    embeddings = create_embeddings()
    connection = get_oracle_connection()

    # Only retrieve, don't ingest again
    vs = retrieve_from_oracle_vs(connection, embeddings)

    llm = create_llm()
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}"""
    chain = create_retrieval_chain(vs, llm, template)

    print("API is ready.")

@app.post("/ask")
def ask_question(request: QueryRequest):
    docs = vs.similarity_search(request.question, k=5)
    response = chain.invoke(request.question)

    return {
        #"question": request.question,
        #"contexts": [doc.page_content for doc in docs],
        "answer": response
    }

@app.on_event("shutdown")
def shutdown_event():
    if connection:
        connection.close()
        print("Connection closed.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
