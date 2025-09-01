import os
from langchain_community.embeddings import OCIGenAIEmbeddings

def create_embeddings(model_name: str = "cohere.embed-english-v3.0") -> OCIGenAIEmbeddings:
    """Creates OCI Generative AI embeddings from .env"""
    embedding = OCIGenAIEmbeddings(
        model_id=model_name,
        service_endpoint=os.getenv("OCI_AI_EMBEDDING_ENDPOINT"),
        truncate="NONE",
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        auth_type="API_KEY",   # or SECURITY_TOKEN, INSTANCE_PRINCIPAL, etc.
        auth_profile="DEFAULT",   # or any other profile in config
        # auth_file_location=r"C:\Users\Fayeque\.oci\config.txt"  # full path to config file
        auth_file_location="oci_config"
    )
    return embedding