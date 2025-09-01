import os
from langchain_community.chat_models import ChatOCIGenAI

def create_llm() -> ChatOCIGenAI:
    """Creates OCI Generative AI LLM."""
    llm = ChatOCIGenAI(
        model_id="ocid1.generativeaimodel.oc1.uk-london-1.amaaaaaask7dceyach2dyu6g5w5ocnvbkto2g76wxitj3rpddplsqoxqh2lq",
        service_endpoint=os.getenv("OCI_AI_EMBEDDING_ENDPOINT"),
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        provider="meta",
        model_kwargs={
            "temperature": 1,
            "max_tokens": 600,
            "frequency_penalty": 0,
            "presence_penalty": 0,  
            "top_p": 0.75,
        },
        auth_type="API_KEY",   # or SECURITY_TOKEN, INSTANCE_PRINCIPAL, etc.
        auth_profile="DEFAULT",   # or any other profile in config
        # auth_file_location=r"C:\Users\Fayeque\.oci\config.txt"  # full path to config file
        auth_file_location="oci_config"
    )
    return llm