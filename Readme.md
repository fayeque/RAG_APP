Here’s a complete README.md for your ingestion.py that explains everything clearly — including local Hugging Face embedding model setup, pipenv usage, and application run instructions.

markdown
Copy
Edit
# 📄 PDF Ingestion & RAG with Oracle Vector DB + Ollama LLM

This project ingests PDF documents, chunks them, generates embeddings locally using a Hugging Face model, stores them in **Oracle 23AI Vector DB**, and uses **Ollama** for Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
- Load and split PDF documents into smaller chunks.
- Generate embeddings locally using Hugging Face (offline capability).
- Store/retrieve embeddings in **Oracle Vector Store (23AI)**.
- Perform semantic search and answer queries with **Ollama LLM**.
- Fully environment-variable driven configuration.

---

## 📂 Project Structure
.
├── ingestion.py # Main script to run the ingestion + query
├── db_connection.py # Oracle DB connection helper
├── ELCM.pdf # Sample PDF for ingestion
├── .env # Environment variables
├── local_model/ # Local Hugging Face embedding model
├── Pipfile # pipenv dependencies
└── README.md

yaml
Copy
Edit

---

## 🔧 Requirements

- **Python 3.10+**
- **pipenv** for virtual environment & dependencies
- **Oracle Database 23AI** with Vector Store enabled
- **Ollama** installed locally
- **Hugging Face account** (to download model)

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_access_token

DB_USERNAME=VEC23AI
DB_PASSWORD=VEC23AI
HOST=ofss-mum-5891.snbomprshared2.gbucdsint02bom.oraclevcn.com
SERVICE_NAME=VEC23AIPDB
PORT=1521
📥 Installing Hugging Face Embedding Model Locally
Login to Hugging Face

bash
Copy
Edit
huggingface-cli login
Enter your HF_TOKEN.

Download the model
Replace MODEL_NAME with your desired embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2):

bash
Copy
Edit
mkdir local_model
huggingface-cli download MODEL_NAME --local-dir local_model
The local_model directory will be used in ingestion.py.

📦 Installation
Using pipenv (recommended)
Install pipenv (if not installed):

bash
Copy
Edit
pip install pipenv
Install all dependencies from Pipfile:

bash
Copy
Edit
pipenv install
Activate environment:

bash
Copy
Edit
pipenv shell
▶️ Running the Application
Step 1: Ingest PDF into Oracle Vector DB
Place your PDF (ELCM.pdf) in the project root.

Open ingestion.py and uncomment the ingestion block:

python
Copy
Edit
knowledge_base = OracleVS.from_documents(
    final_documents,
    embeddings,
    client=connection,
    table_name="RAG_DOCUMENTS",
    distance_strategy=DistanceStrategy.DOT_PRODUCT
)
connection.commit()
connection.close()
Run the script:

bash
Copy
Edit
python ingestion.py
Step 2: Query with Ollama + RAG
Ensure Ollama is running locally and the model (gemma3:1b in script) is available.

Run:

bash
Copy
Edit
python ingestion.py
You will see:

csharp
Copy
Edit
User question was -> What is revolving facility?
LLM response is -> ...