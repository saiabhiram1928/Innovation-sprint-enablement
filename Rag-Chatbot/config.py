from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    DocumentsDir = os.getenv("DOCUMENTS_DIR", "documents")
    OllamaModel = os.getenv("OLLAMA_MODEL", "mistral:latest")
    PORT = int(os.getenv("PORT", 5000))
    ChunkSize = int(os.getenv("CHUNK_SIZE", 1000))
    ChunkOverlap = int(os.getenv("CHUNK_OVERLAP", 200))
    OllamaEmbeddingsModel = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "mxbai-embed-large:latest")
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")


os.makedirs("documents", exist_ok=True)  # Ensure the documents directory exists
os.makedirs("vector_db", exist_ok=True)  # Ensure the documents directory exists