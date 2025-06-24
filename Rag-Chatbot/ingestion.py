from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
import logging
import bs4
import os
from config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def get_loaders(doc_dir: str):
    if doc_dir.startswith("http://") or doc_dir.startswith("https://"):
        # If the document directory is a URL, use WebBaseLoader
        return WebBaseLoader(web_path=doc_dir)
    elif doc_dir.endswith('.txt'):
        return TextLoader(doc_dir, encoding='utf-8')  
    elif doc_dir.endswith('.pdf'):
        return PyPDFLoader(doc_dir)
    else:
        logging.error("Unsupported document format. Please provide a .txt or .pdf file or a URL.")
        raise ValueError("Unsupported document format. Please provide a .txt or .pdf file or a URL.")
        return none

def ingest_documents():
    documents = []
    for root, _, files in os.walk(Config.DocumentsDir):
        for file in files:
            file_path = os.path.join(Config.DocumentsDir, file)
            print(file_path)
            try:
                loader = get_loaders(file_path)
                if loader:
                    documents.extend(loader.load())
                    logging.info(f"Loaded {len(documents)} documents from {file_path}")
                else:
                    logging.error(f"Unsupported file{file_path}")

            except Exception as e:
                logging.error(f"Error loading document {file_path}: {e}")
    if not documents:
        logging.error("No documents found to ingest.")
        return
    txtsplitter =  RecursiveCharacterTextSplitter(chunk_size=Config.ChunkSize, chunk_overlap=Config.ChunkOverlap,add_start_index=True)
    split_docs = txtsplitter.split_documents(documents)
    logging.info(f"Split documents into {len(split_docs)} chunks.")
    try:
        embeddings = OllamaEmbeddings(model=Config.OllamaEmbeddingsModel)
        _ = embeddings.embed_query("test")
        logging.info("Ollama embeddings model is working correctly.")
    except Exception as e:
        logging.error(f"Error with Ollama embeddings model: {e}")
        raise
    try:
        if os.path.exists(Config.VECTOR_DB_DIR) and os.listdir(Config.VECTOR_DB_DIR):
            store = FAISS.load_local(Config.VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization= True)
            store.add_documents(split_docs)
            logging.info(f"Added new Chunks to existing FAISS index")
        else:
            store = FAISS.from_documents(split_docs, embeddings)
            logging.info(f"New FAISS index created with {len(split_docs)} chunks.")
        store.save_local(Config.VECTOR_DB_DIR)
    except Exception as e:
        logging.error(f"Error creating or saving FAISS index: {e}")
        raise


if __name__ == "__main__":
    ingest_documents()
    logging.basicConfig(level=logging.INFO)
    logging.info("Ingestion completed successfully.")