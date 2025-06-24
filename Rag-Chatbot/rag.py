from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging
import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

def load_chat_history_from_file(file_path="chat_history.txt"):
    """
    Load chat history from a text file and convert to LangChain message format.
    
    Expected file format (each line should be one of these):
    Human: <message>
    AI: <message>
    
    Args:
        file_path (str): Path to the chat history file
        
    Returns:
        list: List of LangChain message objects (HumanMessage, AIMessage)
    """
    chat_history = []
    
    if not os.path.exists(file_path):
        logging.info(f"No chat history file found at {file_path}. Starting with empty history.")
        return chat_history
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
            if not content:
                logging.info(f"Chat history file {file_path} is empty.")
                return chat_history
            
            # Parse as plain text format
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Human:') or line.startswith('User:'):
                    message_content = line.split(':', 1)[1].strip()
                    chat_history.append(HumanMessage(content=message_content))
                elif line.startswith('AI:') or line.startswith('Assistant:'):
                    message_content = line.split(':', 1)[1].strip()
                    chat_history.append(AIMessage(content=message_content))
            
            logging.info(f"Loaded {len(chat_history)} messages from {file_path}")
            return chat_history
            
    except Exception as e:
        logging.error(f"Error loading chat history from {file_path}: {e}")
        return []

def save_chat_history_to_file(chat_history, file_path="chat_history.txt"):
    """
    Save chat history to a text file.
    
    Args:
        chat_history (list): List of LangChain message objects
        file_path (str): Path to save the chat history
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    file.write(f"Human: {msg.content}\n")
                elif isinstance(msg, AIMessage):
                    file.write(f"AI: {msg.content}\n")
        
        logging.info(f"Saved {len(chat_history)} messages to {file_path}")
    except Exception as e:
        logging.error(f"Error saving chat history to {file_path}: {e}")

def interactive_rag_query(rag_chain, user_question, chat_history_file="chat_history.txt"):
    """
    Process a single RAG query with chat history from file.
    
    Args:
        rag_chain: The initialized RAG chain
        user_question (str): The user's question
        chat_history_file (str): Path to the chat history file
        
    Returns:
        dict: Response containing answer and context
    """
    # Load existing chat history
    chat_history = load_chat_history_from_file(chat_history_file)
    
    # Process the query
    response = rag_chain.invoke({"input": user_question, "chat_history": chat_history})
    
    # Update chat history with new interaction
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response['answer']))
    
    # Save updated chat history
    save_chat_history_to_file(chat_history, chat_history_file)
    
    return response

def rag_chain():
    llm  = OllamaLLM(model = Config.OllamaModel)
    try:
        # FAISS requires the embedding function used to create it when loading
        embeddings = OllamaEmbeddings(model=Config.OllamaEmbeddingsModel)
        vectorstore = FAISS.load_local(
            Config.VECTOR_DB_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True # Required for loading components like InMemoryDocstore
        )
        retriever = vectorstore.as_retriever()
        logging.info(f"FAISS vector store loaded from {Config.VECTOR_DB_DIR}. Retriever ready.")
    except Exception as e:
        logging.error(f"Failed to load FAISS vector store from {Config.VECTOR_DB_DIR}. Has ingestion been run? Error: {e}")
        raise

    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=context_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's question ONLY based on the provided context.\n"
                   "If the answer is not in the context, state that you don't have enough information.\n"
                   "Context: {context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    document_retriever = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever=history_retriever, combine_docs_chain=document_retriever)
    return retrieval_chain

if __name__ == "__main__":
    rag = rag_chain()
    logging.info("RAG chain created successfully.")
    
    # Load chat history from file
    chat_history = load_chat_history_from_file("chat_history.txt")
    
    print("\n--- RAG Test with File-based Chat History ---")
    print(f"Loaded {len(chat_history)} messages from chat history file")
    
    if chat_history:
        print("\n--- Current Chat History ---")
        for i, msg in enumerate(chat_history):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"{i+1}. {msg_type}: {msg.content}")
    
    # Test query with loaded chat history
    user_question = "Tell me more about the document content"
    print(f"\n--- User Question: {user_question} ---")
    
    response = rag.invoke({"input": user_question, "chat_history": chat_history})
    print(f"\n--- AI Response ---")
    print(f"{response['answer']}")
    
    # Update chat history with new interaction
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response['answer']))
    
    # Save updated chat history back to file
    save_chat_history_to_file(chat_history, "chat_history.txt")
    
    print(f"\n--- Updated Chat History Saved ---")
    print("Source documents:", response['context'])
