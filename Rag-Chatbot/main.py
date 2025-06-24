import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
import logging
import os
from rag import rag_chain, load_chat_history_from_file, save_chat_history_to_file
from config import Config
def create_app():
    app = Flask(__name__)
    with app.app_context():
        rag_chain()
    CORS(app, origins="*", allow_headers=["Content-Type"], supports_credentials=True,  expose_headers=['Set-Cookie'],
         allow_methods=['GET', 'POST'])
    return app
app = create_app()
def intialize_rag_chain():
    os.makedirs("chat_history", exist_ok=True)  # Ensure the documents directory exists
    # Read chat history from chat.txt file
    chat_history_file = "chat_history/chat.txt"
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r', encoding='utf-8') as file:
            chat_content = file.read()
            logging.info(f"Loaded chat history from {chat_history_file}")
    else:
        logging.info(f"No existing chat history found at {chat_history_file}")
    try:
        rag_chain()
        logging.info("RAG chain initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RAG chain: {e}")
        raise


@app.route('/health' , methods = ["GET"])
def main():
    try:
        # payload = {
        #     "model": MODEL,
        #     "prompt": "Hii are you alive ?",
        #     "stream": False
        # }
        # res = requests.post(url= OLLAMA_URL, json=payload) 
        # print(res.json())
        template = "Answer the question: {question} "
        prompt = ChatPromptTemplate.from_template(template) 
        model =  OllamaLLM(model= Config.OllamaModel)
        chain = prompt | model
        res = chain.invoke({"question": "Hii are you healthy?"}) 
        if res is not None:
            return jsonify({
                "message": res,
                "status": 200,
            }), 200
        else:
            return jsonify({
                "error": "Failed to connect to the model",
                "status": res,
            }), 400 
    except Exception as e:
        print("Issue with the connectivity")
        return jsonify({
            "error": "An Exception Occuured",
            "status": 500,
        }), 500
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Invalid request, 'question' is required"}), 400
        
        question = data['prompt']
        chat_history_file = data.get('chat_history_file', 'chat_history.txt')
        
        # Load existing chat history from file
        chat_history = load_chat_history_from_file(chat_history_file)
        
        # Initialize RAG chain
        rag = rag_chain()
        
        # Process the question with chat history
        response = rag.invoke({"input": question, "chat_history": chat_history})
        
        # Update chat history with new interaction
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response['answer']))
        
        # Save updated chat history back to file
        save_chat_history_to_file(chat_history, chat_history_file)
        
        return jsonify({
            "answer": response['answer'], 
            "source_documents": [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                } for doc in response['context']
            ],
            "chat_history_length": len(chat_history)
        }), 200
        
    except Exception as e:
        logging.error(f"Error in query endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    """Get the current chat history from file"""
    try:
        chat_history_file = request.args.get('file', 'chat_history.txt')
        chat_history = load_chat_history_from_file(chat_history_file)
        
        history_data = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_data.append({"type": "ai", "content": msg.content})
        
        return jsonify({
            "chat_history": history_data,
            "total_messages": len(history_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Error getting chat history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_chat_history():
    """Clear the chat history file"""
    try:
        data = request.get_json() or {}
        chat_history_file = data.get('file', 'chat_history.txt')
        
        if os.path.exists(chat_history_file):
            os.remove(chat_history_file)
            return jsonify({"message": f"Chat history cleared from {chat_history_file}"}), 200
        else:
            return jsonify({"message": f"No chat history file found at {chat_history_file}"}), 200
            
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)