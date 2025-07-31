import asyncio
from dotenv import load_dotenv
import subprocess
import json
from typing import Dict, List, Optional, TypedDict , Annotated , Any
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage , HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START , END
from langgraph.prebuilt.tool_node import ToolNode
from langchain_openai import ChatOpenAI
load_dotenv()

class McpClient():
    def __init__(self):
        self.process = subprocess.Popen(
            ["python", "server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0 
        )
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send message
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush() 
            # Read response
            response_line = self.process.stdout.readline()
            print(response_line)
            if response_line:
                response = json.loads(response_line.strip())
                if "result" in response:
                    return response["result"]
                elif "error" in response:
                    return f"Error: {response['error']}"
            
            return "No response received"
            
        except Exception as e:
            return f"Communication error: {str(e)}"
    def close(self):
        """Close the MCP server connection"""
        if self.process:
            self.process.terminate()
            self.process.wait()

mcpClient = McpClient()

@tool
def add_user_data(query:str) ->str:
    """Add user data using sql query statemetns"""
    print( "---"*10 ,query)
    result = mcpClient.call_tool("add_data", {"query": query})
    print(result , "---"*10)
    return str(result)
@tool
def query_user_data(query: str) -> str:
    """Query data from the users table using SQL SELECT statements"""
    result = mcpClient.call_tool("query_data", {"query": query})
    return str(result)


def AgentWorkflow():
    class State(TypedDict):
        messages : Annotated[list[BaseMessage] , add_messages ]
    graph = StateGraph(State)
    tools = [query_user_data, add_user_data]
    llm = ChatOpenAI(model= "gpt-4.1-nano").bind_tools(tools)
    def Agent(state : State) -> State:
        system_message = SystemMessage(content="""
                        You are a helpful database assistant. You can help users interact with a SQLite database 
                        that contains a 'users' table with columns: id, name, email, created_at.
                        Available tools:
                            1. add_user_data - Use this to insert data with SQL INSERT statements
                            2. query_user_data - Use this to retrieve data with SQL SELECT statements
                        When users ask to add data, generate proper INSERT SQL statements.
                        When users ask to query data, generate proper SELECT SQL statements.
                        Always explain what you're doing and format results nicely.""")
        message = state["messages"] + [system_message]
        response = llm.invoke(state["messages"])
        return {"messages" : [response]}
    def should_continue(state : State) :
        return "tools" if state["messages"][-1].tool_calls else END
    graph.add_node("Agent" , Agent)
    graph.add_node("Tools" , ToolNode(tools))
    graph.set_entry_point("Agent")
    graph.add_edge("Tools", "Agent")
    graph.add_conditional_edges(
        "Agent",
        should_continue,
        {
            "tools" : "Tools",
            END : END
        }
    )
    app = graph.compile()
    with open("Agent_grpah.png" , "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    
    system_message = """You are a helpful database assistant. You can help users interact with a SQLite database 
    that contains a 'users' table with columns: id, name, email, created_at.
    When a user asks you to do something, you must first generate the correct SQL query and then call the appropriate tool with that query.
    The table structure is   id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL, email TEXT UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    For example, if a user says 'add sai', you should first think 'I need to create an INSERT statement'."""

    print("Agent is ready. Type your request or 'quit'.")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
            
            # Invoke the agent with the history and the new message
            response = app.invoke({
                "messages": [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_input)
                ]
            })
            
            # Print the final response from the LLM
            final_message = response["messages"][-1]
            print(f"Agent: {final_message.content}")

    finally:
        print("\nShutting down MCP client.")
        mcpClient.close()
    
if __name__ == "__main__":
    AgentWorkflow()