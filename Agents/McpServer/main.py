from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio

load_dotenv()

client = MultiServerMCPClient({
    "database": {
        "command": "python",
        "args": ["server.py"],
        "transport": "stdio"
    } 
}) 

async def create_async_agent():
    """Create an async agent that works with MCP tools"""
    
    # Get tools from MCP server (these are async tools)
    tools = await client.get_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Create LLM with valid model name
    llm = ChatOpenAI(model="gpt-4-turbo")
    llm_with_tools = llm.bind_tools(tools)
    
    # Define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    # Define agent node
    async def agent_node(state: State):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}
    
    # Create tool node (async)
    tool_node = ToolNode(tools)
    
    # Define conditional logic
    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    
    return graph.compile()

async def AgentWorkflowWithMCP():
    """Main workflow function"""
    # Create the async agent
    agent = await create_async_agent()
    
    # Run the agent
    result = await agent.ainvoke({
        # "messages": [HumanMessage(content="add user name sai with email test@gmail.com")]
         "messages": [HumanMessage(content="fetch user with name sai with email test@gmail.com from table users")]
    })
    
    print("--" * 20)
    print("Conversation:")
    
    for message in result["messages"]:
        if hasattr(message, 'content') and message.content:
            msg_type = message.__class__.__name__
            if msg_type == "HumanMessage":
                print(f"User: {message.content}")
            elif msg_type == "AIMessage":
                print(f"Assistant: {message.content}")
            elif msg_type == "ToolMessage":
                print(f"Tool Result: {message.content}")
    
    print("--" * 20)
    
    return result

if __name__ == "__main__":
    asyncio.run(AgentWorkflowWithMCP())