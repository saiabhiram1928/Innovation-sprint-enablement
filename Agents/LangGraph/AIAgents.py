from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage , HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START ,END
from langgraph.prebuilt import ToolNode , tools_condition
from IPython.display import display , Image
import gradio as gr
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from playwright.sync_api import sync_playwright
import nest_asyncio
import textwrap
load_dotenv()
def ReactAgent():
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    graph = StateGraph(State)
    @tool
    def add(a: int, b : int) -> int:
        """This tool is used for addition of two a and b which are passed as arguments"""
        return a+b
    @tool
    def mul(a : int , b:int) ->int:
        """This tool is used for multiplication of two numbers a and b  which are passed as arguments"""
        return a*b
    
    tools = [add, mul]
    llm = ChatOpenAI(model= "gpt-4.1-nano" ).bind_tools(tools)
    def agent(state: State) -> State:
        system_prompt = SystemMessage(content= "You are help ful assistent, please use the tool provided to solve the query")
        response = llm.invoke([system_prompt] + state["messages"])
        return {"messages" : [response]}
    
    def should_continue(state: State) -> str: 
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls: 
            return "end"
        else:
            return "continue"
    graph.add_node("agent" , agent)
    graph.add_node("tools", ToolNode(tools) )

    graph.set_entry_point("agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "end" : END,
            "continue" : "tools"
        }
    )
    app = graph.compile()

    with open("ReactAgent.png" , "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
    inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
    print_stream(app.stream(inputs, stream_mode="values"))

def memory_testing():
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    graph = StateGraph(State)
    memory = MemorySaver()
    llm = ChatOpenAI(model= "gpt-4.1")
    def agent(state : State) -> State:
        return {"messages" : [llm.invoke(state["messages"])]}
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    app = graph.compile(checkpointer= memory)

    config = {"configurable" : {"thread_id" : "1"}}
    with open("memory_implementation.png" , "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    def chat(user_input : str, history):
        res = app.invoke({"messages" : [HumanMessage(content= user_input)]}, config = config)
        print(f"After '{user_input}':")
        print("Current state:", app.get_state(config=config))
        print("Number of states in history:", len(list(app.get_state_history(config=config))))
        return res['messages'][-1].content
   
    gr.ChatInterface(chat, type= "messages").launch()
    # print(app.get_state(config= config) , list(app.get_state_history(config)))

def broswer_automation():

    class State(TypedDict):
        messages : Annotated[list[BaseMessage] , add_messages]
    graph = StateGraph(State)  
    nest_asyncio.apply()
    async_browser =  create_async_playwright_browser(headless=False)  
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    for tool in tools:
        print(f"Tool Name : {tool.name} and tool : {tool}\n ")
    llm = ChatOpenAI(model= "gpt-4o-mini").bind_tools(tools)

    def chatbot(state: State) -> State:
        res = llm.invoke(state["messages"])
        print(res)
        return {"messages" : [res]}

    
    graph.add_node("Agent" , chatbot)
    graph.add_node("tools" , ToolNode(tools= tools))
    graph.set_entry_point("Agent")
    graph.add_conditional_edges("Agent", tools_condition, {
      "tools": "tools",                 
       END : END
    } )

    graph.add_edge("tools" , "Agent")
    memory = MemorySaver()

    app = graph.compile(checkpointer=memory)
    
    with open("BrowserAutomation.png" , "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    config = {"configurable" : {"thread_id" : "1"}}
    async def chat(user_input, history) : 
        try:
            print(f"Processing: {user_input}")
            print("==="*100)
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            return result["messages"][-1].content
        except Exception as e:
            print(f"Error occurred: {e}")
            return f"Error: {str(e)}"
    
    gr.ChatInterface(chat,type="messages" ).launch()


    
if __name__ == "__main__":
    # ReactAgent()
    # memory_testing()
    broswer_automation()