from langgraph.graph import START, END , StateGraph 
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import gradio as gr

load_dotenv()
class MyState(TypedDict):
    Messages: Annotated[list[dict],  add_messages  ]


graphBuilder = StateGraph(MyState)

def chatbot(state : MyState) -> MyState:
    print(state["Messages"])
    response = model.invoke(state["Messages"])
    new_messages = [{"role": "assistant", "content": response.content}]
    new_state = MyState(Messages = new_messages)
    return new_state

model = init_chat_model("gpt-4o", temperature=0.1)

graphBuilder.add_node("chatbot", chatbot)
graphBuilder.add_edge(START , "chatbot")
graphBuilder.add_edge("chatbot", END)

graph = graphBuilder.compile()

    

def chat(user_input: str, history):
    message = {"role": "user", "content": user_input}
    messages = [message]
    state = MyState(Messages=messages)
    result = graph.invoke(state)
    print(result , "State " , state)
    return result["Messages"][-1].content

print("Mystate" , MyState)

gr.ChatInterface(chat, type="messages").launch()