from typing import TypedDict, Annotated
from langgraph.graph import StateGraph , START, END
from IPython.display import Image, display
import inquirer as inq
import math
from langgraph.graph.message import add_messages
import random
def pratice_1()  : 
    class State(TypedDict):
        name : str
    def node_1(state: State) -> State :
        state["name"] =  state["name"] + " you are doing amazing"
        return state
    graph = StateGraph(State)
    graph.add_node("node_1" , node_1)

    graph.set_entry_point("node_1")
    graph.set_finish_point("node_1")
    app = graph.compile()
    result = app.invoke({"name" : "Bob"})
    display(Image(app.get_graph().draw_mermaid_png()))
    return result

def practice_2() -> dict:
    class State(TypedDict):
        name : str
        values : list[int]
        operation : str
        result : str
    
    def processor(state: State) -> State:
        ans = "Operation Invalid"
        if state["operation"] == "*" :
            ans = f"Hi {state['name']} your answer is : {math.prod(state['values'])}"
        elif state["operation"] == "+":
            ans = f"Hi {state['name']} your name is : {sum(state['values'])}"
        return {"result" : ans} 
    graph = StateGraph(State)
    graph.add_node("processor", processor)
    graph.add_edge(START, "processor")
    graph.add_edge("processor", END)
    graph.set_finish_point("processor")

    app = graph.compile()

    return app.invoke({"name" : "Bob" , "values" : [1, 3 ,4, 5] , "operation" : "+"}) 

def practice_3() -> dict:
    class State(TypedDict) :
        name : str
        age : int
        skills : list[str]
        result : str
    graph = StateGraph(State)
    def node_1(state : State) -> State:
       return {"result" : f"Hi {state['name']} !, "} 
    def node_2(state: State) -> State:
        return {"result" : state["result"]+f"Your Age is : {state['age']}"}
    def node_3(state: State) -> State:
        return {"result" : state["result"] + f" And your Skills are : {" ".join(state['skills'])}"}
    graph.add_node("node_1" , node_1)
    graph.add_node("node_2" , node_2)
    graph.add_node("node_3" , node_3)

    graph.set_entry_point("node_1")
    graph.add_edge("node_1" , "node_2")
    graph.add_edge("node_2", "node_3")
    graph.set_finish_point("node_3")

    app = graph.compile()
    return app.invoke({"age" : 10, "name" : "Bob" , "skills" : ["Python", "C++"]})

def practice_4() -> dict:
    class State(TypedDict):
        num1 : int
        num2: int
        num3 : int 
        num4: int
        operation : str
        operation_2 : str
        final_result_1 : int
        final_result_2 : int
    graph = StateGraph(State)
    def router_1(state : State) -> State:
        if state["operation"] == "+" :
            return "add"
        elif state["operation"] == "*" :
            return "mul"
    
    def add_node_1(state : State) -> State :
        return {"final_result_1" : state["num1"] + state["num2"] }
    def mul_node_1(state: State) -> State:
        return {"final_result_2" : state["num3"] * state["num4"]}
    # Router function doesn't validate to node
    def router_2(state: State) -> str:
        if state["operation_2"] == "+":
            return "add_2"
        elif state["operation_2"] == "*":
            return "mul_2"
    
    def mul_node_2(state : State) -> State:
        return {"final_result_2" : state["num3"] * state["num4"]}
    def add_node_2(state: State) -> State:
        return {"final_result_2" : state["num3"] + state["num4"]}
    graph.add_node("add_1", add_node_1)
    graph.add_node("add_2", add_node_2)
    graph.add_node("mul_1", mul_node_1)
    graph.add_node("mul_2", mul_node_2)
    graph.add_node("router_1", lambda state:state)
    graph.add_node("router_2", lambda state:state)
    
    graph.add_edge(START, "router_1")
    graph.add_conditional_edges(
        "router_1",
        router_1,
        {
            # edge return in the router_1 function : node to processed
            "add" : "add_1",
            "mul" : "mul_1"
        }
    )
    graph.add_edge("add_1" , "router_2")
    graph.add_edge("mul_1", "router_2")
    graph.add_conditional_edges(
        "router_2",
        router_2,
        {
            "add_2" : "add_2",
            "mul_2" : "mul_2"
        }
    )
    graph.add_edge("add_2" , END) 
    graph.add_edge("mul_2", END)
    app = graph.compile()
    with open("practice_4_graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png()) 
    return app.invoke({"num1" : 1 , "num2" : 2 , "operation": "+", "num3" : 4 , "num4" : 5 , "operation_2" : "*"}) 

def practice_5():
    class State(TypedDict):
        player_name : str
        guesses : list[int]
        attempts : int
        lower_bound : int
        upper_bound : int
        ans : int
        hint : str
        verdict : str
    graph = StateGraph(State)
    def setUp(state: State) -> State:
        state["attempts"] = 0
        state["verdict"] = state["hint"] = ""
        state["player_name"] = f"Hi there {state['player_name']}"
        state["guesses"] = []
        return state
    def guess(state: State) -> State:
        print(state)
        if state["hint"] == "higher":
            state["lower_bound"] =state["guesses"][-1] + 1
        elif state["hint"] == "lower":
            state["upper_bound"] =state["guesses"][-1] - 1
        current_number = random.randint(state["lower_bound"], state["upper_bound"])
        state["attempts"] +=1
        state["guesses"].append(current_number)
        return state
    def hint_node(state: State) -> State:
        if state["attempts"] >= 7 :
            state["verdict"] = "Lost"
            return state
        elif state["ans"] > state["guesses"][-1]  or state["ans"] < state["guesses"][-1] :
            state["hint"] = "higher" if state["guesses"][-1] < state["ans"] else "lower"
            return state
        state["verdict"] = "Won"
        return state
    # This is a router function doesn't used as node
    def router_function(state: State) -> str:
        if state.get("verdict"):
            return "end"
        return "continue"
    graph.add_node("setUp", setUp)
    graph.add_node("guess", guess)
    graph.add_node("hint", hint_node)

    graph.set_entry_point("setUp")
    graph.add_edge("setUp", "guess")
    graph.add_edge("guess", "hint")
    graph.add_conditional_edges(
        "hint",
        router_function,
        {
            "continue" : "guess",
            "end" : END
        }
    )
    graph.add_edge("hint", END)
    app = graph.compile()
    with open("practice_5_graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png()) 
    return app.invoke({"lower_bound": 1 , "upper_bound" : 10 , "ans" : 3, "player_name" : "Bob" }) 

if __name__ == "__main__":
    questions = [inq.List("file" , "What file you want to run" , choices=["practice_1", "practice_2", "practice_3" , "practice_4", "practice_5"] , default= "practice_1")]
    response = inq.prompt(questions)
    if response["file"] == "practice_1" :
        ans = pratice_1() 
    elif response['file'] == "practice_2":
        ans = practice_2()
    elif response["file"] == "practice_3":
        ans = practice_3()
    elif response["file"] == "practice_4":
        ans = practice_4()
    elif response["file"] == "practice_5":
        ans = practice_5()
    print(ans)
    