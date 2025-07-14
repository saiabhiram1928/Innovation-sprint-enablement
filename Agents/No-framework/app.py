from langchain_ollama.llms import OllamaLLM
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate  

MODEL_NAME = "mistral:latest"

def get_web_search_results(query: str):
    return f"Web search results for: {query} , this websearch is simulated."

def get_details_from_another(query: str):
    return f"Details from another source for: {query}"

def main():
    model = OllamaLLM(model=MODEL_NAME)
    prompt = PromptTemplate(
        input_variables=["input"  , "agent_scratchpad" , "tools", "tool_names"],
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    tools = [
        Tool(
            name="web_search",
            description="Search the web for information",
            func=get_web_search_results,
        ),
        Tool(
            name="details_search",
            description="Get details from another source",
            func=get_details_from_another,  
        ),
    ]
    
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt= prompt,
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    query = "What is the weather like today?"
    # Run the agent
    response = agent_executor.invoke({"input": query })
    print(response['output'])

if __name__ == "__main__":
   main()