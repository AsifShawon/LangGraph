from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    
    state["messages"].append(AIMessage(content = response.content))
    print(f'\nAI: {response.content}')
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

conversational_history = []
human = input("Enter a message: ")

while human not in ['exit', 'e']:
    conversational_history.append(HumanMessage(content = human))
    result = app.invoke({"messages": conversational_history})
    conversational_history.append(result["messages"][-1])
    human = input("Enter another message: ")