from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 
    # messages that contains all the messages like HumanMessage, Toolsmessage, SystemMessage with their contents
    
@tool
def add(a: int, b: int):
    """Addition function that adds two numbers"""
    return a+b

@tool
def subtract(a: int, b: int):   
    """Subtraction function that subtracts two numbers"""
    return a-b

@tool
def multiplication(a: int, b: int):
    """Multiplication function that multiplies two numbers"""
    return a*b

tools = [add, subtract, multiplication]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = "You are a helpful assistant that can use the following tools to answer questions.")
    response = model.invoke([system_prompt] + state["messages"])
    
    return {"messages": [response]}
    
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)

graph.add_edge("tool", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "what is the result of the expression ((20+30)*2)-10 and tell me a joke")]}

print_stream(app.stream(inputs, stream_mode = "values"))