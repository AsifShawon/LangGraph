from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def update(content: str) -> str:
    """Update the document content with the new content"""
    global document_content
    document_content = content
    
    return "Document content updated successfully"

@tool
def save_tool(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args: 
        filename: Name for the text file
    """
    
    if not filename.endswith(".txt"):
        filename = filename + ".txt"
    
    try: 
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\nDocument saved to {filename}")
        return "Document saved successfully to {filename}."
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update, save_tool]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
).bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = """
     You are a helpful assistant that can update the document content and save it to a text file using provided tools.

        - If the user wants to update or modify the document content, you should use the 'update' tool.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after mofifications.state
        
        The current document content is:
        {document_content}
    """)
    
    if not state["messages"]:
        user = "I'm ready to help your update a document. What would you like to create?"
        user_message = HumanMessage(content = user)
    
    else:
        user = input("\nWhat would you like to do with the document?")
        print(f"\nUser: {user}")
        user_message = HumanMessage(content = user)
        
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    
    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages" : list(state['messages']) + [user_message, response]}
        
        
def should_continue(state: AgentState):
    messages = state["messages"]
    
    if not messages:
        return 'continue'
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 'saved' in message.content.lower() and
            'document' in message.content.lower()):
            return "end"
        
    return "continue"

def print_message(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tool", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tool")

graph.add_conditional_edges(
    "tool",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n======== DRAFTER =========")
    
    state = {"message" : []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
            
    print("\n======== DRAFTER ENDED =========")
    
if __name__ == "__main__":
    run_document_agent()