import asyncio
from typing import Sequence, TypedDict, Annotated, List, Any, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import logging

from config import Config
from tools import create_tools
# We will use this utility to format the conversation for the summarizer
from utils import extract_tool_calls, format_messages_for_display
from memory_manager import memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for our agent with summarization-based memory."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    error_count: int
    last_error: str
    # 'summary' is now the primary context carrier
    summary: str

class LangGraphAgent:
    """Main agent class with integrated summarization memory"""

    def __init__(self):
        Config.validate()
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            timeout=Config.TIMEOUT
        )
        self.tools = create_tools(Config.BRAVE_API_KEY)
        
        # --- MODIFICATION: System prompt now uses a 'summary' ---
        self.system_prompt_template = """You are an intelligent and efficient assistant.
Here is a summary of our past conversation:
{summary}

Based on this summary and the latest message, provide a helpful response. You have access to tools for searching, getting the time, and checking weather.
Guidelines:
- Use tools in sequence if a query requires multiple steps.
- Interpret and present tool data clearly.
- Be accurate, polite, and context-aware.
"""
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph_definition = self._build_graph_definition()
        logger.info("Agent with summarization memory initialized successfully")

    def _build_graph_definition(self) -> StateGraph:
        """
        Builds the graph structure with a summarization step.
        """
        graph = StateGraph(AgentState)
        
        # Create a runnable for the new summarization node
        summarize_runnable = RunnableLambda(self._summarize_conversation_node)
        
        # The graph now starts with summarization
        graph.add_node("summarize_conversation", summarize_runnable)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("error_handler", self._error_handler_node)

        graph.add_edge(START, "summarize_conversation")
        graph.add_edge("summarize_conversation", "agent")
        graph.add_edge("tools", "agent")
        graph.add_edge("error_handler", "agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END, "error": "error_handler"}
        )
        return graph

    def _summarize_conversation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Summarizes the conversation history to provide context to the agent.
        """
        history = state.get("messages", [])
        # We only summarize if there's a history to process
        if len(history) <= 1:
            return {"summary": "No prior conversation history."}

        # Summarize all messages except the very last one (the user's current input)
        messages_to_summarize = history[:-1]
        history_str = format_messages_for_display(messages_to_summarize)
        
        summarization_prompt = f"""
Please create a concise summary of the following conversation.
Focus on key facts, user preferences, and important topics discussed.
This summary will be used to give an AI assistant context for its next response.

Conversation History:
{history_str}

Concise Summary:
"""
        try:
            summary_response = self.llm.invoke(summarization_prompt)
            summary = summary_response.content
            logger.info(f"Generated summary: {summary}")
            return {"summary": summary}
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return {"summary": "Error summarizing conversation."}

    def _agent_node(self, state: AgentState) -> AgentState:
        """The main agent logic node, now using the summary for context."""
        try:
            # Use the generated summary for context
            system_prompt = SystemMessage(
                content=self.system_prompt_template.format(
                    summary=state.get("summary", "No summary available.")
                )
            )
            
            # Send only the system prompt and the latest user message
            messages_to_send = [system_prompt, state["messages"][-1]]
            
            response = self.llm_with_tools.invoke(messages_to_send)
            
            return {
                "messages": [response],
                "tool_calls": extract_tool_calls(response),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error_count": 0,
                "last_error": ""
            }
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            return {"error_count": state.get("error_count", 0) + 1, "last_error": str(e)}

    def _error_handler_node(self, state: AgentState) -> AgentState:
        error_message = state.get("last_error", "Unknown error")
        error_response = AIMessage(content=f"I encountered an error: {error_message}. Let me try again.")
        return {"messages": [error_response]}

    def _should_continue(self, state: AgentState) -> str:
        if state.get("last_error"):
            return "error" if state.get("error_count", 0) < 3 else "end"
        if state.get("tool_calls"):
            return "continue"
        if state.get("iteration_count", 0) >= Config.MAX_ITERATIONS:
            return "end"
        return "end"

    def invoke(self, message: str, thread_id: str):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpointer = memory_manager.get_checkpointer(thread_id)
            thread_graph = self.graph_definition.compile(checkpointer=checkpointer)
            
            print(f"Invoking agent for thread {thread_id} with message: {message}")
            result = thread_graph.invoke({"messages": [HumanMessage(content=message)]}, config)
            
            print(f"Agent invocation result: {result}")
            
            # Extract the final message from the result directly
            final_messages = result.get("messages", [])
            final_message = final_messages[-1] if final_messages else None
            
            print(f"Final messages extracted: {len(final_messages)} messages found.")
            print(f"Final message: {final_message.content if final_message else 'No final message'}")
            
            return {
                "success": True,
                "response": final_message.content if final_message else "No response.",
                "thread_id": thread_id
            }
        except Exception as e:
            logger.error(f"Error invoking agent for thread {thread_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "thread_id": thread_id}

    async def stream(self, message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        checkpointer = await memory_manager.get_async_checkpointer(thread_id)
        thread_graph = self.graph_definition.compile(checkpointer=checkpointer)

        try:
            async for event in thread_graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config,
                version="v1"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
            
            # Memory is persisted automatically by the checkpointer.
            logger.info(f"Streaming finished for thread {thread_id}. Checkpointer has saved the state.")

        except Exception as e:
            logger.error(f"Error during agent streaming for thread {thread_id}: {e}", exc_info=True)
            yield f"\n--- ERROR ---\n{str(e)}"
        finally:
            if checkpointer and hasattr(checkpointer.conn, 'close'):
                await checkpointer.conn.close()

def create_agent() -> LangGraphAgent:
    return LangGraphAgent()