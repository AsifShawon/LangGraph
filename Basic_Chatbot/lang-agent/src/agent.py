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
from utils import extract_tool_calls
from memory_manager import memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for our agent with long-term memory support."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    error_count: int
    last_error: str
    summary: str
    retrieved_context: str

class LangGraphAgent:
    """Main agent class with integrated smart memory"""

    def __init__(self):
        Config.validate()
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            timeout=Config.TIMEOUT
        )
        self.tools = create_tools(Config.BRAVE_API_KEY)
        self.system_prompt_template = """You are an intelligent and efficient assistant.
Here is some relevant context from our past conversations:
{retrieved_context}

Use this context to inform your response. You have access to tools for searching, getting the time, and checking weather.
Guidelines:
- Use tools in sequence if a query requires multiple steps.
- Interpret and present tool data clearly.
- Be accurate, polite, and context-aware.
"""
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph_definition = self._build_graph_definition()
        logger.info("Agent with smart memory initialized successfully")

    def _build_graph_definition(self) -> StateGraph:
        """
        Builds the graph structure without compiling it.
        This allows us to attach a different checkpointer for each run.
        """
        graph = StateGraph(AgentState)
        
        retrieve_context_runnable = RunnableLambda(self._retrieve_context_node)
        
        graph.add_node("retrieve_context", retrieve_context_runnable)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("error_handler", self._error_handler_node)

        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "agent")
        graph.add_edge("tools", "agent")
        graph.add_edge("error_handler", "agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END, "error": "error_handler"}
        )
        return graph

    def _retrieve_context_node(self, state: AgentState, config: Dict) -> AgentState:
        thread_id = config["configurable"]["thread_id"]
        last_user_message = state["messages"][-1].content
        retriever = memory_manager.get_retriever(thread_id)
        retrieved_context = ""
        if retriever:
            try:
                retrieved_docs = retriever.invoke(last_user_message)
                retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved context for thread {thread_id}: {retrieved_context}")
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
        return {"retrieved_context": retrieved_context}

    def _agent_node(self, state: AgentState) -> AgentState:
        try:
            system_prompt = SystemMessage(
                content=self.system_prompt_template.format(
                    retrieved_context=state.get("retrieved_context", "No context found.")
                )
            )
            messages_with_system_prompt = [system_prompt] + state["messages"]
            response = self.llm_with_tools.invoke(messages_with_system_prompt)
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

    def _get_final_messages_safely(self, checkpointer, config, result):
        """
        Safely extract final messages from different checkpointer formats
        """
        try:
            # Try the new format first
            checkpoint = checkpointer.get(config)
            if checkpoint and hasattr(checkpoint, 'channel_values'):
                return checkpoint.channel_values.get('messages', [])
            
            # Try dictionary format
            elif isinstance(checkpoint, dict):
                if 'messages' in checkpoint:
                    return checkpoint['messages']
                elif 'channel_values' in checkpoint and 'messages' in checkpoint['channel_values']:
                    return checkpoint['channel_values']['messages']
            
            # Fall back to result messages if checkpoint doesn't work
            if result and 'messages' in result:
                return result['messages']
                
            logger.warning("Could not extract messages from checkpoint, using empty list")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting messages from checkpoint: {e}")
            # Fall back to result messages
            return result.get('messages', []) if result else []

    def invoke(self, message: str, thread_id: str):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpointer = memory_manager.get_checkpointer(thread_id)
            thread_graph = self.graph_definition.compile(checkpointer=checkpointer)
            
            result = thread_graph.invoke({"messages": [HumanMessage(content=message)]}, config)
            
            # Safely get final messages
            final_messages = self._get_final_messages_safely(checkpointer, config, result)
            
            # Update vector store with the last 2 messages if available
            if len(final_messages) >= 2:
                memory_manager.update_vector_store(thread_id, final_messages[-2:])
                logger.info(f"Updated vector store for thread {thread_id} with {len(final_messages[-2:])} messages")
            elif final_messages:
                memory_manager.update_vector_store(thread_id, final_messages)
                logger.info(f"Updated vector store for thread {thread_id} with {len(final_messages)} messages")
            
            final_message = result["messages"][-1] if result.get("messages") else None
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
        # --- MODIFIED: Use the async checkpointer ---
        checkpointer = await memory_manager.get_async_checkpointer(thread_id)
        thread_graph = self.graph_definition.compile(checkpointer=checkpointer)

        try:
            full_response = []
            async for event in thread_graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config,
                version="v1"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        full_response.append(chunk.content)
                        yield chunk.content
            
            # --- UPDATED: Retrieve final state after stream and update memory ---
            final_graph_state = await thread_graph.aget_state(config)
            final_messages = final_graph_state.values.get("messages", [])

            if len(final_messages) >= 2:
                memory_manager.update_vector_store(thread_id, final_messages[-2:])
                logger.info(f"Updated vector store for thread {thread_id} after streaming.")
            elif final_messages:
                memory_manager.update_vector_store(thread_id, final_messages)
                logger.info(f"Updated vector store for thread {thread_id} after streaming with {len(final_messages)} messages.")

        except Exception as e:
            logger.error(f"Error during agent streaming for thread {thread_id}: {e}", exc_info=True)
            yield f"\n--- ERROR ---\n{str(e)}"
        finally:
            # Ensure the async connection is closed
            if checkpointer and hasattr(checkpointer.conn, 'close'):
                await checkpointer.conn.close()

def create_agent() -> LangGraphAgent:
    return LangGraphAgent()