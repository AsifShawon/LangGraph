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
from utils import extract_tool_calls, format_messages_for_display
from memory_manager import memory_manager
# Import the specific memory instances
from vector_memory import conversation_memory, physics_kb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    error_count: int
    last_error: str

class PhysicsBotAgent:
    def __init__(self):
        Config.validate()
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            timeout=Config.TIMEOUT
        )
        self.tools = create_tools(Config.BRAVE_API_KEY)
        
        self.system_prompt_template = """You are a specialized Physics Bot. Your primary knowledge base is "The Theoretical Minimum" by Leonard Susskind.
Answer the user's physics questions based on the following retrieved knowledge.
Also, be mindful of our recent conversation history for context.

---
RETRIEVED PHYSICS KNOWLEDGE:
{physics_context}
---
RETRIEVED CONVERSATION HISTORY:
{conversation_context}
---

Based on all the information above, provide a clear and accurate answer to the user's latest message.
"""
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph_definition = self._build_graph_definition()
        logger.info("Physics Bot Agent initialized successfully")

    def _build_graph_definition(self) -> StateGraph:
        graph = StateGraph(AgentState)
        
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("error_handler", self._error_handler_node)

        graph.add_edge(START, "agent")
        graph.add_edge("tools", "agent")
        graph.add_edge("error_handler", "agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END, "error": "error_handler"}
        )
        return graph

    def _agent_node(self, state: AgentState) -> AgentState:
        try:
            history = state.get("messages", [])
            thread_id = state.get('config', {}).get('thread_id', 'global')
            user_question = history[-1].content
            
            # 1. Get recent conversation context from the vector store
            conversation_context = conversation_memory.search_sync(user_question, k=3, thread_id=thread_id)
            conversation_context_text = "\n\n---\n\n".join([r['content'] for r in conversation_context]) or "No recent conversation history found."
            print(f"Conversation context: {conversation_context_text}")
            
            # 2. Search the physics knowledge base
            physics_context = physics_kb.search_sync(user_question, k=5)
            physics_context_text = "\n\n---\n\n".join([r['content'] for r in physics_context]) or "No relevant physics knowledge found."
            print(f"Physics context: {physics_context_text}")

            # 3. Construct the system prompt
            system_prompt = SystemMessage(
                content=self.system_prompt_template.format(
                    physics_context=physics_context_text,
                    conversation_context=conversation_context_text
                )
            )
            
            messages_to_send = [system_prompt, history[-1]]
            
            response = self.llm_with_tools.invoke(messages_to_send)
            
            return {
                "messages": [response],
                "tool_calls": extract_tool_calls(response),
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            return {"error_count": state.get("error_count", 0) + 1, "last_error": str(e)}

    def _error_handler_node(self, state: AgentState) -> AgentState:
        error_message = state.get("last_error", "Unknown error")
        error_response = AIMessage(content=f"I encountered an error: {error_message}. Please try again.")
        return {"messages": [error_response]}

    def _should_continue(self, state: AgentState) -> str:
        if state.get("last_error"):
            return "error" if state.get("error_count", 0) < 3 else "end"
        if state.get("tool_calls"):
            return "continue"
        return "end"
        
    def invoke(self, message: str, thread_id: str):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpointer = memory_manager.get_checkpointer(thread_id)
            thread_graph = self.graph_definition.compile(checkpointer=checkpointer)
            
            print(f"Invoking agent for thread {thread_id} with message: {message}")
            result = thread_graph.invoke({"messages": [HumanMessage(content=message)]}, config)
            
            final_messages = result.get("messages", [])
            final_message = final_messages[-1] if final_messages else None
            
            # Store the conversational turn in the conversation memory
            if final_message:
                conversation_memory.upsert_sync(thread_id, user_message=message, ai_reply=final_message.content)

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
        
        final_response_content = []

        try:
            async for event in thread_graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config,
                version="v1"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        final_response_content.append(chunk.content)
                        yield chunk.content
            
            if final_response_content:
                full_response = "".join(final_response_content)
                # Store the conversational turn in the conversation memory
                await conversation_memory.upsert(thread_id, user_message=message, ai_reply=full_response)

            logger.info(f"Streaming finished for thread {thread_id}. Checkpointer has saved the state.")

        except Exception as e:
            logger.error(f"Error during agent streaming for thread {thread_id}: {e}", exc_info=True)
            yield f"\n--- ERROR ---\n{str(e)}"
        finally:
            if checkpointer and hasattr(checkpointer.conn, 'close'):
                await checkpointer.conn.close()

def create_agent() -> PhysicsBotAgent:
    return PhysicsBotAgent()
