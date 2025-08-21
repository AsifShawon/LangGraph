import asyncio
import logging
import json
import time
from typing import Sequence, TypedDict, Annotated, List, Any, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import Config
from tools import create_tools
from memory_manager import memory_manager
from memory_utils import (
    get_short_term_context,
    get_long_term_context
)

from conversation_memory import conversation_memory
from retrieve_physics_query import query_physics

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- STATE ---------- #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: List[Dict[str, Any]]
    iteration_count: int
    error_count: int
    last_error: str
    thinking_completed: bool
    final_answer: str


# ---------- AGENT ---------- #
class PhysicsBotAgent:
    def __init__(self):
        Config.validate()

        # LLM setup with enhanced configuration for better performance
        self.llm = ChatGoogleGenerativeAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.1,  # Lower temperature for more consistent calculations
            max_tokens=Config.MAX_TOKENS,
            timeout=Config.TIMEOUT,
            max_retries=Config.MAX_RETRIES,
            api_key=Config.GEMINI_API_KEY,
        )

        # Tool setup
        self.tools = create_tools(Config.BRAVE_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Enhanced System Prompts focused on calculation capability
        self.think_prompt_template = """
You are an expert Physics Problem Solver that MUST provide numerical answers when asked.

======================
PHASE 1: THINKING
======================
You MUST perform calculations step-by-step. You CAN and SHOULD:
- Do arithmetic operations (addition, subtraction, multiplication, division)
- Apply physics formulas directly with given numbers
- Convert units as needed (km/h to m/s, etc.)
- Calculate exact numerical results

For numerical problems, follow this structure:
1. Identify the physics principle/formula needed
2. List all given values with proper units
3. Apply the formula step-by-step with actual numbers
4. Perform the arithmetic calculations
5. State the final numerical result with units

STRICT FORMAT:
<THINK>
- Formula: [physics equation to use]
- Given: [list all values with units]
- Calculate: [show actual arithmetic step by step]
- Result: [final number with units]
</THINK>

CRITICAL: You MUST perform calculations. Do NOT say you cannot calculate.

### Context:
{physics_context}
{conversation_context}
"""

        self.answer_prompt_template = """
======================
PHASE 2: FINAL ANSWER
======================
You MUST provide clear numerical answers for physics problems.

For numerical questions:
1. Start with the exact numerical result: "**Answer: [NUMBER] [UNITS]**"
2. Then provide clear explanation with steps
3. Use proper LaTeX formatting for equations

For conceptual questions:
1. Give concise, accurate explanations
2. Focus on the core physics concept
3. Use proper scientific terminology

STRICT FORMAT:
<ANSWER>
**Answer: [NUMERICAL_RESULT]** (for problems asking for calculations)

[Clear explanation with proper formatting]
</ANSWER>

Remember:
- Always include the numerical result prominently for calculation problems
- Show key steps but be concise
- Use proper units throughout

### Your Prior Calculations:
{thinking_phase_output}

### Context:
{physics_context}
{conversation_context}
"""

        # Graph
        self.graph_definition = self._build_graph_definition()

        logger.info("✅ Enhanced PhysicsBotAgent initialized.")

    # ---------- GRAPH BUILD ---------- #
    def _build_graph_definition(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("think", self._think_node)
        graph.add_node("answer", self._answer_node)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "think")
        graph.add_edge("tools", "answer")
        graph.add_conditional_edges(
            "think",
            self._should_call_tools,
            {"continue": "tools", "end": "answer"},
        )
        graph.add_edge("answer", END)
        return graph

    # ---------- NODES ---------- #
    def _get_context(self, state: AgentState):
        history = state.get("messages", [])
        user_msg = history[-1]
        thread_id = state.get("config", {}).get("thread_id", "global")

        short_term = get_short_term_context(conversation_memory, thread_id, user_msg.content, k=5)
        long_term = get_long_term_context(conversation_memory, user_msg.content, k=8)
        conversation_context_text = short_term
        if long_term:
            conversation_context_text = (conversation_context_text + "\n\n---LONG TERM---\n\n" + long_term) if conversation_context_text else long_term

        physics_context = query_physics(user_msg.content)
        if isinstance(physics_context, str):
            physics_text = physics_context
        else:
            physics_text = "\n\n---\n\n".join(
                r.get("content") if isinstance(r, dict) else str(r)
                for r in physics_context
            )
        if not physics_text.strip():
            physics_text = "No relevant physics knowledge found."

        return physics_text, conversation_context_text

    def _think_node(self, state: AgentState) -> AgentState:
        physics_context, conversation_context = self._get_context(state)
        
        prompt = self.think_prompt_template.format(
            physics_context=physics_context,
            conversation_context=conversation_context
        )
        
        messages_to_send = [SystemMessage(content=prompt)] + state["messages"]
        
        try:
            response = self.llm_with_tools.invoke(messages_to_send)
            tool_calls = response.tool_calls or []
            return {
                "messages": [response],
                "tool_calls": tool_calls
            }
        except Exception as e:
            logger.warning(f"Think node error (likely API quota): {e}")
            # Create a fallback response that encourages calculation
            user_question = state["messages"][-1].content if state["messages"] else ""
            fallback_content = f"""<THINK>
- Problem: {user_question}
- Approach: Will solve step-by-step using physics principles
- Note: Encountered API limitation but will provide best solution
</THINK>
I'll solve this physics problem step by step."""
            
            fallback_response = AIMessage(content=fallback_content)
            return {
                "messages": [fallback_response],
                "tool_calls": []
            }

    def _answer_node(self, state: AgentState) -> AgentState:
        # Extract THINK phase output and original user message
        thinking_phase_output = ""
        original_user_message = ""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                original_user_message = msg.content
            if isinstance(msg, AIMessage):
                thinking_phase_output = msg.content

        physics_context, conversation_context = self._get_context(state)

        logger.info(f"Thinking Phase Output: {thinking_phase_output}")
        prompt = self.answer_prompt_template.format(
            thinking_phase_output=thinking_phase_output,
            physics_context=physics_context,
            conversation_context=conversation_context
        )

        messages_to_send = [
            SystemMessage(content=prompt)
        ]
        if original_user_message:
            messages_to_send.append(HumanMessage(content=original_user_message))
        if thinking_phase_output:
            messages_to_send.append(AIMessage(content=thinking_phase_output))

        try:
            response = self.llm.invoke(messages_to_send)
            
            # Enhanced response processing
            answer_text = ""
            if isinstance(response.content, list):
                # Find the first string containing <ANSWER>
                for item in response.content:
                    if isinstance(item, str) and "<ANSWER>" in item:
                        answer_text = item
                        break
            elif isinstance(response.content, str):
                answer_text = response.content
            
            # Extract only the text inside <ANSWER>...</ANSWER>
            import re
            match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", answer_text)
            final_answer = match.group(0) if match else f"<ANSWER>\n{answer_text}\n</ANSWER>"
            
            logger.info(f"Final Answer: {final_answer}")
            return {"final_answer": final_answer}
            
        except Exception as e:
            logger.error(f"Answer node error (likely API quota): {e}")
            # Provide intelligent fallback based on question type
            fallback_answer = self._create_fallback_answer(original_user_message, thinking_phase_output)
            return {"final_answer": fallback_answer}

    def _create_fallback_answer(self, question: str, thinking: str) -> str:
        """Create an intelligent fallback answer when API fails"""
        if not question:
            return "<ANSWER>\nI encountered an API error. Please try again.\n</ANSWER>"
        
        # Try to extract key information from the question for a basic response
        question_lower = question.lower()
        
        if "gravitational acceleration" in question_lower or "g =" in question_lower:
            return "<ANSWER>\n**Answer: 9.8 m/s²**\n\nThe gravitational acceleration on Earth near the surface is approximately 9.8 m/s².\n</ANSWER>"
        
        elif "speed of light" in question_lower:
            return "<ANSWER>\n**Answer: 3.0 × 10⁸ m/s**\n\nThe speed of light in vacuum is approximately 3.0 × 10⁸ m/s.\n</ANSWER>"
        
        elif "newton" in question_lower and "second law" in question_lower:
            return "<ANSWER>\nNewton's second law states that the net force acting on an object equals the mass of the object times its acceleration: F = ma.\n</ANSWER>"
        
        else:
            return f"<ANSWER>\nI encountered an API limitation while processing: '{question}'\n\nPlease try again in a moment as this may be due to temporary rate limiting.\n</ANSWER>"

    def _should_call_tools(self, state: AgentState) -> str:
        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "continue"
        return "end"

    # ---------- INVOCATION ---------- #
    async def stream(self, message: str, thread_id: str):
        """Enhanced streaming response generator with better error handling."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpointer = await memory_manager.get_async_checkpointer(thread_id)
        graph = self.graph_definition.compile(checkpointer=checkpointer)

        try:
            # Yield the thread_id first
            yield {"type": "thread_id", "thread_id": thread_id}

            async for event in graph.astream(
                {"messages": [HumanMessage(content=message)]}, config,
            ):
                if "think" in event:
                    content = event["think"]["messages"][-1].content
                    yield {"type": "thinking", "content": content, "thread_id": thread_id}
                if "answer" in event:
                    content = event["answer"]["final_answer"]
                    yield {"type": "answer", "content": content, "thread_id": thread_id}
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            # Provide more informative error messages
            if "quota" in str(e).lower() or "429" in str(e):
                error_msg = "API quota exceeded. Please wait a moment and try again."
            else:
                error_msg = f"Error processing request: {str(e)}"
            yield {"type": "error", "content": error_msg, "thread_id": thread_id}
        finally:
            if checkpointer and hasattr(checkpointer.conn, "close"):
                await checkpointer.conn.close()


# ---------- FACTORY ---------- #
def create_agent() -> PhysicsBotAgent:
    return PhysicsBotAgent()
