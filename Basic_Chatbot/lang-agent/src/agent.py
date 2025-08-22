import asyncio
import logging
import json
import time
import re
from typing import Sequence, TypedDict, Annotated, List, Any, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
# from langchain_groq import ChatGroq
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
from memory_utils import (
    get_short_term_context,
    get_long_term_context,
    upsert_turn_async
)

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

        # LLM setup with Groq
        # self.llm = ChatGroq(
        #     model=Config.DEFAULT_MODEL,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     timeout=Config.TIMEOUT,
        #     max_retries=0,  # We'll handle retries manually
        #     api_key=Config.GROQ_API_KEY,
        # )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Lower temperature for more consistent calculations
            max_tokens=Config.MAX_TOKENS,
            timeout=Config.TIMEOUT,
            max_retries=0,  # We'll handle retries manually
            api_key=Config.GEMINI_API_KEY,
        )

        # API rate limiting settings
        self.max_retries = 3
        self.base_wait_time = 15  # Increased wait time for quota errors
        self.last_api_call = 0
        self.min_time_between_calls = 8  # Increased time between calls to avoid rate limits

        # Tool setup
        self.tools = create_tools(Config.BRAVE_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Two-Phase System Prompts
        self.think_prompt_template = """
You are an expert Physics Problem Solver in PHASE 1: PLANNING.

PHASE 1: THINKING & PLANNING
Your job is to PLAN how to solve the user's question. DO NOT provide the final answer yet.

Analyze:
1. What is the user asking?
2. What information do I need?
3. Should I search for more information?
4. What approach will I take in the final answer?

FORMAT:
<THINK>
- Question: [What is being asked?]
- Approach: [How will I solve this?]
- Information Needed: [What do I need to know?]
- Tools: [Should I search for information?]
</THINK>

DO NOT give the final answer here. Only plan and use tools if needed.

Context:
{physics_context}
{conversation_context}
"""

        self.answer_prompt_template = """
You are an expert Physics Problem Solver in PHASE 2: FINAL ANSWER.

PHASE 2: FINAL ANSWER
Now provide the complete final answer based on your planning and available context.

For physics calculations:
- Show step-by-step work with numbers
- Include formulas and units
- Give clear final answer

For concepts:
- Provide thorough explanations
- Use proper scientific terms
- Include examples when helpful

FORMAT:
<ANSWER>
[Your complete final answer here - this is what the user sees]
</ANSWER>

Your Planning:
{thinking_phase_output}

Context:
{physics_context}
{conversation_context}
"""

        # Graph
        self.graph_definition = self._build_graph_definition()

        logger.info("✅ Enhanced PhysicsBotAgent initialized.")

    # ---------- API RATE LIMITING ---------- #
    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_time_between_calls:
            wait_time = self.min_time_between_calls - time_since_last_call
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_api_call = time.time()

    def _extract_retry_delay(self, error_message: str) -> int:
        """Extract retry delay from API error message"""
        # Look for retry_delay in error message
        match = re.search(r'retry_delay.*?seconds: (\d+)', str(error_message))
        if match:
            return int(match.group(1))
        return self.base_wait_time

    async def _call_llm_with_retry(self, messages, use_tools=False, retry_count=0):
        """Call LLM with intelligent retry logic"""
        await self._wait_for_rate_limit()
        
        try:
            if use_tools:
                response = self.llm_with_tools.invoke(messages)
            else:
                response = self.llm.invoke(messages)
            
            # Check for empty response
            if not response or not response.content:
                logger.warning(f"Received empty response from LLM (retry {retry_count})")
                if retry_count < self.max_retries:
                    wait_time = self.base_wait_time * (retry_count + 1)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    return await self._call_llm_with_retry(messages, use_tools, retry_count + 1)
                else:
                    # Create a fallback response
                    from langchain_core.messages import AIMessage
                    fallback_content = "I apologize, but I'm experiencing temporary issues. Please try asking your question again."
                    return AIMessage(content=fallback_content)
            
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a quota/rate limit error
            if any(keyword in error_str for keyword in ['quota', '429', 'rate limit', 'exceeded']):
                if retry_count < self.max_retries:
                    # Extract suggested wait time from error or use default
                    wait_time = self._extract_retry_delay(str(e))
                    logger.warning(f"API quota exceeded. Waiting {wait_time} seconds before retry {retry_count + 1}/{self.max_retries}")
                    await asyncio.sleep(wait_time)
                    return await self._call_llm_with_retry(messages, use_tools, retry_count + 1)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for API quota")
                    raise e
            else:
                # Non-quota error, don't retry
                raise e

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

        short_term = get_short_term_context(conversation_memory, thread_id, user_msg.content, k=3)  # Reduced from 5
        long_term = get_long_term_context(conversation_memory, user_msg.content, k=5)  # Reduced from 8
        
        # Truncate contexts to prevent overly long prompts
        max_context_length = 1000  # characters
        if short_term and len(short_term) > max_context_length:
            short_term = short_term[:max_context_length] + "..."
        if long_term and len(long_term) > max_context_length:
            long_term = long_term[:max_context_length] + "..."
            
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
        
        # Truncate physics context as well
        if len(physics_text) > max_context_length:
            physics_text = physics_text[:max_context_length] + "..."

        return physics_text, conversation_context_text

    def _think_node(self, state: AgentState) -> AgentState:
        physics_context, conversation_context = self._get_context(state)
        
        prompt = self.think_prompt_template.format(
            physics_context=physics_context,
            conversation_context=conversation_context
        )
        
        messages_to_send = [SystemMessage(content=prompt)] + state["messages"]
        
        try:
            # Use async call with retry logic
            import asyncio
            response = asyncio.run(self._call_llm_with_retry(messages_to_send, use_tools=True))
            tool_calls = response.tool_calls or []
            return {
                "messages": [response],
                "tool_calls": tool_calls
            }
        except Exception as e:
            logger.warning(f"Think node error after retries: {e}")
            # Create a fallback response that encourages calculation
            user_question = state["messages"][-1].content if state["messages"] else ""
            fallback_content = f"""<THINK>
- Problem: {user_question}
- Approach: Will solve step-by-step using physics principles
- Note: Encountered API limitation after retries, providing fallback solution
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
            # Use async call with retry logic
            import asyncio
            response = asyncio.run(self._call_llm_with_retry(messages_to_send, use_tools=False))
            
            # Debug logging for response
            logger.info(f"Raw response type: {type(response)}")
            logger.info(f"Raw response content: {response.content}")
            
            # Enhanced response processing
            answer_text = ""
            if isinstance(response.content, list):
                # Handle list content (multimodal responses)
                text_parts = [item for item in response.content if isinstance(item, str)]
                answer_text = "".join(text_parts)
            elif isinstance(response.content, str):
                answer_text = response.content
            else:
                answer_text = str(response.content) if response.content else ""
            
            # Check if we got an empty response
            if not answer_text.strip():
                logger.warning("Received empty response from LLM")
                answer_text = "I apologize, but I received an empty response. Let me try to answer based on the available context."
                if thinking_phase_output:
                    # Try to extract useful information from thinking phase
                    answer_text += f"\n\nBased on my analysis: {thinking_phase_output}"
            
            # Extract only the text inside <ANSWER>...</ANSWER> or wrap if not present
            import re
            match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", answer_text)
            if match:
                answer_content = match.group(1).strip()
                if answer_content:  # If there's content inside the tags
                    final_answer = f"<ANSWER>\n{answer_content}\n</ANSWER>"
                else:
                    # If the ANSWER tags are empty, check if there's content before them
                    before_answer = answer_text.split("<ANSWER>")[0].strip()
                    if before_answer:
                        final_answer = f"<ANSWER>\n{before_answer}\n</ANSWER>"
                    else:
                        final_answer = f"<ANSWER>\n{answer_text.strip()}\n</ANSWER>"
            else:
                # Wrap the response in ANSWER tags if not present
                final_answer = f"<ANSWER>\n{answer_text.strip()}\n</ANSWER>"
            
            logger.info(f"Final Answer: {final_answer}")
            return {"final_answer": final_answer}
            
        except Exception as e:
            logger.error(f"Answer node error after retries: {e}")
            # Provide intelligent fallback based on question type
            fallback_answer = self._create_fallback_answer(original_user_message, thinking_phase_output)
            return {"final_answer": fallback_answer}

    def _create_fallback_answer(self, question: str, thinking: str) -> str:
        """Create an intelligent fallback answer when API fails"""
        if not question:
            return "<ANSWER>\nI encountered an API error. Please try again.\n</ANSWER>"
        
        # Try to extract key information from the question for a basic response
        question_lower = question.lower()
        
        # Physics constants
        if "gravitational acceleration" in question_lower or "g =" in question_lower:
            return "<ANSWER>\n**Answer: 9.8 m/s²**\n\nThe gravitational acceleration on Earth near the surface is approximately 9.8 m/s².\n</ANSWER>"
        
        elif "speed of light" in question_lower:
            return "<ANSWER>\n**Answer: 3.0 × 10⁸ m/s**\n\nThe speed of light in vacuum is approximately 3.0 × 10⁸ m/s.\n</ANSWER>"
        
        # Simple calculations we can do without API
        elif "force" in question_lower and "mass" in question_lower and "acceleration" in question_lower:
            # Try to extract numbers for F = ma
            import re
            masses = re.findall(r'(\d+(?:\.\d+)?)\s*kg', question)
            accelerations = re.findall(r'(\d+(?:\.\d+)?)\s*m/s', question)
            
            if masses and accelerations:
                try:
                    mass = float(masses[0])
                    accel = float(accelerations[0])
                    force = mass * accel
                    return f"<ANSWER>\n**Answer: {force} N**\n\nUsing Newton's second law: F = ma = {mass} kg × {accel} m/s² = {force} N\n</ANSWER>"
                except:
                    pass
        
        elif "kinetic energy" in question_lower:
            # Try to extract mass and height for mgh calculation
            masses = re.findall(r'(\d+(?:\.\d+)?)\s*kg', question)
            heights = re.findall(r'(\d+(?:\.\d+)?)\s*m(?:eter)?', question)
            
            if masses and heights:
                try:
                    mass = float(masses[0])
                    height = float(heights[0])
                    # Use energy conservation: PE = KE = mgh
                    g = 9.8
                    ke = mass * g * height
                    return f"<ANSWER>\n**Answer: {ke} J**\n\nUsing energy conservation: KE = PE = mgh = {mass} kg × 9.8 m/s² × {height} m = {ke} J\n</ANSWER>"
                except:
                    pass
        
        elif "spring" in question_lower and "potential energy" in question_lower:
            # Try to extract k and x for (1/2)kx²
            k_matches = re.findall(r'k\s*=\s*(\d+(?:\.\d+)?)', question)
            x_matches = re.findall(r'(\d+(?:\.\d+)?)\s*m', question)
            
            if k_matches and x_matches:
                try:
                    k = float(k_matches[0])
                    x = float(x_matches[0])
                    pe = 0.5 * k * x * x
                    return f"<ANSWER>\n**Answer: {pe} J**\n\nUsing spring potential energy formula: PE = ½kx² = ½ × {k} N/m × ({x} m)² = {pe} J\n</ANSWER>"
                except:
                    pass
        
        # Physics concepts
        elif "newton" in question_lower and "second law" in question_lower:
            return "<ANSWER>\nNewton's second law states that the net force acting on an object equals the mass of the object times its acceleration: F = ma.\n</ANSWER>"
        
        elif "speed" in question_lower and "velocity" in question_lower and "difference" in question_lower:
            return "<ANSWER>\nSpeed is a scalar quantity that measures how fast an object moves, while velocity is a vector quantity that includes both speed and direction.\n</ANSWER>"
        
        elif "first law" in question_lower and "thermodynamics" in question_lower:
            return "<ANSWER>\nThe first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another. In equation form: ΔU = Q - W, where ΔU is the change in internal energy, Q is heat added to the system, and W is work done by the system.\n</ANSWER>"
        
        elif "frequency" in question_lower and "wavelength" in question_lower:
            return "<ANSWER>\nFrequency and wavelength are inversely proportional for electromagnetic waves. Their relationship is given by c = fλ, where c is the speed of light, f is frequency, and λ is wavelength. As frequency increases, wavelength decreases.\n</ANSWER>"
        
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

        final_answer = ""
        
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
                    final_answer = content
                    yield {"type": "answer", "content": content, "thread_id": thread_id}
            
            # Store the conversation turn in memory after successful completion
            if final_answer:
                try:
                    # Extract clean answer from the <ANSWER> tags
                    import re
                    match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", final_answer)
                    clean_answer = match.group(1).strip() if match else final_answer
                    
                    # Store in conversation memory
                    await upsert_turn_async(conversation_memory, thread_id, message, clean_answer)
                    logger.info(f"Stored conversation turn for thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Failed to store conversation turn: {e}")
                    
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
