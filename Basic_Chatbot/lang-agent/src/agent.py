import asyncio
import logging
import json
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

        # LLM setup
        self.llm = ChatGoogleGenerativeAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            timeout=Config.TIMEOUT,
            max_retries=Config.MAX_RETRIES,
            api_key=Config.GEMINI_API_KEY,
        )

        # Tool setup
        self.tools = create_tools(Config.BRAVE_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # System Prompts
        self.think_prompt_template = """
You must always operate in **two explicit phases**: THINK → ANSWER.

======================
PHASE 1: THINKING
======================
When the system requests THINK MODE, you must:
- Carefully reason step by step.
- Write your thought process in 3–7 concise bullet points.
- Mention which retrieved knowledge (lecture/section/page) you’ll use.
- If tools were used, briefly note their role.
- Highlight assumptions or gaps.
- DO NOT give the final answer here.

STRICT FORMAT:
<THINK>
- bullet 1
- bullet 2
...
</THINK>

Rules:
- Max 120 words inside <THINK>.
- This block is safe to show to the user.
- No extra text outside <THINK>.

### Context:
{physics_context}
{conversation_context}
"""

        self.answer_prompt_template = """
======================
PHASE 2: FINAL ANSWER
======================
When the system requests ANSWER MODE, you must:
- Use your own prior THINK plan.
- Combine with retrieved physics knowledge.
- Give a clear, well-structured explanation in Markdown.
- Use headings (##, ###), lists, and LaTeX equations (`$...$`).
- If insufficient info, explicitly say so, then use best-effort reasoning.
- Reference book metadata (if available) without fabricating.

STRICT FORMAT:
<ANSWER>
[final answer here]
</ANSWER>

Rules:
- No raw tool dumps; summarize tool results.
- Do not include your hidden chain-of-thought reasoning.
- Only output inside <ANSWER>.

### Prior Thought Process:
{thinking_phase_output}

### Context:
{physics_context}
{conversation_context}
"""

        # Graph
        self.graph_definition = self._build_graph_definition()

        logger.info("✅ PhysicsBotAgent initialized.")

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
        
        response = self.llm_with_tools.invoke(messages_to_send)
        
        tool_calls = response.tool_calls or []

        return {
            "messages": [response],
            "tool_calls": tool_calls
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

        # Always send: [SystemMessage(prompt), HumanMessage(user), AIMessage(thinking)]
        messages_to_send = [
            SystemMessage(content=prompt)
        ]
        if original_user_message:
            messages_to_send.append(HumanMessage(content=original_user_message))
        if thinking_phase_output:
            messages_to_send.append(AIMessage(content=thinking_phase_output))

        response = self.llm.invoke(messages_to_send)
        # Gemini sometimes returns a list, sometimes a string
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
        final_answer = match.group(0) if match else answer_text
        logger.info(f"Final Answer: {final_answer}")
        return {"final_answer": final_answer}

    def _should_call_tools(self, state: AgentState) -> str:
        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "continue"
        return "end"

    # ---------- INVOCATION ---------- #
    async def stream(self, message: str, thread_id: str):
        """Streaming response generator that yields structured JSON data."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpointer = await memory_manager.get_async_checkpointer(thread_id)
        graph = self.graph_definition.compile(checkpointer=checkpointer)

        try:
            # Yield the thread_id first, so the client can use it immediately
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
            yield {"type": "error", "content": str(e), "thread_id": thread_id}
        finally:
            if checkpointer and hasattr(checkpointer.conn, "close"):
                await checkpointer.conn.close()


# ---------- FACTORY ---------- #
def create_agent() -> PhysicsBotAgent:
    return PhysicsBotAgent()
