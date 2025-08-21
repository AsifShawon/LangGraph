import re
import logging
from typing import Optional, List, Dict, Any
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

logger = logging.getLogger(__name__)


def _short_text_summary(text: str, max_chars: int = 240) -> str:
    # Try to pick the first 1-2 sentences, fallback to truncation
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        summary = " ".join(sentences[:2])
        if len(summary) <= max_chars:
            return summary
    # fallback truncation
    return (text[: max_chars - 3] + "...") if len(text) > max_chars else text


def _summarize_with_llm(user_message: str, ai_reply: str, max_chars: int = 240) -> Optional[str]:
    """Use the Gemini model to create a concise 1-2 sentence summary."""
    api_key = getattr(Config, "GEMINI_API_KEY", None)
    model = getattr(Config, "DEFAULT_MODEL", None)
    if not api_key or not model:
        return None

    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.0, max_tokens=128, api_key=api_key)
        system = SystemMessage(content=(
            "You are a concise summarization assistant.\n"
            "Given a single user/assistant conversation turn, produce a 1-2 sentence summary (max {max_chars} characters) that captures the main intent and any key facts."
        ))
        human = HumanMessage(content=f"User: {user_message}\nAI: {ai_reply}\n\nProvide a short summary:")
        resp = llm.invoke([system, human])
        # resp may be an AIMessage or similar
        text = getattr(resp, 'content', None) or str(resp)
        text = text.strip()
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        return None


def summarize_turn(user_message: str, ai_reply: str, max_chars: int = 240) -> str:
    """Create a short summary for a conversational turn combining user and AI text.

    The summary is intentionally short so it works well as a long-term memory index.
    """
    # Try LLM-based summarization first, fall back to heuristic
    llm_summary = _summarize_with_llm(user_message, ai_reply, max_chars=max_chars)
    if llm_summary:
        return llm_summary

    combined = f"User: {user_message}\nAI: {ai_reply}"
    return _short_text_summary(combined, max_chars=max_chars)


def upsert_turn_sync(conversation_memory, thread_id: str, user_message: str, ai_reply: str) -> Optional[str]:
    """Store both the original turn and a short summary into the vector DB synchronously.

    We store two records:
      - type: "turn"  -> full text (useful for exact retrieval)
      - type: "summary" -> short summary (useful for long-term semantic search)

    Returns the uuid of the summary entry on success, or None on failure.
    """
    if conversation_memory is None:
        logger.debug("No conversation_memory available; skipping upsert_sync")
        return None

    try:
        full_text = f"User: {user_message}\nAI: {ai_reply}"
        summary = summarize_turn(user_message, ai_reply)

        # Upsert full turn (metadata indicates original turn)
        try:
            conversation_memory.upsert_sync(thread_id, user_message=user_message, ai_reply=ai_reply, metadata={"type": "turn"})
        except Exception as e:
            logger.warning(f"Failed to upsert full turn: {e}")

        # Upsert short summary with metadata so long-term searches can prefer these
        try:
            summary_id = conversation_memory.upsert_sync(thread_id, user_message=summary, ai_reply="", metadata={"type": "summary"})
            return summary_id
        except Exception as e:
            logger.warning(f"Failed to upsert summary: {e}")
            return None

    except Exception as e:
        logger.exception(f"Unexpected error in upsert_turn_sync: {e}")
        return None


async def upsert_turn_async(conversation_memory, thread_id: str, user_message: str, ai_reply: str) -> Optional[str]:
    """Async version of upsert_turn_sync."""
    if conversation_memory is None:
        logger.debug("No conversation_memory available; skipping upsert_async")
        return None

    try:
        full_text = f"User: {user_message}\nAI: {ai_reply}"
        # Run LLM summarization in threadpool to avoid blocking
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, summarize_turn, user_message, ai_reply)

        try:
            await conversation_memory.upsert(thread_id, user_message=user_message, ai_reply=ai_reply, metadata={"type": "turn"})
        except Exception as e:
            logger.warning(f"Failed to async upsert full turn: {e}")

        try:
            summary_id = await conversation_memory.upsert(thread_id, user_message=summary, ai_reply="", metadata={"type": "summary"})
            return summary_id
        except Exception as e:
            logger.warning(f"Failed to async upsert summary: {e}")
            return None

    except Exception as e:
        logger.exception(f"Unexpected error in upsert_turn_async: {e}")
        return None


def _combine_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    return "\n\n---\n\n".join([r.get("content", "") for r in results])


def get_short_term_context(conversation_memory, thread_id: str, query: str, k: int = 5) -> str:
    """Retrieve recent/relevant turns for the provided thread.

    Prefers full turns (type==turn) but will return whatever the vector search gives.
    Falls back to empty string when memory isn't available.
    """
    if conversation_memory is None:
        return ""

    try:
        results = conversation_memory.search_sync(query, k=k, thread_id=thread_id)
        return _combine_results(results)
    except Exception as e:
        logger.warning(f"Short-term search failed: {e}")
        return ""


def get_long_term_context(conversation_memory, query: str, k: int = 10) -> str:
    """Retrieve long-term context across threads, preferring summary-type entries.

    This searches globally (no thread_id) and filters for metadata.type=="summary" to prefer compact facts.
    """
    if conversation_memory is None:
        return ""

    try:
        results = conversation_memory.search_sync(query, k=k)
        # Filter for summary-type results if metadata available
        summary_results = [r for r in results if (r.get("metadata") or {}).get("type") == "summary"]
        if summary_results:
            return _combine_results(summary_results)
        # fallback to any results
        return _combine_results(results)
    except Exception as e:
        logger.warning(f"Long-term search failed: {e}")
        return ""
