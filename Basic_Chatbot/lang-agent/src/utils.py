from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_messages_for_display(messages: List[BaseMessage]) -> str:
    """Format messages for display purposes"""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            formatted.append(f"Tool: {msg.content}")
    return "\n".join(formatted)

def extract_tool_calls(ai_message: AIMessage) -> List[Dict[str, Any]]:
    """Extract tool calls from AI message"""
    if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        return ai_message.tool_calls
    return []

def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}