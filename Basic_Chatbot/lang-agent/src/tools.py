from langchain_community.tools import BraveSearch
from langchain_core.tools import tool
from typing import List, Dict, Any
import json
import requests
from datetime import datetime

class BraveSearchTool:
    """Wrapper for Brave Search with enhanced functionality"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_tool = BraveSearch.from_api_key(api_key)
    
    def get_search_tool(self):
        """Get the LangChain-compatible Brave Search tool"""
        return self.search_tool

# Custom tool for getting current time
@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    Returns:
        str: Current date and time in ISO format
    """
    return f"Current time: {datetime.now().isoformat()}"

# Custom tool for weather information (example)
@tool
def get_weather(location: str) -> str:
    """
    Get weather information for a specific location.
    Args:
        location (str): The location to get weather for (e.g., "New York, NY")
    Returns:
        str: Weather information or error message
    """
    try:
        # This is a mock implementation - in real usage, you'd call a weather API
        return f"Weather information for {location}: Partly cloudy, 72°F"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

def create_tools(api_key: str) -> List[Any]:
    """Create and return all available tools"""
    brave_tool = BraveSearchTool(api_key)
    
    return [
        brave_tool.get_search_tool(),
        get_current_time,
        get_weather,
    ]