# LangGraphAgent Usage Guide

This guide shows you how to call and run the LangGraphAgent.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file in the `lang-agent` directory with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   BRAVE_API_KEY=your_brave_api_key_here
   ```

## How to Run the Agent

### Method 1: Interactive Chat (Recommended)
```bash
python interactive_chat.py
```
This starts an interactive chat session where you can have a conversation with the agent.

### Method 2: Run Examples
```bash
python run_agent_example.py
```
This runs various examples showing different ways to use the agent.

### Method 3: Direct Usage in Python
```python
import sys
import os
sys.path.append('src')
from agent import create_agent

# Create the agent
agent = create_agent()

# Ask a question
response = agent.invoke("What is the weather in Tokyo?")
print(response["response"])
```

## Available Methods

### 1. Basic Invocation
```python
response = agent.invoke("Your question here")
if response["success"]:
    print(response["response"])
else:
    print(response["error"])
```

### 2. Thread-based Conversations
```python
# Use different thread IDs to maintain separate conversations
response1 = agent.invoke("What is the capital of France?", thread_id="user_123")
response2 = agent.invoke("What is its population?", thread_id="user_123")
```

### 3. Streaming Responses
```python
for event in agent.stream("Tell me about AI"):
    if "messages" in event and event["messages"]:
        latest_message = event["messages"][-1]
        print(latest_message.content)
```

### 4. Error Handling
The agent includes built-in error handling:
- Maximum iteration limits
- Error recovery
- Graph recursion protection

## Configuration

You can modify the agent behavior in `src/config.py`:
- `DEFAULT_MODEL`: Change the LLM model
- `TEMPERATURE`: Adjust creativity (0.0-1.0)
- `MAX_ITERATIONS`: Limit tool usage iterations
- `TIMEOUT`: Set request timeout

## Troubleshooting

1. **API Key Errors**: Make sure your `.env` file has the correct API keys
2. **Import Errors**: Ensure you're running from the correct directory
3. **Network Issues**: Check your internet connection for API calls

## Example Usage

```python
from agent import create_agent

agent = create_agent()

# Simple question
response = agent.invoke("What's the latest news about AI?")

# Multi-turn conversation
thread_id = "my_conversation"
response1 = agent.invoke("What is machine learning?", thread_id=thread_id)
response2 = agent.invoke("How does it differ from deep learning?", thread_id=thread_id)

# Streaming for real-time responses
for event in agent.stream("Explain quantum computing"):
    # Process streaming events
    pass
``` 