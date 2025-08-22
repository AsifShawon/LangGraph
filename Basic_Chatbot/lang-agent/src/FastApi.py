import uvicorn
import uuid
import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from agent import create_agent, PhysicsBotAgent
from langchain_core.messages import BaseMessage
from memory_manager import memory_manager

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

class MessageModel(BaseModel):
    type: str
    content: str

class ConversationHistory(BaseModel):
    thread_id: str
    messages: List[MessageModel]

# --- FastAPI Application Setup ---
logging.getLogger("httpx").setLevel(logging.WARNING)
app = FastAPI(
    title="Smart Memory Agent API",
    description="API for an agent with persistent, context-aware memory.",
    version="2.0.0",
)

agent_executor: Optional[PhysicsBotAgent] = None

@app.on_event("startup")
async def startup_event():
    global agent_executor
    try:
        print("🔧 Initializing agent...")
        agent_executor = create_agent()
        print("✅ Agent has been successfully initialized.")
    except Exception as e:
        print(f"❌ FATAL: Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - let the server start but mark agent as unavailable
        agent_executor = None

@app.on_event("shutdown") 
async def shutdown_event():
    print("🛑 Server shutting down...")

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint", tags=["Status"])
def read_root():
    return {"status": "online", "message": "Welcome to the Smart Memory Agent API!"}

@app.get("/chats", response_model=List[str], summary="List All Conversations", tags=["Memory"])
async def list_all_chats():
    return memory_manager.list_conversations()

@app.get("/chats/{thread_id}", response_model=ConversationHistory, summary="Get Conversation History", tags=["Memory"])
async def get_chat_history(thread_id: str):
    checkpointer = memory_manager.get_checkpointer(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    conversation = checkpointer.get(config)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = [
        MessageModel(type=msg.__class__.__name__, content=msg.content)
        for msg in conversation.get("channel_values", {}).get("messages", [])
    ]
    return ConversationHistory(thread_id=thread_id, messages=messages)

@app.post("/chat", summary="Streaming Chat Endpoint", tags=["Chat"])
async def stream_chat_with_agent(request: ChatRequest):
    """
    Handles a user's message and streams the agent's response back as JSON objects.
    It will stream the THINKING phase, then the FINAL ANSWER using Server-Sent Events.
    """
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agent is not initialized. Please check server logs for initialization errors.")

    thread_id = request.thread_id or str(uuid.uuid4())

    async def response_generator():
        """A generator that yields JSON objects from the agent's async stream."""
        try:
            async for chunk in agent_executor.stream(request.message, thread_id=thread_id):
                # Each chunk is a dictionary, convert it to a JSON string
                # and format for Server-Sent Events (SSE)
                yield f"data: {json.dumps(chunk)}\n\n"
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logging.info(f"Request cancelled for thread {thread_id}")
            error_payload = {"type": "error", "content": "Request was cancelled", "thread_id": thread_id}
            yield f"data: {json.dumps(error_payload)}\n\n"
        except Exception as e:
            logging.error(f"Error during agent streaming for thread {thread_id}: {e}")
            error_payload = {"type": "error", "content": str(e), "thread_id": thread_id}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
