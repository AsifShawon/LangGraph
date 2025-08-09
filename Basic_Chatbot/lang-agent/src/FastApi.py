import uvicorn
import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from agent import create_agent, LangGraphAgent
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

agent_executor: Optional[LangGraphAgent] = None

@app.on_event("startup")
def startup_event():
    global agent_executor
    try:
        agent_executor = create_agent()
        print("✅ Agent has been successfully initialized.")
    except Exception as e:
        print(f"❌ FATAL: Error initializing agent: {e}")
        raise e

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint", tags=["Status"])
def read_root():
    return {"status": "online", "message": "Welcome to the Smart Memory Agent API!"}

@app.get("/chats", response_model=List[str], summary="List All Conversations", tags=["Memory"])
async def list_all_chats():
    return memory_manager.list_conversations()

@app.get("/chats/{thread_id}", response_model=ConversationHistory, summary="Get Conversation History", tags=["Memory"])
async def get_chat_history(thread_id: str):
    print("Therad_id", thread_id)
    checkpointer = memory_manager.get_checkpointer(thread_id)
    print("Checkpointer", checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    conversation = checkpointer.get(config)
    print("Conversation", conversation)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # --- FIXED: Access messages via the 'channel_values' attribute ---
    messages = [
        MessageModel(type=msg.__class__.__name__, content=msg.content)
        for msg in conversation.get("channel_values", {}).get("messages", [])
    ]
    return ConversationHistory(thread_id=thread_id, messages=messages)

@app.post("/chat", response_model=ChatResponse, summary="Standard Chat Endpoint", tags=["Chat"])
async def chat_with_agent(request: ChatRequest):
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        result = agent_executor.invoke(request.message, thread_id=thread_id)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Agent failed."))
        return ChatResponse(reply=result.get("response", ""), thread_id=thread_id)
    except Exception as e:
        logging.error(f"Error during agent execution for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- UPDATED: Streaming endpoint for real-time responses ---
@app.post("/chat/stream", summary="Streaming Chat Endpoint", tags=["Chat"])
async def stream_chat_with_agent(request: ChatRequest):
    """
    Handles a user's message and streams the agent's response back token by token.
    """
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")

    thread_id = request.thread_id or str(uuid.uuid4())

    async def response_generator():
        """A generator that yields tokens from the agent's async stream."""
        try:
            async for token in agent_executor.stream(request.message, thread_id=thread_id):
                yield token
        except Exception as e:
            logging.error(f"Error during agent streaming for thread {thread_id}: {e}")
            yield f"\n\n--- ERROR ---\n{str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)