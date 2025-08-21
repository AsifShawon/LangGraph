import uvicorn
import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

# --- MODIFIED: Import the correct agent class ---
from agent import create_agent, PhysicsBotAgent
from memory_manager import memory_manager

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

# --- FastAPI Application Setup ---
logging.getLogger("httpx").setLevel(logging.WARNING)
app = FastAPI(
    title="Physics Bot API",
    description="API for a specialized physics agent with RAG.",
    version="1.0.0",
)

# --- MODIFIED: Update the type hint for the agent executor ---
agent_executor: Optional[PhysicsBotAgent] = None

@app.on_event("startup")
def startup_event():
    global agent_executor
    try:
        agent_executor = create_agent()
        print("✅ Physics Bot Agent has been successfully initialized.")
    except Exception as e:
        print(f"❌ FATAL: Error initializing agent: {e}")
        raise e

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint", tags=["Status"])
def read_root():
    return {"status": "online", "message": "Welcome to the Physics Bot API!"}

@app.get("/chats", response_model=List[str], summary="List All Conversations", tags=["Memory"])
async def list_all_chats():
    return memory_manager.list_conversations()

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
