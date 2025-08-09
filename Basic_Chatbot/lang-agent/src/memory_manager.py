# memory_manager.py

import os
import sqlite3
import aiosqlite  # <-- Add this import
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # <-- Add this import

from config import Config

class MemoryManager:
    """
    Manages persistent memory for conversations using SQLite for checkpoints
    and FAISS for long-term vector-based retrieval.
    """
    def __init__(self):
        self.memory_path = Path(Config.MEMORY_PATH)
        self.memory_path.mkdir(exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        print(f"MemoryManager initialized. Storing data in: {self.memory_path.resolve()}")

    def _get_db_path(self, thread_id: str) -> Path:
        """Returns the path to the SQLite database for a given thread."""
        return self.memory_path / f"{thread_id}.db"

    def _get_faiss_path(self, thread_id: str) -> Path:
        """Returns the path to the FAISS index for a given thread."""
        return self.memory_path / f"{thread_id}.faiss"

    def get_checkpointer(self, thread_id: str) -> SqliteSaver:
        """
        Returns a SqliteSaver instance for the given thread_id.
        A new database file is created if one doesn't exist.
        """
        db_path = self._get_db_path(thread_id)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn=conn)

    # --- NEW: Asynchronous checkpointer for streaming ---
    async def get_async_checkpointer(self, thread_id: str) -> AsyncSqliteSaver:
        """
        Returns an AsyncSqliteSaver instance for the given thread_id.
        """
        db_path = self._get_db_path(thread_id)
        conn = await aiosqlite.connect(db_path)
        return AsyncSqliteSaver(conn=conn)

    def get_retriever(self, thread_id: str):
        """
        Returns a FAISS retriever for the given thread_id.
        Loads from disk if it exists, otherwise returns None.
        """
        faiss_path = self._get_faiss_path(thread_id)
        if faiss_path.exists():
            try:
                vector_store = FAISS.load_local(
                    folder_path=str(self.memory_path),
                    index_name=f"{thread_id}.faiss",
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True # Required for FAISS
                )
                return vector_store.as_retriever(search_kwargs={'k': 3})
            except Exception as e:
                print(f"Error loading FAISS index for {thread_id}: {e}")
                return None
        return None

    def update_vector_store(self, thread_id: str, messages: list[BaseMessage]):
        """
        Updates the FAISS vector store with new messages.
        Creates a new store if one doesn't exist for the thread_id.
        """
        if not messages:
            return

        faiss_path = self._get_faiss_path(thread_id)
        
        # We only want to embed the actual content
        texts_to_embed = [msg.content for msg in messages if hasattr(msg, 'content')]
        if not texts_to_embed:
            return

        if faiss_path.exists():
            try:
                vector_store = FAISS.load_local(
                    folder_path=str(self.memory_path),
                    index_name=f"{thread_id}.faiss",
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_store.add_texts(texts_to_embed)
            except Exception as e:
                print(f"Error updating FAISS index for {thread_id}: {e}")
                return
        else:
            vector_store = FAISS.from_texts(texts_to_embed, embedding=self.embeddings)

        vector_store.save_local(folder_path=str(self.memory_path), index_name=f"{thread_id}.faiss")
        print(f"Updated vector store for thread_id: {thread_id}")

    def list_conversations(self) -> list[str]:
        """Lists all conversation thread_ids based on saved .db files."""
        return [f.stem for f in self.memory_path.glob("*.db")]

# Singleton instance to be used across the application
memory_manager = MemoryManager()