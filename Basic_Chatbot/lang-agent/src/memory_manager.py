import sqlite3
import aiosqlite
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import Config

class MemoryManager:
    """
    Manages persistent memory for conversations using SQLite for checkpoints.
    The vector store logic has been removed in favor of summarization.
    """
    def __init__(self):
        self.memory_path = Path(Config.MEMORY_PATH)
        self.memory_path.mkdir(exist_ok=True)
        print(f"MemoryManager initialized. Storing checkpoint data in: {self.memory_path.resolve()}")

    def _get_db_path(self, thread_id: str) -> Path:
        """Returns the path to the SQLite database for a given thread."""
        return self.memory_path / f"{thread_id}.db"

    def get_checkpointer(self, thread_id: str) -> SqliteSaver:
        """
        Returns a SqliteSaver instance for the given thread_id.
        A new database file is created if one doesn't exist.
        """
        db_path = self._get_db_path(thread_id)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn=conn)

    async def get_async_checkpointer(self, thread_id: str) -> AsyncSqliteSaver:
        """
        Returns an AsyncSqliteSaver instance for the given thread_id.
        """
        db_path = self._get_db_path(thread_id)
        conn = await aiosqlite.connect(db_path)
        return AsyncSqliteSaver(conn=conn)

    def list_conversations(self) -> list[str]:
        """Lists all conversation thread_ids based on saved .db files."""
        return [f.stem for f in self.memory_path.glob("*.db")]

# Singleton instance to be used across the application
memory_manager = MemoryManager()
