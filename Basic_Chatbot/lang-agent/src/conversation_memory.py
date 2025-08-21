import os
import json
import time
import uuid
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_ollama import OllamaEmbeddings
import faiss

class ConversationMemory:
    """
    A class to manage conversation memory using FAISS for vector search and SQLite for metadata storage.
    """
    def __init__(self, base_dir: str, model_name: str = "nomic-embed-text:v1.5"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "faiss.index"
        self.meta_db_path = self.base_dir / "meta.db"
        self.model = OllamaEmbeddings(model=model_name)

        # Dynamically determine embedding dimension
        sample_embedding = self.model.embed_query("test")
        self.dim = len(sample_embedding)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dim)
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as e:
                print(f"[ConversationMemory] Failed to load FAISS index: {e}. Using a new index.")

        # Initialize SQLite database
        self._init_meta_db()

    def _init_meta_db(self):
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                uuid TEXT UNIQUE,
                content TEXT,
                metadata TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Insert or update a memory entry."""
        metadata = metadata or {}
        embedding = self.model.embed_query(content)
        faiss.normalize_L2(embedding.reshape(1, -1))

        # Add to FAISS index
        uuid_str = str(uuid.uuid4())
        self.index.add_with_ids(embedding.reshape(1, -1), [int(uuid_str[:8], 16)])
        faiss.write_index(self.index, str(self.index_path))

        # Add to SQLite database
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memory (uuid, content, metadata) VALUES (?, ?, ?)",
            (uuid_str, content, json.dumps(metadata))
        )
        conn.commit()
        conn.close()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories using FAISS."""
        embedding = self.model.embed_query(query)
        faiss.normalize_L2(embedding.reshape(1, -1))
        distances, indices = self.index.search(embedding.reshape(1, -1), k)

        # Retrieve metadata from SQLite
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            cursor.execute("SELECT uuid, content, metadata FROM memory WHERE id = ?", (idx,))
            row = cursor.fetchone()
            if row:
                results.append({
                    "uuid": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2])
                })
        conn.close()
        return results

# Singleton instance
conversation_memory = ConversationMemory(base_dir="./memory/conversation")
