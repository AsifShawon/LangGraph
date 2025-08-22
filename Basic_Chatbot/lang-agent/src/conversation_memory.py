import os
import json
import time
import uuid
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from config import Config

class ConversationMemory:
    """
    A class to manage conversation memory using FAISS for vector search and SQLite for metadata storage.
    """
    def __init__(self, base_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "faiss.index"
        self.meta_db_path = self.base_dir / "meta.db"
        
        # Use HuggingFace embedding model
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            # Dynamically determine embedding dimension
            sample_embedding = self.model.embed_query("test")
            self.dim = len(sample_embedding)
        except Exception as e:
            print(f"[ConversationMemory] Failed to initialize HuggingFace embeddings: {e}")
            # Fallback to a simpler solution - we'll use simple text matching
            self.model = None
            self.dim = 384  # Standard dimension for sentence transformers

        # Initialize FAISS index
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as e:
                print(f"[ConversationMemory] Failed to load FAISS index: {e}. Using a new index.")
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))

        # Initialize SQLite database
        self._init_meta_db()

    def _init_meta_db(self):
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with fallback to simple hash-based embedding."""
        if self.model:
            try:
                embedding = self.model.embed_query(text)
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                print(f"[ConversationMemory] Embedding failed: {e}")
        
        # Fallback: simple hash-based embedding
        hash_value = hash(text.lower().strip())
        # Create a simple embedding based on text features
        words = text.lower().split()
        word_count = len(words)
        char_count = len(text)
        
        # Create a simple feature vector
        features = np.zeros(self.dim, dtype=np.float32)
        features[0] = word_count / 100.0  # normalized word count
        features[1] = char_count / 1000.0  # normalized char count
        features[2] = abs(hash_value) % 1000 / 1000.0  # normalized hash
        
        # Fill remaining dimensions with character frequency features
        for i, char in enumerate(text.lower()[:self.dim-3]):
            features[i+3] = ord(char) / 255.0
            
        return features

    def upsert_sync(self, thread_id: str, user_message: str, ai_reply: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Insert or update a memory entry synchronously."""
        metadata = metadata or {}
        metadata["thread_id"] = thread_id
        
        # Combine user message and AI reply
        content = f"User: {user_message}"
        if ai_reply:
            content += f"\nAI: {ai_reply}"
            
        embedding_array = self._get_embedding(content).reshape(1, -1)
        faiss.normalize_L2(embedding_array)

        # Generate a unique ID
        uuid_str = str(uuid.uuid4())
        # Use a consistent ID for FAISS based on current time and thread_id
        faiss_id = abs(hash(f"{thread_id}_{time.time()}_{uuid_str}")) % (2**31)
        
        # Add to FAISS index
        self.index.add_with_ids(embedding_array, np.array([faiss_id], dtype=np.int64))
        faiss.write_index(self.index, str(self.index_path))

        # Add to SQLite database with the same ID for mapping
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memory (uuid, content, metadata) VALUES (?, ?, ?)",
            (f"{faiss_id}_{uuid_str}", content, json.dumps(metadata))
        )
        conn.commit()
        conn.close()
        return uuid_str

    async def upsert(self, thread_id: str, user_message: str, ai_reply: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Async version - run sync version in executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upsert_sync, thread_id, user_message, ai_reply, metadata)

    def search_sync(self, query: str, k: int = 5, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant memories using FAISS with optional thread filtering."""
        if self.index.ntotal == 0:
            return []
            
        embedding_array = self._get_embedding(query).reshape(1, -1)
        faiss.normalize_L2(embedding_array)
        
        # Search more results initially to allow for filtering
        search_k = k * 3 if thread_id else k
        distances, indices = self.index.search(embedding_array, min(search_k, self.index.ntotal))

        # Retrieve metadata from SQLite
        conn = sqlite3.connect(self.meta_db_path)
        cursor = conn.cursor()
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
                
            # The FAISS index returns the stored ID
            # Find the record with matching FAISS ID in the UUID field
            cursor.execute("SELECT uuid, content, metadata FROM memory WHERE uuid LIKE ? ORDER BY id DESC LIMIT 1", (f"{idx}_%",))
            row = cursor.fetchone()
            
            if row:
                try:
                    metadata = json.loads(row[2]) if row[2] else {}
                    
                    # Filter by thread_id if specified
                    if thread_id and metadata.get("thread_id") != thread_id:
                        continue
                        
                    results.append({
                        "uuid": row[0],
                        "content": row[1],
                        "metadata": metadata,
                        "distance": float(distances[0][i])
                    })
                    
                    # Stop when we have enough results
                    if len(results) >= k:
                        break
                except (json.JSONDecodeError, IndexError):
                    continue
                    
        conn.close()
        return results

    def search(self, query: str, k: int = 5, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant memories using FAISS."""
        return self.search_sync(query, k, thread_id)

# Singleton instance
conversation_memory = ConversationMemory(base_dir="./memory/conversation")
