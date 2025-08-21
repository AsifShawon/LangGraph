import os
import json
import time
import uuid
import asyncio
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import aiosqlite
import logging

from config import Config

# --- Sync/async helper -------------------------------------------------

def sync_await(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container = {}

    def _run():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(coro)
            result_container['result'] = result
        finally:
            new_loop.close()

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    return result_container.get('result')


# --- Vector Memory Implementation --------------------------------------

class VectorMemory:
    """
    A generic Vector Memory class that can be instantiated for different
    knowledge bases or conversation histories.
    """
    def __init__(
        self,
        base_dir: str,
        model_name: str = Config.EMBEDDING_MODEL_NAME,
        dim: Optional[int] = None,
        executor_workers: int = 4,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "faiss.index"
        self.meta_db_path = self.base_dir / "meta.db"
        self.model = SentenceTransformer(model_name)
        self.dim = dim or self.model.get_sentence_embedding_dimension()
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)

        flat = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(flat)

        if self.index_path.exists():
            try:
                loaded = faiss.read_index(str(self.index_path))
                if not isinstance(loaded, faiss.IndexIDMap):
                    loaded = faiss.IndexIDMap(loaded)
                self.index = loaded
                print(f"[VectorMemory] Loaded FAISS index from {self.index_path}")
            except Exception as e:
                print(f"[VectorMemory] Failed to load FAISS index: {e}. Using fresh index.")

        sync_await(self._init_meta_db())
        logging.info(f"VectorMemory initialized for directory: {base_dir}")

    async def _init_meta_db(self):
        async with aiosqlite.connect(self.meta_db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    uuid TEXT UNIQUE,
                    thread_id TEXT,
                    content TEXT,
                    created_at INTEGER,
                    metadata TEXT
                )
                """
            )
            await db.commit()

    def _embed_sync(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    async def embed(self, texts: List[str]):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._embed_sync, texts)

    def upsert_sync(self, thread_id: str, user_message: str, ai_reply: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return sync_await(self.upsert(thread_id, user_message, ai_reply, metadata))

    def search_sync(self, query: str, k: int = 5, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return sync_await(self.search(query, k, thread_id))

    async def upsert(self, thread_id: str, user_message: str, ai_reply: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        metadata = metadata or {}
        
        chunk_content = f"User: {user_message}\nAI: {ai_reply}" if ai_reply else user_message
        
        embeddings = await self.embed([chunk_content])
        vec = embeddings[0]
        faiss.normalize_L2(vec.reshape(1, -1))
        uuid_str = str(uuid.uuid4())
        ts = int(time.time())

        async with aiosqlite.connect(self.meta_db_path) as db:
            cursor = await db.execute(
                "INSERT INTO chunks (uuid, thread_id, content, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (uuid_str, thread_id, chunk_content, ts, json.dumps(metadata))
            )
            await db.commit()
            numeric_id = cursor.lastrowid

        ids = np.array([numeric_id], dtype='int64')
        self.index.add_with_ids(vec.reshape(1, -1), ids)

        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            logging.warning(f"Failed to write FAISS index to disk: {e}")
        return uuid_str

    async def search(self, query: str, k: int = 5, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
            
        q_emb = (await self.embed([query]))[0]
        faiss.normalize_L2(q_emb.reshape(1, -1))
        
        D, I = self.index.search(q_emb.reshape(1, -1), k)
        
        ids = I[0].tolist()
        scores = D[0].tolist()
        
        valid_ids = [i for i, s in zip(ids, scores) if i != -1 and s >= Config.SIMILARITY_THRESHOLD]
        if not valid_ids:
            return []
            
        placeholders = ','.join('?' for _ in valid_ids)
        async with aiosqlite.connect(self.meta_db_path) as db:
            # If a thread_id is provided, filter by it.
            sql_query = f"SELECT id, uuid, thread_id, content, created_at, metadata FROM chunks WHERE id IN ({placeholders})"
            params = valid_ids
            if thread_id:
                sql_query += " AND thread_id = ?"
                params.append(thread_id)

            cursor = await db.execute(sql_query, params)
            rows = await cursor.fetchall()
            
        rows_by_id = {row[0]: row for row in rows}
        
        results = []
        for id_val, score in zip(ids, scores):
            if id_val in rows_by_id:
                row = rows_by_id[id_val]
                results.append({
                    'id': row[0],
                    'uuid': row[1],
                    'thread_id': row[2],
                    'content': row[3],
                    'created_at': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {},
                    'score': float(score)
                })
        return results

    def close(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

# --- Singleton instances for different memory types ---
# For storing conversation history
conversation_memory = VectorMemory(base_dir=Config.CONVERSATION_VECTOR_PATH)

# For storing the physics knowledge base
physics_kb = VectorMemory(base_dir=Config.PHYSICS_VECTOR_STORE_PATH)
