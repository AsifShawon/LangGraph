# physics_query.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config
import os

# Validate config (if needed)
Config.validate()

# Paths
MEMORY_PATH = "./physics_memory"

# Try to initialize embeddings and vectorstore with fallback
vectorstore = None
try:
    # Try HuggingFace embeddings
    print("🔍 Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Load FAISS vectorstore once
    print("🔍 Loading FAISS vectorstore...")
    if os.path.exists(MEMORY_PATH):
        vectorstore = FAISS.load_local(
            MEMORY_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Physics vectorstore loaded successfully")
    else:
        print("⚠️ Physics memory path not found, physics queries will return empty")
        
except Exception as e:
    print(f"⚠️ Failed to load physics vectorstore: {e}")
    print("Physics queries will return fallback responses")

def query_physics(question: str, k: int = 3) -> str:
    """
    Query the physics FAISS vectorstore and return relevant context.
    """
    if vectorstore is None:
        return "Physics knowledge base not available. Proceeding with general knowledge."
    
    try:
        results = vectorstore.similarity_search(question, k=k)
        
        if not results:
            return "No relevant physics knowledge found."
        
        # Combine results into a single context string
        context_pieces = []
        for doc in results:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_pieces.append(content)
        
        return "\n\n---\n\n".join(context_pieces)
        
    except Exception as e:
        print(f"⚠️ Error querying physics vectorstore: {e}")
        return "Error retrieving physics knowledge. Proceeding with general knowledge."


if __name__ == "__main__":
    query = "What are the principles of quantum mechanics?"
    context = query_physics(query, k=3)
    print(f"\nQ: {query}\n\nContext:\n{context}\n")
