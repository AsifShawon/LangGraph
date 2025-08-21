# physics_query.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config

# Validate config (if needed)
Config.validate()

# Paths
MEMORY_PATH = "./physics_memory"

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Load FAISS vectorstore once
print("🔍 Loading FAISS vectorstore...")
vectorstore = FAISS.load_local(
    MEMORY_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def query_physics(question: str, k: int = 3) -> str:
    """
    Query the physics FAISS vectorstore and return relevant context.
    """
    results = vectorstore.similarity_search(question, k=k)

    if not results:
        return "No relevant physics knowledge found."

    # Join retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context_text


if __name__ == "__main__":
    query = "What are the principles of quantum mechanics?"
    context = query_physics(query, k=3)
    print(f"\nQ: {query}\n\nContext:\n{context}\n")
