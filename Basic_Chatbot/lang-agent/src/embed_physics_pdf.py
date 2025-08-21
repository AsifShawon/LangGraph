from langchain_ollama import OllamaEmbeddings
from config import Config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import time

Config.validate()

PHYSICS_PDF_PATH = "./theoretical_minimum.pdf"
MEMORY_PATH = "./physics_memory"

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

print("Loading PDF document...")

loader = PyMuPDFLoader(PHYSICS_PDF_PATH)
docs = loader.load()

print("Splitting document into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"Document split into {len(chunks)} chunks.")

print("Creating FAISS vector store with batch processing...")

# Define the size of each batch
batch_size = 50 

# Create the vector store with the first batch
vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)

# Loop through the rest of the chunks in batches
for i in range(batch_size, len(chunks), batch_size):
    # Get the next batch of chunks
    batch = chunks[i:i + batch_size]
    print(f"Processing batch {i // batch_size + 1}...")
    
    # Add the batch to the existing vector store
    vectorstore.add_documents(batch)

print("FAISS vector store created successfully.")
vectorstore.save_local(MEMORY_PATH)

def get_physics_context(query: str, k: int = 3) -> str:
    """
    Retrieve physics context from the vector store based on the query.
    """
    vectorstore = FAISS.load_local(MEMORY_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever  = vectorstore.similarity_search(query, k=k)
    
    if not retriever :
        return "No relevant physics knowledge found."
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retriever ])
    return context_text

if __name__ == "__main__":
    query = "What is the principle of least action?"
    context = get_physics_context(query)
    print(f"Context for query '{query}':\n{context}")