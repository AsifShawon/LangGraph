import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_memory import VectorMemory
from config import Config

def ingest_physics_knowledge():
    """
    Reads 'The Theoretical Minimum' from a PDF, splits it into chunks,
    and ingests them into a dedicated vector store.
    """
    # 1. Initialize a dedicated VectorMemory instance for the physics knowledge
    print("Initializing physics knowledge base...")
    physics_kb = VectorMemory(
        base_dir=Config.PHYSICS_VECTOR_STORE_PATH,
        model_name=Config.EMBEDDING_MODEL_NAME
    )

    # 2. Check if the source PDF file exists
    source_file = "theoretical_minimum.pdf" # Changed to .pdf
    if not os.path.exists(source_file):
        print(f"❌ Error: Source PDF file not found at '{source_file}'")
        print("Please ensure you have placed the PDF in the root directory.")
        return

    # 3. Load the content from the PDF
    print(f"Loading content from '{source_file}'...")
    loader = PyPDFLoader(source_file)
    documents = loader.load()
    # Combine the content of all pages into a single string
    content = "\n".join(doc.page_content for doc in documents)
    print(f"Successfully loaded {len(documents)} pages from the PDF.")

    # 4. Split the text into chunks
    # This splitter tries to keep paragraphs together, which is great for context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    print(f"Split content into {len(chunks)} chunks.")

    # 5. Ingest each chunk into the vector store
    print("Ingesting chunks into the vector store. This may take a few minutes...")
    for i, chunk in enumerate(chunks):
        # For the knowledge base, we use the content as a single chunk
        # with a 'knowledge' role for clarity.
        physics_kb.upsert_sync(
            thread_id="physics_knowledge_base",  # A single, consistent ID for all docs
            user_message=chunk, # We store the chunk as the "user_message"
            ai_reply="" # The AI reply is empty for knowledge chunks
        )
        print(f"  -> Ingested chunk {i + 1}/{len(chunks)}")

    print("✅ Ingestion complete! Your Physics Bot is now ready.")

if __name__ == "__main__":
    ingest_physics_knowledge()
