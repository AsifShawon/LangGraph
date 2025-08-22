import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    HUGGINGFACE_API_KEY_READ_ONLY = os.getenv("HUGGINGFACE_API_KEY_READ_ONLY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Model Configuration
    DEFAULT_MODEL = "openai/gpt-oss-120b"  # Use Groq model
    TEMPERATURE = 0.1
    MAX_TOKENS = 8192  # Set reasonable limit
    TIMEOUT = 60  # Set timeout in seconds
    MAX_RETRIES = 2

    # Agent Configuration
    MAX_ITERATIONS = 10
    TIMEOUT = 30

    # --- NEW: Memory and Embedding Configuration ---
    MEMORY_PATH = "./memory"  
    # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
    MODEL_KWARGS = {'device': 'cpu'}
    ENCODE_KWARGS = {'normalize_embeddings': False} 

    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        if not cls.BRAVE_API_KEY:
            raise ValueError("BRAVE_API_KEY is required")
