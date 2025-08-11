import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    HUGGINGFACE_API_KEY_READ_ONLY = os.getenv("HUGGINGFACE_API_KEY_READ_ONLY")

    # Model Configuration
    DEFAULT_MODEL = "llama3-70b-8192"
    TEMPERATURE = 0.1

    # Agent Configuration
    MAX_ITERATIONS = 10
    TIMEOUT = 30

    # --- NEW: Memory and Embedding Configuration ---
    MEMORY_PATH = "./memory"  
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        if not cls.BRAVE_API_KEY:
            raise ValueError("BRAVE_API_KEY is required")
