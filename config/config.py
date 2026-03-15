import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# Web Search Configuration
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4

# Vector Store
VECTORSTORE_DIR = "vectorstore"