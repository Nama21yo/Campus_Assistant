import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

LLM_MODEL_NAME = "models/gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001"

DATA_FILE_PATH = "data/university_info.txt"
CHROMA_PERSIST_DIRECTORY = "chroma_db_persistent" 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
