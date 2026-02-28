"""Global configuration for the RAG demo."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or OPENAI_API_KEY
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or OPENAI_BASE_URL
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

CHUNK_SIZES = {
    "character": {"chunk_size": 300, "chunk_overlap": 30},
    "recursive": {"chunk_size": 300, "chunk_overlap": 30},
    "semantic": {"breakpoint_threshold_type": "percentile"},
    "paragraph": {},
}
