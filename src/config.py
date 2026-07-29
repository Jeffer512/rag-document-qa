from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "llama3.2:3b"
EMBED_MODEL: str = "embeddinggemma:300m"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
RETRIEVAL_K: int = 5
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db/"
DATA_DIR: Path = PROJECT_ROOT / "data/"
