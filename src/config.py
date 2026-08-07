from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ollama server and models
OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "llama3.2:3b"
EMBED_MODEL: str = "embeddinggemma:300m"

# Text chunking
CHUNK_SIZE: int = 500       # Characters per chunk 
CHUNK_OVERLAP: int = 50     # Overlap between adjacent chunks

# Number of top-k chunks retrieved per query
RETRIEVAL_K: int = 5

# Persistence directories (auto-created on first run)
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db/"
DATA_DIR: Path = PROJECT_ROOT / "data/"
CONVERSATIONS_DIR: Path = PROJECT_ROOT / "conversations/"

# Opens PDFs in the OS default viewer from the sidebar (requires the app and
# its files to run on the user's own machine); set to False to show downloads.
OPEN_PDF_SYSTEM_VIEWER: bool = True
