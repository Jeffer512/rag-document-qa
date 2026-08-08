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

# Generation defaults (overridable from the app's Settings sidebar)
TEMPERATURE: float = 0.0
MULTI_TURN: bool = True
HISTORY_WINDOW: int = 0                  # 0 = unlimited (current behavior)
REWRITE_ENABLED: bool = True
REWRITE_HISTORY_WINDOW: int = 0          # 0 = unlimited
RETRIEVAL_HISTORY_MESSAGES: int = 1      # last N messages embedded for retrieval when rewrite is off

# User-tunable settings (auto-created on first change)
SETTINGS_PATH: Path = PROJECT_ROOT / "settings.json"

# Persistence directories (auto-created on first run)
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db/"
DATA_DIR: Path = PROJECT_ROOT / "data/"
CONVERSATIONS_DIR: Path = PROJECT_ROOT / "conversations/"

# Opens PDFs in the OS default viewer from the sidebar (requires the app and
# its files to run on the user's own machine); set to False to show downloads.
OPEN_PDF_SYSTEM_VIEWER: bool = True
