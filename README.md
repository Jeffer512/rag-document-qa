# rag-document-qa

A local RAG (Retrieval-Augmented Generation) document question-answering app using **Ollama** for embeddings and LLM, **LangChain** for text chunking, **ChromaDB** for vector storage, and **Streamlit** for the UI.

![Python](https://img.shields.io/badge/python-3.11+-blue)

## Overview

Upload PDF documents, ask questions in natural language, and get answers generated strictly from the document content — all running locally. No API keys, no cloud dependencies.

### Features

- Local LLM via Ollama (no internet required after setup)
- Streaming responses — text appears token by token
- Conversation persistence — multiple named chats saved to disk, restored on restart
- Hash-based deduplication — same content never re-indexed
- File management via sidebar — per-document removal with disk cleanup
- Self-contained tests — all mocks, no Ollama needed for `pytest`

## Architecture

```
PDF upload → ingestion (pypdf + chunking) → ChromaDB (Ollama embeddings)
                                                     ↓
User question  →  retrieve top-k chunks  →  Ollama LLM  →  streamed answer
```

1. **Ingestion** — PDF parsed page by page with pypdf, text split into chunks via LangChain `RecursiveCharacterTextSplitter`, each chunk assigned a hash-based ID for dedup
2. **Retrieval** — query embedded with Ollama, top-k chunks retrieved from ChromaDB by cosine similarity
3. **Generation** — chunks formatted as context, sent to Ollama chat endpoint with `stream=True`, tokens rendered incrementally in the chat UI

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

Pull the required models (or use other models and update config.py):

```bash
ollama pull llama3.2:3b
ollama pull embeddinggemma:300m
```

## Quick start

```bash
git clone <repo-url>
cd rag-document-qa

python3 -m venv venv
source venv/bin/activate

pip install -e ".[dev]"

streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. **Upload a PDF** — use the file uploader in the main area
2. **Ask a question** — type in the chat input and press Enter
3. **View sources** — expand the "Sources" section under each answer to see which document and page the answer came from
4. **Manage documents** — use the sidebar to view indexed documents, remove individual files, or clear everything
5. **Manage conversations** — the sidebar lets you create, rename, delete, and switch between named conversations; chats persist across app restarts

## Project structure

```
rag-document-qa/
├── app.py                 # Streamlit UI entry point
├── pyproject.toml         # Project config and dependencies
├── src/
│   ├── config.py          # Configuration constants
│   ├── conversation.py    # Conversation persistence (JSON files)
│   ├── ingestion.py       # PDF parsing and text chunking
│   ├── vector_store.py    # ChromaDB index and retrieval
│   └── generator.py       # Ollama LLM integration (streaming + sync)
├── tests/
│   ├── test_conversation.py  # Conversation CRUD
│   ├── test_ingestion.py  # PDF loading, chunking, hashing
│   ├── test_vector_store.py  # Index, retrieve, source management
│   └── test_generator.py  # Formatting, source extraction, streaming
├── chroma_db/             # Vector store persistence (auto-created)
├── conversations/         # Conversation history (auto-created)
└── data/                  # Uploaded PDFs (auto-created)
```

## Testing

All tests use mocks for both Ollama and ChromaDB — no external dependencies required:

```bash
pytest -v
```

Linting:

```bash
ruff check src/ tests/ app.py
```

## How it works

### Ingestion pipeline

Files are identified by a SHA-256 content hash (first 16 characters), not by filename. This means:

- Renaming a file and re-uploading it skips re-indexing
- Uploading two files with the same name but different content indexes both independently
- IDs follow the pattern `{hash}_p{page}_c{chunk}`

### Retrieval

The `retrieve_context` function embeds the user's query and finds the closest `RETRIEVAL_K` chunks. Results are returned as a `list[dict]` with `id`, `text`, and `metadata` keys.

### Generation

Two modes:
- **Streaming** (`generate_stream`): returns an `Iterator[str]` and renders tokens incrementally via `st.write_stream`. Handles connection errors before the request and parsing errors mid-stream gracefully.
- **Sync** (`generate_response`): returns the full answer as a dict.

### Error handling

- Connection errors (Ollama not running) show "Connection failed" with the exception message
- Mid-stream parsing errors yield an error token appended to partial output
- A 30-second read timeout prevents hanging on a dead connection

## Configuration

All settings in `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.2:3b` | Model for answer generation |
| `EMBED_MODEL` | `embeddinggemma:300m` | Model for embeddings |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `5` | Top-k chunks retrieved per query |
| `CHROMA_DIR` | `chroma_db/` | Vector store persistence directory |
| `DATA_DIR` | `data/` | Uploaded file storage directory |
| `CONVERSATIONS_DIR` | `conversations/` | Conversation history storage directory |
