from typing import cast

import requests
from chromadb import Collection, EmbeddingFunction, PersistentClient
from chromadb.api import ClientAPI

from src.config import CHROMA_DIR, EMBED_MODEL, OLLAMA_BASE_URL, RETRIEVAL_K


class _OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def __call__(self, input: list[str]) -> list[list[float]]:  
        res = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": input},
        )
        res.raise_for_status()
        return res.json()["embeddings"]


def _get_client() -> ClientAPI:
    return PersistentClient(path=str(CHROMA_DIR))


def _get_collection() -> Collection:
    client = _get_client()
    embedding_fn = _OllamaEmbeddingFunction(EMBED_MODEL, OLLAMA_BASE_URL)
    return client.get_or_create_collection(
        name="rag_document_collection",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(documents: list[dict]) -> list[str]:
    collection = _get_collection()
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return ids


def retrieve_context(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    results = cast(dict[str, list[list]], results)
    documents = []
    for i in range(len(results["ids"][0])):
        documents.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )
    return documents


def get_indexed_sources() -> dict[str, dict]:
    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])
    all_data = cast(dict[str, list[dict]], all_data)
    result: dict[str, dict] = {}
    if all_data and all_data.get("metadatas"):
        seen_keys: set[str] = set()
        for meta in all_data["metadatas"]:
            if meta and "file_hash" in meta:
                key = meta["file_hash"]
                if key not in seen_keys:
                    seen_keys.add(key)
                    result[key] = {
                        "total_pages": meta.get("total_pages", 0),
                        "source": meta.get("source", ""),
                    }
    return result


def remove_source(file_hash: str) -> None:
    collection = _get_collection()
    collection.delete(where={"file_hash": file_hash})


def clear_index() -> None:
    try:
        _get_client().delete_collection("rag_document_collection")
    except ValueError:
        pass