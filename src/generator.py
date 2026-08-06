import json
from collections.abc import Iterator

import requests

from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.vector_store import retrieve_context

SYSTEM_PROMPT = (
    "You are a professional document assistant. Answer the question based "
    "strictly on the provided context below. If the answer cannot be found "
    'in the context, say "I do not have enough information in the uploaded '
    'documents."'
)


def _format_context(documents: list[dict]) -> str:
    blocks = []
    for doc in documents:
        meta = doc["metadata"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        blocks.append(f"[Source: {source}, Page {page}]\n{doc['text']}")
    return "\n\n".join(blocks)


def _extract_sources(documents: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for doc in documents:
        meta = doc["metadata"]
        key = (meta.get("source"), meta.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


def generate_response(question: str) -> dict:
    documents = retrieve_context(question)
    context = _format_context(documents)
    sources = _extract_sources(documents)

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        },
    )
    res.raise_for_status()
    answer = res.json()["message"]["content"]
    return {"answer": answer, "sources": sources}


def generate_stream(question: str) -> tuple[Iterator[str], list[dict]]:
    documents = retrieve_context(question)
    context = _format_context(documents)
    sources = _extract_sources(documents)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {"temperature": 0.0},
        },
        stream=True,
        timeout=(5, 30),
    )
    res.raise_for_status()

    def _tokens() -> Iterator[str]:
        for line in res.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content

    return _tokens(), sources