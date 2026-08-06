import json
from collections.abc import Iterator

import requests

from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.vector_store import retrieve_context

SYSTEM_PROMPT = (
    "You are a professional document assistant. Answer the question based "
    "strictly on the provided context, using the conversation history "
    "for continuity. If the answer cannot be found in the context, say "
    '"I do not have enough information in the uploaded documents."'
)

REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriter for a document retrieval system. Rewrite the "
    "user's latest question into a standalone search query that captures its "
    "full intent, resolving pronouns and implicit references using the "
    "conversation history. If the question is already self-contained, return "
    "it unchanged. Output only the rewritten query."
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


def _prior_messages(history: list[dict]) -> list[dict]:
    prior = []
    for msg in history:
        if msg["role"] == "assistant" and "error" in msg:
            if prior and prior[-1]["role"] == "user":
                prior.pop()
            continue
        prior.append({"role": msg["role"], "content": msg["content"]})
    return prior


def rewrite_query(question: str, history: list[dict]) -> str:
    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                *_prior_messages(history),
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=(5, 30),
    )
    res.raise_for_status()
    return res.json()["message"]["content"].strip() or question


def generate_stream(
    question: str, history: list[dict] | None = None
) -> tuple[Iterator[str], list[dict]]:
    query = rewrite_query(question, history) if history else question
    documents = retrieve_context(query)
    context = _format_context(documents)
    sources = _extract_sources(documents)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *(_prior_messages(history) if history else []),
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