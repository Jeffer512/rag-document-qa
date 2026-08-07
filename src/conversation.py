import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src.config import CONVERSATIONS_DIR


def _path(conversation_id: str) -> Path:
    if not conversation_id or Path(conversation_id).name != conversation_id:
        raise ValueError(f"Invalid conversation id: {conversation_id!r}")
    return CONVERSATIONS_DIR / f"{conversation_id}.json"


def _load_raw(conversation_id: str) -> dict:
    path = _path(conversation_id)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def conversation_exists(conversation_id: str) -> bool:
    return _path(conversation_id).exists()


def list_conversations() -> list[dict]:
    if not CONVERSATIONS_DIR.exists():
        return []
    conversations = []
    for path in CONVERSATIONS_DIR.glob("*.json"):
        data = _load_raw(path.stem)
        if data:
            conversations.append(
                {
                    "conversation_id": path.stem,
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                }
            )
    return sorted(conversations, key=lambda c: c["updated_at"], reverse=True)


def create_conversation(conversation_id: str, title: str = "Untitled") -> str:
    save_conversation(conversation_id, title, [])
    return conversation_id


def load_conversation(conversation_id: str) -> list[dict]:
    return _load_raw(conversation_id).get("messages", [])


def save_conversation(
    conversation_id: str, title: str, messages: list[dict], updated_at: str | None = None
) -> None:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "conversation_id": conversation_id,
        "title": title,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "messages": messages,
    }
    path = _path(conversation_id)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def rename_conversation(conversation_id: str, title: str) -> None:
    data = _load_raw(conversation_id)
    if not data:
        return
    save_conversation(
        conversation_id,
        title,
        data.get("messages", []),
        updated_at=data.get("updated_at"),
    )


def delete_conversation(conversation_id: str) -> None:
    _path(conversation_id).unlink(missing_ok=True)