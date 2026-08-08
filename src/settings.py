import json
import os
from typing import Any

from src.config import (
    HISTORY_WINDOW,
    MULTI_TURN,
    RETRIEVAL_HISTORY_MESSAGES,
    RETRIEVAL_K,
    REWRITE_ENABLED,
    REWRITE_HISTORY_WINDOW,
    SETTINGS_PATH,
    TEMPERATURE,
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "temperature": TEMPERATURE,
    "top_k": RETRIEVAL_K,
    "multi_turn": MULTI_TURN,
    "history_window": HISTORY_WINDOW,
    "rewrite_enabled": REWRITE_ENABLED,
    "rewrite_history_window": REWRITE_HISTORY_WINDOW,
    "retrieval_history_messages": RETRIEVAL_HISTORY_MESSAGES,
}

_SETTING_KEYS = tuple(DEFAULT_SETTINGS)


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        with SETTINGS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)
    if not isinstance(data, dict):
        return dict(DEFAULT_SETTINGS)
    return {key: data.get(key, DEFAULT_SETTINGS[key]) for key in _SETTING_KEYS}


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {key: settings.get(key, DEFAULT_SETTINGS[key]) for key in _SETTING_KEYS}
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)
