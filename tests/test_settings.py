import json

import pytest

from src import settings as settings_module


@pytest.fixture
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", tmp_path / "settings.json")
    yield


def test_load_settings_missing_file_defaults(_isolated):
    assert settings_module.load_settings() == settings_module.DEFAULT_SETTINGS


def test_load_settings_corrupt_file_defaults(_isolated):
    settings_module.SETTINGS_PATH.write_text("{not json", encoding="utf-8")
    assert settings_module.load_settings() == settings_module.DEFAULT_SETTINGS


def test_save_load_round_trip(_isolated):
    custom = {
        "temperature": 0.7,
        "top_k": 10,
        "multi_turn": False,
        "history_window": 4,
        "rewrite_enabled": False,
        "rewrite_history_window": 2,
        "retrieval_history_messages": 3,
    }
    settings_module.save_settings(custom)
    assert settings_module.load_settings() == custom
    assert json.loads(settings_module.SETTINGS_PATH.read_text(encoding="utf-8")) == custom


def test_load_settings_partial_merged_with_defaults(_isolated):
    settings_module.SETTINGS_PATH.write_text(
        json.dumps({"temperature": 0.9}), encoding="utf-8"
    )
    loaded = settings_module.load_settings()
    assert loaded["temperature"] == 0.9
    for key in (
        "top_k",
        "multi_turn",
        "history_window",
        "rewrite_enabled",
        "rewrite_history_window",
        "retrieval_history_messages",
    ):
        assert loaded[key] == settings_module.DEFAULT_SETTINGS[key]


def test_save_settings_fills_missing_keys_with_defaults(_isolated):
    settings_module.save_settings({"temperature": 0.7})
    loaded = settings_module.load_settings()
    assert loaded["temperature"] == 0.7
    for key in ("top_k", "history_window", "rewrite_enabled"):
        assert loaded[key] == settings_module.DEFAULT_SETTINGS[key]


def test_load_settings_ignores_unknown_keys(_isolated):
    settings_module.SETTINGS_PATH.write_text(
        json.dumps({"temperature": 0.5, "bogus": True}), encoding="utf-8"
    )
    assert "bogus" not in settings_module.load_settings()
