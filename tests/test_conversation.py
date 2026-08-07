import time

import pytest

from src import conversation


def test_create_save_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    cid = conversation.create_conversation("My chat")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there", "sources": []},
    ]
    conversation.save_conversation(cid, "My chat", messages)
    assert conversation.load_conversation(cid) == messages


def test_create_default_title(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    conversation.create_conversation("chat-1")
    assert conversation.list_conversations()[0]["title"] == "Untitled"


def test_list_conversations_sorted_newest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    first = conversation.create_conversation("First")
    time.sleep(0.01)
    second = conversation.create_conversation("Second")
    result = conversation.list_conversations()
    assert [c["conversation_id"] for c in result] == [second, first]


def test_list_conversations_reads_metadata_only(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    cid = conversation.create_conversation("Meta only")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there", "sources": [{"source": "a.pdf", "page": 1}]},
    ]
    conversation.save_conversation(cid, "Meta only", messages)
    meta = conversation._load_meta(cid)
    assert meta["title"] == "Meta only"
    assert "messages" not in meta
    assert conversation.list_conversations()[0]["title"] == "Meta only"


def test_rename_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    cid = conversation.create_conversation("Old")
    conversation.save_conversation(cid, "Old", [{"role": "user", "content": "hi"}])
    before = conversation._load_raw(cid)["updated_at"]
    conversation.rename_conversation(cid, "New")
    assert conversation.list_conversations()[0]["title"] == "New"
    assert conversation.load_conversation(cid) == [{"role": "user", "content": "hi"}]
    assert conversation._load_raw(cid)["updated_at"] == before


def test_delete_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    cid = conversation.create_conversation("chat-2")
    conversation.delete_conversation(cid)
    assert conversation.list_conversations() == []
    assert not (tmp_path / f"{cid}.json").exists()


def test_load_missing_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    assert conversation.load_conversation("does-not-exist") == []


def test_invalid_conversation_id_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation, "CONVERSATIONS_DIR", tmp_path)
    with pytest.raises(ValueError):
        conversation.save_conversation("../evil", "Title", [])
