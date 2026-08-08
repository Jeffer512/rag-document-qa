import json
from unittest.mock import Mock

import pytest

from src.generator import (
    _extract_sources,
    _format_context,
    _history_augmented_query,
    _prior_messages,
    generate_stream,
    rewrite_query,
)


def test_format_context():
    docs = [
        {
            "id": "geo.pdf_p1_c0",
            "text": "Paris is the capital.",
            "metadata": {"source": "geo.pdf", "page": 1},
        },
    ]
    result = _format_context(docs)
    assert "[Source: geo.pdf, Page 1]" in result
    assert "Paris is the capital." in result


def test_extract_sources():
    docs = [
        {"id": "a", "text": "A", "metadata": {"source": "a.pdf", "page": 1}},
        {"id": "b", "text": "B", "metadata": {"source": "a.pdf", "page": 1}},  # duplicate
        {"id": "c", "text": "C", "metadata": {"source": "b.pdf", "page": 2}},
        {"id": "d", "text": "D", "metadata": {"source": "a.pdf", "page": 3}},  # same source, diff page
    ]
    sources = _extract_sources(docs)
    assert len(sources) == 3
    assert {"source": "a.pdf", "page": 1} in sources
    assert {"source": "b.pdf", "page": 2} in sources
    assert {"source": "a.pdf", "page": 3} in sources


_SAMPLE_DOCS = [
    {
        "id": "aaa_p1_c0",
        "text": "Paris is the capital of France.",
        "metadata": {"source": "geo.pdf", "page": 1, "file_hash": "aaa"},
    },
]


def _mock_retrieve(question: str, k: int | None = None) -> list[dict]:
    return _SAMPLE_DOCS


def test_generate_stream_decodes_tokens(monkeypatch):
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)

    lines = [
        '{"message":{"content":"Hello "}}',
        '{"message":{"content":"world"}}',
        '{"done":true,"message":{"content":""}}',
    ]
    mock_resp = Mock()
    mock_resp.iter_lines.return_value = lines
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    tokens, _sources = generate_stream("test")
    assert "".join(tokens) == "Hello world"


def test_generate_stream_connection_error(monkeypatch):
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)
    from requests.exceptions import ConnectionError

    def _raise(*a, **kw):
        raise ConnectionError("No server")

    monkeypatch.setattr("src.generator.requests.post", _raise)

    with pytest.raises(ConnectionError):
        generate_stream("test")


def test_generate_stream_mid_stream_error(monkeypatch):
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)

    lines = [
        '{"message":{"content":"Partial "}}',
        "not json",
    ]
    mock_resp = Mock()
    mock_resp.iter_lines.return_value = lines
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    tokens, _sources = generate_stream("test")
    assert next(tokens) == "Partial "
    with pytest.raises(json.JSONDecodeError):
        next(tokens)


def test_prior_messages_drops_failed_turn():
    history = [
        {"role": "user", "content": "Failed?"},
        {"role": "assistant", "content": "partial", "error": "interrupted"},
        {"role": "user", "content": "Good?"},
        {"role": "assistant", "content": "Good answer"},
    ]
    assert _prior_messages(history) == [
        {"role": "user", "content": "Good?"},
        {"role": "assistant", "content": "Good answer"},
    ]


def test_generate_stream_payload(monkeypatch):
    retrieved = []

    def _capture_retrieve(question, k=None):
        retrieved.append(question)
        return _SAMPLE_DOCS

    monkeypatch.setattr("src.generator.retrieve_context", _capture_retrieve)
    monkeypatch.setattr(
        "src.generator.rewrite_query",
        lambda question, history, history_window=0: "standalone query",
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    history = [
        {"role": "user", "content": "Failed?"},
        {"role": "assistant", "content": "partial", "error": "interrupted"},
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Answer one", "sources": [{"source": "a.pdf", "page": 1}]},
    ]
    tokens, _sources = generate_stream("Second?", history)
    list(tokens)

    assert retrieved == ["standalone query"]
    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "First?"}
    assert messages[2] == {"role": "assistant", "content": "Answer one"}
    assert messages[3]["role"] == "user"
    assert "QUESTION:\nSecond?" in messages[3]["content"]
    assert messages[3]["content"].startswith("CONTEXT:")


def test_rewrite_query_payload(monkeypatch):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "standalone query"}}

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    history = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Answer one", "sources": [{"source": "a.pdf", "page": 1}]},
    ]
    result = rewrite_query("what about it?", history)

    assert result == "standalone query"
    payload = captured["json"]
    assert payload["stream"] is False
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "First?"}
    assert messages[2] == {"role": "assistant", "content": "Answer one"}
    assert messages[3] == {"role": "user", "content": "what about it?"}


def test_rewrite_query_empty_falls_back(monkeypatch):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "   "}}
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    assert rewrite_query("what about it?", []) == "what about it?"


def test_generate_stream_forwards_k_and_temperature(monkeypatch):
    retrieved = {}

    def _capture_retrieve(question, k=None):
        retrieved["k"] = k
        return _SAMPLE_DOCS

    monkeypatch.setattr("src.generator.retrieve_context", _capture_retrieve)
    monkeypatch.setattr(
        "src.generator.rewrite_query", lambda question, history, history_window=0: "query"
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    tokens, _sources = generate_stream("Q", temperature=0.8, top_k=3)
    list(tokens)

    assert retrieved["k"] == 3
    assert captured["json"]["options"] == {"temperature": 0.8}


def test_generate_stream_history_window_truncates_answer(monkeypatch):
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)
    monkeypatch.setattr(
        "src.generator.rewrite_query", lambda question, history, history_window=0: question
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    tokens, _sources = generate_stream("Three?", history, history_window=2)
    list(tokens)

    prior = captured["json"]["messages"][1:-1]
    assert prior == [
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]


def test_generate_stream_rewrite_history_window_forwards(monkeypatch):
    seen = {}

    def _fake_rewrite(question, history, history_window=0):
        seen["window"] = history_window
        return "query"

    monkeypatch.setattr("src.generator.rewrite_query", _fake_rewrite)
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    tokens, _sources = generate_stream("Three?", history, rewrite_history_window=2)
    list(tokens)

    assert seen["window"] == 2


def test_rewrite_query_history_window_truncates(monkeypatch):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "standalone query"}}

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    rewrite_query("Three?", history, history_window=2)

    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == [
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    assert messages[3] == {"role": "user", "content": "Three?"}


def test_history_augmented_query_skips_failed_turn():
    history = [
        {"role": "user", "content": "Bad?"},
        {"role": "assistant", "content": "partial", "error": "interrupted"},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    assert _history_augmented_query("Three?", history, n=4) == "Two?\nTwo\nThree?"


def test_generate_stream_multi_turn_disabled_ignores_history(monkeypatch):
    retrieved = []
    rewrite_calls = []
    monkeypatch.setattr(
        "src.generator.retrieve_context",
        lambda question, k=None: retrieved.append(question) or _SAMPLE_DOCS,
    )
    monkeypatch.setattr(
        "src.generator.rewrite_query",
        lambda question, history, history_window=0: rewrite_calls.append(question) or "rewritten",
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None

    captured = {}

    def _post(*a, **kw):
        captured["json"] = kw["json"]
        return mock_resp

    monkeypatch.setattr("src.generator.requests.post", _post)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
    ]
    tokens, _sources = generate_stream(
        "Two?", history, multi_turn=False, rewrite_history_window=1
    )
    list(tokens)

    assert retrieved == ["Two?"]
    assert rewrite_calls == []
    prior = captured["json"]["messages"][1:-1]
    assert prior == []


def test_generate_stream_rewrite_disabled_uses_raw_question(monkeypatch):
    retrieved = []
    rewrite_calls = []
    monkeypatch.setattr(
        "src.generator.retrieve_context",
        lambda question, k=None: retrieved.append(question) or _SAMPLE_DOCS,
    )
    monkeypatch.setattr(
        "src.generator.rewrite_query",
        lambda question, history, history_window=0: rewrite_calls.append(question) or "rewritten",
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
    ]
    tokens, _sources = generate_stream("Three?", history, rewrite_enabled=False)
    list(tokens)

    assert retrieved == ["Three?"]
    assert rewrite_calls == []


def test_generate_stream_rewrite_disabled_with_retrieval_messages(monkeypatch):
    retrieved = []
    monkeypatch.setattr(
        "src.generator.retrieve_context",
        lambda question, k=None: retrieved.append(question) or _SAMPLE_DOCS,
    )
    monkeypatch.setattr(
        "src.generator.rewrite_query", lambda question, history, history_window=0: "rewritten"
    )

    mock_resp = Mock()
    mock_resp.iter_lines.return_value = ['{"done":true,"message":{"content":""}}']
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr("src.generator.requests.post", lambda *a, **kw: mock_resp)

    history = [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Two"},
    ]
    tokens, _sources = generate_stream(
        "Three?", history, rewrite_enabled=False, retrieval_history_messages=3
    )
    list(tokens)

    assert retrieved == ["Two?\nTwo\nThree?"]