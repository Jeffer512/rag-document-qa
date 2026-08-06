import json
from unittest.mock import Mock

import pytest

from src.generator import _extract_sources, _format_context, _prior_messages, generate_stream


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
    monkeypatch.setattr("src.generator.retrieve_context", _mock_retrieve)

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

    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "First?"}
    assert messages[2] == {"role": "assistant", "content": "Answer one"}
    assert messages[3]["role"] == "user"
    assert "QUESTION:\nSecond?" in messages[3]["content"]
    assert messages[3]["content"].startswith("CONTEXT:")