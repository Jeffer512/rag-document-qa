from unittest.mock import Mock

from src.generator import _extract_sources, _format_context, generate_stream


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

    tokens, _sources = generate_stream("test")
    result = "".join(tokens)
    assert _sources == []
    assert "❌" in result
    assert "No server" in result


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
    result = "".join(tokens)
    assert "Partial" in result
    assert "❌" in result