from src.generator import _extract_sources, _format_context


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