import pytest
from chromadb import EmbeddingFunction, EphemeralClient

import src.vector_store as vs2


class _TestEmbedding(EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in input]


_ephemeral_client = EphemeralClient()


def _ephemeral_client_fn():
    return _ephemeral_client


def _ephemeral_collection():
    return _ephemeral_client.get_or_create_collection(
        name="rag_document_collection",
        embedding_function=_TestEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture(autouse=True)
def _patch_vs(monkeypatch):
    monkeypatch.setattr(vs2, "_get_client", _ephemeral_client_fn)
    monkeypatch.setattr(vs2, "_get_collection", _ephemeral_collection)


SAMPLE_DOCS = [
    {
        "id": "aaa_p1_c0",
        "text": "The capital of France is Paris.",
        "metadata": {"source": "geo.pdf", "page": 1, "total_pages": 1, "chunk_index": 0, "file_hash": "aaa"},
    },
    {
        "id": "aaa_p1_c1",
        "text": "The capital of Japan is Tokyo.",
        "metadata": {"source": "geo.pdf", "page": 1, "total_pages": 1, "chunk_index": 1, "file_hash": "aaa"},
    },
    {
        "id": "bbb_p3_c0",
        "text": "The chemical symbol for water is H2O.",
        "metadata": {"source": "sci.pdf", "page": 3, "total_pages": 3, "chunk_index": 0, "file_hash": "bbb"},
    },
]


def test_index_then_retrieve():
    ids = vs2.index_documents(SAMPLE_DOCS[:1])
    assert len(ids) == 1

    results = vs2.retrieve_context("France", k=1)
    assert len(results) == 1
    assert results[0]["text"] == SAMPLE_DOCS[0]["text"]


def test_get_indexed_sources():
    vs2.index_documents(SAMPLE_DOCS)
    sources = vs2.get_indexed_sources()
    assert sources == {
        "aaa": {"total_pages": 1, "source": "geo.pdf"},
        "bbb": {"total_pages": 3, "source": "sci.pdf"},
    }
