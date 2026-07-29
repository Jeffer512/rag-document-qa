from pathlib import Path

from src.ingestion import _compute_hash, chunk_documents, load_pdf


def _create_dummy_pdf(path: Path, text: str = "Hello World. ") -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 700, text)
    c.save()


def test_load_pdf_metadata_fields(tmp_path):
    pdf = tmp_path / "r.pdf"
    _create_dummy_pdf(pdf, "X. " * 50)

    docs = load_pdf(pdf, "r.pdf", "abc123")
    assert len(docs) == 1
    meta = docs[0]["metadata"]
    assert meta["source"] == "r.pdf"
    assert meta["page"] == 1
    assert meta["total_pages"] == 1
    assert meta["file_hash"] == "abc123"
    assert meta["id"] == "abc123_p1"


def test_load_pdf_multiple_pages(tmp_path):
    pdf = tmp_path / "m.pdf"
    _create_dummy_pdf(pdf, "A. " * 10)

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.append(pdf)
    writer.append(pdf)
    multi = tmp_path / "multi.pdf"
    writer.write(multi)

    docs = load_pdf(multi, "multi.pdf", "xyz789")
    assert len(docs) == 2
    assert docs[0]["id"] == "xyz789_p1"
    assert docs[1]["id"] == "xyz789_p2"
    assert docs[1]["metadata"]["page"] == 2
    assert docs[1]["metadata"]["total_pages"] == 2


def test_chunk_documents_sets_id_and_index():
    docs = [
        {
            "id": "abc123_p1",
            "text": "A " * 600,
            "metadata": {"source": "a.pdf", "page": 1, "total_pages": 1, "file_hash": "abc123", "id": "abc123_p1"},
        }
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c["metadata"]["chunk_index"] == i
        assert c["id"] == f"abc123_p1_c{i}"
        assert c["metadata"]["file_hash"] == "abc123"
        assert c["metadata"]["source"] == "a.pdf"
        assert c["metadata"]["page"] == 1


def test_compute_hash_deterministic(tmp_path):
    pdf = tmp_path / "a.pdf"
    _create_dummy_pdf(pdf, "Same content. " * 30)
    h1 = _compute_hash(pdf)
    h2 = _compute_hash(pdf)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_hash_different_content(tmp_path):
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    _create_dummy_pdf(pdf1, "Content A. " * 30)
    _create_dummy_pdf(pdf2, "Content B. " * 30)
    assert _compute_hash(pdf1) != _compute_hash(pdf2)


def test_compute_hash_bytesio(tmp_path):
    pdf = tmp_path / "t.pdf"
    _create_dummy_pdf(pdf, "Hello. " * 20)
    raw = pdf.read_bytes()
    from io import BytesIO

    h = _compute_hash(BytesIO(raw))
    assert len(h) == 16



