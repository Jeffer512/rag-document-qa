import hashlib
from io import BytesIO
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(pdf_source: str | Path | BytesIO, source_name: str, file_hash: str) -> list[dict]:
    reader = PdfReader(pdf_source)
    documents: list[dict] = []
    total_pages = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            cleaned_text = " ".join(text.split())
            doc_id = f"{file_hash}_p{i}"
            documents.append(
                {
                    "id": doc_id,
                    "text": cleaned_text,
                    "metadata": {
                        "source": source_name,
                        "page": i,
                        "total_pages": total_pages,
                        "file_hash": file_hash,
                        "id": doc_id,
                    },
                }
            )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunked: list[dict] = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for idx, chunk_text in enumerate(chunks):
            chunked.append(
                {
                    "id": f"{doc['id']}_c{idx}",
                    "text": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                }
            )
    return chunked


def _compute_hash(pdf_source: str | Path | BytesIO) -> str:
    if isinstance(pdf_source, BytesIO):
        raw = pdf_source.read()
        pdf_source.seek(0)
    else:
        raw = Path(pdf_source).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_and_chunk(
    pdf_source: str | Path | BytesIO, source_name: str, file_hash: str | None = None
) -> list[dict]:
    fh = file_hash or _compute_hash(pdf_source)
    docs = load_pdf(pdf_source, source_name, fh)
    return chunk_documents(docs)
