import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st

from src.config import DATA_DIR
from src.generator import generate_response
from src.ingestion import load_and_chunk
from src.vector_store import clear_index, get_indexed_sources, index_documents, remove_source

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A - RAG Pipeline")


def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = get_indexed_sources()
        

def _index_pdf(raw_bytes: bytes, name:str, file_hash:str):
    with st.spinner(f"Indexing {name}..."):
        chunks = load_and_chunk(BytesIO(raw_bytes), name, file_hash)
        index_documents(chunks) 
    total_pages = chunks[0]["metadata"]["total_pages"] if chunks else 0
    st.session_state.sources[file_hash] = {"source": name, "total_pages": total_pages}


def render_app():
    st.header("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded:
        raw_bytes = uploaded.read()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        uploaded.seek(0)

        if file_hash not in st.session_state.sources:
            file_path = Path(uploaded.name)
            save_path: Path = DATA_DIR / f"{file_path.stem}_{file_hash}{file_path.suffix}"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(raw_bytes)
            _index_pdf(raw_bytes, uploaded.name, file_hash)


def render_chat():
    st.header("Ask a Question")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- Page {s['page']} of **{s['source']}**")

    if question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.sources:
            st.warning("No documents indexed yet.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = generate_response(question)
                except Exception as e:      # noqa: BLE001
                    st.error(f"Generation failed: {e}") 
                    st.stop()
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.write(f"- Page {s['page']} of **{s['source']}**")
        st.session_state.messages.append({"content": question, "role": "user"})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})


def render_sidebar():
    st.sidebar.markdown("### Indexed Documents")
    if not st.session_state.sources:
        st.sidebar.caption("No documents indexed.")
        return

    for file_hash, info in sorted(st.session_state.sources.items(), key=lambda x: x[1]["source"]):
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        col1.write(info["source"])
        col2.write(f"{info['total_pages']}p")
        if col3.button("❌", key=f"rm_{file_hash}"):
            remove_source(file_hash)
            file_path = Path(info["source"])
            pdf_path = DATA_DIR / f"{file_path.stem}_{file_hash}{file_path.suffix}"
            pdf_path.unlink(missing_ok=True)
            st.session_state.sources.pop(file_hash, None)
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Remove all"):
        clear_index()
        for f in DATA_DIR.iterdir():
            if f.is_file():
                f.unlink()
        st.session_state.sources.clear()
        st.rerun()


def main():
    _init_session()
    render_app()
    render_chat()
    render_sidebar()


if __name__ == "__main__":
    main()
