import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.config import DATA_DIR
from src.conversation import (
    delete_conversation,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
)
from src.generator import generate_stream
from src.ingestion import load_and_chunk
from src.vector_store import clear_index, get_indexed_sources, index_documents, remove_source

st.set_page_config(page_title="Document Q&A", page_icon="📄")


def _init_session():
    st.session_state.setdefault("sources", get_indexed_sources())
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("active_conversation", None)
    st.session_state.setdefault("rename_target", None)
    if not st.session_state.conversations:
        conversations = list_conversations()
        for conversation in conversations:
            st.session_state.conversations[conversation["conversation_id"]] = {
                "title": conversation["title"]
            }
        if conversations:
            st.session_state.active_conversation = conversations[0]["conversation_id"]
        else:
            _new_conversation()


def _get_conversation(conversation_id: str) -> dict:
    conversation = st.session_state.conversations.setdefault(conversation_id, {})
    conversation.setdefault("title", "Untitled")
    if "messages" not in conversation:
        conversation["messages"] = load_conversation(conversation_id)
    return conversation


def _active() -> dict:
    return _get_conversation(st.session_state.active_conversation)


def _open_conversation(conversation_id: str):
    st.session_state.active_conversation = conversation_id
    st.session_state.rename_target = None


def _new_conversation():
    conversation_id = uuid4().hex
    st.session_state.conversations[conversation_id] = {"title": "Untitled"}
    st.session_state.active_conversation = conversation_id
    st.session_state.rename_target = None


def _start_rename(conversation_id: str):
    st.session_state.rename_target = conversation_id
    st.session_state.rename_input = _get_conversation(conversation_id)["title"]
    st.session_state.active_conversation = conversation_id


def _save_rename():
    title = st.session_state.rename_input.strip()
    if title:
        rename_conversation(st.session_state.rename_target, title)
        _get_conversation(st.session_state.rename_target)["title"] = title
    st.session_state.rename_target = None


def _cancel_rename():
    st.session_state.rename_target = None


def _delete_conversation(conversation_id: str):
    delete_conversation(conversation_id)
    st.session_state.conversations.pop(conversation_id, None)
    if st.session_state.rename_target == conversation_id:
        st.session_state.rename_target = None
    if st.session_state.active_conversation == conversation_id:
        remaining = list_conversations()
        if remaining:
            st.session_state.active_conversation = remaining[0]["conversation_id"]
        else:
            _new_conversation()


def _pdf_save_path(name: str, file_hash: str) -> Path:
    return DATA_DIR / f"{Path(name).stem}_{file_hash}{Path(name).suffix}"


def _remove_document(file_hash: str):
    remove_source(file_hash)
    info = st.session_state.sources[file_hash]
    _pdf_save_path(info["source"], file_hash).unlink(missing_ok=True)
    st.session_state.sources.pop(file_hash, None)


def _remove_all_documents():
    clear_index()
    for f in DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            f.unlink()
    st.session_state.sources.clear()


def _index_pdf(raw_bytes: bytes, name: str, file_hash: str):
    with st.spinner(f"Indexing {name}..."):
        chunks = load_and_chunk(BytesIO(raw_bytes), name, file_hash)
        index_documents(chunks)
    total_pages = chunks[0]["metadata"]["total_pages"] if chunks else 0
    st.session_state.sources[file_hash] = {"source": name, "total_pages": total_pages}


def render_upload():
    st.header("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded:
        raw_bytes = uploaded.read()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        uploaded.seek(0)

        if file_hash not in st.session_state.sources:
            save_path = _pdf_save_path(uploaded.name, file_hash)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(raw_bytes)
            _index_pdf(raw_bytes, uploaded.name, file_hash)


def render_chat():
    conversation = _active()
    for msg in conversation["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- Page {s['page']} of **{s['source']}**")

    if question := st.chat_input("Ask a question about your documents..."):
        if not conversation["messages"]:
            conversation["title"] = question.strip().splitlines()[0][:50] or "Untitled"

        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.sources:
            st.warning("No documents indexed yet.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context..."):
                token_stream, sources = generate_stream(question)
            answer = st.write_stream(token_stream)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- Page {s['page']} of **{s['source']}**")

        conversation["messages"].append({"content": question, "role": "user"})
        conversation["messages"].append({"role": "assistant", "content": answer, "sources": sources})
        save_conversation(
            st.session_state.active_conversation,
            conversation["title"],
            conversation["messages"],
        )


def render_sidebar():
    st.sidebar.button(
        "New conversation",
        icon=":material/add:",
        on_click=_new_conversation,
        width="stretch",
    )

    with st.sidebar.expander("History", expanded=True, icon=":material/history:"):
        active = st.session_state.active_conversation
        for conversation in list_conversations():
            conversation_id = conversation["conversation_id"]
            title = ("▸ " if conversation_id == active else "") + conversation["title"]

            if conversation_id == st.session_state.rename_target:
                with st.container(horizontal=True):
                    st.text_input(
                        "Conversation title",
                        key="rename_input",
                        placeholder="Conversation title",
                        label_visibility="collapsed",
                    )
                    with st.container(horizontal=True, width="content"):
                        st.button(
                            "Save",
                            icon=":material/check:",
                            key=f"save_{conversation_id}",
                            on_click=_save_rename,
                            type="primary",
                        )
                        st.button(
                            "Cancel",
                            icon=":material/close:",
                            key=f"cancel_{conversation_id}",
                            on_click=_cancel_rename,
                        )
            else:
                row = st.columns([8, 1, 1], gap="xsmall")
                with row[0]:
                    st.button(
                        title,
                        key=f"open_{conversation_id}2",
                        on_click=_open_conversation,
                        args=(conversation_id,),
                        width="stretch",
                    )
                with row[1]:
                    st.button(
                        "",
                        icon=":material/edit:",
                        key=f"ren_{conversation_id}2",
                        on_click=_start_rename,
                        args=(conversation_id,),
                    )
                with row[2]:
                    st.button(
                        "",
                        icon=":material/delete:",
                        key=f"del_{conversation_id}2",
                        on_click=_delete_conversation,
                        args=(conversation_id,),
                    )

    with st.sidebar.expander("Indexed Documents", icon=":material/description:"):
        if not st.session_state.sources:
            st.caption("No documents indexed.")
        else:
            for file_hash, info in sorted(
                st.session_state.sources.items(), key=lambda x: x[1]["source"]
            ):
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    col1.write(info["source"])
                    col2.write(f"{info['total_pages']}p")
                    st.button(
                        "Remove",
                        icon=":material/delete:",
                        key=f"rm_{file_hash}",
                        on_click=_remove_document,
                        args=(file_hash,),
                        width="stretch",
                    )

            st.button(
                "Remove all",
                icon=":material/delete_sweep:",
                on_click=_remove_all_documents,
                width="stretch",
            )


def main():
    title_container = st.empty()
    _init_session()
    render_upload()
    render_chat()
    render_sidebar()
    conversation = _active()
    with title_container:    
        st.title("New conversation" if not conversation["messages"] else conversation["title"])


if __name__ == "__main__":
    main()
