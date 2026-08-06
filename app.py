import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import cast
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


class PartialStreamError(Exception):
    def __init__(self, partial: str, sources: list[dict], cause: Exception):
        super().__init__(f"Stream interrupted after partial output: {cause}")
        self.partial = partial
        self.sources = sources
        self.cause = cause


class NoTextError(Exception):
    pass


def _init_session():
    st.session_state.setdefault("sources", get_indexed_sources())
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("active_conversation", None)
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("regenerate_index", None)
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
    st.session_state.regenerate_index = None


def _new_conversation():
    conversation_id = uuid4().hex
    st.session_state.conversations[conversation_id] = {"title": "Untitled"}
    st.session_state.active_conversation = conversation_id
    st.session_state.rename_target = None
    st.session_state.regenerate_index = None


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
        if not chunks:
            raise NoTextError(name)
        index_documents(chunks)
    st.session_state.sources[file_hash] = {"source": name, "total_pages": chunks[0]["metadata"]["total_pages"]}


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
            try:
                save_path.write_bytes(raw_bytes)
                _index_pdf(raw_bytes, uploaded.name, file_hash)
            except NoTextError as e:
                save_path.unlink(missing_ok=True)
                st.error(f"No extractable text found in {e}; it won't be searchable.")
            except Exception as e:  # noqa: BLE001
                save_path.unlink(missing_ok=True)
                st.error(f"Failed to index {uploaded.name}: {e}")


def _request_regenerate(index: int):
    st.session_state.regenerate_index = index


def _stream_answer(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    with st.spinner("Retrieving context..."):
        token_stream, sources = generate_stream(question, history)

    with st.chat_message("assistant"):
        collected: list[str] = []

        def _tee() -> Iterator[str]:
            for token in token_stream:
                collected.append(token)
                yield token

        try:
            answer = cast(str, st.write_stream(_tee()))
        except Exception as e:
            if collected:
                raise PartialStreamError("".join(collected), sources, e) from None
            raise
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- Page {s['page']} of **{s['source']}**")
    return answer, sources


def _generate_answer(question: str, target_index: int | None):
    conversation = _active()
    messages = conversation["messages"]
    start = (target_index - 1) if target_index is not None else (len(messages) - 1)
    history = messages[:start]
    try:
        answer, sources = _stream_answer(question, history)
    except PartialStreamError as e:
        st.error(f"Answer was interrupted: {e.cause}")
        content, srcs, error = e.partial, e.sources, f"Answer was interrupted: {e.cause}"
    except Exception as e:  # noqa: BLE001
        st.error(f"An internal error has occurred: {e}")
        content, srcs, error = "", [], f"An internal error has occurred: {e}"
    else:
        content, srcs, error = answer, sources, None

    assistant_msg = {"role": "assistant", "content": content, "sources": srcs}
    if error:
        assistant_msg["error"] = error
    if target_index is None:
        messages.append(assistant_msg)
    else:
        messages[target_index] = assistant_msg
    save_conversation(st.session_state.active_conversation, conversation["title"], messages)


def render_chat():
    conversation = _active()
    messages = conversation["messages"]
    regenerate_index = st.session_state.regenerate_index

    for i, msg in enumerate(messages):
        if i == regenerate_index and msg["role"] == "assistant":
            continue
        with st.chat_message(msg["role"]):
            if msg["content"]:
                st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- Page {s['page']} of **{s['source']}**")
            if msg.get("error"):
                st.error(msg["error"])

    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.regenerate_index = None
        if not messages:
            conversation["title"] = question.strip().splitlines()[0][:50] or "Untitled"
        with st.chat_message("user"):
            st.markdown(question)
        if not st.session_state.sources:
            st.warning("No documents indexed yet.")
            return
        messages.append({"content": question, "role": "user"})
        _generate_answer(question, None)
    elif regenerate_index is not None:
        st.session_state.regenerate_index = None
        if 0 < regenerate_index < len(messages):
            _generate_answer(messages[regenerate_index - 1]["content"], regenerate_index)

    if messages and messages[-1]["role"] == "assistant":
        st.button(
            "Regenerate",
            icon=":material/refresh:",
            key="regenerate_last",
            on_click=_request_regenerate,
            args=(len(messages) - 1,),
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
