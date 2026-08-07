import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import uuid4

import streamlit as st

try:
    from streamlit.runtime.scriptrunner_utils.exceptions import StopException
except ImportError:
    StopException = BaseException
    
from src.config import DATA_DIR, OPEN_PDF_SYSTEM_VIEWER
from src.conversation import (
    conversation_exists,
    delete_conversation,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
)
from src.generator import generate_stream
from src.ingestion import load_and_chunk
from src.pdf import open_pdf
from src.vector_store import clear_index, get_indexed_sources, index_documents, remove_source

st.set_page_config(page_title="Document Q&A", page_icon="📄")


class PartialStreamError(Exception):
    def __init__(self, partial: str, sources: list[dict], cause: BaseException):
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
    st.session_state.chat_input = ""


def _new_conversation():
    conversation_id = uuid4().hex
    st.session_state.conversations[conversation_id] = {"title": "Untitled"}
    st.session_state.active_conversation = conversation_id
    st.session_state.rename_target = None
    st.session_state.regenerate_index = None
    st.session_state.chat_input = ""


def _start_rename(conversation_id: str):
    st.session_state.rename_target = conversation_id
    st.session_state.rename_input = _get_conversation(conversation_id)["title"]
    st.session_state.active_conversation = conversation_id
    st.session_state.chat_input = ""


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
        st.session_state.chat_input = ""


def _pdf_save_path(name: str, file_hash: str) -> Path:
    return DATA_DIR / f"{Path(name).stem}_{file_hash}{Path(name).suffix}"


def _remove_document(file_hash: str):
    remove_source(file_hash)
    info = st.session_state.sources[file_hash]
    _pdf_save_path(info["source"], file_hash).unlink(missing_ok=True)
    st.session_state.sources.pop(file_hash, None)


def _open_pdf(file_hash: str):
    info = st.session_state.sources[file_hash]
    path = _pdf_save_path(info["source"], file_hash)
    if not path.exists():
        st.error(f"PDF not found on disk: {info['source']}")
        return
    open_pdf(path)


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
    uploaded = st.file_uploader("Choose a PDF", type="pdf", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            raw_bytes = file.read()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            file.seek(0)

            if file_hash not in st.session_state.sources:
                save_path = _pdf_save_path(file.name, file_hash)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    save_path.write_bytes(raw_bytes)
                    _index_pdf(raw_bytes, file.name, file_hash)
                except NoTextError as e:
                    save_path.unlink(missing_ok=True)
                    st.error(f"No extractable text found in {e}; it won't be searchable.")
                except Exception as e:  # noqa: BLE001
                    save_path.unlink(missing_ok=True)
                    st.error(f"Failed to index {file.name}: {e}")


def _request_regenerate(index: int):
    st.session_state.regenerate_index = index


def _stream_answer(question: str, history: list[dict] | None, index: int) -> tuple[str, list[dict]]:
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption("Thinking...")
        token_stream, sources = generate_stream(question, history)
        placeholder.empty()
        collected: list[str] = []

        def _tee() -> Iterator[str]:
            for token in token_stream:
                collected.append(token)
                yield token

        try:
            answer = cast(str, st.write_stream(_tee()))
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- Page {s['page']} of **{s['source']}**")
            _render_actions(index)
        except Exception as e:
            if collected:
                raise PartialStreamError("".join(collected), sources, e) from None
            raise
        except StopException as e:
            if collected:
                raise PartialStreamError("".join(collected), sources, e) from None
            raise

    return answer, sources


def _generate_answer(question: str, target_index: int | None):
    conversation = _active()
    messages = conversation["messages"]
    start = (target_index - 1) if target_index is not None else (len(messages))
    history = messages[:start]
    assistant_index = target_index if target_index is not None else len(messages) + 1
    conversation_id = st.session_state.active_conversation
    try:
        answer, sources = _stream_answer(question, history, assistant_index)
    except PartialStreamError as e:
        if isinstance(e.cause, StopException):
            content, srcs, error = e.partial, e.sources, "Answer was stopped"
        else:
            content, srcs, error = e.partial, e.sources, f"Answer was interrupted: {e.cause}"
            st.error(f"Answer was interrupted: {e.cause}")
        
    except Exception as e:  # noqa: BLE001
        st.error(f"An internal error has occurred: {e}")
        content, srcs, error = "", [], f"An internal error has occurred: {e}"
    else:
        content, srcs, error = answer, sources, None

    assistant_msg = {"role": "assistant", "content": content, "sources": srcs}
    if error:
        assistant_msg["error"] = error
    if target_index is None:
        messages.append({"content": question, "role": "user"})
        messages.append(assistant_msg)
    else:
        messages[target_index] = assistant_msg
    save_conversation(conversation_id, conversation["title"], messages)


def _rollback(index: int):
    conversation = _active()
    if conversation["messages"][index]["role"] == "user":
        st.session_state.chat_input = conversation["messages"][index]["content"]
        conversation["messages"] = conversation["messages"][:index]
    else:
        conversation["messages"] = conversation["messages"][:index+1]
    st.session_state.regenerate_index = None
    save_conversation(st.session_state.active_conversation, conversation["title"], conversation["messages"])


def _fork(index: int):
    current = _active()
    _new_conversation()

    if index == 0:
        st.session_state.chat_input = current["messages"][index]["content"]
        return
    
    new = _active()
    if current["messages"][index]["role"] == "user":
        new["messages"] = current["messages"][:index]
        st.session_state.chat_input = current["messages"][index]["content"]
    else:
        new["messages"] = current["messages"][:index+1]
    new_title = f"Branch of {current['title']}"
    new["title"] = new_title
    save_conversation(st.session_state.active_conversation, new_title, new["messages"])


def _render_actions(index: int):
    conversation_id = st.session_state.active_conversation
    with st.container(horizontal=True, border=False):
        st.button("Rollback", icon=":material/undo:", type="tertiary",
                key=f"rollback_{conversation_id}_{index}", on_click=_rollback, args=(index,),
                help="Remove this message and everything after it")
        st.button("Fork", icon=":material/call_split:", type="tertiary",
                key=f"fork_{conversation_id}_{index}", on_click=_fork, args=(index,),
                help="Start a new conversation from this point")


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
            if msg.get("sources", []):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- Page {s['page']} of **{s['source']}**")
            if msg.get("error"):
                st.error(msg["error"])
            _render_actions(i)

    if question := st.chat_input("Ask a question about your documents...", key="chat_input", submit_mode="stop"):
        st.session_state.regenerate_index = None
        new = not conversation_exists(st.session_state.active_conversation)
        if new:
            conversation["title"] = question.strip().splitlines()[0][:50] or "Untitled"
        user_index = len(messages)
        with st.chat_message("user"):
            st.markdown(question)
            if st.session_state.sources:
                _render_actions(user_index)
        if not st.session_state.sources:
            st.warning("No documents indexed yet.")
            return
        
        _generate_answer(question, None)

        if new:
            st.rerun()
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
                    if OPEN_PDF_SYSTEM_VIEWER:
                        with st.container(horizontal=True):
                            st.button(
                                "Open",
                                icon=":material/open_in_new:",
                                key=f"open_pdf_{file_hash}",
                                on_click=_open_pdf,
                                args=(file_hash,),
                                width="stretch",
                            )
                            st.button(
                                "Remove",
                                icon=":material/delete:",
                                key=f"rm_{file_hash}",
                                on_click=_remove_document,
                                args=(file_hash,),
                                width="stretch",
                            )
                    else:
                        st.download_button(
                            "Download",
                            data=_pdf_save_path(info["source"], file_hash).read_bytes(),
                            file_name=info["source"],
                            mime="application/pdf",
                            key=f"dl_{file_hash}",
                            width="stretch",
                        )
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


def _header_title() -> str:
    conversation_id = st.session_state.active_conversation
    if not conversation_exists(conversation_id):
        return "New conversation"
    return _get_conversation(conversation_id)["title"]


def main():
    title_container = st.empty()
    _init_session()
    with title_container:
        st.title(_header_title())
    render_sidebar()
    render_upload()
    render_chat()
    with title_container:
        st.title(_header_title())


if __name__ == "__main__":
    main()
