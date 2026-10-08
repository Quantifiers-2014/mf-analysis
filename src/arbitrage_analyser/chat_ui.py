"""The "Ask the analyst" chat tab and its sidebar (conversation list)."""

from __future__ import annotations

import os

import streamlit as st

from arbitrage_analyser import db
from arbitrage_analyser.agent import agent, providers, store, tracing
from arbitrage_analyser.agent.prompt import PROMPT_VERSION
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.runtime import db_path

CONVERSATION_KEY = "chat_conversation_id"
USER_KEY = "chat_user"
EXAMPLES = [
    "Which arbitrage fund has the highest 3-year return?",
    "How consistent were Kotak and SBI over rolling 1-year periods?",
    "Is any of the data out of date or flagged?",
]


def _error_text(exc: Exception) -> str:
    """Plain-words message for a failed answer."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return (
            "The AI model's usage limit was reached (free tiers allow only a few requests per "
            "minute and per day). Wait a minute and ask again, or try again tomorrow."
        )
    if status in (401, 403):
        return "The API key was rejected. Check the key in your `.env` file, then restart the app."
    return f"Sorry, I couldn't answer that: {exc}"


def chat_sidebar() -> None:
    with st.sidebar:
        st.header("Ask the analyst")
        user = st.text_input(
            "Your name",
            value="local",
            key=USER_KEY,
            help="Conversations are saved under this name, so you can come back to them.",
        )
        if st.button("New conversation", icon=":material/add:", width="stretch"):
            st.session_state[CONVERSATION_KEY] = None
        with store.connect() as conn:
            conversations = store.list_conversations(conn, user.strip() or "local")
        if conversations:
            st.caption("Recent conversations")
        current = st.session_state.get(CONVERSATION_KEY)
        for conv in conversations:
            label = conv["title"] if conv["id"] != current else f"▶ {conv['title']}"
            if st.button(label, key=f"conv_{conv['id']}", width="stretch"):
                st.session_state[CONVERSATION_KEY] = conv["id"]
                st.rerun()
        info = providers.selected_provider()
        model = os.environ.get(providers.MODEL_ENV) or (info.default_model if info else "")
        st.caption(f"Model: {info.label} · {model}" if info else "Model: not set up")
        st.caption(
            "Langfuse tracing: on" if tracing.enabled() else "Langfuse tracing: off (no keys set)"
        )


def _show_message(message: store.StoredMessage) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.role != "assistant":
            return
        if message.tool_calls:
            with st.expander(f"Data used ({len(message.tool_calls)} tool call(s))"):
                for call in message.tool_calls:
                    status = " (error)" if call.get("is_error") else ""
                    st.markdown(f"**{call['name']}**{status} · input `{call['input']}`")
                    st.json(call["output"], expanded=False)
        rating = st.feedback("thumbs", key=f"fb_{message.id}")
        if rating is not None:
            with store.connect() as conn:
                if store.read_feedback(conn, message.id) != rating:
                    store.set_feedback(conn, message.id, int(rating))
                    tracing.score_feedback(message.trace_id, int(rating))


def chat_tab(config: AppConfig) -> None:
    problems = providers.setup_problems()
    if problems:
        st.info(
            "The chat needs a little setup first:\n\n" + "\n\n".join(f"- {p}" for p in problems)
        )
        return

    user_id = (st.session_state.get(USER_KEY) or "local").strip() or "local"
    conversation_id = st.session_state.get(CONVERSATION_KEY)
    history: list[store.StoredMessage] = []
    if conversation_id:
        with store.connect() as conn:
            history = store.read_messages(conn, conversation_id)
    else:
        st.caption(
            "Ask about the funds in the database. Answers come only from the stored data, with "
            "sources. For example:\n\n" + "\n".join(f"- *{q}*" for q in EXAMPLES)
        )
    for message in history:
        _show_message(message)

    question = st.chat_input("Ask about the funds...", key="chat_input")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with store.connect() as conn:
        if not conversation_id:
            conversation_id = store.create_conversation(conn, user_id, question)
            st.session_state[CONVERSATION_KEY] = conversation_id
        store.add_message(conn, conversation_id, "user", question)

    with st.chat_message("assistant"), st.spinner("Looking at the data..."):
        try:
            with db.connect(db_path()) as fund_conn:
                result = agent.answer(
                    agent.make_provider(),
                    fund_conn,
                    config,
                    history,
                    question,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
        except Exception as exc:  # API, network or key problems: show them, keep the question
            st.error(_error_text(exc))
            return
    with store.connect() as conn:
        store.add_message(
            conn,
            conversation_id,
            "assistant",
            result.answer or "(no answer)",
            tool_calls=result.tool_calls,
            model=result.model,
            prompt_version=PROMPT_VERSION,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            trace_id=result.trace_id,
        )
    st.rerun()
