"""Chat history storage on SQLite (always) and on Postgres (when TEST_DATABASE_URL points to a
throwaway Postgres database; its chat tables are dropped and recreated)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arbitrage_analyser import db
from arbitrage_analyser.agent import agent, store
from arbitrage_analyser.agent.providers import AnthropicProvider
from tests.test_agent import FakeClient, seed, text_reply
from tests.test_cli_app import APP_PATH, env  # noqa: F401  (env is a fixture)

TEST_URL = os.environ.get("TEST_DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def postgres(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """DATABASE_URL set to the test Postgres, with empty chat tables."""
    import psycopg

    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS feedback, messages, conversations")
    store.close_pools()
    monkeypatch.setenv(store.DATABASE_URL_ENV, TEST_URL)
    yield TEST_URL
    store.close_pools()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=needs_postgres)])
def chat_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path | None]:
    """The `path` to pass to store.connect: a SQLite file, or None (DATABASE_URL)."""
    if request.param == "sqlite":
        yield tmp_path / "chat.db"
    else:
        request.getfixturevalue("postgres")
        yield None


def test_round_trip_and_feedback(chat_db: Path | None) -> None:
    with store.connect(chat_db) as conn:
        cid = store.create_conversation(conn, "asha", "  Which fund   is cheapest? ")
        first = store.add_message(conn, cid, "user", "Which fund is cheapest?")
        mid = store.add_message(
            conn, cid, "assistant", "Alpha.", tool_calls=[{"name": "list_funds"}], trace_id="tr1"
        )
        store.set_feedback(conn, mid, 1)
        store.set_feedback(conn, mid, 0)  # changing your mind overwrites
    assert mid > first
    with store.connect(chat_db) as conn:  # a new connection sees it all
        assert store.list_conversations(conn, "asha")[0]["title"] == "Which fund is cheapest?"
        assert store.list_conversations(conn, "someone else") == []
        messages = store.read_messages(conn, cid)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].tool_calls == [{"name": "list_funds"}] and messages[1].trace_id == "tr1"
        assert store.read_feedback(conn, mid) == 0


def test_a_failed_turn_saves_nothing(chat_db: Path | None) -> None:
    with pytest.raises(RuntimeError), store.connect(chat_db) as conn:
        cid = store.create_conversation(conn, "asha", "lost question")
        raise RuntimeError("the save failed half-way")
    with store.connect(chat_db) as conn:
        assert store.list_conversations(conn, "asha") == []
        assert store.read_messages(conn, cid) == []


def test_conversations_are_listed_newest_first(chat_db: Path | None) -> None:
    with store.connect(chat_db) as conn:
        old = store.create_conversation(conn, "asha", "old")
        new = store.create_conversation(conn, "asha", "new")
        conn.execute("UPDATE conversations SET updated_at='2026-01-01T00:00:00' WHERE id=?", (old,))
        assert [c["id"] for c in store.list_conversations(conn, "asha")] == [new, old]


def test_storage_label(monkeypatch: pytest.MonkeyPatch) -> None:
    assert store.storage_label() == "this machine only"
    monkeypatch.setenv(store.DATABASE_URL_ENV, "postgresql://x")
    assert store.storage_label() == "hosted database"


@needs_postgres
def test_postgres_survives_a_new_pool(postgres: str) -> None:
    """What an app reboot does: a new process, a new pool, the same database."""
    with store.connect() as conn:
        cid = store.create_conversation(conn, "asha", "kept?")
        store.add_message(conn, cid, "user", "kept?")
    store.close_pools()
    with store.connect() as conn:
        assert [m.content for m in store.read_messages(conn, cid)] == ["kept?"]


@needs_postgres
def test_chat_tab_saves_to_postgres(
    postgres: str,
    env: Path,  # noqa: F811
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arbitrage_analyser.config import load_config

    with db.connect(env) as conn:
        seed(conn, load_config(config_file))
    monkeypatch.setenv("ARBITRAGE_AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = FakeClient([text_reply("Alpha leads.")])
    monkeypatch.setattr(agent, "make_provider", lambda: AnthropicProvider(client, "claude-test"))

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.chat_input(key="chat_input").set_value("Who leads?").run()
    assert not app.exception
    assert any("hosted database" in c.value for c in app.sidebar.caption)
    with store.connect() as conn:
        conversation = store.list_conversations(conn, "local")[0]
        messages = store.read_messages(conn, conversation["id"])
    assert [m.content for m in messages] == ["Who leads?", "Alpha leads."]
    assert not env.with_name("chat.db").exists()  # nothing written to the local file


def test_chat_tab_reports_an_unreachable_database(
    env: Path,  # noqa: F811
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arbitrage_analyser.config import load_config

    with db.connect(env) as conn:
        seed(conn, load_config(config_file))
    monkeypatch.setenv("ARBITRAGE_AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(store.DATABASE_URL_ENV, "postgresql://nobody@127.0.0.1:1/none")
    monkeypatch.setattr(store, "POOL_TIMEOUT_SECONDS", 2)
    store.close_pools()
    client = FakeClient([text_reply("never sent")])
    monkeypatch.setattr(agent, "make_provider", lambda: AnthropicProvider(client, "claude-test"))

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert any("could not be loaded" in w.value for w in app.sidebar.warning)
    app.chat_input(key="chat_input").set_value("Who leads?").run()
    assert not app.exception
    assert any("could not be saved or loaded" in e.value for e in app.error)
    assert client.requests == []  # the question is not sent when it cannot be saved
    store.close_pools()
