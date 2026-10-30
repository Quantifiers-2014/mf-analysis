"""Conversation storage: a hosted Postgres database when DATABASE_URL is set (the live app, on
Neon), else a SQLite file on this machine (default data/chat.db; tests pass their own file).

Streamlit Cloud wipes its disk when the app reboots, so the live app must use DATABASE_URL, or
its conversations are lost. Kept apart from the fund database so fund data can be rebuilt
without losing conversations. Every user message and every answer is stored, with the tool
calls behind it, the model, token counts, timing and the Langfuse trace id, so any answer can
be audited later.

Postgres: one small connection pool per process (Streamlit reruns the script on every click),
connecting through Neon's pooled connection string. A connection is checked before use, so one
dropped while the database slept is replaced; the first request after a sleep waits for Neon to
wake (POOL_TIMEOUT_SECONDS).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from arbitrage_analyser.db import DEFAULT_DB_PATH

CHAT_DB_ENV = "ARBITRAGE_CHAT_DB"
DATABASE_URL_ENV = "DATABASE_URL"
DEFAULT_CHAT_DB_PATH = DEFAULT_DB_PATH.parent / "chat.db"
POOL_MAX_SIZE = 4  # connections per app process; Neon's pooler shares them further
POOL_TIMEOUT_SECONDS = 30  # wait for a connection, e.g. while Neon wakes from sleep

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    tool_calls       TEXT,            -- JSON list of {name, input, output, is_error}
    model            TEXT,
    prompt_version   TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    latency_ms       INTEGER,
    trace_id         TEXT,            -- Langfuse trace, when tracing is on
    error            TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);
CREATE TABLE IF NOT EXISTS feedback (
    message_id  INTEGER PRIMARY KEY REFERENCES messages(id),
    rating      INTEGER NOT NULL CHECK (rating IN (0, 1)),   -- 1 = thumbs up
    comment     TEXT,
    created_at  TEXT NOT NULL
);
"""


# The same tables in Postgres: an identity column instead of AUTOINCREMENT.
POSTGRES_SCHEMA = SCHEMA.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
).replace("message_id  INTEGER PRIMARY KEY", "message_id  BIGINT PRIMARY KEY")


@dataclass(frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]]
    trace_id: str | None


def chat_db_path() -> Path:
    return Path(os.environ.get(CHAT_DB_ENV, DEFAULT_CHAT_DB_PATH))


def database_url() -> str | None:
    return os.environ.get(DATABASE_URL_ENV, "").strip() or None


def storage_label() -> str:
    """Where conversations are kept, for the sidebar."""
    return "hosted database" if database_url() else "this machine only"


class ChatDB:
    """A connection to the chat tables. SQL is written with ? placeholders (SQLite style);
    for Postgres they become %s."""

    def __init__(self, conn: Any, postgres: bool) -> None:
        self.conn = conn
        self.postgres = postgres

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if self.postgres:
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)


_pools: dict[str, Any] = {}
_pool_lock = threading.Lock()


def _pool(url: str) -> Any:
    """One connection pool per database URL for this process, with the tables created once."""
    with _pool_lock:
        if url not in _pools:
            from psycopg_pool import ConnectionPool  # only needed with DATABASE_URL

            pool = ConnectionPool(
                url,
                min_size=0,
                max_size=POOL_MAX_SIZE,
                timeout=POOL_TIMEOUT_SECONDS,
                check=ConnectionPool.check_connection,
                # no server-side prepared statements: Neon's pooler (PgBouncer) shares
                # connections between clients
                kwargs={"prepare_threshold": None},
                open=True,
            )
            with pool.connection() as conn:
                conn.execute(POSTGRES_SCHEMA)
            _pools[url] = pool
        return _pools[url]


def close_pools() -> None:
    """Close the Postgres pools (tests; the app keeps them for its lifetime)."""
    with _pool_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


@contextmanager
def connect(path: Path | None = None) -> Iterator[ChatDB]:
    """The chat tables: Postgres when DATABASE_URL is set (and no `path` is given), else the
    SQLite file. Commits on success, rolls back on error."""
    url = None if path is not None else database_url()
    if url is not None:
        with _pool(url).connection() as conn:  # commits, or rolls back on error
            yield ChatDB(conn, postgres=True)
        return
    path = path or chat_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        yield ChatDB(conn, postgres=False)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_conversation(conn: ChatDB, user_id: str, first_question: str) -> str:
    conversation_id = uuid.uuid4().hex
    title = " ".join(first_question.split())[:60] or "New conversation"
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conversation_id, user_id, title, now, now),
    )
    return conversation_id


def list_conversations(conn: ChatDB, user_id: str, limit: int = 30) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations WHERE user_id=? "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def add_message(
    conn: ChatDB,
    conversation_id: str,
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
    trace_id: str | None = None,
    error: str | None = None,
) -> int:
    row = conn.execute(
        """INSERT INTO messages (conversation_id, role, content, tool_calls, model, prompt_version,
             input_tokens, output_tokens, latency_ms, trace_id, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
        (
            conversation_id,
            role,
            content,
            json.dumps(tool_calls, default=str) if tool_calls is not None else None,
            model,
            prompt_version,
            input_tokens,
            output_tokens,
            latency_ms,
            trace_id,
            error,
            _now(),
        ),
    ).fetchone()
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))
    return int(row[0])


def read_messages(conn: ChatDB, conversation_id: str) -> list[StoredMessage]:
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, trace_id FROM messages "
        "WHERE conversation_id=? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [StoredMessage(r[0], r[1], r[2], json.loads(r[3]) if r[3] else [], r[4]) for r in rows]


def set_feedback(conn: ChatDB, message_id: int, rating: int, comment: str = "") -> None:
    conn.execute(
        """INSERT INTO feedback (message_id, rating, comment, created_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating,
             comment=excluded.comment, created_at=excluded.created_at""",
        (message_id, rating, comment, _now()),
    )


def read_feedback(conn: ChatDB, message_id: int) -> int | None:
    row = conn.execute("SELECT rating FROM feedback WHERE message_id=?", (message_id,)).fetchone()
    return None if row is None else int(row[0])
