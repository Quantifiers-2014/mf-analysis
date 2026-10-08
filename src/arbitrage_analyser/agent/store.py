"""Conversation storage, in its own SQLite file (default data/chat.db).

Kept apart from the fund database so fund data can be rebuilt without losing conversations.
Every user message and every answer is stored, with the tool calls behind it, the model,
token counts, timing and the Langfuse trace id, so any answer can be audited later.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from arbitrage_analyser.db import DEFAULT_DB_PATH

CHAT_DB_ENV = "ARBITRAGE_CHAT_DB"
DEFAULT_CHAT_DB_PATH = DEFAULT_DB_PATH.parent / "chat.db"

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


@dataclass(frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]]
    trace_id: str | None


def chat_db_path() -> Path:
    return Path(os.environ.get(CHAT_DB_ENV, DEFAULT_CHAT_DB_PATH))


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = path or chat_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_conversation(conn: sqlite3.Connection, user_id: str, first_question: str) -> str:
    conversation_id = uuid.uuid4().hex
    title = " ".join(first_question.split())[:60] or "New conversation"
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conversation_id, user_id, title, now, now),
    )
    return conversation_id


def list_conversations(
    conn: sqlite3.Connection, user_id: str, limit: int = 30
) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations WHERE user_id=? "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def add_message(
    conn: sqlite3.Connection,
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
    cursor = conn.execute(
        """INSERT INTO messages (conversation_id, role, content, tool_calls, model, prompt_version,
             input_tokens, output_tokens, latency_ms, trace_id, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    )
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))
    return int(cursor.lastrowid or 0)


def read_messages(conn: sqlite3.Connection, conversation_id: str) -> list[StoredMessage]:
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, trace_id FROM messages "
        "WHERE conversation_id=? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [StoredMessage(r[0], r[1], r[2], json.loads(r[3]) if r[3] else [], r[4]) for r in rows]


def set_feedback(conn: sqlite3.Connection, message_id: int, rating: int, comment: str = "") -> None:
    conn.execute(
        """INSERT INTO feedback (message_id, rating, comment, created_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating,
             comment=excluded.comment, created_at=excluded.created_at""",
        (message_id, rating, comment, _now()),
    )


def read_feedback(conn: sqlite3.Connection, message_id: int) -> int | None:
    row = conn.execute("SELECT rating FROM feedback WHERE message_id=?", (message_id,)).fetchone()
    return None if row is None else int(row[0])
