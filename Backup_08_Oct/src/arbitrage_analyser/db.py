"""SQLite storage: schema, writes and reads. All dates are stored as ISO strings (YYYY-MM-DD)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "arbitrage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS funds (
    amfi_code    INTEGER PRIMARY KEY,
    scheme_name  TEXT NOT NULL,
    isin         TEXT NOT NULL,
    category     TEXT,
    launch_date  TEXT
);
CREATE TABLE IF NOT EXISTS nav (
    amfi_code  INTEGER NOT NULL,
    date       TEXT NOT NULL,
    nav        REAL NOT NULL,
    PRIMARY KEY (amfi_code, date)
);
CREATE TABLE IF NOT EXISTS benchmark (
    index_name TEXT NOT NULL,
    date       TEXT NOT NULL,
    value      REAL NOT NULL,
    PRIMARY KEY (index_name, date)
);
CREATE TABLE IF NOT EXISTS ter (
    amfi_code         INTEGER NOT NULL,
    date              TEXT NOT NULL,
    base_ter          REAL NOT NULL,
    brokerage         REAL,
    transaction_cost  REAL,
    statutory_levies  REAL,
    total_ter         REAL NOT NULL,
    PRIMARY KEY (amfi_code, date)
);
CREATE TABLE IF NOT EXISTS aaum (
    amfi_code     INTEGER NOT NULL,
    quarter_end   TEXT NOT NULL,
    aaum_crore    REAL NOT NULL,
    PRIMARY KEY (amfi_code, quarter_end)
);
CREATE TABLE IF NOT EXISTS validation_flags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset     TEXT NOT NULL,
    subject     TEXT NOT NULL,
    date        TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    detail      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'reviewed')),
    created_at  TEXT NOT NULL,
    UNIQUE (dataset, subject, date, check_name)
);
CREATE TABLE IF NOT EXISTS refresh_log (
    dataset      TEXT PRIMARY KEY,
    refreshed_at TEXT NOT NULL,
    rows         INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class Flag:
    """A validation finding. `subject` is the fund code or index name it relates to."""

    dataset: str
    subject: str
    date: str
    check_name: str
    detail: str


@contextmanager
def connect(path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open the database, create tables if needed, commit on success, roll back on error."""
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


# ---------- writes ----------


def upsert_fund(
    conn: sqlite3.Connection,
    amfi_code: int,
    scheme_name: str,
    isin: str,
    category: str | None,
    launch_date: date | None,
) -> None:
    conn.execute(
        """INSERT INTO funds (amfi_code, scheme_name, isin, category, launch_date)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(amfi_code) DO UPDATE SET scheme_name=excluded.scheme_name,
             isin=excluded.isin, category=excluded.category, launch_date=excluded.launch_date""",
        (amfi_code, scheme_name, isin, category, launch_date.isoformat() if launch_date else None),
    )


def upsert_nav(conn: sqlite3.Connection, amfi_code: int, series: pd.Series) -> int:
    """Insert or replace NAVs. `series` is indexed by date with float values."""
    rows = [(amfi_code, d.date().isoformat(), float(v)) for d, v in series.items()]
    conn.executemany("INSERT OR REPLACE INTO nav (amfi_code, date, nav) VALUES (?, ?, ?)", rows)
    return len(rows)


def insert_benchmark(conn: sqlite3.Connection, index_name: str, series: pd.Series) -> int:
    """Insert benchmark values for dates not already stored. Existing dates are left unchanged."""
    rows = [(index_name, d.date().isoformat(), float(v)) for d, v in series.items()]
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO benchmark (index_name, date, value) VALUES (?, ?, ?)", rows
    )
    return conn.total_changes - before


def upsert_ter(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """`frame` columns: amfi_code, date, base_ter, brokerage, transaction_cost,
    statutory_levies, total_ter (percent values)."""
    cols = [
        "amfi_code",
        "date",
        "base_ter",
        "brokerage",
        "transaction_cost",
        "statutory_levies",
        "total_ter",
    ]
    rows = [
        (
            int(r.amfi_code),
            r.date.date().isoformat(),
            float(r.base_ter),
            _opt(r.brokerage),
            _opt(r.transaction_cost),
            _opt(r.statutory_levies),
            float(r.total_ter),
        )
        for r in frame[cols].itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO ter (amfi_code, date, base_ter, brokerage, transaction_cost,
           statutory_levies, total_ter) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def upsert_aaum(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """`frame` columns: amfi_code, quarter_end (Timestamp), aaum_crore."""
    rows = [
        (int(r.amfi_code), r.quarter_end.date().isoformat(), float(r.aaum_crore))
        for r in frame[["amfi_code", "quarter_end", "aaum_crore"]].itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO aaum (amfi_code, quarter_end, aaum_crore) VALUES (?, ?, ?)", rows
    )
    return len(rows)


def record_flags(conn: sqlite3.Connection, flags: Iterable[Flag]) -> int:
    """Store new flags. A flag already stored (same dataset, subject, date, check) is kept as is,
    so a reviewed flag is not reopened by the next refresh."""
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO validation_flags
           (dataset, subject, date, check_name, detail, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
        [(f.dataset, f.subject, f.date, f.check_name, f.detail, _now()) for f in flags],
    )
    return conn.total_changes - before


def mark_flag_reviewed(conn: sqlite3.Connection, flag_id: int) -> None:
    conn.execute("UPDATE validation_flags SET status='reviewed' WHERE id=?", (flag_id,))


def log_refresh(conn: sqlite3.Connection, dataset: str, rows: int) -> None:
    conn.execute(
        """INSERT INTO refresh_log (dataset, refreshed_at, rows) VALUES (?, ?, ?)
           ON CONFLICT(dataset) DO UPDATE SET refreshed_at=excluded.refreshed_at,
             rows=excluded.rows""",
        (dataset, _now(), rows),
    )


def _opt(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


# ---------- reads ----------


def read_nav(conn: sqlite3.Connection, amfi_code: int) -> pd.Series:
    frame = pd.read_sql_query(
        "SELECT date, nav FROM nav WHERE amfi_code=? ORDER BY date", conn, params=(amfi_code,)
    )
    return _to_series(frame, "nav")


def read_benchmark(conn: sqlite3.Connection, index_name: str) -> pd.Series:
    frame = pd.read_sql_query(
        "SELECT date, value FROM benchmark WHERE index_name=? ORDER BY date",
        conn,
        params=(index_name,),
    )
    return _to_series(frame, "value")


def _to_series(frame: pd.DataFrame, column: str) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
    return pd.Series(frame[column].astype(float).to_numpy(), index=index, name=column)


def read_funds(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM funds", conn)


def read_latest_ter(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT t.* FROM ter t
           JOIN (SELECT amfi_code, MAX(date) AS date FROM ter GROUP BY amfi_code) m
             ON t.amfi_code = m.amfi_code AND t.date = m.date""",
        conn,
    )


def read_latest_aaum(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT a.* FROM aaum a
           JOIN (SELECT amfi_code, MAX(quarter_end) AS q FROM aaum GROUP BY amfi_code) m
             ON a.amfi_code = m.amfi_code AND a.quarter_end = m.q""",
        conn,
    )


def read_flags(conn: sqlite3.Connection, status: str = "open") -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM validation_flags WHERE status=? ORDER BY date DESC, id DESC",
        conn,
        params=(status,),
    )


def read_refresh_log(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM refresh_log", conn)
