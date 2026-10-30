"""The data store: CSV files in published/ hold every dataset; the database is a cache of them.

The daily GitHub job (update-data) loads the files, fetches new data, and writes the files back;
the app loads them on start. Streamlit Cloud keeps no database between restarts, so nothing that
only lives in the database survives.

Files (one row per key, sorted, so a change shows as a small diff):
  funds.csv          scheme name, category and launch date from the NAV source
  nav.csv            NAV per fund per day
  ter.csv            AMFI TER (Direct Plan) per fund per day
  benchmark.csv      benchmark index values (uploaded in the app)
  factsheet_values.csv  AUM, AAUM, turnover, Direct BER per fund per month (see factsheets.py)
  factsheet_chunks.csv  text of each fund's factsheet page, in chunks, per month (chatbot search)
  factsheet_embeddings.csv  an embedding (vector) per chunk text and model (for search by meaning)
  reviewed_flags.csv flags marked reviewed on the Data health screen

Loading checks every line; a bad line is skipped and reported, the rest loads, and that file is
then not written back until it is fixed. Validation flags are worked out again from the loaded
data, so they need no file of their own; only the review decisions are kept.
"""

from __future__ import annotations

import io
import math
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from arbitrage_analyser import db, factsheets
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.ingest import LoadResult, ter_flags
from arbitrage_analyser.validation import check_series

BENCHMARK_FILE = "benchmark.csv"
FACTSHEET_FILE = "factsheet_values.csv"
REVIEWED_FILE = "reviewed_flags.csv"
CHUNKS_FILE = "factsheet_chunks.csv"
EMBEDDINGS_FILE = "factsheet_embeddings.csv"

Parser = Callable[[str], object]


def _text(value: str) -> str:
    if not value.strip():
        raise ValueError("is empty")
    return value.strip()


def _optional_text(value: str) -> str | None:
    return value.strip() or None


def _day(value: str) -> str:
    text = value.strip()
    if len(text) != 10:
        raise ValueError(f"'{text}' is not YYYY-MM-DD")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"'{text}' is not YYYY-MM-DD") from None


def _optional_day(value: str) -> str | None:
    return _day(value) if value.strip() else None


def _code(value: str) -> int:
    text = value.strip()
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"'{text}' is not a scheme code")
    return int(text)


def _number(value: str) -> float:
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"'{text}' is not a number")
    return number


def _optional_number(value: str) -> float | None:
    return _number(value) if value.strip() else None


@dataclass(frozen=True)
class Table:
    """A database table kept as one CSV file."""

    name: str
    file: str
    columns: dict[str, Parser]  # column -> parser of the CSV text (raises ValueError)
    keys: tuple[str, ...]
    upsert: str  # SQL: insert a row, or update the stored row with the same keys

    def select(self) -> str:
        return f"SELECT {', '.join(self.columns)} FROM {self.name} ORDER BY {', '.join(self.keys)}"


def _upsert(name: str, columns: Sequence[str], keys: Sequence[str]) -> str:
    values = [c for c in columns if c not in keys]
    changed = " OR ".join(f"{c} IS NOT excluded.{c}" for c in values)
    return (
        f"INSERT INTO {name} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
        f"ON CONFLICT({', '.join(keys)}) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in values)
        + f" WHERE {changed}"
    )


def _table(name: str, file: str, columns: dict[str, Parser], keys: tuple[str, ...]) -> Table:
    return Table(name, file, columns, keys, _upsert(name, list(columns), keys))


TABLES = (
    _table(
        "funds",
        "funds.csv",
        {
            "amfi_code": _code,
            "scheme_name": _text,
            "isin": _text,
            "category": _optional_text,
            "launch_date": _optional_day,
        },
        ("amfi_code",),
    ),
    _table(
        "nav",
        "nav.csv",
        {"amfi_code": _code, "date": _day, "nav": _number},
        ("amfi_code", "date"),
    ),
    _table(
        "ter",
        "ter.csv",
        {
            "amfi_code": _code,
            "date": _day,
            "base_ter": _number,
            "brokerage": _optional_number,
            "transaction_cost": _optional_number,
            "statutory_levies": _optional_number,
            "total_ter": _number,
        },
        ("amfi_code", "date"),
    ),
    _table(
        "benchmark",
        BENCHMARK_FILE,
        {"index_name": _text, "date": _day, "value": _number},
        ("index_name", "date"),
    ),
    _table(
        "factsheet_chunks",
        CHUNKS_FILE,
        {
            "amfi_code": _code,
            "as_of": _day,
            "chunk_no": _code,
            "amc": _text,
            "page": _code,
            "section": _text,
            "text": _text,
            "text_hash": _text,
            "source_url": _text,
        },
        ("amfi_code", "as_of", "chunk_no"),
    ),
    _table(
        "factsheet_embeddings",
        EMBEDDINGS_FILE,
        {"text_hash": _text, "model": _text, "vector": _text},
        ("text_hash", "model"),
    ),
)
FILES = (*(t.file for t in TABLES), FACTSHEET_FILE, REVIEWED_FILE)
# Who writes which file: the data job never writes what the app saves, and the other way round,
# so their commits cannot overwrite each other.
JOB_FILES = ("funds.csv", "nav.csv", "ter.csv", FACTSHEET_FILE, CHUNKS_FILE, EMBEDDINGS_FILE)
APP_FILES = (BENCHMARK_FILE, REVIEWED_FILE)


# ---------- load ----------


@dataclass
class StoreLoad(LoadResult):
    """A load result, plus the files that did not load cleanly (they must not be written over
    from the database: rows that failed to load would be lost)."""

    damaged: set[str] = field(default_factory=set)


def load(conn: sqlite3.Connection, config: AppConfig, folder: Path, today: date) -> StoreLoad:
    """Replace the database tables with the files in `folder` (a missing file leaves its table
    as it is), work out the validation flags again and apply the review decisions.

    Rows for funds no longer in config are kept: removing a fund from config hides it, it does
    not delete its history.
    """
    result = StoreLoad()
    for table in TABLES:
        result.rows += _load_table(conn, table, folder / table.file, result)
    path = folder / FACTSHEET_FILE
    if path.exists():
        conn.execute("DELETE FROM factsheet")
        sheet = factsheets.import_csv(conn, config, path, today)
        result.rows += sheet.rows
        result.errors += sheet.errors
        if sheet.errors:
            result.damaged.add(FACTSHEET_FILE)
    _load_reviews(conn, folder / REVIEWED_FILE, result)
    conn.execute("DELETE FROM validation_flags")  # worked out again from the data just loaded
    result.new_flags += _flag_stored(conn, config)
    db.apply_reviews(conn)
    return result


def _read(path: Path, columns: Sequence[str], result: StoreLoad) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        result.errors.append(f"{path.name}: cannot be read ({exc})")
        result.damaged.add(path.name)
        return None
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        result.errors.append(f"{path.name}: missing columns {missing}; file not loaded")
        result.damaged.add(path.name)
        return None
    records: list[dict[str, str]] = frame.to_dict("records")
    return records


def _parse_lines(
    path: Path, columns: dict[str, Parser], records: list[dict[str, str]], result: StoreLoad
) -> list[tuple[object, ...]]:
    rows = []
    for line, raw in enumerate(records, start=2):
        try:
            rows.append(tuple(_parse(column, parse, raw) for column, parse in columns.items()))
        except ValueError as exc:
            result.errors.append(f"{path.name} line {line}: {exc}; line skipped")
            result.damaged.add(path.name)
    return rows


def _load_table(conn: sqlite3.Connection, table: Table, path: Path, result: StoreLoad) -> int:
    records = _read(path, list(table.columns), result)
    if records is None:
        return 0
    rows = _parse_lines(path, table.columns, records, result)
    conn.execute(f"DELETE FROM {table.name}")  # the file is the whole table
    conn.executemany(table.upsert, rows)  # upsert: a repeated key keeps its last line
    return len(rows)


def _parse(column: str, parse: Parser, raw: dict[str, str]) -> object:
    try:
        return parse(raw[column])
    except ValueError as exc:
        raise ValueError(f"{column} {exc}") from None


def _flag_stored(conn: sqlite3.Connection, config: AppConfig) -> int:
    """Validation flags for the stored data (the same checks as when it was fetched)."""
    settings = config.settings
    flags: list[db.Flag] = []
    for fund in config.funds:
        nav = db.read_nav(conn, fund.amfi_code)
        flags += check_series(nav, "nav", str(fund.amfi_code), settings, fund.name)
    bench = db.read_benchmark(conn, settings.benchmark_name)
    flags += check_series(
        bench, "benchmark", settings.benchmark_name, settings, settings.benchmark_name
    )
    ter = pd.read_sql_query("SELECT * FROM ter", conn, parse_dates=["date"])
    flags += ter_flags(ter[ter["amfi_code"].isin([f.amfi_code for f in config.funds])])
    return db.record_flags(conn, flags) + factsheets.check_latest(conn, config)


_REVIEW_COLUMNS: dict[str, Parser] = {
    "dataset": _text,
    "subject": _text,
    "date": _day,
    "check_name": _text,
}


def _load_reviews(conn: sqlite3.Connection, path: Path, result: StoreLoad) -> None:
    records = _read(path, list(_REVIEW_COLUMNS), result)
    if records is None:
        return
    rows = _parse_lines(path, _REVIEW_COLUMNS, records, result)
    conn.execute("DELETE FROM reviewed_flags")
    conn.executemany(
        f"INSERT OR IGNORE INTO reviewed_flags ({', '.join(db.FLAG_KEY)}) VALUES (?, ?, ?, ?)",
        rows,
    )


# ---------- save ----------


def render(conn: sqlite3.Connection, file: str) -> bytes:
    """The CSV text of one store file, from the database."""
    if file == FACTSHEET_FILE:
        frame = db.read_factsheets(conn)[list(db.FACTSHEET_COLUMNS)]
    elif file == REVIEWED_FILE:
        frame = pd.DataFrame(db.reviewed_flag_keys(conn), columns=list(db.FLAG_KEY))
    else:
        table = next(t for t in TABLES if t.file == file)
        frame = pd.read_sql_query(table.select(), conn)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode()


def save(conn: sqlite3.Connection, folder: Path, files: Sequence[str] = FILES) -> list[str]:
    """Write store files whose content changed (none for a dataset with no rows yet).
    Returns the files written."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for file in files:
        content = render(conn, file)
        path = folder / file
        if path.exists() and path.read_bytes() == content:
            continue
        if not path.exists() and content.count(b"\n") <= 1:
            continue  # no rows yet: no file
        path.write_bytes(content)
        written.append(file)
    return written


# ---------- app start ----------

_lock = threading.Lock()
_last: dict[str, StoreLoad] = {}


def version(folder: Path) -> str:
    """Size and modified time of each store file: changes when any file changes."""
    found = []
    for file in FILES:
        path = folder / file
        if path.exists():
            stat = path.stat()
            found.append(f"{file}:{stat.st_size}:{stat.st_mtime_ns}")
    return f"{folder.resolve()}|" + ",".join(found)


def sync(database: Path, config: AppConfig, folder: Path, today: date) -> StoreLoad:
    """Load the store into `database` unless the database already holds this version of it
    (kept in the database itself, so a new or deleted database loads again). One load at a
    time: the app's sessions share the database. Returns the result of the last load."""
    current = version(folder)
    key = str(database.resolve())
    with _lock, db.connect(database) as conn:
        loaded = conn.execute("SELECT value FROM store_meta WHERE key='version'").fetchone()
        if loaded is not None and loaded[0] == current and key in _last:
            return _last[key]
        result = load(conn, config, folder, today)
        conn.execute(
            "INSERT INTO store_meta (key, value) VALUES ('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (current,),
        )
    _last[key] = result
    return result
