"""Factsheet values (AUM, AAUM, portfolio turnover, Direct BER): fetch, check, store, and the
values file kept in the repository.

Both ways in, a fetch from the AMC sites and the values file, go through the same checks:
- a value outside its range is not stored and is reported as an error;
- each as-of date is its own row, so older data never replaces newer;
- for the same as-of date, a newer read replaces the stored row, except when it lacks a value
  the stored row has: then the stored row (and its PDF link) is kept and this is reported;
  changed values are reported;
- flags: AAUM far from AUM, factsheet BER far from the AMFI BER (with and without GST).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from arbitrage_analyser import db, factsheet_search
from arbitrage_analyser.config import AppConfig, FactsheetSource
from arbitrage_analyser.ingest import LoadResult
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.factsheet_links import Download, download_latest
from arbitrage_analyser.sources.factsheet_pdf import FactsheetValues, read_factsheet
from arbitrage_analyser.sources.factsheet_text import Chunk, read_chunks

FactsheetDownloader = Callable[[FactsheetSource, date], Download]
FactsheetReader = Callable[[bytes, str, dict[str, str]], FactsheetValues]
Chunker = Callable[[bytes, int], list[Chunk]]  # (PDF, page) -> the page's text chunks

MAX_CRORE = 1_000_000.0  # Rs 10 lakh crore: far above any single scheme's AUM
MAX_BER_PCT = 2.5  # above SEBI's highest TER limit (2.25%)
# Factsheet vs AMFI BER: a flag when the factsheet value is more than this from the AMFI BER and
# from the AMFI BER plus 18% GST (some AMCs print BER with GST: Kotak 0.39% = AMFI 0.33% x 1.18).
BER_TOLERANCE_PCT = 0.02
GST = 0.18
AAUM_TO_AUM = (0.5, 2.0)  # AAUM outside this multiple of AUM raises a flag

_VALUES = ("aum_crore", "aaum_crore", "turnover", "direct_ber")
_LABELS = {"aum_crore": "AUM", "aaum_crore": "AAUM", "turnover": "Turnover", "direct_ber": "BER"}
_RANGES = {"aum_crore": MAX_CRORE, "aaum_crore": MAX_CRORE, "direct_ber": MAX_BER_PCT}


@dataclass(frozen=True)
class FactsheetRow:
    """One fund's values for one as-of date, as stored."""

    amfi_code: int
    as_of: date
    aum_crore: float | None
    aaum_crore: float | None
    turnover: str | None
    direct_ber: float | None
    source_url: str
    page: int
    fetched_at: str  # ISO timestamp in UTC, with offset

    def record(self) -> dict[str, object]:
        return {c: getattr(self, c) for c in db.FACTSHEET_COLUMNS} | {
            "as_of": self.as_of.isoformat()
        }


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------- fetch ----------


def fetch_factsheets(
    conn: sqlite3.Connection,
    config: AppConfig,
    today: date,
    download: FactsheetDownloader | None = None,
    read: FactsheetReader | None = None,
    chunk: Chunker | None = None,
) -> LoadResult:
    """Download each fund's latest factsheet, read its values, check them and store them, and
    store the fund page's text in chunks for the chatbot's search. A fund that fails is
    reported; the others still load. Flags are also listed in the notes.

    `download`, `read` and `chunk` default to the AMC sites and the PDF rules; tests pass fakes.
    """
    download = download or download_latest
    read = read or read_factsheet
    chunk = chunk or read_chunks
    result = LoadResult()
    cache: dict[tuple[str | None, str | None, str | None], Download] = {}
    rows: list[FactsheetRow] = []
    chunk_rows = 0
    fetched_at = now_utc()
    for fund in config.funds:
        source = fund.factsheet
        if source is None:
            continue
        key = (source.link, source.listing, source.link_text)
        try:
            if key not in cache:
                cache[key] = download(source, today)
            pdf = cache[key]
            values = read(pdf.content, source.page_title, dict(source.turnover))
        except SourceError as exc:
            result.errors.append(f"{fund.name}: {exc}")
            continue
        if values.as_of > today:
            result.errors.append(f"{fund.name}: as-of date {values.as_of} is in the future")
            continue
        if values.missing:
            result.notes.append(
                f"{fund.name}: not found in the factsheet: {', '.join(values.missing)}"
            )
        try:
            chunks = chunk(pdf.content, values.page)
            if not chunks:
                raise SourceError(f"no text found on page {values.page}")
            chunk_rows += factsheet_search.store_chunks(
                conn, fund, values.as_of, values.page, pdf.url, chunks
            )
        except SourceError as exc:
            result.errors.append(f"{fund.name}: factsheet text not stored for search: {exc}")
        rows.append(
            FactsheetRow(
                fund.amfi_code,
                values.as_of,
                values.aum_crore,
                values.aaum_crore,
                values.turnover,
                values.direct_ber_pct,
                pdf.url,
                values.page,
                fetched_at,
            )
        )
    _store(conn, config, rows, result, report_flags=True)
    if chunk_rows:
        result.notes.append(f"Factsheet text: {chunk_rows} chunk(s) stored for search")
    not_set = [f.name for f in config.funds if f.factsheet is None]
    if not_set:
        result.notes.append("No factsheet set in config: " + ", ".join(not_set))
    return result


# ---------- checks and storage ----------


def _store(
    conn: sqlite3.Connection,
    config: AppConfig,
    rows: Iterable[FactsheetRow],
    result: LoadResult,
    report_flags: bool,
) -> None:
    flags: list[db.Flag] = []
    to_write: list[dict[str, object]] = []
    names = {f.amfi_code: f.name for f in config.funds}
    for row in rows:
        name = names.get(row.amfi_code, str(row.amfi_code))  # kept even if not in config
        record = _checked(conn, name, row, result)
        if record is None:
            continue
        if row.amfi_code in names:  # no flags for funds no longer in config
            flags += _flags(conn, name, record)
        if (
            report_flags
            and record["direct_ber"] is not None
            and db.ber_on(conn, row.amfi_code, str(record["as_of"])) is None
        ):
            result.notes.append(
                f"{name}: BER not checked against AMFI (no AMFI TER on or before {row.as_of})"
            )
        to_write.append(record)
    if to_write:  # when values were read is kept per row (fetched_at)
        result.rows += db.upsert_factsheet(conn, pd.DataFrame(to_write))
    result.new_flags += db.record_flags(conn, flags)
    if report_flags:
        result.notes += [f"Flag: {f.detail}" for f in flags]


def _checked(
    conn: sqlite3.Connection, name: str, row: FactsheetRow, result: LoadResult
) -> dict[str, object] | None:
    """The record to store, or None when nothing changed."""
    record = row.record()
    for column, high in _RANGES.items():
        value = record[column]
        if isinstance(value, float) and not 0 < value <= high:
            result.errors.append(
                f"{name}: {_LABELS[column]} {value} is outside 0-{high:g}; not stored"
            )
            record[column] = None

    as_of = str(record["as_of"])
    stored = _stored(conn, row.amfi_code, as_of)
    if stored is None:
        return record
    lost = [_LABELS[c] for c in _VALUES if record[c] is None and stored[c] is not None]
    if lost:  # never mix values from two PDFs in one row
        result.notes.append(
            f"{name}: kept the stored values for {as_of}; the new read lacks {', '.join(lost)}"
        )
        return None
    if all(record[c] == stored[c] for c in (*_VALUES, "source_url", "page")):
        return None  # unchanged: keeps the values file free of no-op updates
    for column in _VALUES:
        if stored[column] is not None and record[column] != stored[column]:
            result.notes.append(
                f"{name}: {_LABELS[column]} for {as_of} changed "
                f"{stored[column]} -> {record[column]}"
            )
    return record


def _stored(conn: sqlite3.Connection, code: int, as_of: str) -> dict[str, object] | None:
    found = conn.execute(
        f"SELECT {', '.join(db.FACTSHEET_COLUMNS)} FROM factsheet WHERE amfi_code=? AND as_of=?",
        (code, as_of),
    ).fetchone()
    return None if found is None else dict(zip(db.FACTSHEET_COLUMNS, found, strict=True))


def check_latest(conn: sqlite3.Connection, config: AppConfig) -> int:
    """Flag each fund's latest stored values against the data now in the database (e.g. AMFI
    TER fetched after the values file was loaded). Returns the number of new flags."""
    flags: list[db.Flag] = []
    for record in db.read_latest_factsheet(conn).astype(object).to_dict("records"):
        record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        try:
            name = config.fund_by_code(int(str(record["amfi_code"]))).name
        except KeyError:
            continue
        flags += _flags(conn, name, record)
    return db.record_flags(conn, flags)


def _flags(conn: sqlite3.Connection, name: str, record: dict[str, object]) -> list[db.Flag]:
    code, as_of = int(str(record["amfi_code"])), str(record["as_of"])
    aum, aaum, ber = (record[c] for c in ("aum_crore", "aaum_crore", "direct_ber"))
    flags = []
    ratio_ok = not (isinstance(aum, float) and isinstance(aaum, float)) or (
        AAUM_TO_AUM[0] <= aaum / aum <= AAUM_TO_AUM[1]
    )
    if not ratio_ok:
        flags.append(
            db.Flag(
                "factsheet",
                str(code),
                as_of,
                "aaum_vs_aum",
                f"{name}: AAUM {aaum:,.2f} Cr vs AUM {aum:,.2f} Cr; check the factsheet",
            )
        )
    amfi = db.ber_on(conn, code, as_of)
    if (
        isinstance(ber, float)
        and amfi is not None
        and min(abs(ber - amfi), abs(ber - amfi * (1 + GST))) > BER_TOLERANCE_PCT
    ):
        flags.append(
            db.Flag(
                "factsheet",
                str(code),
                as_of,
                "ber_vs_amfi",
                f"{name}: factsheet Direct BER {ber:.2f}% vs AMFI BER {amfi:.2f}% "
                f"({amfi * (1 + GST):.2f}% with GST)",
            )
        )
    return flags


# ---------- values file ----------


def import_csv(conn: sqlite3.Connection, config: AppConfig, path: Path, today: date) -> LoadResult:
    """Load the values file kept in the data store (written by the data job), with the same
    checks as a fetch. A malformed line is skipped and reported; the others load."""
    result = LoadResult()
    if not path.exists():
        return result
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        result.errors.append(f"cannot read {path.name}: {exc}")
        return result
    missing = [c for c in db.FACTSHEET_COLUMNS if c not in frame.columns]
    if missing:
        result.errors.append(f"{path.name} lacks columns {missing}")
        return result
    rows = []
    for line, raw in enumerate(frame.to_dict("records"), start=2):
        try:
            row = _parse_line(raw)
            if row.as_of > today:
                raise ValueError(f"as_of {row.as_of} is in the future")
            rows.append(row)
        except ValueError as exc:
            result.errors.append(f"{path.name} line {line}: {exc}; line skipped")
    _store(conn, config, rows, result, report_flags=False)
    return result


def _parse_line(raw: dict[str, str]) -> FactsheetRow:
    def number(column: str) -> float | None:
        text = raw[column].strip()
        try:
            return float(text) if text else None
        except ValueError:
            raise ValueError(f"{column} '{text}' is not a number") from None

    def whole(column: str) -> int:
        text = raw[column].strip()
        if not text.isdigit() or int(text) < 1:
            raise ValueError(f"{column} '{text}' is not a positive whole number")
        return int(text)

    as_of_text = raw["as_of"].strip()
    try:
        as_of = date.fromisoformat(as_of_text)
    except ValueError:
        raise ValueError(f"as_of '{as_of_text}' is not YYYY-MM-DD") from None
    if len(as_of_text) != 10:
        raise ValueError(f"as_of '{as_of_text}' is not YYYY-MM-DD")
    url = raw["source_url"].strip()
    if not url.startswith("https://"):
        raise ValueError(f"source_url '{url}' is not an https link")
    fetched_at = raw["fetched_at"].strip()
    try:
        stamp = datetime.fromisoformat(fetched_at)
    except ValueError:
        raise ValueError(f"fetched_at '{fetched_at}' is not an ISO timestamp") from None
    if not fetched_at.endswith("+00:00") or stamp.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"fetched_at '{fetched_at}' is not in UTC (+00:00)")
    turnover = raw["turnover"].strip()
    return FactsheetRow(
        whole("amfi_code"),
        as_of,
        number("aum_crore"),
        number("aaum_crore"),
        turnover or None,
        number("direct_ber"),
        url,
        whole("page"),
        fetched_at,
    )
