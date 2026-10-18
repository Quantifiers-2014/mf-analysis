"""Load data into the database: fetch or parse, validate, store, record flags."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from arbitrage_analyser import db
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.amfi_ter_api import fetch_ter_records
from arbitrage_analyser.sources.benchmark import parse_benchmark_csv
from arbitrage_analyser.sources.mfapi import SchemeHistory, fetch_history
from arbitrage_analyser.sources.ter import TerImport, parse_ter, parse_ter_records
from arbitrage_analyser.validation import (
    check_benchmark_upload,
    check_duplicate_dates,
    check_fund_mapping,
    check_series,
)

HistoryFetcher = Callable[[int], SchemeHistory]


@dataclass
class LoadResult:
    rows: int = 0
    new_flags: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


FileImporter = Callable[[sqlite3.Connection, AppConfig, bytes], LoadResult]


def refresh_nav(
    conn: sqlite3.Connection, config: AppConfig, fetch: HistoryFetcher | None = None
) -> LoadResult:
    """Download NAV history for every configured fund and store it from history_start.

    A fund whose source data fails the mapping check is skipped; the others still load.
    `fetch` defaults to mfapi.in; tests pass a fake.
    """
    fetch = fetch or fetch_history
    result = LoadResult()
    start = pd.Timestamp(config.settings.history_start)
    for fund in config.funds:
        try:
            history = fetch(fund.amfi_code)
        except SourceError as exc:
            result.errors.append(f"{fund.name}: {exc}")
            continue
        mapping_errors = check_fund_mapping(fund, history)
        if mapping_errors:
            result.errors.extend(mapping_errors)
            continue

        nav = history.nav
        duplicates = check_duplicate_dates(nav.index.to_series())
        if duplicates:  # a source quirk, not a data problem: the last value is kept
            result.notes.append(
                f"{fund.name}: the source lists {', '.join(duplicates[:5])} more than once; "
                "the last value is used"
            )
        nav = nav[~nav.index.duplicated(keep="last")]

        db.upsert_fund(
            conn,
            fund.amfi_code,
            history.scheme_name,
            fund.isin,
            history.scheme_category,
            nav.index[0].date(),
        )
        result.rows += db.upsert_nav(conn, fund.amfi_code, nav[nav.index >= start])

        stored = db.read_nav(conn, fund.amfi_code)
        flags = check_series(stored, "nav", str(fund.amfi_code), config.settings, fund.name)
        result.new_flags += db.record_flags(conn, flags)
    return result


def import_benchmark(conn: sqlite3.Connection, config: AppConfig, content: bytes) -> LoadResult:
    """Load a niftyindices.com CSV. The whole file is rejected if any hard check fails."""
    result = LoadResult()
    try:
        upload = parse_benchmark_csv(content)
    except SourceError as exc:
        result.errors.append(str(exc))
        return result

    start = pd.Timestamp(config.settings.history_start)
    before_start = int((upload["date"] < start).sum())
    upload = upload[upload["date"] >= start]
    if before_start:
        result.notes.append(f"Skipped {before_start} row(s) dated before {start.date()}")
    if upload.empty:
        result.errors.append(f"The file has no rows on or after {start.date()}")
        return result

    name = config.settings.benchmark_name
    stored = db.read_benchmark(conn, name)
    check = check_benchmark_upload(upload, stored, config.settings)
    if not check.accepted:
        result.errors.extend(check.rejections)
        return result

    series = pd.Series(upload["value"].to_numpy(), index=pd.DatetimeIndex(upload["date"]))
    result.rows = db.insert_benchmark(conn, name, series.sort_index())
    result.new_flags = db.record_flags(conn, check.flags)
    already_stored = len(upload) - result.rows
    if already_stored:
        result.notes.append(f"{already_stored} row(s) were already stored and matched")
    return result


def import_ter(conn: sqlite3.Connection, config: AppConfig, content: bytes) -> LoadResult:
    """Import the AMFI TER Excel download."""
    try:
        parsed = parse_ter(content, config.funds)
    except SourceError as exc:
        return LoadResult(errors=[str(exc)])
    return _store_ter(conn, [parsed], "in the file")


TerFetcher = Callable[[str, int, int], list[dict[str, Any]]]  # month, category id, MF_ID
_MONTH = re.compile(r"(0[1-9]|1[0-2])-\d{4}")


def previous_month(today: date) -> str:
    """MM-YYYY of the month before `today`: the latest month with a full set of daily TER rows."""
    first = today.replace(day=1)
    last_month = first - timedelta(days=1)
    return last_month.strftime("%m-%Y")


def fetch_ter(
    conn: sqlite3.Connection,
    config: AppConfig,
    month: str,
    fetch: TerFetcher | None = None,
    month_in_progress: bool = False,
) -> LoadResult:
    """Download one month (MM-YYYY) of TER from AMFI and store it.

    AMFI returns one fund house per request, so there is one request per category and
    amfi_mf_id. A failed request is reported; the other fund houses still load.
    `month_in_progress`: the current month, which may have no rows yet; that is a note, not an
    error. `fetch(month, category_id, mf_id)` defaults to AMFI's page request; tests pass a fake.
    """
    if not _MONTH.fullmatch(month):
        return LoadResult(errors=[f"Month must be MM-YYYY, got '{month}'"])
    fetch = fetch or (lambda m, c, f: fetch_ter_records(m, c, f))
    category_ids = config.settings.amfi_ter_category_ids
    parsed: list[TerImport] = []
    errors: list[str] = []
    no_mf_id = [f.name for f in config.funds if f.amfi_mf_id is None]
    for category in config.categories():
        funds = [f for f in config.funds_in(category) if f.amfi_mf_id is not None]
        if not funds:
            continue
        if category not in category_ids:
            errors.append(
                f"No AMFI TER category id for '{category}'. Add it under "
                "[settings.amfi_ter_category_ids] in config/funds.toml."
            )
            continue
        for mf_id in sorted({f.amfi_mf_id for f in funds if f.amfi_mf_id is not None}):
            house = [f for f in funds if f.amfi_mf_id == mf_id]
            try:
                records = fetch(month, category_ids[category], mf_id)
                parsed.append(parse_ter_records(records, house))
            except SourceError as exc:
                errors.append(f"{', '.join(f.name for f in house)}: {exc}")

    if parsed:
        result = _store_ter(conn, parsed, f"for {month}", month_in_progress)
    else:
        result = LoadResult()
        if not errors:
            errors.append("No fund has an amfi_mf_id in config/funds.toml")
    result.errors += errors
    if no_mf_id:
        result.notes.append("Not fetched (no amfi_mf_id in config): " + ", ".join(no_mf_id))
    if errors:
        result.notes.append(
            "For failed funds, download the Excel from the AMFI page and import it."
        )
    return result


def _store_ter(
    conn: sqlite3.Connection, parsed: list[TerImport], source: str, empty_ok: bool = False
) -> LoadResult:
    """Validate and store parsed TER rows. `source` names the data in messages."""
    result = LoadResult()
    rows = pd.concat([p.rows for p in parsed], ignore_index=True)
    unmatched = [name for p in parsed for name in p.unmatched_funds]
    blank = [name for p in parsed for name in p.blank_rows]
    if rows.empty:
        if empty_ok:
            result.notes.append(f"No TER values published yet {source}")
            return result
        result.errors.append(f"No TER values {source} for any configured fund")
        result.notes += _ter_notes(unmatched, blank)
        return result

    result.rows = db.upsert_ter(conn, rows)
    result.new_flags = db.record_flags(conn, ter_flags(rows))
    result.notes += _ter_notes(unmatched, blank)
    return result


def ter_flags(rows: pd.DataFrame) -> list[db.Flag]:
    """Flags for TER rows (amfi_code, date as Timestamp, base_ter, total_ter): values outside
    0-5% (wrong units) and Total TER below BER."""
    flags: list[db.Flag] = []
    for row in rows.itertuples(index=False):
        day = row.date.date().isoformat()
        for column in ("base_ter", "total_ter"):
            value = getattr(row, column)
            if not 0 <= value <= 5:
                flags.append(
                    db.Flag(
                        "ter",
                        str(row.amfi_code),
                        day,
                        f"{column}_range",
                        f"{column} {value}% is outside 0-5%; check the units",
                    )
                )
        if row.total_ter < row.base_ter:
            flags.append(
                db.Flag(
                    "ter", str(row.amfi_code), day, "total_below_base", "Total TER is below BER"
                )
            )
    return flags


def _ter_notes(unmatched: list[str], blank: list[str]) -> list[str]:
    notes = []
    if unmatched:
        notes.append("No rows found for: " + ", ".join(unmatched))
    if blank:
        notes.append("Skipped rows with blank BER or Total TER: " + ", ".join(blank))
    return notes
