"""Load data into the database: fetch or parse, validate, store, record flags."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from arbitrage_analyser import db
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.aaum import parse_aaum
from arbitrage_analyser.sources.benchmark import parse_benchmark_csv
from arbitrage_analyser.sources.mfapi import SchemeHistory, fetch_history
from arbitrage_analyser.sources.ter import parse_ter
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

        flags: list[db.Flag] = []
        nav = history.nav
        for day in check_duplicate_dates(nav.index.to_series()):
            flags.append(
                db.Flag(
                    "nav",
                    str(fund.amfi_code),
                    day,
                    "duplicate_date",
                    f"{fund.name}: the source lists {day} more than once",
                )
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
        flags.extend(check_series(stored, "nav", str(fund.amfi_code), config.settings, fund.name))
        result.new_flags += db.record_flags(conn, flags)

    if result.rows:
        db.log_refresh(conn, "nav", result.rows)
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
    db.log_refresh(conn, "benchmark", result.rows)
    return result


def import_ter(conn: sqlite3.Connection, config: AppConfig, content: bytes) -> LoadResult:
    result = LoadResult()
    try:
        parsed = parse_ter(content, config.funds)
    except SourceError as exc:
        result.errors.append(str(exc))
        return result
    if parsed.rows.empty:
        result.errors.append("The file has no rows for any configured fund")
        return result

    flags: list[db.Flag] = []
    for row in parsed.rows.itertuples(index=False):
        for column in ("base_ter", "total_ter"):
            value = getattr(row, column)
            if not 0 <= value <= 5:
                flags.append(
                    db.Flag(
                        "ter",
                        str(row.amfi_code),
                        row.date.date().isoformat(),
                        f"{column}_range",
                        f"{column} {value}% is outside 0-5%; check the units",
                    )
                )
        if row.total_ter < row.base_ter:
            flags.append(
                db.Flag(
                    "ter",
                    str(row.amfi_code),
                    row.date.date().isoformat(),
                    "total_below_base",
                    "Total TER is below Base TER",
                )
            )
    result.rows = db.upsert_ter(conn, parsed.rows)
    result.new_flags = db.record_flags(conn, flags)
    if parsed.unmatched_funds:
        result.notes.append("No rows found for: " + ", ".join(parsed.unmatched_funds))
    db.log_refresh(conn, "ter", result.rows)
    return result


def import_aaum(conn: sqlite3.Connection, config: AppConfig, content: bytes) -> LoadResult:
    result = LoadResult()
    try:
        parsed = parse_aaum(content, config.funds)
    except SourceError as exc:
        result.errors.append(str(exc))
        return result
    if parsed.rows.empty:
        result.errors.append("The file has no rows for any configured fund code")
        return result
    result.rows = db.upsert_aaum(conn, parsed.rows)
    result.notes.append(
        f"Quarter ending {parsed.quarter_end.date()}: " + ", ".join(parsed.funds_found)
    )
    db.log_refresh(conn, "aaum", result.rows)
    return result
