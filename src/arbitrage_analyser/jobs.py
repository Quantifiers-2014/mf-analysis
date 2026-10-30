"""The data refresh job: load the store, fetch new data from the APIs, save the store.

Run daily by .github/workflows/refresh-data.yml (python -m arbitrage_analyser update-data).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date

from arbitrage_analyser import embeddings, factsheets, ingest
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.ingest import LoadResult

DATASETS = ("nav", "ter", "factsheets")
DAILY = ("nav", "ter")


def current_month(today: date) -> str:
    return today.strftime("%m-%Y")


def fetch(
    conn: sqlite3.Connection, config: AppConfig, today: date, datasets: Sequence[str]
) -> dict[str, LoadResult]:
    """Fetch the chosen datasets into the database (already loaded from the store). TER covers
    last month and this month (TER can change mid-month); factsheets run after TER so their BER
    is checked against it. Factsheet chunks without an embedding are embedded on every run.
    Datasets switched off in config are skipped."""
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choose from {list(DATASETS)}")
    active = config.settings.active_datasets()
    results: dict[str, LoadResult] = {}
    if "nav" in datasets:
        results["nav"] = ingest.refresh_nav(conn, config)
    if "ter" in datasets and "ter" in active:
        for month, in_progress in ((ingest.previous_month(today), False),
                                   (current_month(today), True)):  # fmt: skip
            results[f"ter {month}"] = ingest.fetch_ter(
                conn, config, month, month_in_progress=in_progress
            )
    if "factsheets" in datasets and "factsheet" in active:
        results["factsheets"] = factsheets.fetch_factsheets(conn, config, today)
    if "ter" in datasets and "factsheet" in active:
        # the latest factsheet BER against the TER just fetched (also for unchanged factsheets)
        results["factsheet BER check"] = LoadResult(new_flags=factsheets.check_latest(conn, config))
    if "factsheet" in active:
        # every run, so chunks left without an embedding (no key yet, Mistral down) catch up
        results["factsheet embeddings"] = embeddings.update(conn, config)
    return results
