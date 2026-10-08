"""Shared fixtures: a small config, synthetic NAV series and file builders."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from arbitrage_analyser import db
from arbitrage_analyser.config import AppConfig, load_config
from arbitrage_analyser.sources.mfapi import SchemeHistory

CONFIG_TEXT = """
[settings]
history_start = "2019-01-01"
benchmark_name = "NIFTY 50 Arbitrage"
exit_load_flag_above_pct = 0.25
outlier_multiplier = 10.0
outlier_lookback_days = 250
outlier_min_history = 60
max_gap_days = 5
overlap_tolerance = 0.01

[settings.features]
ter = true
aaum = true

[settings.freshness]
nav = [4, 7]
ter = [45, 60]
aaum = [110, 140]
benchmark = [45, 60]

[[funds]]
name = "Alpha Arbitrage Fund"
amc = "Alpha"
category = "Arbitrage"
amfi_code = 100001
isin = "INF000A01AA1"
ter_match = "Alpha Arbitrage Fund"
exit_load_pct = 0.25
exit_load_days = 15
fund_managers = [{ name = "A. Manager", since = "2019-10-03" }]

[[funds]]
name = "Beta Arbitrage Fund"
amc = "Beta"
category = "Arbitrage"
amfi_code = 100002
isin = "INF000B01BB2"
ter_match = "Beta Arbitrage Fund"
exit_load_pct = 0.5
exit_load_days = 15
fund_managers = ["B. Manager", { name = "C. Manager", since = "2014-12" }]
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "funds.toml"
    path.write_text(CONFIG_TEXT)
    return path


@pytest.fixture
def config(config_file: Path) -> AppConfig:
    return load_config(config_file)


@pytest.fixture
def db_file(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_file: Path) -> Iterator[db.sqlite3.Connection]:
    with db.connect(db_file) as connection:
        yield connection


def growth_series(
    start: str,
    end: str,
    annual_rate: float,
    start_value: float = 10.0,
    noise: float = 0.0,
    seed: int = 1,
) -> pd.Series:
    """Weekday series growing at `annual_rate` per calendar year, optional small noise."""
    dates = pd.bdate_range(start, end)
    days = (dates - dates[0]).days.to_numpy()
    values = start_value * np.power(1 + annual_rate, days / 365.25)
    if noise:
        rng = np.random.default_rng(seed)
        values = values * (1 + rng.normal(0, noise, len(values)))
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"))


def history_for(code: int, isin: str, series: pd.Series, name: str | None = None) -> SchemeHistory:
    return SchemeHistory(
        scheme_code=code,
        scheme_name=name or f"Fund {code} - Direct Plan - Growth",
        isin_growth=isin,
        scheme_category="Hybrid Schemes - Arbitrage Fund",
        nav=series,
    )


def benchmark_csv(series: pd.Series, date_format: str = "%d-%b-%Y") -> bytes:
    lines = ["Date,Open,High,Low,Close"]
    for ts, value in series.items():
        lines.append(f"{ts.strftime(date_format)},{value:.4f},{value:.4f},{value:.4f},{value:.4f}")
    return ("\n".join(lines) + "\n").encode()


def xlsx_bytes(rows: list[list[object]]) -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


TER_HEADER = [
    "Scheme Name",
    "Date (DD/MM/YYYY)",
    "Base Expense Ratio (BER) (%)",
    "Brokerage cost (%)",
    "Transaction Cost incurred for the purpose of execution of trade (%)",
    "Statutory Levies (including GST) (%)",
    "Total TER (%)",
]


def ter_rows(data: list[tuple[str, str, float, float]]) -> list[list[object]]:
    """data: (scheme name, dd/mm/yyyy, direct base, direct total). Regular plan = direct + 0.5."""
    group = [
        None,
        None,
        "Regular Plan",
        None,
        None,
        None,
        None,
        "Direct Plan",
        None,
        None,
        None,
        None,
        None,
    ]
    header = [*TER_HEADER, *TER_HEADER[2:], "NSDL_Number"]
    rows: list[list[object]] = [group, header]
    for name, day, base, total in data:
        rows.append(
            [
                name,
                day,
                base + 0.5,
                0.14,
                0.09,
                1.7,
                total + 0.5,
                base,
                0.14,
                0.09,
                round(total - base - 0.23, 4),
                total,
                "X/1",
            ]
        )
    return rows


def aaum_rows(
    entries: list[tuple[int, str, float]], title: str | None = None
) -> list[list[object]]:
    title = title or (
        "Average Assets under Management (AAUM) for the quarter of "
        "July - September 2026 (Rs in Lakhs)"
    )
    rows: list[list[object]] = [
        [title, None, None, None],
        ["AMFI Code", "Scheme NAV Name", "Average AUM for The Month", None],
        [
            None,
            None,
            "Excluding Fund of Funds - Domestic but including Fund of Funds - Overseas",
            "Fund Of Funds - Domestic",
        ],
    ]
    rows.extend([code, name, value, 0] for code, name, value in entries)
    return rows


CONFIG_TEXT_OFF = CONFIG_TEXT.replace("ter = true\naaum = true", "ter = false\naaum = false")


@pytest.fixture
def config_file_off(tmp_path: Path) -> Path:
    """Config with the optional TER and AAUM datasets switched off."""
    path = tmp_path / "funds_off.toml"
    path.write_text(CONFIG_TEXT_OFF)
    return path
