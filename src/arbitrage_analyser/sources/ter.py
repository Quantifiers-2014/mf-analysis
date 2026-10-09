"""TER (Total Expense Ratio) data from AMFI: the downloaded Excel, or the JSON behind the page.

Page: https://www.amfiindia.com/ter-of-mf-schemes > Category "Hybrid Scheme",
Sub Category "Arbitrage Fund" > GO > Download Excel.

Excel layout (AMFI file for Sep-2026, checked 08-Oct-2026): one header row, one row per scheme per
day, then disclaimer rows. Header cells used:
  Scheme Name | TER Date | Direct Plan - Base Expense Ratio (BER) (%) |
  Direct Plan - Brokerage cost (%) | Direct Plan - Transaction Cost ... (%) |
  Direct Plan - Statutory Levies (including GST) (%) | Direct Plan - Total TER (%)

JSON (the request the page makes on GO; see sources/amfi_ter_api.py) has the same values with keys
Scheme_Name, TER_Date, D_BER, D_BrokerageCost, D_TransactionCost, D_StatutoryLevies, D_TER.

Only Direct Plan values are read. Rows with a blank BER or Total TER are skipped and reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from arbitrage_analyser.config import Fund
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.tables import cell_text, find_row, read_grid, to_number

FIELDS = ("base_ter", "brokerage", "transaction_cost", "statutory_levies", "total_ter")
_REQUIRED = ("base_ter", "total_ter")

# Excel header keyword (within the "Direct Plan - ..." columns) -> field
_EXCEL_KEYWORDS = {
    "base expense": "base_ter",
    "brokerage": "brokerage",
    "transaction": "transaction_cost",
    "statutory": "statutory_levies",
    "total ter": "total_ter",
}
# JSON key -> field
_JSON_KEYS = {
    "D_BER": "base_ter",
    "D_BrokerageCost": "brokerage",
    "D_TransactionCost": "transaction_cost",
    "D_StatutoryLevies": "statutory_levies",
    "D_TER": "total_ter",
}


@dataclass(frozen=True)
class TerImport:
    rows: pd.DataFrame  # amfi_code, date, base_ter, brokerage, transaction_cost, ...
    unmatched_funds: list[str]  # configured funds with no row in the data
    blank_rows: list[str]  # funds with rows skipped for a blank BER or Total TER


def _parse_date(value: object) -> pd.Timestamp:
    if isinstance(value, datetime | pd.Timestamp):
        return pd.Timestamp(value).normalize()
    text = cell_text(value)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)
    # ISO timestamp from the JSON, e.g. 2026-09-01T00:00:00.000Z: the date part is the TER date.
    parsed = pd.to_datetime(text[:10], format="%Y-%m-%d", errors="coerce")
    if len(text) > 10 and text[10] == "t" and pd.notna(parsed):
        return pd.Timestamp(parsed)
    raise SourceError(f"TER data has an unreadable date: {value!r}")


def parse_ter(content: bytes, funds: Sequence[Fund]) -> TerImport:
    """Read the AMFI TER Excel download."""
    grid = read_grid(content)
    header_row = find_row(grid, "scheme name")
    headers = {int(c): cell_text(v) for c, v in grid.iloc[header_row].items()}

    name_col = next(c for c, t in headers.items() if t == "scheme name")
    date_cols = [c for c, t in headers.items() if "date" in t]
    if not date_cols:
        raise SourceError("TER file has no TER Date column")

    field_cols: dict[str, int] = {}
    for c, text in headers.items():
        if not text.startswith("direct plan"):
            continue
        for keyword, field in _EXCEL_KEYWORDS.items():
            if keyword in text and field not in field_cols:
                field_cols[field] = c
    missing = [f for f in _REQUIRED if f not in field_cols]
    if missing:
        raise SourceError(f"TER file has no Direct Plan columns for: {missing}")

    body = grid.iloc[header_row + 1 :]
    rows = (
        {"name": row[name_col], "date": row[date_cols[0]]}
        | {field: row[col] for field, col in field_cols.items()}
        for _, row in body.iterrows()
    )
    return _match_funds(rows, funds)


def parse_ter_records(records: Iterable[Mapping[str, Any]], funds: Sequence[Fund]) -> TerImport:
    """Read the rows of AMFI's TER JSON (the `data` lists of every page)."""
    rows = []
    for record in records:
        if "Scheme_Name" not in record or "TER_Date" not in record or "D_BER" not in record:
            raise SourceError(
                "AMFI TER response rows lack Scheme_Name, TER_Date or D_BER; "
                "the AMFI page may have changed"
            )
        rows.append(
            {"name": record["Scheme_Name"], "date": record["TER_Date"]}
            | {field: record.get(key) for key, field in _JSON_KEYS.items()}
        )
    return _match_funds(rows, funds)


def _match_funds(rows: Iterable[Mapping[str, object]], funds: Sequence[Fund]) -> TerImport:
    """Pick each configured fund's rows by its ter_match text.

    `rows` have name, date and the FIELDS (raw cell values).
    """
    table = pd.DataFrame(list(rows), columns=["name", "date", *FIELDS]).astype(object)
    names = table["name"].map(cell_text)

    records: list[dict[str, object]] = []
    unmatched: list[str] = []
    blank: list[str] = []
    for fund in funds:
        mask = names.str.contains(fund.ter_match.lower(), regex=False)
        if not mask.any():
            unmatched.append(fund.name)
            continue
        distinct = sorted(set(names[mask]))
        if len(distinct) > 1:
            raise SourceError(
                f"ter_match '{fund.ter_match}' matches several schemes: {distinct}. "
                "Make it more specific in config/funds.toml."
            )
        skipped = 0
        for _, row in table[mask].iterrows():
            values = {field: to_number(row[field]) for field in FIELDS}
            if values["base_ter"] is None or values["total_ter"] is None:
                skipped += 1
                continue
            records.append({"amfi_code": fund.amfi_code, "date": _parse_date(row["date"])} | values)
        if skipped:
            blank.append(f"{fund.name} ({skipped} day(s))")

    frame = pd.DataFrame(records, columns=["amfi_code", "date", *FIELDS])
    if frame.duplicated(["amfi_code", "date"]).any():
        raise SourceError("TER data has more than one row for the same fund and date")
    return TerImport(rows=frame, unmatched_funds=unmatched, blank_rows=blank)
