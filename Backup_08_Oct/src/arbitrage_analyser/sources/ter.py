"""Parse the TER (Total Expense Ratio) file downloaded from AMFI.

Download: https://www.amfiindia.com/ter-of-mf-schemes (choose month, then export).
Expected layout (seen in the TER file reviewed on 07-Oct-2026):
  a group row with "Regular Plan" and "Direct Plan" labels, then a header row with
  Scheme Name | Date (DD/MM/YYYY) | Base Expense Ratio (BER) (%) | Brokerage cost (%) |
  Transaction Cost ... (%) | Statutory Levies (including GST) (%) | Total TER (%)
  repeated per plan. Only the Direct Plan columns are read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from arbitrage_analyser.config import Fund
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.tables import cell_text, find_row, read_grid, to_number

# header keyword -> output column; first two are required
_TER_FIELDS = {
    "base expense": "base_ter",
    "total ter": "total_ter",
    "brokerage": "brokerage",
    "transaction": "transaction_cost",
    "statutory": "statutory_levies",
}
_REQUIRED = ("base_ter", "total_ter")


@dataclass(frozen=True)
class TerImport:
    rows: pd.DataFrame  # amfi_code, date, base_ter, brokerage, transaction_cost, ...
    unmatched_funds: list[str]  # configured funds with no row in the file


def _direct_plan_columns(grid: pd.DataFrame, header_row: int) -> range:
    """Columns under the 'Direct Plan' group label, found in the rows above the header."""
    for r in range(max(0, header_row - 3), header_row):
        labels = {c: cell_text(v) for c, v in grid.iloc[r].items() if cell_text(v)}
        starts = [c for c, t in labels.items() if "direct plan" in t]
        if starts:
            start = int(starts[0])
            later = [int(c) for c in labels if int(c) > start]
            end = min(later) if later else grid.shape[1]
            return range(start, end)
    raise SourceError("TER file has no 'Direct Plan' column group above the header row")


def _parse_date(value: object) -> pd.Timestamp:
    if isinstance(value, datetime | pd.Timestamp):
        return pd.Timestamp(value).normalize()
    text = cell_text(value)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)
    raise SourceError(f"TER file has an unreadable date: {value!r}")


def parse_ter(content: bytes, funds: Sequence[Fund]) -> TerImport:
    grid = read_grid(content)
    header_row = find_row(grid, "scheme name")
    headers = {c: cell_text(v) for c, v in grid.iloc[header_row].items()}

    name_col = next(c for c, t in headers.items() if t == "scheme name")
    date_cols = [c for c, t in headers.items() if t.startswith("date")]
    if not date_cols:
        raise SourceError("TER file has no Date column")

    direct = _direct_plan_columns(grid, header_row)
    field_cols: dict[str, int] = {}
    for c in direct:
        for keyword, field in _TER_FIELDS.items():
            if keyword in headers.get(c, "") and field not in field_cols:
                field_cols[field] = int(c)
    missing = [f for f in _REQUIRED if f not in field_cols]
    if missing:
        raise SourceError(f"TER file Direct Plan group lacks columns: {missing}")

    body = grid.iloc[header_row + 1 :]
    names = body[name_col].map(cell_text)

    records: list[dict[str, object]] = []
    unmatched: list[str] = []
    for fund in funds:
        mask = names.str.contains(fund.ter_match.lower(), regex=False)
        matched = body[mask]
        if matched.empty:
            unmatched.append(fund.name)
            continue
        distinct = sorted(set(names[mask]))
        if len(distinct) > 1:
            raise SourceError(
                f"ter_match '{fund.ter_match}' matches several schemes: {distinct}. "
                "Make it more specific in config/funds.toml."
            )
        for _, row in matched.iterrows():
            record: dict[str, object] = {
                "amfi_code": fund.amfi_code,
                "date": _parse_date(row[date_cols[0]]),
            }
            for field in _TER_FIELDS.values():
                record[field] = to_number(row[field_cols[field]]) if field in field_cols else None
            if record["base_ter"] is None or record["total_ter"] is None:
                raise SourceError(f"TER file row for {fund.name} has no Base or Total TER")
            records.append(record)

    columns = ["amfi_code", "date", *_TER_FIELDS.values()]
    frame = pd.DataFrame(records, columns=columns)
    if frame.duplicated(["amfi_code", "date"]).any():
        raise SourceError("TER file has more than one row for the same fund and date")
    return TerImport(rows=frame, unmatched_funds=unmatched)
