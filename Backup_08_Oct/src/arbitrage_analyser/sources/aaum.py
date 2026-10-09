"""Parse the scheme-wise Average AUM (AAUM) export from AMFI.

Download: https://www.amfiindia.com/aum-data/average-aum
  Select Data: Schemewise, Select Type: Categorywise, Mutual Fund, Financial Year, Period
  (a quarter), Go, then the Excel button. Repeat for each fund house in the config.
Layout seen on 07-Oct-2026: title "Average Assets under Management (AAUM) for the quarter of
July - September 2026 (Rs in Lakhs)", columns AMFI Code | Scheme NAV Name |
"Excluding Fund of Funds - Domestic but including Fund of Funds - Overseas" | "Fund Of Funds -
Domestic". The first value column is used and converted from lakhs to crore.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from arbitrage_analyser.config import Fund
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.tables import cell_text, find_row, read_grid, to_number

_QUARTER = re.compile(r"quarter of\s+([a-z]+)\s*-\s*([a-z]+)\s+(\d{4})")
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}


@dataclass(frozen=True)
class AaumImport:
    rows: pd.DataFrame  # amfi_code, quarter_end, aaum_crore
    quarter_end: pd.Timestamp
    funds_found: list[str]


def _title_text(grid: pd.DataFrame, before_row: int) -> str:
    cells = [cell_text(v) for r in range(before_row) for v in grid.iloc[r]]
    return " ".join(c for c in cells if c)


def _quarter_end(title: str) -> pd.Timestamp:
    match = _QUARTER.search(title)
    if not match:
        raise SourceError(
            "AAUM file title does not name the quarter (e.g. 'July - September 2026')"
        )
    month = _MONTHS.get(match.group(2))
    if month is None:
        raise SourceError(f"AAUM file title has an unknown month: {match.group(2)!r}")
    year = int(match.group(3))
    return pd.Timestamp(year, month, calendar.monthrange(year, month)[1])


def _divisor_to_crore(title: str) -> float:
    if "lakh" in title:
        return 100.0
    if "crore" in title:
        return 1.0
    raise SourceError("AAUM file title does not say whether values are in lakhs or crore")


def _scheme_code(value: object) -> int | None:
    """AMFI codes may arrive as 120401, 120401.0 or '120401'; anything else is not a data row."""
    text = cell_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return int(text) if text.isdigit() else None


def parse_aaum(content: bytes, funds: Sequence[Fund]) -> AaumImport:
    grid = read_grid(content)
    header_row = find_row(grid, "amfi code")
    title = _title_text(grid, header_row + 1)
    quarter_end = _quarter_end(title)
    divisor = _divisor_to_crore(title)

    code_col = next(c for c, v in grid.iloc[header_row].items() if cell_text(v) == "amfi code")
    value_col = None
    for r in range(header_row, min(header_row + 3, len(grid))):
        for c, v in grid.iloc[r].items():
            if cell_text(v).startswith("excluding fund of funds"):
                value_col = c
                break
        if value_col is not None:
            break
    if value_col is None:
        raise SourceError("AAUM file has no 'Excluding Fund of Funds' value column")

    wanted = {f.amfi_code: f.name for f in funds}
    records: list[dict[str, object]] = []
    for _, row in grid.iloc[header_row + 1 :].iterrows():
        code = _scheme_code(row[code_col])
        if code is None or code not in wanted:
            continue
        amount = to_number(row[value_col])
        if amount is None or amount < 0:
            raise SourceError(f"AAUM file has no valid value for scheme {code}")
        records.append(
            {"amfi_code": code, "quarter_end": quarter_end, "aaum_crore": amount / divisor}
        )

    frame = pd.DataFrame(records, columns=["amfi_code", "quarter_end", "aaum_crore"])
    if frame["amfi_code"].duplicated().any():
        raise SourceError("AAUM file lists the same scheme code more than once")
    found = [wanted[int(c)] for c in frame["amfi_code"]]
    return AaumImport(rows=frame, quarter_end=quarter_end, funds_found=found)
