"""Parse the benchmark index CSV downloaded from niftyindices.com.

Download: https://www.niftyindices.com/reports/historical-data
  Index type Equity -> Strategy Indices -> NIFTY 50 ARBITRAGE -> date range -> CSV.
The page lists columns Date, Open, High, Low, Close, ... The Close column is used.
The NIFTY 50 Arbitrage index already includes dividends and interest (index methodology),
so there is a single version of it.
"""

from __future__ import annotations

import io

import pandas as pd

from arbitrage_analyser.sources import SourceError

DATE_FORMATS = ("%d-%b-%Y", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%B-%Y", "%d %B %Y")


def _find_column(columns: list[str], wanted: str) -> str:
    for column in columns:
        if column.strip().lower() == wanted:
            return column
    for column in columns:
        if column.strip().lower().startswith(wanted):
            return column
    raise SourceError(f"Benchmark file has no '{wanted}' column. Columns found: {columns}")


def _parse_dates(raw: pd.Series) -> pd.Series:
    text = raw.astype(str).str.strip()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    unreadable = text[pd.to_datetime(text, format=DATE_FORMATS[0], errors="coerce").isna()]
    raise SourceError(f"Benchmark file has dates in an unknown format, e.g. {unreadable.iloc[0]!r}")


def parse_benchmark_csv(content: bytes) -> pd.DataFrame:
    """Return a frame with columns date (Timestamp) and value (float), in file order.

    Duplicates and bad values are NOT removed here; validation decides what to reject.
    """
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise SourceError(f"Benchmark file is not a readable CSV: {exc}") from exc
    if frame.empty:
        raise SourceError("Benchmark file has no rows")

    columns = [str(c) for c in frame.columns]
    date_col = _find_column(columns, "date")
    close_col = _find_column(columns, "close")

    values = pd.to_numeric(frame[close_col].str.replace(",", "", regex=False), errors="coerce")
    if values.isna().any():
        row = int(values.isna().to_numpy().nonzero()[0][0]) + 2  # +2: header row, 1-based
        raise SourceError(f"Benchmark file has a non-numeric Close value on line {row}")
    return pd.DataFrame({"date": _parse_dates(frame[date_col]), "value": values.astype(float)})
