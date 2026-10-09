"""Read a downloaded spreadsheet into a raw grid, whatever its real format.

Website "Excel" exports come as .xlsx, legacy .xls, HTML tables saved as .xls, or CSV.
The format is detected from the file content, not the file name.
"""

from __future__ import annotations

import io

import pandas as pd

from arbitrage_analyser.sources import SourceError

_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"


def read_grid(content: bytes) -> pd.DataFrame:
    """Return the first sheet (or largest HTML table) with no header row, all cells as objects."""
    if not content:
        raise SourceError("The file is empty")
    head = content[:512].lstrip().lower()
    try:
        if content.startswith(_XLSX_MAGIC):
            grid = pd.read_excel(io.BytesIO(content), header=None, engine="openpyxl")
        elif content.startswith(_XLS_MAGIC):
            grid = pd.read_excel(io.BytesIO(content), header=None, engine="xlrd")
        elif head.startswith((b"<", b"\xef\xbb\xbf<")):
            tables = pd.read_html(io.StringIO(_decode(content)), header=None)
            if not tables:
                raise SourceError("The file has no table")
            grid = max(tables, key=lambda t: t.size)
        else:
            grid = pd.read_csv(io.BytesIO(content), header=None, dtype=str, keep_default_na=False)
    except SourceError:
        raise
    except Exception as exc:  # reader libraries raise many unrelated types
        raise SourceError(f"Could not read the file as a spreadsheet: {exc}") from exc
    grid = grid.astype(object)
    grid.columns = range(grid.shape[1])
    return grid.reset_index(drop=True)


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1")


def cell_text(value: object) -> str:
    """Normalised lower-case text of a cell; empty string for blanks."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).split()).lower()


def find_row(grid: pd.DataFrame, text: str, max_rows: int = 30) -> int:
    """Index of the first row (within the first `max_rows`) with a cell equal to `text`."""
    for i in range(min(max_rows, len(grid))):
        if any(cell_text(v) == text for v in grid.iloc[i]):
            return i
    raise SourceError(f"Could not find a '{text}' header in the first {max_rows} rows")


def to_number(value: object) -> float | None:
    """Parse '1,234.5', '0.34%', 0.34 -> float; blanks and dashes -> None."""
    text = cell_text(value).replace(",", "").replace("%", "").strip()
    if text in ("", "-", "--", "na", "n/a", "nan"):
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise SourceError(f"Expected a number, found {value!r}") from exc
