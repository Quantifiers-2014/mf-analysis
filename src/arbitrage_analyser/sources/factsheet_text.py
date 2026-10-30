"""Split a fund's factsheet page into text chunks for the chatbot's factsheet search.

Layout-aware: factsheet pages are boxes laid out in columns, so plain text extraction mixes
neighbouring boxes. The page is cut recursively at blank space (an "XY cut"):

- first across the page, at blank rows (a gap of at least ROW_GAP points), giving bands;
- when a band has no blank row, down the page, at blank columns (COLUMN_GAP points) - except
  when the pieces' lines sit on the same rows (a table: "Regular  2.00%  0.78%"), which is then
  kept whole, so a label stays with its value.

Each piece that cannot be cut further is a block (a box or a section of one), read row by row.
Blocks are packed in reading order into chunks of at most MAX_TOKENS (a block is never split
unless it alone is too big; then it is split by lines, with OVERLAP_LINES lines repeated). A
heading (a short one-line block) is moved to the chunk of the block after it. Each chunk's
section is its first line.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pdfplumber

from arbitrage_analyser.sources import SourceError

Word = dict[str, Any]

MAX_TOKENS = 250  # agreed cap: about 200-300 tokens
CHARS_PER_TOKEN = 4  # rough size of a token in English text; no tokenizer needed
OVERLAP_LINES = 2  # lines repeated when a block is split
ROW_GAP = 4.0  # points of blank space that separate two bands
COLUMN_GAP = 6.0  # points of blank space that separate two columns
SAME_ROW = 2.5  # points: line centres this close sit on the same row
TABLE_ROWS = 0.6  # share of lines on shared rows that makes pieces a table (see _is_table)
TABLE_ROWS_ALWAYS = 0.9
CELL_GAP = 12.0  # points between words of a line read as a table cell break (" | ")
MAX_SECTION_CHARS = 80
MAX_HEADING_WORDS = 6  # a one-line block this short is kept with the block after it
MIN_LETTERS = 3  # a chunk with fewer letters (a page number) is dropped


# A line of single digits ("7 4 2") or of two or more small whole numbers ("92 21 | 70"): bar
# labels of a chart, printed vertically.
_CHART_DIGITS = re.compile(r"\d( \d)*|\d{1,2}(( \|)? \d{1,2})+")


@dataclass(frozen=True)
class Chunk:
    section: str  # first line of the chunk
    text: str  # lines separated by "\n"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def text_hash(text: str) -> str:
    """Short fingerprint of a text: a chunk is embedded again only when its text changes."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_chunks(content: bytes, page: int, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Chunks of one page (1-based) of the PDF. Raises SourceError when it cannot be read."""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            if not 1 <= page <= len(pdf.pages):
                raise SourceError(f"the PDF has no page {page}")
            words = pdf.pages[page - 1].extract_words(x_tolerance=1.5, extra_attrs=["size"])
    except SourceError:
        raise
    except Exception as exc:  # a damaged or unusual PDF
        raise SourceError(f"the PDF cannot be read: {type(exc).__name__}: {exc}") from exc
    return page_chunks(words, max_tokens)


def page_chunks(words: Sequence[Word], max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    words = [w for w in words if str(w["text"]).strip() and float(w.get("size", 6)) >= 3]
    blocks = [lines for lines in (_lines(b) for b in _cut(words)) if lines]
    return _pack(blocks, max_tokens)


# ---------- layout ----------


def _top(w: Word) -> float:
    return float(w["top"])


def _bottom(w: Word) -> float:
    return float(w["bottom"])


def _centre(w: Word) -> float:
    return (_top(w) + _bottom(w)) / 2


def _split(words: Sequence[Word], lo: str, hi: str, gap: float) -> list[list[Word]]:
    """Groups of words separated by at least `gap` of blank space along one axis."""
    ordered = sorted(words, key=lambda w: float(w[lo]))
    groups: list[list[Word]] = []
    end = -math.inf
    for word in ordered:
        if not groups or float(word[lo]) - end >= gap:
            groups.append([])
            end = -math.inf
        groups[-1].append(word)
        end = max(end, float(word[hi]))
    return groups


def _row_centres(words: Sequence[Word]) -> list[float]:
    return [sum(_centre(w) for w in line) / len(line) for line in _rows(words)]


def _is_table(left: Sequence[Word], right: Sequence[Word]) -> bool:
    """True when two side-by-side pieces have their lines on shared rows: nearly all of them
    (TABLE_ROWS_ALWAYS), or most (TABLE_ROWS) when neither has a blank row of its own. Page
    columns share rows only by chance (measured 0.25-0.82 on the Sep-2026 factsheets), the
    columns of a table share nearly all."""
    banded = any(len(_split(p, "top", "bottom", ROW_GAP)) > 1 for p in (left, right))
    a, b = _row_centres(left), _row_centres(right)
    fewer, other = (a, b) if len(a) <= len(b) else (b, a)
    shared = sum(1 for c in fewer if any(abs(c - d) <= SAME_ROW for d in other))
    return shared / len(fewer) >= (TABLE_ROWS_ALWAYS if banded else TABLE_ROWS)


def _cut(words: list[Word]) -> list[list[Word]]:
    """Blocks in reading order: top to bottom, then left to right within a band."""
    if len(words) <= 1:
        return [words]
    bands = _split(words, "top", "bottom", ROW_GAP)
    if len(bands) > 1:
        return [block for band in bands for block in _cut(band)]
    groups: list[list[Word]] = []  # columns, with the columns of one table kept together
    previous: list[Word] = []
    for column in _split(words, "x0", "x1", COLUMN_GAP):
        if groups and _is_table(previous, column):
            groups[-1] += column
        else:
            groups.append(list(column))
        previous = column
    if len(groups) > 1:
        return [block for group in groups for block in _cut(group)]
    return [words]


def _rows(words: Sequence[Word]) -> list[list[Word]]:
    rows: list[list[Word]] = []
    for word in sorted(words, key=_centre):
        if rows and abs(_centre(word) - _centre(rows[-1][0])) <= SAME_ROW:
            rows[-1].append(word)
        else:
            rows.append([word])
    return rows


def _lines(words: Sequence[Word]) -> list[str]:
    lines = []
    for row in _rows(words):
        row.sort(key=lambda w: float(w["x0"]))
        text = str(row[0]["text"])
        for before, word in itertools.pairwise(row):
            gap = float(word["x0"]) - float(before["x1"])
            text += (" | " if gap >= CELL_GAP else " ") + str(word["text"])
        text = re.sub(r"\s+", " ", text).strip()
        if text and not _CHART_DIGITS.fullmatch(text):
            lines.append(text)
    return lines


# ---------- chunks ----------


def _split_block(lines: list[str], max_tokens: int) -> list[list[str]]:
    """A block too big for one chunk, as pieces of whole lines with OVERLAP_LINES repeated."""
    pieces: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and estimate_tokens("\n".join([*current, line])) > max_tokens:
            pieces.append(current)
            keep = current[-OVERLAP_LINES:] if len(current) > OVERLAP_LINES else []
            if estimate_tokens("\n".join([*keep, line])) > max_tokens:
                keep = []
            current = keep
        current = [*current, line]
    if current:
        pieces.append(current)
    return pieces


def _is_heading(lines: Sequence[str]) -> bool:
    """A one-line block of a few words: likely the heading of the block after it."""
    return len(lines) == 1 and len(lines[0].split()) <= MAX_HEADING_WORDS


def _pack(blocks: Sequence[list[str]], max_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[str] = []
    heading: list[str] = []  # the last block, when it is a heading: moves with the next block

    def flush() -> None:
        nonlocal current
        if sum(ch.isalpha() for ch in "".join(current)) >= MIN_LETTERS:  # not a stray number
            chunks.append(Chunk(current[0][:MAX_SECTION_CHARS], "\n".join(current)))
        current = []

    for lines in blocks:
        if estimate_tokens("\n".join([*heading, *lines])) > max_tokens:  # too big for one chunk
            if heading and current[-1:] == heading:
                current = current[:-1]
            flush()
            for piece in _split_block([*heading, *lines], max_tokens):
                current = piece
                flush()
            heading = []
            continue
        if current and estimate_tokens("\n".join([*current, *lines])) > max_tokens:
            carried = heading if heading and current[-1:] == heading else []
            current = current[: len(current) - len(carried)]
            flush()
            current = list(carried)
        current = [*current, *lines]
        heading = list(lines) if _is_heading(lines) else []
    flush()
    return chunks
