"""Read AUM, AAUM, portfolio turnover and the Direct Plan BER from an AMC factsheet PDF.

Rules, checked on the Sep-2026 factsheets of the 8 configured AMCs (tests/data/factsheets):

- Fund page: the page whose largest-font text is the fund's `page_title` (e.g. "SBI Arbitrage
  Fund"). Other pages that only mention the fund (fund of funds, index pages) are skipped. When
  several pages carry the title (continuation pages), the one with an AUM value is used; two
  pages with an AUM value is an error, not a guess.
- AUM / AAUM: the first "<number> Cr / crs / crores" in the label's own column, on the label's
  row or just below it. Labels: "Closing AUM", "AUM" / "AuM"; "AAUM" / "AAuM", "Average AUM".
- As-of date: the date printed in the AUM label ("AUM as on 30/09/2026"), else the page's
  "Data as on" / "Date as on" line. Both present and different is an error. The factsheet's
  file name or edition month is never used: an "October" edition can hold September data.
- Turnover: for each configured label (e.g. "Equity Turnover"), the nearest number on the
  label's row to the right, kept as printed; "%" is added only when the factsheet prints it with
  the value or in the label ("(%)").
- Direct BER: under the "Expense Ratio" / "Base Expense Ratio" heading, the row that starts with
  "Direct" ("Direct", "Direct Plan:", "Direct Plan: BER"); its % value. When the rows sit under
  a "TER | BER" header (SBI), the value in the BER column; several % values without such a
  header leave the BER blank.

A value that is not found is returned as None. Nothing is estimated or carried over.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pdfplumber
import pypdfium2

from arbitrage_analyser.sources import SourceError

logging.getLogger("pdfminer").setLevel(logging.ERROR)  # pdfminer logs every odd font

Word = dict[str, Any]  # pdfplumber word: text, x0, x1, top, bottom, size

_ROW = 3.0  # points: words whose vertical centres are this close sit on the same row
# "₹ 24,539.33 Cr", "`75,274.99 crs", "Rs. 35,576.34 crores", "₹ 17532.65 Cr." (` is the
# rupee sign in some AMC fonts)
_AMOUNT = re.compile(
    r"(?:₹|`|rs\.?)?\s*(\d[\d,]*\.\d{1,2})\s*(?:crores?|crs|cr)\b\.?", re.IGNORECASE
)
_COLUMN_WIDTH = 165.0  # points; measured: values end <= 155pt right of their label, the next
# column starts >= 166pt right (Sep-2026 factsheets)
_NUMBER = re.compile(r"^-?\d[\d,]*\.\d+%?$")
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE_PATTERNS = (
    # 30th September, 2026 | 30-Sep-26 | 30 September 2026
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s-]*{_MONTH}[\s,-]*(\d{{4}}|\d{{2}})\b", re.I),
    # September 30, 2026
    re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})\b", re.I),
    # 30/09/2026
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
)
_PAGE_DATE = re.compile(r"\b(?:data|date)\s+as\s+on\b", re.IGNORECASE)

AUM_LABELS = (("closing", "aum"), ("aum",))  # not when "Average" / "Avg" comes just before
AAUM_LABELS = (("monthly", "aaum"), ("aaum",), ("average", "aum"), ("avg", "aum"))
_AVERAGE = frozenset({"average", "avg"})
EXPENSE_HEADINGS = (("base", "expense", "ratio"), ("expense", "ratio"))
DIRECT_LABELS = (("direct", "plan", "ber"), ("direct", "plan"), ("direct",))


@dataclass(frozen=True)
class FactsheetValues:
    """Values read from one fund's factsheet page. None = not printed or not found."""

    page: int  # 1-based page number in the PDF
    as_of: date
    aum_crore: float | None
    aaum_crore: float | None
    turnover: str | None  # as printed, e.g. "Equity 1.60, Gross 11.57"
    direct_ber_pct: float | None
    missing: tuple[str, ...] = field(default=())  # fields not found on the page


def read_factsheet(
    content: bytes, page_title: str, turnover_labels: dict[str, str]
) -> FactsheetValues:
    """Find the fund's page in the PDF and read its values.

    `turnover_labels` maps the name shown in the app to the label printed on the factsheet,
    e.g. {"Equity": "Equity Turnover", "Gross": "Gross Exposure Turnover"}.
    """
    if not content.startswith(b"%PDF"):
        raise SourceError("the file is not a PDF")
    candidates = _pages_mentioning(content, page_title)
    if not candidates:
        raise SourceError(f"no page mentions '{page_title}'")
    found: list[FactsheetValues] = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for number in candidates:
                words = pdf.pages[number - 1].extract_words(x_tolerance=1.5, extra_attrs=["size"])
                if not _is_title(words, page_title):
                    continue
                values = read_page(words, number, turnover_labels)
                if values is not None:
                    found.append(values)
    except SourceError:
        raise
    except Exception as exc:  # a damaged or unusual PDF: report it for this fund only
        raise SourceError(f"the PDF cannot be read: {type(exc).__name__}: {exc}") from exc
    if not found:
        raise SourceError(f"no page titled '{page_title}' shows an AUM value and as-of date")
    if len(found) > 1:
        pages = ", ".join(str(v.page) for v in found)
        raise SourceError(f"pages {pages} are all titled '{page_title}' and show AUM")
    return found[0]


def read_page(
    words: Sequence[Word], page: int, turnover_labels: dict[str, str]
) -> FactsheetValues | None:
    """Values from one page's words, or None when the page shows no AUM value (not the fund's
    main page). Raises SourceError when the page is ambiguous (two different as-of dates)."""
    aum = _amount_after(words, AUM_LABELS, stop=AAUM_LABELS)
    if aum is None:
        return None
    aum_crore, aum_text = aum
    as_of = _as_of(words, aum_text)
    if as_of is None:
        return None
    aaum = _amount_after(words, AAUM_LABELS, stop=AUM_LABELS)
    turnover = _turnover(words, turnover_labels)
    ber, ber_problem = _direct_ber(words)
    missing = tuple(
        name
        for name, value in (("AAUM", aaum), ("Turnover", turnover), (ber_problem, ber))
        if value is None
    )
    return FactsheetValues(
        page=page,
        as_of=as_of,
        aum_crore=aum_crore,
        aaum_crore=aaum[0] if aaum else None,
        turnover=turnover,
        direct_ber_pct=ber,
        missing=missing,
    )


# ---------- page finding ----------


def _squash(text: str) -> str:
    """Lower case, letters and digits only: 'FundDetails' and 'Fund Details' compare equal."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pages_mentioning(content: bytes, title: str) -> list[int]:
    """1-based pages whose text contains the title (fast pass with pdfium)."""
    key = _squash(title)
    try:
        doc = pypdfium2.PdfDocument(content)
    except pypdfium2.PdfiumError as exc:
        raise SourceError(f"the PDF cannot be read: {exc}") from exc
    try:
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            if key in _squash(textpage.get_text_bounded()):
                pages.append(i + 1)
            textpage.close()
            page.close()
        return pages
    finally:
        doc.close()


def _is_title(words: Sequence[Word], title: str) -> bool:
    """True when the page's largest-font text is exactly the title."""
    if not words:
        return False
    largest = max(round(float(w["size"]), 1) for w in words)
    heading = " ".join(w["text"] for w in words if round(float(w["size"]), 1) == largest)
    return _squash(heading) == _squash(title)


# ---------- labels ----------


def _norm(word: str) -> str:
    """'AUM:' -> 'aum', 'Ratio2' -> 'ratio' (footnote digits), 'Turnover$$' -> 'turnover'."""
    text = re.sub(r"[^a-z0-9]", "", word.lower())
    return re.sub(r"(?<=[a-z])\d+$", "", text)


def _centre(word: Word) -> float:
    return (float(word["top"]) + float(word["bottom"])) / 2


def _same_row(a: Word, b: Word, tolerance: float = _ROW) -> bool:
    return abs(_centre(a) - _centre(b)) <= tolerance


def _find_labels(words: Sequence[Word], tokens: Sequence[str]) -> list[list[Word]]:
    """Every run of consecutive words on one row whose normalised texts equal `tokens`.

    Words that normalise to nothing (':', '@@') are skipped inside a run.
    """
    hits: list[list[Word]] = []
    for start, first in enumerate(words):
        if _norm(first["text"]) != tokens[0]:
            continue
        run, i, t = [first], start + 1, 1
        while t < len(tokens) and i < len(words):
            word = words[i]
            i += 1
            if not _same_row(first, word):
                break
            norm = _norm(word["text"])
            if not norm:
                continue
            if norm != tokens[t]:
                break
            run.append(word)
            t += 1
        if t == len(tokens):
            hits.append(run)
    return hits


def _starts_row(words: Sequence[Word], label: list[Word], gap: float = 12) -> bool:
    """No word on the label's row ends just before it ('Regular/Other than Direct' is not a
    'Direct' row; text from a neighbouring column further left does not count)."""
    x0 = float(label[0]["x0"])
    return not any(
        w is not label[0] and _same_row(w, label[0]) and 0 <= x0 - float(w["x1"]) <= gap
        for w in words
    )


def _first_label(words: Sequence[Word], choices: Iterable[Sequence[str]]) -> Iterable[list[Word]]:
    for tokens in choices:
        yield from _find_labels(words, tokens)


# ---------- AUM, AAUM, as-of date ----------


def _column_text(
    words: Sequence[Word], label: list[Word], stop_tops: Sequence[float], depth: float = 45
) -> str:
    """Text in the label's column: words starting up to _COLUMN_WIDTH right of the label, on
    its row and the rows below (up to `depth` points), stopping at the next label."""
    x0, top = float(label[0]["x0"]), float(label[0]["top"])
    end = min([t for t in stop_tops if t > top + _ROW] + [top + depth])
    region = [
        w
        for w in words
        if top - _ROW <= float(w["top"]) < end
        and x0 - 20 <= float(w["x0"]) <= x0 + _COLUMN_WIDTH
        and w not in label
    ]
    region.sort(key=lambda w: (round(_centre(w) / _ROW), float(w["x0"])))
    return " ".join(w["text"] for w in region)


def _amount_after(
    words: Sequence[Word], labels: Sequence[Sequence[str]], stop: Sequence[Sequence[str]]
) -> tuple[float, str] | None:
    """(crore value, column text after the amount is removed) for the first label with an
    amount in its column."""
    stop_tops = [float(s[0]["top"]) for s in _first_label(words, stop)]
    for label in _first_label(words, labels):
        if labels is AUM_LABELS and _after_average(words, label):
            continue  # "Monthly Average AUM" is AAUM
        text = _column_text(words, label, stop_tops)
        match = _AMOUNT.search(text)
        if match:
            remaining = text[: match.start()] + " " + text[match.end() :]
            return float(match.group(1).replace(",", "")), remaining
    return None


def _after_average(words: Sequence[Word], label: list[Word], gap: float = 12) -> bool:
    x0 = float(label[0]["x0"])
    return any(
        _norm(w["text"]) in _AVERAGE and _same_row(w, label[0]) and 0 <= x0 - float(w["x1"]) <= gap
        for w in words
    )


def parse_date(text: str) -> date | None:
    """First date in the text ('30th September, 2026', 'September 30, 2026', '30-Sep-26',
    '30/09/2026'), or None."""
    found: list[tuple[int, date]] = []
    for index, pattern in enumerate(_DATE_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        a, b, c = match.groups()
        try:
            if index == 0:
                year = int(c) + (2000 if len(c) == 2 else 0)
                found.append((match.start(), date(year, _MONTHS[b[:3].lower()], int(a))))
            elif index == 1:
                found.append((match.start(), date(int(c), _MONTHS[a[:3].lower()], int(b))))
            else:
                found.append((match.start(), date(int(c), int(b), int(a))))
        except ValueError:
            continue
    return min(found)[1] if found else None


def _as_of(words: Sequence[Word], aum_text: str) -> date | None:
    """Date printed in the AUM label ('AUM as on ...'), else after the page's 'Data as on'.
    An 'as on' further down the column belongs to another label and is not used."""
    after = re.match(r"\W*as\s+on\b", aum_text, re.IGNORECASE)
    near_aum = parse_date(aum_text[after.end() : after.end() + 40]) if after else None
    page_text = " ".join(w["text"] for w in words)
    header = _PAGE_DATE.search(page_text)
    on_page = parse_date(page_text[header.end() : header.end() + 40]) if header else None
    if near_aum and on_page and near_aum != on_page:
        raise SourceError(
            f"AUM is dated {near_aum:%d-%b-%Y} but the page says data as on {on_page:%d-%b-%Y}"
        )
    return near_aum or on_page


# ---------- turnover ----------


def _numbers_right_of(words: Sequence[Word], label: list[Word], max_dx: float) -> list[Word]:
    x1 = float(label[-1]["x1"])
    return [
        w
        for w in words
        if _NUMBER.match(w["text"])
        and 0 <= float(w["x0"]) - x1 <= max_dx
        and _same_row(w, label[0], tolerance=5)
    ]


def _turnover(words: Sequence[Word], labels: dict[str, str]) -> str | None:
    parts = []
    taken: set[int] = set()
    for name, printed in labels.items():
        tokens = [t for t in (_norm(p) for p in printed.split()) if t]
        value = None
        for label in _find_labels(words, tokens):
            options = [w for w in _numbers_right_of(words, label, max_dx=160) if id(w) not in taken]
            if options:
                best = min(options, key=lambda w: float(w["x0"]))
                taken.add(id(best))
                value = best["text"]
                if not value.endswith("%") and _percent_label(words, label):
                    value += "%"
                break
        if value is not None:
            parts.append(f"{name} {value}")
    return ", ".join(parts) if parts else None


def _percent_label(words: Sequence[Word], label: list[Word]) -> bool:
    """'(%)' printed after the label on its row or the row below, within the label's column."""
    x0, x1, top = float(label[0]["x0"]), float(label[-1]["x1"]), float(label[0]["top"])
    return any(
        w["text"] == "(%)"
        and x0 - 15 <= float(w["x0"]) <= x1 + 20
        and -_ROW <= float(w["top"]) - top <= 20
        for w in words
    )


# ---------- Direct BER ----------


def _direct_ber(words: Sequence[Word]) -> tuple[float | None, str]:
    """(BER, what to call it when missing): 'Direct BER', or why it was left blank."""
    for heading in _first_label(words, EXPENSE_HEADINGS):
        hx, htop = float(heading[0]["x0"]), float(heading[0]["top"])
        rows = [
            label
            for label in _first_label(words, DIRECT_LABELS)
            if 0 < float(label[0]["top"]) - htop <= 80
            and abs(float(label[0]["x0"]) - hx) <= 60
            and _starts_row(words, label)
        ]
        if not rows:
            continue
        row = min(rows, key=lambda r: float(r[0]["top"]))
        values = [
            w
            for w in _numbers_right_of(words, row, max_dx=_COLUMN_WIDTH)
            if w["text"].endswith("%")
        ]
        if not values:
            continue
        column = _ber_column(words, heading, row)
        if column is not None:
            chosen = min(values, key=lambda w: abs((float(w["x0"]) + float(w["x1"])) / 2 - column))
        elif len(values) == 1:
            chosen = values[0]
        else:
            shown = ", ".join(w["text"] for w in values)
            return None, f"Direct BER (the Direct row shows {shown}; which is BER is unclear)"
        return float(chosen["text"].rstrip("%").replace(",", "")), ""
    return None, "Direct BER"


def _ber_column(words: Sequence[Word], heading: list[Word], row: list[Word]) -> float | None:
    """x centre of a 'BER' column header between the heading and the Direct row, if the
    rows are laid out as a TER | BER table."""
    htop, rtop = float(heading[0]["top"]), float(row[0]["top"])
    between = [w for w in words if htop < float(w["top"]) < rtop]
    headers = [
        w
        for w in between
        if _norm(w["text"]) == "ber"
        and any(_norm(t["text"]) == "ter" and _same_row(t, w) for t in between)
    ]
    if not headers:
        return None
    header = headers[-1]
    return (float(header["x0"]) + float(header["x1"])) / 2
