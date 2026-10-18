"""Find and download a fund's latest factsheet PDF.

A link with placeholders ({month}, {year}, ...) is tried for this month's edition, then the two
before, newest first; the first one that returns a PDF is used. A listing page is read for the
first PDF link containing `link_text` for those editions. Only https links are followed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from html import unescape
from urllib.parse import urljoin

import requests

from arbitrage_analyser.config import FactsheetSource
from arbitrage_analyser.sources import SourceError

TIMEOUT_SECONDS = 120
MAX_BYTES = 80 * 1024 * 1024  # largest factsheet seen: 22 MB (Edelweiss, Oct-2026)
EDITIONS_TRIED = 3
_HEADERS = {"User-Agent": "Mozilla/5.0 (arbitrage-analyser; monthly factsheet check)"}
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

# get(url) -> (status code, content type, body); tests pass a fake.
Getter = Callable[[str], tuple[int, str, bytes]]


@dataclass(frozen=True)
class Download:
    url: str
    content: bytes


def edition_months(today: date, count: int = EDITIONS_TRIED) -> list[date]:
    """First day of this month and the months before it, newest first."""
    months, year, month = [], today.year, today.month
    for _ in range(count):
        months.append(date(year, month, 1))
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return months


def fill(template: str, edition: date) -> str:
    following = date(edition.year + edition.month // 12, edition.month % 12 + 1, 1)
    name = edition.strftime("%B")
    return template.format(
        month=name.lower(),
        Month=name,
        mm=f"{edition.month:02d}",
        year=str(edition.year),
        next_mm=f"{following.month:02d}",
        next_year=str(following.year),
    )


def candidate_links(source: FactsheetSource, today: date) -> Iterator[str]:
    """Links to try for `source.link`, newest edition first (one link if it has no
    placeholders)."""
    if source.link is None:
        return
    seen: set[str] = set()
    for edition in edition_months(today):
        url = fill(source.link, edition)
        if url not in seen:
            seen.add(url)
            yield url


def links_on_listing(html: str, base_url: str, source: FactsheetSource, today: date) -> list[str]:
    """PDF links on a listing page matching `link_text`, newest edition first."""
    if source.link_text is None:
        return []
    hrefs = [urljoin(base_url, unescape(h).strip()) for h in _HREF.findall(html)]
    pdfs = [h for h in hrefs if h.startswith("https://") and ".pdf" in h.lower()]
    found: list[str] = []
    for edition in edition_months(today):
        wanted = fill(source.link_text, edition).lower()
        for href in pdfs:
            if wanted in href.lower() and href not in found:
                found.append(href)
    return found


def http_get(url: str) -> tuple[int, str, bytes]:
    """GET with a size cap. Raises SourceError on network errors."""
    try:
        with requests.get(url, headers=_HEADERS, timeout=TIMEOUT_SECONDS, stream=True) as resp:
            chunks, size = [], 0
            if resp.status_code == 200:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise SourceError(f"{url} is larger than {MAX_BYTES // 2**20} MB")
                    chunks.append(chunk)
            return resp.status_code, resp.headers.get("Content-Type", ""), b"".join(chunks)
    except requests.RequestException as exc:
        raise SourceError(f"could not download {url}: {exc}") from exc


def download_latest(source: FactsheetSource, today: date, get: Getter = http_get) -> Download:
    """The newest factsheet PDF for `source`. Raises SourceError listing what was tried."""
    if source.listing is not None:
        status, _, body = get(source.listing)
        if status != 200:
            raise SourceError(f"factsheet listing page returned HTTP {status}: {source.listing}")
        links = links_on_listing(body.decode("utf-8", "replace"), source.listing, source, today)
        if not links:
            raise SourceError(
                f"no PDF link containing '{fill(source.link_text or '', today)}' (or the two "
                f"months before) on {source.listing}"
            )
    else:
        links = list(candidate_links(source, today))
    tried = []
    for url in links:
        status, _, body = get(url)
        if status == 200 and body.startswith(b"%PDF"):
            return Download(url, body)
        tried.append(f"{url} (HTTP {status}{', not a PDF' if status == 200 else ''})")
    raise SourceError("no factsheet PDF found. Tried: " + "; ".join(tried))
