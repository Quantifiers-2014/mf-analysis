"""Download TER rows from the JSON request behind AMFI's "TER of MF Schemes" page.

This is AMFI's internal page request, not a published API: AMFI can change or block it without
notice. When it fails, download the Excel from the page and import it instead.

Request (seen 08-Oct-2026 by clicking GO on https://www.amfiindia.com/ter-of-mf-schemes):
    GET /api/populate-te-rdata-revised?MF_ID=17&Month=09-2026&strCat=46&strType=-1
        &page=1&pageSize=15
    MF_ID: fund house (17 = Kotak; -1 returns no rows, so each fund house is fetched separately),
    Month: MM-YYYY,
    strCat: sub category (46 = Arbitrage Fund), strType: fund type (-1 = all).
Response:
    {"data": [{"Scheme_Name": "Kotak Arbitrage Fund", "TER_Date": "2026-09-01T00:00:00.000Z",
               "D_BER": "0.3400", "D_TER": "2.3282", ...}, ...],
     "meta": {"page": 1, "pageSize": 15, "total": 30, "pageCount": 2}}
"""

from __future__ import annotations

from typing import Any

import requests

from arbitrage_analyser.sources import SourceError

URL = "https://www.amfiindia.com/api/populate-te-rdata-revised"
PAGE_URL = "https://www.amfiindia.com/ter-of-mf-schemes"
TIMEOUT_SECONDS = 30
PAGE_SIZE = 100
MAX_PAGES = 200  # stops a runaway loop if the paging metadata is wrong
_HEADERS = {
    "Accept": "application/json",
    "Referer": PAGE_URL,
    "User-Agent": "Mozilla/5.0 (arbitrage-analyser)",
}


def fetch_ter_records(
    month: str,
    category_id: int,
    mf_id: int,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """All TER rows for one fund house and month (MM-YYYY), following the page count."""
    http = session or requests.Session()
    records: list[dict[str, Any]] = []
    page, page_count, total = 1, 1, 0
    while page <= page_count:
        if page > MAX_PAGES:
            raise SourceError(f"AMFI TER response reports more than {MAX_PAGES} pages")
        payload = _get_page(http, month, category_id, mf_id, page)
        data, meta = payload.get("data"), payload.get("meta")
        if not isinstance(data, list) or not isinstance(meta, dict):
            raise SourceError("AMFI TER response is missing 'data' or 'meta'")
        try:
            page_count, total = int(meta["pageCount"]), int(meta["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceError("AMFI TER response 'meta' has no pageCount/total") from exc
        records.extend(r for r in data if isinstance(r, dict))
        page += 1
    if len(records) != total:
        raise SourceError(f"AMFI TER response said {total} rows but returned {len(records)}")
    return records


def _get_page(
    http: requests.Session, month: str, category_id: int, mf_id: int, page: int
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "MF_ID": mf_id,
        "Month": month,
        "strCat": category_id,
        "strType": -1,
        "page": page,
        "pageSize": PAGE_SIZE,
    }
    try:
        response = http.get(URL, params=params, headers=_HEADERS, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SourceError(f"Could not download TER from AMFI: {exc}") from exc
    except ValueError as exc:
        raise SourceError("AMFI returned something other than JSON for the TER request") from exc
    if not isinstance(payload, dict):
        raise SourceError("AMFI TER response is not a JSON object")
    return payload
