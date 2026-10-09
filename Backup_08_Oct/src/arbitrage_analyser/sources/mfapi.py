"""NAV history from mfapi.in.

mfapi.in is a free, unofficial API that republishes AMFI NAV data. AMFI's own NAV history
download is limited to 90 days per request, so mfapi.in is used for the full history.
Response format (checked 07-Oct-2026 for scheme 120401):
    {"meta": {"scheme_code": 120401, "scheme_name": "...", "isin_growth": "INF205K01KR8",
              "scheme_category": "Hybrid Schemes - Arbitrage Fund", ...},
     "data": [{"date": "06-10-2026", "nav": "37.48850"}, ...]}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from arbitrage_analyser.sources import SourceError

BASE_URL = "https://api.mfapi.in/mf"
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SchemeHistory:
    scheme_code: int
    scheme_name: str
    isin_growth: str | None
    scheme_category: str | None
    nav: pd.Series  # full history, ascending dates, float values


def parse_history(payload: dict[str, Any]) -> SchemeHistory:
    """Turn an mfapi.in JSON response into a sorted NAV series."""
    meta = payload.get("meta")
    data = payload.get("data")
    if not isinstance(meta, dict) or not isinstance(data, list):
        raise SourceError("mfapi.in response is missing 'meta' or 'data'")
    if not data:
        raise SourceError(f"mfapi.in returned no NAV data for scheme {meta.get('scheme_code')}")

    frame = pd.DataFrame(data)
    if not {"date", "nav"} <= set(frame.columns):
        raise SourceError("mfapi.in NAV rows must have 'date' and 'nav'")
    dates = pd.to_datetime(frame["date"], format="%d-%m-%Y", errors="coerce")
    navs = pd.to_numeric(frame["nav"], errors="coerce")
    bad = dates.isna() | navs.isna()
    if bad.any():
        sample = frame.loc[bad].head(3).to_dict("records")
        raise SourceError(f"mfapi.in returned unreadable NAV rows, e.g. {sample}")

    series = pd.Series(navs.to_numpy(dtype=float), index=pd.DatetimeIndex(dates, name="date"))
    series = series.sort_index()

    try:
        code = int(meta["scheme_code"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceError("mfapi.in meta has no valid scheme_code") from exc
    return SchemeHistory(
        scheme_code=code,
        scheme_name=str(meta.get("scheme_name") or ""),
        isin_growth=meta.get("isin_growth") or None,
        scheme_category=meta.get("scheme_category") or None,
        nav=series,
    )


def fetch_history(amfi_code: int, session: requests.Session | None = None) -> SchemeHistory:
    """Download the full NAV history for one scheme."""
    http = session or requests.Session()
    try:
        response = http.get(f"{BASE_URL}/{amfi_code}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SourceError(f"Could not download NAV history for {amfi_code}: {exc}") from exc
    except ValueError as exc:
        raise SourceError(f"mfapi.in returned invalid JSON for {amfi_code}") from exc
    if not isinstance(payload, dict):
        raise SourceError(f"mfapi.in returned an unexpected response for {amfi_code}")
    history = parse_history(payload)
    if history.scheme_code != amfi_code:
        raise SourceError(
            f"mfapi.in returned scheme {history.scheme_code} when {amfi_code} was requested"
        )
    return history
