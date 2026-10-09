"""Fetching TER from AMFI's page request, and storing it."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

from arbitrage_analyser import cli, db, ingest
from arbitrage_analyser.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config
from arbitrage_analyser.runtime import CONFIG_ENV, DB_ENV
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.amfi_ter_api import fetch_ter_records
from arbitrage_analyser.sources.ter import parse_ter_records
from tests.conftest import CONFIG_TEXT

# Page 1 of AMFI's response for Kotak, Sep-2026 (MF_ID=17, strCat=46, pageSize=15).
PAGE1 = json.loads(
    (Path(__file__).parent / "data" / "amfi_ter_api_kotak_2026-09_page1.json").read_text()
)


def _row(name: str, day: str, ber: str | None, ter: str | None) -> dict[str, Any]:
    return {
        "Scheme_Name": name,
        "TER_Date": f"{day}T00:00:00.000Z",
        "R_BER": "0.9",
        "R_TER": "2.9",
        "D_BER": ber,
        "D_BrokerageCost": "0.14",
        "D_TransactionCost": "0.08",
        "D_StatutoryLevies": "1.7",
        "D_TER": ter,
    }


class FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload, self.status = payload, status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    """Returns one queued response per GET and records the query parameters."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.params: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], **_: Any) -> FakeResponse:
        self.params.append(params)
        return self.responses.pop(0)


def _two_pages() -> FakeSession:
    page2 = {
        "data": [
            _row("Kotak Arbitrage Fund", f"2026-09-{d}", "0.33", "2.3182") for d in range(16, 31)
        ],
        "meta": {"page": 2, "pageSize": 15, "total": 30, "pageCount": 2},
    }
    return FakeSession(FakeResponse(PAGE1), FakeResponse(page2))


# ---------- parsing the JSON rows ----------


def test_parse_real_amfi_response_page() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    parsed = parse_ter_records(PAGE1["data"], config.funds)
    rows = parsed.rows.set_index("date")
    assert len(rows) == 15 and set(rows["amfi_code"]) == {119771}
    first, tenth = rows.loc["2026-09-01"], rows.loc["2026-09-10"]
    assert (first["base_ter"], first["total_ter"]) == (0.34, 2.3282)
    assert (tenth["base_ter"], tenth["total_ter"]) == (0.33, 2.3182)
    assert tenth["statutory_levies"] == pytest.approx(1.76)
    assert len(parsed.unmatched_funds) == len(config.funds) - 1


def test_parse_records_skips_blanks(config: AppConfig) -> None:
    records = [
        _row("Alpha Arbitrage Fund", "2026-09-01", "0.33", "2.3"),
        _row("Alpha Arbitrage Fund", "2026-09-02", None, None),
        _row("Beta Arbitrage Fund", "2026-09-01", "", "1.6"),
    ]
    parsed = parse_ter_records(records, config.funds)
    assert len(parsed.rows) == 1
    assert parsed.blank_rows == [
        "Alpha Arbitrage Fund (1 day(s))",
        "Beta Arbitrage Fund (1 day(s))",
    ]


def test_parse_records_rejects_changed_keys(config: AppConfig) -> None:
    with pytest.raises(SourceError, match="may have changed"):
        parse_ter_records([{"SchemeName": "x"}], config.funds)


# ---------- the HTTP request ----------


def test_fetch_follows_pages_and_sends_filters() -> None:
    session = _two_pages()
    records = fetch_ter_records("09-2026", 46, 17, session=session)  # type: ignore[arg-type]
    assert len(records) == 30
    assert [p["page"] for p in session.params] == [1, 2]
    assert session.params[0] | {"page": 0} == {
        "MF_ID": 17,
        "Month": "09-2026",
        "strCat": 46,
        "strType": -1,
        "page": 0,
        "pageSize": 100,
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse({}, status=403), "Could not download"),
        (FakeResponse(ValueError("not json")), "other than JSON"),
        (FakeResponse({"data": []}), "missing 'data' or 'meta'"),
        (FakeResponse({"data": [], "meta": {"page": 1}}), "pageCount"),
        (FakeResponse({"data": [], "meta": {"pageCount": 1, "total": 5}}), "said 5 rows"),
    ],
)
def test_fetch_errors(response: FakeResponse, message: str) -> None:
    with pytest.raises(SourceError, match=message):
        fetch_ter_records("09-2026", 46, 17, session=FakeSession(response))  # type: ignore[arg-type]


# ---------- ingest ----------


def _fake_fetch(records: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, int, int]] = []

    def fetch(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        calls.append((month, category_id, mf_id))
        return records

    return fetch, calls


def test_fetch_ter_stores_rows(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    fetch, calls = _fake_fetch(
        [
            _row("Alpha Arbitrage Fund", "2026-09-29", "0.33", "2.30"),
            _row("Alpha Arbitrage Fund", "2026-09-30", "0.32", "2.31"),
        ]
    )
    result = ingest.fetch_ter(conn, config, "09-2026", fetch)
    assert result.ok and result.rows == 2
    assert calls == [("09-2026", 46, 17), ("09-2026", 46, 22)]  # one request per fund house
    assert result.notes == ["No rows found for: Beta Arbitrage Fund"]
    latest = db.read_latest_ter(conn).set_index("amfi_code")
    assert latest.loc[100001, "base_ter"] == pytest.approx(0.32)
    assert latest.loc[100001, "date"] == "2026-09-30"


def test_fetch_ter_errors(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    fetch, _ = _fake_fetch([])
    assert "MM-YYYY" in ingest.fetch_ter(conn, config, "2026-09", fetch).errors[0]
    assert "No TER values for 09-2026" in ingest.fetch_ter(conn, config, "09-2026", fetch).errors[0]

    def broken(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        raise SourceError("blocked")

    result = ingest.fetch_ter(conn, config, "09-2026", broken)
    assert result.errors == ["Alpha Arbitrage Fund: blocked", "Beta Arbitrage Fund: blocked"]
    assert "download the Excel" in result.notes[-1]


def test_fetch_ter_keeps_houses_that_worked(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    def half_broken(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        if mf_id == 22:
            raise SourceError("403 Forbidden")
        return [_row("Alpha Arbitrage Fund", "2026-09-30", "0.33", "2.3")]

    result = ingest.fetch_ter(conn, config, "09-2026", half_broken)
    assert result.rows == 1
    assert result.errors == ["Beta Arbitrage Fund: 403 Forbidden"]
    assert set(db.read_latest_ter(conn)["amfi_code"]) == {100001}


def test_fetch_ter_skips_funds_without_mf_id(tmp_path: Path, conn: db.sqlite3.Connection) -> None:
    path = tmp_path / "c.toml"
    path.write_text(CONFIG_TEXT.replace("amfi_mf_id = 22\n", ""))
    fetch, calls = _fake_fetch([_row("Alpha Arbitrage Fund", "2026-09-30", "0.33", "2.3")])
    result = ingest.fetch_ter(conn, load_config(path), "09-2026", fetch)
    assert result.ok and calls == [("09-2026", 46, 17)]
    assert "Not fetched (no amfi_mf_id in config): Beta Arbitrage Fund" in result.notes

    path.write_text(CONFIG_TEXT.replace("amfi_mf_id = 22\n", "").replace("amfi_mf_id = 17\n", ""))
    assert (
        "No fund has an amfi_mf_id"
        in ingest.fetch_ter(conn, load_config(path), "09-2026", fetch).errors[0]
    )


def test_mf_id_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(CONFIG_TEXT.replace("amfi_mf_id = 17", "amfi_mf_id = 0"))
    with pytest.raises(ConfigError, match="amfi_mf_id must be positive"):
        load_config(path)


def test_fetch_ter_needs_category_id(tmp_path: Path, conn: db.sqlite3.Connection) -> None:
    path = tmp_path / "c.toml"
    path.write_text(CONFIG_TEXT.replace("[settings.amfi_ter_category_ids]\nArbitrage = 46\n", ""))
    fetch, calls = _fake_fetch([])
    result = ingest.fetch_ter(conn, load_config(path), "09-2026", fetch)
    assert "No AMFI TER category id for 'Arbitrage'" in result.errors[0]
    assert calls == []


@pytest.mark.parametrize("value", ["0", "true", '"46"'])
def test_category_id_validation(tmp_path: Path, value: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(CONFIG_TEXT.replace("Arbitrage = 46", f"Arbitrage = {value}"))
    with pytest.raises(ConfigError, match="positive whole number"):
        load_config(path)


@pytest.mark.parametrize(
    ("today", "expected"),
    [(date(2026, 10, 8), "09-2026"), (date(2026, 1, 1), "12-2025"), (date(2026, 3, 31), "02-2026")],
)
def test_previous_month(today: date, expected: str) -> None:
    assert ingest.previous_month(today) == expected


# ---------- CLI ----------


def test_cli_fetch_ter(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    db_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(config_file))
    monkeypatch.setenv(DB_ENV, str(db_file))
    seen: list[tuple[str, int]] = []

    def fake(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        seen.append((month, mf_id))
        return [_row("Alpha Arbitrage Fund", "2026-08-31", "0.33", "2.3")]

    monkeypatch.setattr(ingest, "fetch_ter_records", fake)
    assert cli.main(["fetch-ter", "--month", "08-2026"]) == 0
    assert "Rows stored: 1" in capsys.readouterr().out
    assert seen == [("08-2026", 17), ("08-2026", 22)]
    assert cli.main(["fetch-ter"]) == 0  # default: last month
    assert seen[-1][0] == ingest.previous_month(date.today())


# ---------- app ----------


def test_app_fetch_ter_button(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, db_file: Path
) -> None:
    from streamlit.testing.v1 import AppTest

    from tests.test_cli_app import APP_PATH

    monkeypatch.setenv(CONFIG_ENV, str(config_file))
    monkeypatch.setenv(DB_ENV, str(db_file))
    seen: list[str] = []

    def fake(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        seen.append(month)
        return [_row("Alpha Arbitrage Fund", "2026-08-31", "0.33", "2.3")]

    monkeypatch.setattr(ingest, "fetch_ter_records", fake)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert app.text_input(key="ter_month").value == ingest.previous_month(date.today())
    app.text_input(key="ter_month").set_value("08-2026")
    app.button(key="ter_fetch").click().run()
    assert not app.exception
    assert seen == ["08-2026", "08-2026"]
    assert any("Stored 1 row(s)" in s.value for s in app.success)
