import pandas as pd
import pytest
import requests

from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.aaum import parse_aaum
from arbitrage_analyser.sources.benchmark import parse_benchmark_csv
from arbitrage_analyser.sources.mfapi import fetch_history, parse_history
from arbitrage_analyser.sources.tables import read_grid, to_number
from arbitrage_analyser.sources.ter import parse_ter
from tests.conftest import aaum_rows, benchmark_csv, growth_series, ter_rows, xlsx_bytes

MFAPI_PAYLOAD = {
    "meta": {
        "fund_house": "Invesco Mutual Fund",
        "scheme_category": "Hybrid Schemes - Arbitrage Fund",
        "scheme_code": 120401,
        "scheme_name": "Invesco India Arbitrage Fund - Direct Plan - Growth",
        "isin_growth": "INF205K01KR8",
        "isin_div_reinvestment": None,
    },
    "data": [
        {"date": "06-10-2026", "nav": "37.48850"},
        {"date": "05-10-2026", "nav": "37.47250"},
        {"date": "01-10-2026", "nav": "37.42770"},
    ],
    "status": "SUCCESS",
}


# ---------- mfapi ----------


def test_parse_history_sorts_ascending() -> None:
    history = parse_history(MFAPI_PAYLOAD)
    assert history.scheme_code == 120401
    assert history.isin_growth == "INF205K01KR8"
    assert list(history.nav.index.strftime("%Y-%m-%d")) == [
        "2026-10-01",
        "2026-10-05",
        "2026-10-06",
    ]
    assert history.nav.iloc[-1] == pytest.approx(37.4885)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"meta": {}}, "missing"),
        ({"meta": {"scheme_code": 1}, "data": []}, "no NAV data"),
        ({"meta": {"scheme_code": 1}, "data": [{"date": "31-02-2026", "nav": "1"}]}, "unreadable"),
        (
            {"meta": {"scheme_code": 1}, "data": [{"date": "01-02-2026", "nav": "N.A."}]},
            "unreadable",
        ),
        ({"meta": {}, "data": [{"date": "01-02-2026", "nav": "1"}]}, "scheme_code"),
    ],
)
def test_parse_history_errors(payload: dict, message: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(SourceError, match=message):
        parse_history(payload)


class _FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> object:
        return self._payload


class _FakeSession(requests.Session):
    def __init__(self, response: _FakeResponse) -> None:
        super().__init__()
        self.response = response
        self.urls: list[str] = []

    def get(self, url, **kwargs):  # type: ignore[no-untyped-def, override]
        self.urls.append(url)
        return self.response


def test_fetch_history_checks_code_and_errors() -> None:
    session = _FakeSession(_FakeResponse(MFAPI_PAYLOAD))
    assert fetch_history(120401, session).scheme_code == 120401
    assert session.urls == ["https://api.mfapi.in/mf/120401"]
    with pytest.raises(SourceError, match="was requested"):
        fetch_history(999, _FakeSession(_FakeResponse(MFAPI_PAYLOAD)))
    with pytest.raises(SourceError, match="Could not download"):
        fetch_history(120401, _FakeSession(_FakeResponse({}, status=500)))
    with pytest.raises(SourceError, match="unexpected response"):
        fetch_history(120401, _FakeSession(_FakeResponse([1, 2])))


# ---------- benchmark ----------


@pytest.mark.parametrize("fmt", ["%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y"])
def test_parse_benchmark_date_formats(fmt: str) -> None:
    series = growth_series("2026-01-01", "2026-01-31", 0.07, start_value=1000)
    frame = parse_benchmark_csv(benchmark_csv(series, fmt))
    assert list(frame["date"]) == list(series.index)
    assert frame["value"].iloc[0] == pytest.approx(1000.0)


def test_parse_benchmark_handles_spaces_and_commas() -> None:
    content = b'Date ,Open ,High ,Low ,Close \n02-Jan-2026,1,1,1,"2,345.10"\n'
    frame = parse_benchmark_csv(content)
    assert frame["value"].tolist() == [2345.1]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "not a readable CSV"),
        (b"Date,Open\n02-Jan-2026,1\n", "no 'close' column"),
        (b"Date,Close\n02-Jan-2026,abc\n", "line 2"),
        (b"Date,Close\nyesterday,1\n", "unknown format"),
    ],
)
def test_parse_benchmark_errors(content: bytes, message: str) -> None:
    with pytest.raises(SourceError, match=message):
        parse_benchmark_csv(content)


# ---------- generic tables ----------


def test_read_grid_formats() -> None:
    rows = [["a", "b"], ["1", "2"]]
    assert read_grid(xlsx_bytes(rows)).iloc[1, 1] in (2, "2")
    assert read_grid(b"a,b\n1,2\n").iloc[1, 1] == "2"
    html = b"<html><table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table></html>"
    assert read_grid(html).shape == (2, 2)
    with pytest.raises(SourceError, match="empty"):
        read_grid(b"")
    with pytest.raises(SourceError, match="spreadsheet"):
        read_grid(b"\xd0\xcf\x11\xe0 not really xls")


def test_to_number() -> None:
    assert to_number("0.34%") == pytest.approx(0.34)
    assert to_number("2,125,927.9") == pytest.approx(2125927.9)
    assert to_number(None) is None
    assert to_number("-") is None
    with pytest.raises(SourceError):
        to_number("abc")


# ---------- TER ----------


def test_parse_ter_reads_direct_plan_only(config: AppConfig) -> None:
    content = xlsx_bytes(
        ter_rows(
            [
                ("Alpha Arbitrage Fund", "03/10/2026", 0.33, 2.30),
                ("Alpha Arbitrage Fund", "04/10/2026", 0.33, 2.32),
                ("Beta Arbitrage Fund", "04/10/2026", 0.34, 1.63),
                ("Gamma Liquid Fund", "04/10/2026", 0.10, 0.20),
            ]
        )
    )
    parsed = parse_ter(content, config.funds)
    assert parsed.unmatched_funds == []
    beta = parsed.rows[parsed.rows["amfi_code"] == 100002].iloc[0]
    assert beta["base_ter"] == pytest.approx(0.34)
    assert beta["total_ter"] == pytest.approx(1.63)
    assert beta["brokerage"] == pytest.approx(0.14)
    assert beta["date"] == pd.Timestamp("2026-10-04")
    assert len(parsed.rows) == 3


def test_parse_ter_reports_unmatched_and_ambiguous(config: AppConfig) -> None:
    only_alpha = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "04/10/2026", 0.33, 2.32)]))
    assert parse_ter(only_alpha, config.funds).unmatched_funds == ["Beta Arbitrage Fund"]
    ambiguous = xlsx_bytes(
        ter_rows(
            [
                ("Alpha Arbitrage Fund", "04/10/2026", 0.33, 2.32),
                ("Alpha Arbitrage Fund of Funds", "04/10/2026", 0.2, 0.5),
            ]
        )
    )
    with pytest.raises(SourceError, match="matches several schemes"):
        parse_ter(ambiguous, config.funds)


def test_parse_ter_layout_errors(config: AppConfig) -> None:
    no_direct = ter_rows([("Alpha Arbitrage Fund", "04/10/2026", 0.33, 2.32)])
    no_direct[0] = [None] * len(no_direct[0])
    with pytest.raises(SourceError, match="Direct Plan"):
        parse_ter(xlsx_bytes(no_direct), config.funds)
    with pytest.raises(SourceError, match="scheme name"):
        parse_ter(xlsx_bytes([["foo", "bar"]]), config.funds)


# ---------- AAUM ----------


def test_parse_aaum_converts_lakhs_to_crore(config: AppConfig) -> None:
    content = xlsx_bytes(
        aaum_rows(
            [
                (100001, "Alpha Arbitrage Fund - Direct Plan - Growth Option", 2125927.9),
                (999999, "Other fund", 10.0),
            ]
        )
    )
    parsed = parse_aaum(content, config.funds)
    assert parsed.quarter_end == pd.Timestamp("2026-09-30")
    assert parsed.funds_found == ["Alpha Arbitrage Fund"]
    assert parsed.rows["aaum_crore"].iloc[0] == pytest.approx(21259.279)


def test_parse_aaum_html_export(config: AppConfig) -> None:
    rows = aaum_rows(
        [(100002, "Beta - Direct - Growth", 150000.0)],
        title="AAUM for the quarter of January - March 2027 (Rs in Crore)",
    )
    cells = "".join(
        "<tr>" + "".join(f"<td>{'' if v is None else v}</td>" for v in row) + "</tr>"
        for row in rows
    )
    parsed = parse_aaum(f"<html><table>{cells}</table></html>".encode(), config.funds)
    assert parsed.quarter_end == pd.Timestamp("2027-03-31")
    assert parsed.rows["aaum_crore"].iloc[0] == pytest.approx(150000.0)


@pytest.mark.parametrize(
    ("title", "message"),
    [
        ("Average AUM (Rs in Lakhs)", "does not name the quarter"),
        ("AAUM for the quarter of July - Smarch 2026 (Rs in Lakhs)", "unknown month"),
        ("AAUM for the quarter of July - September 2026", "lakhs or crore"),
    ],
)
def test_parse_aaum_title_errors(config: AppConfig, title: str, message: str) -> None:
    content = xlsx_bytes(aaum_rows([(100001, "Alpha", 1.0)], title=title))
    with pytest.raises(SourceError, match=message):
        parse_aaum(content, config.funds)


def test_parse_benchmark_ignores_dashes_in_unused_columns() -> None:
    content = b"Date,Open,High,Low,Close\n01-Oct-2026,-,-,-,2401.35\n"
    assert parse_benchmark_csv(content)["value"].tolist() == [2401.35]
