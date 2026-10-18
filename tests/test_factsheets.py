"""Factsheet values: PDF rules on real factsheet pages, links, config, ingest, values file."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pypdfium2
import pytest

from arbitrage_analyser import db, factsheets, ingest, store
from arbitrage_analyser.config import AppConfig, ConfigError, FactsheetSource, load_config
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.factsheet_links import (
    Download,
    candidate_links,
    download_latest,
    edition_months,
    fill,
    links_on_listing,
)
from arbitrage_analyser.sources.factsheet_pdf import (
    FactsheetValues,
    parse_date,
    read_factsheet,
    read_page,
)
from tests.conftest import CONFIG_TEXT, factsheet_values, fake_factsheets, ter_rows, xlsx_bytes

FIXTURES = Path(__file__).parent / "data" / "factsheets"
SHIPPED = load_config()

# Pages copied from the AMC factsheets (Sep-2026 data; Nippon's September edition holds August
# data). Some fixtures add a page that must be skipped: SBI and Edelweiss a continuation page
# with the same title, Nippon and Tata a fund-of-funds page that names the fund. Expected
# values were read from the printed pages.
# fixture, fund (AMFI code), page, as of, AUM, AAUM, turnover, Direct BER
EXPECTED = [
    ("kotak_2026-09", 119771, 1, "2026-09-30", 75274.99, 75770.41,
     "Equity 491.02%, Total 1707.02%", 0.39),
    ("icici_2026-09", 120364, 1, "2026-09-30", 35576.34, 35244.35, None, 0.34),
    ("sbi_2026-09", 119574, 1, "2026-09-30", 39791.81, 46917.73, "Equity 1.60, Gross 11.57",
     0.34),
    ("invesco_2026-09", 120401, 1, "2026-09-30", 30395.97, 26638.29, "1Y 16.42", 0.34),
    ("absl_2026-09", 119526, 1, "2026-09-30", 26933.29, 27096.47, "Gross 10.33", 0.27),
    ("edelweiss_2026-09", 130206, 1, "2026-09-30", 12406.71, 14470.61,
     "Equity 1.98, Gross 12.02", 0.33),
    ("nippon_2026-08", 118585, 2, "2026-08-31", 17532.65, 17097.46,
     "Equity 1.94, Gross 11.21", 0.33),
    ("tata_2026-09", 145724, 2, "2026-09-30", 24539.33, 24804.26,
     "Equity 275.52%, Gross 1409.69%", 0.26),
]  # fmt: skip


def _source(code: int) -> FactsheetSource:
    source = SHIPPED.fund_by_code(code).factsheet
    assert source is not None
    return source


def _read(name: str, source: FactsheetSource) -> FactsheetValues:
    content = (FIXTURES / f"{name}.pdf").read_bytes()
    return read_factsheet(content, source.page_title, dict(source.turnover))


# ---------- PDF rules ----------


@pytest.mark.parametrize(
    ("name", "code", "page", "as_of", "aum", "aaum", "turnover", "ber"), EXPECTED
)
def test_reads_each_amc_factsheet(
    name: str,
    code: int,
    page: int,
    as_of: str,
    aum: float,
    aaum: float,
    turnover: str | None,
    ber: float,
) -> None:
    values = _read(name, _source(code))
    assert values.page == page
    assert values.as_of == date.fromisoformat(as_of)
    assert values.aum_crore == pytest.approx(aum)
    assert values.aaum_crore == pytest.approx(aaum)
    assert values.turnover == turnover
    assert values.direct_ber_pct == pytest.approx(ber)
    assert values.missing == (() if turnover else ("Turnover",))


def test_every_shipped_fund_has_a_tested_factsheet() -> None:
    assert {f.amfi_code for f in SHIPPED.funds if f.factsheet} == {row[1] for row in EXPECTED}


def test_wrong_title_or_file_is_an_error() -> None:
    content = (FIXTURES / "kotak_2026-09.pdf").read_bytes()
    with pytest.raises(SourceError, match="no page mentions"):
        read_factsheet(content, "Gamma Arbitrage Fund", {})
    # in the text (the benchmark name) but not a page heading
    tata = (FIXTURES / "tata_2026-09.pdf").read_bytes()
    with pytest.raises(SourceError, match="no page titled"):
        read_factsheet(tata, "Nifty 50 Arbitrage Index", {})
    with pytest.raises(SourceError, match="not a PDF"):
        read_factsheet(b"<html>Not found</html>", "Kotak Arbitrage Fund", {})


def test_two_main_pages_for_one_fund_is_an_error(tmp_path: Path) -> None:
    """Never guess between two pages that both show the fund's AUM."""
    source = pypdfium2.PdfDocument(FIXTURES / "icici_2026-09.pdf")
    twice = pypdfium2.PdfDocument.new()
    twice.import_pages(source, [0, 0])
    path = tmp_path / "twice.pdf"
    twice.save(path)
    with pytest.raises(SourceError, match="pages 1, 2"):
        read_factsheet(path.read_bytes(), "ICICI Prudential Arbitrage Fund", {})


def test_turnover_label_not_printed_is_none() -> None:
    values = _read("absl_2026-09", FactsheetSource("Aditya Birla Sun Life Arbitrage Fund",
                   turnover=(("Equity", "Equity Portfolio Turnover"),)))  # fmt: skip
    assert values.turnover is None
    assert values.missing == ("Turnover",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("as on 30th September, 2026:", date(2026, 9, 30)),
        ("as on September 30, 2026", date(2026, 9, 30)),
        ("as on 30-Sep-26 : Rs.", date(2026, 9, 30)),
        ("as on 30/09/2026:", date(2026, 9, 30)),
        ("as on 31 August 2026", date(2026, 8, 31)),
        ("as on 31/02/2026", None),
        ("no date here", None),
    ],
)
def test_parse_date(text: str, expected: date | None) -> None:
    assert parse_date(text) == expected


def _word(text: str, x0: float, top: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x0 + 5 * len(text), "top": top, "bottom": top + 7}


def _row(top: float, *items: tuple[str, float]) -> list[dict[str, object]]:
    return [_word(text, x0, top) for text, x0 in items]


def test_conflicting_as_of_dates_are_an_error() -> None:
    words = [
        *_row(10, ("Data", 300), ("as", 330), ("on", 345), ("31-Aug-2026", 360)),
        *_row(100, ("AUM", 20), ("as", 45), ("on", 60), ("30/09/2026:", 75), ("1,234.50", 135),
              ("Cr", 180)),
    ]  # fmt: skip
    with pytest.raises(SourceError, match="30-Sep-2026 but the page says data as on 31-Aug"):
        read_page(words, 1, {})


def test_as_on_of_another_label_is_not_the_as_of_date() -> None:
    words = [
        *_row(100, ("AUM", 20), ("1,234.50", 60), ("Cr", 110)),
        *_row(110, ("NAV", 20), ("as", 45), ("on", 60), ("29-Sep-2026", 75)),
    ]
    assert read_page(words, 1, {}) is None  # no date for the AUM: not used


def test_average_aum_is_aaum() -> None:
    words = [
        *_row(10, ("Data", 20), ("as", 45), ("on", 60), ("30-Sep-2026", 75)),
        *_row(100, ("Monthly", 20), ("Average", 60), ("AUM:", 105), ("26,638.29", 130),
              ("Cr", 180)),
        *_row(110, ("Month", 20), ("End", 50), ("AUM:", 70), ("30,395.97", 130), ("Cr", 180)),
    ]  # fmt: skip
    values = read_page(words, 1, {})
    assert values is not None
    assert (values.aum_crore, values.aaum_crore) == (30395.97, 26638.29)


def test_two_expense_ratios_without_a_ber_header_leave_ber_blank() -> None:
    words = [
        *_row(10, ("Data", 20), ("as", 45), ("on", 60), ("30-Sep-2026", 75)),
        *_row(20, ("AUM", 20), ("1,234.50", 60), ("Cr", 110)),
        *_row(100, ("Expense", 20), ("Ratio", 60)),
        *_row(110, ("Direct", 20), ("0.43%", 60), ("0.34%", 100)),
    ]
    values = read_page(words, 1, {})
    assert values is not None and values.aum_crore == 1234.5 and values.direct_ber_pct is None
    assert values.missing[-1] == (
        "Direct BER (the Direct row shows 0.43%, 0.34%; which is BER is unclear)"
    )


# ---------- links ----------


def test_fill_and_editions() -> None:
    assert edition_months(date(2026, 2, 18)) == [
        date(2026, 2, 1),
        date(2026, 1, 1),
        date(2025, 12, 1),
    ]
    template = "{month}-{Month}-{mm}-{year}-{next_year}-{next_mm}"
    assert fill(template, date(2026, 12, 1)) == "december-December-12-2026-2027-01"
    tata = _source(145724)
    assert list(candidate_links(tata, date(2026, 10, 18)))[1] == (
        "https://www.tatamutualfund.com/system/files/2026-10/"
        "Tata%20Factsheet%20-%20September%202026.pdf"
    )
    # a link without placeholders is tried once
    assert len(list(candidate_links(_source(119771), date(2026, 10, 18)))) == 1


def test_links_on_listing_page() -> None:
    edelweiss = _source(130206)
    html = """
      <a href="/Files/MF/Downloads/FACTSHEETS/FACTSHEETS/Factsheet__October_2026_17102026141628.pdf">Oct</a>
      <a href='https://www.edelweissmf.com/Files/MF/Downloads/FACTSHEETS/FACTSHEETS/Factsheet__September_2026_15092026120000.pdf?a=1&amp;b=2'>Sep</a>
      <a href="https://www.edelweissmf.com/Files/Other__October_2026_1.pdf">other</a>
      <a href="http://www.edelweissmf.com/Factsheet__October_2026_9.pdf">not https</a>
    """  # noqa: E501
    links = links_on_listing(html, edelweiss.listing or "", edelweiss, date(2026, 10, 18))
    assert links == [
        "https://www.edelweissmf.com/Files/MF/Downloads/FACTSHEETS/FACTSHEETS/"
        "Factsheet__October_2026_17102026141628.pdf",
        "https://www.edelweissmf.com/Files/MF/Downloads/FACTSHEETS/FACTSHEETS/"
        "Factsheet__September_2026_15092026120000.pdf?a=1&b=2",
    ]


def test_download_tries_newest_edition_first() -> None:
    calls: list[str] = []

    def get(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        if "october" in url:
            return 404, "text/html", b""
        if "september" in url:
            return 200, "text/html", b"<html>soft 404</html>"
        return 200, "application/pdf", b"%PDF-1.7"

    source = FactsheetSource("SBI Arbitrage Fund", link="https://x.in/f-{month}-{year}.pdf")
    got = download_latest(source, date(2026, 10, 18), get)
    assert got == Download("https://x.in/f-august-2026.pdf", b"%PDF-1.7")
    assert calls == [f"https://x.in/f-{m}-2026.pdf" for m in ("october", "september", "august")]

    def missing(url: str) -> tuple[int, str, bytes]:
        return 404, "", b""

    with pytest.raises(SourceError, match=r"october-2026.pdf \(HTTP 404\)"):
        download_latest(source, date(2026, 10, 18), missing)


def test_download_from_listing_page() -> None:
    edelweiss = _source(130206)
    pdf = "https://www.edelweissmf.com/Factsheet__October_2026_1.pdf"

    def get(url: str) -> tuple[int, str, bytes]:
        if url == edelweiss.listing:
            return 200, "text/html", f'<a href="{pdf}">x</a>'.encode()
        return (200, "application/pdf", b"%PDF-1") if url == pdf else (404, "", b"")

    assert download_latest(edelweiss, date(2026, 10, 18), get).url == pdf
    with pytest.raises(SourceError, match="no PDF link containing 'Factsheet__October_2026_'"):
        download_latest(edelweiss, date(2026, 10, 18), lambda url: (200, "text/html", b""))
    with pytest.raises(SourceError, match="listing page returned HTTP 503"):
        download_latest(edelweiss, date(2026, 10, 18), lambda url: (503, "", b""))


# ---------- config ----------

_ALPHA_LINK = 'link = "https://example.com/alpha/factsheet-{month}-{year}.pdf"'


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (_ALPHA_LINK, 'link = "http://example.com/a.pdf"', "https:// link"),
        (_ALPHA_LINK, 'link = "https://example.com/{monthname}.pdf"', "unknown placeholders"),
        (_ALPHA_LINK, 'link = "https://example.com/{month.pdf"', "broken"),
        (_ALPHA_LINK, "", "either link or listing"),
        (_ALPHA_LINK, _ALPHA_LINK + '\nlisting = "https://example.com/list"', "either link"),
        (_ALPHA_LINK, 'listing = "https://example.com/list"', "listing needs link_text"),
        (_ALPHA_LINK, _ALPHA_LINK + '\nlink_text = "x"', "only with listing"),
        ('page_title = "Alpha Arbitrage Fund"', 'page_title = " "', "page_title"),
        ('turnover = { Equity = "Equity Turnover"', 'turnover = { Equity = ""', "turnover"),
    ],
)
def test_factsheet_config_is_checked(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in CONFIG_TEXT
    path = tmp_path / "funds.toml"
    path.write_text(CONFIG_TEXT.replace(old, new, 1))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_listing_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "funds.toml"
    path.write_text(
        CONFIG_TEXT.replace(
            _ALPHA_LINK,
            'listing = "https://example.com/list"\nlink_text = "Factsheet__{Month}_{year}_"',
        )
    )
    source = load_config(path).fund_by_code(100001).factsheet
    assert source is not None and source.listing and source.link is None


# ---------- ingest ----------

TODAY = date(2026, 10, 18)


def _fetch(
    conn: db.sqlite3.Connection, config_file: Path, values: FactsheetValues
) -> ingest.LoadResult:
    download, read = fake_factsheets(values)
    return factsheets.fetch_factsheets(conn, load_config(config_file), TODAY, download, read)


def test_fetch_stores_values_with_page_link(conn: db.sqlite3.Connection, config_file: Path) -> None:
    result = _fetch(conn, config_file, factsheet_values())
    assert result.ok and result.rows == 1
    assert result.notes[-1] == "No factsheet set in config: Beta Arbitrage Fund"
    row = db.read_latest_factsheet(conn).iloc[0]
    assert (row["amfi_code"], row["as_of"], row["aum_crore"], row["direct_ber"]) == (
        100001,
        "2026-09-30",
        75274.99,
        0.34,
    )
    assert row["turnover"] == "Equity 1.60, Gross 11.57"
    assert (row["source_url"], row["page"]) == (
        "https://example.com/alpha/factsheet-october-2026.pdf",
        62,
    )


def test_fetch_skips_unchanged_values(conn: db.sqlite3.Connection, config_file: Path) -> None:
    assert _fetch(conn, config_file, factsheet_values()).rows == 1
    fetched = db.read_factsheets(conn)["fetched_at"].tolist()
    again = _fetch(conn, config_file, factsheet_values())
    assert again.ok and again.rows == 0
    assert db.read_factsheets(conn)["fetched_at"].tolist() == fetched
    assert fetched[0].endswith("+00:00")  # UTC, so runs on a laptop and on GitHub compare


def test_fetch_reports_changed_values(conn: db.sqlite3.Connection, config_file: Path) -> None:
    assert _fetch(conn, config_file, factsheet_values()).ok
    result = _fetch(conn, config_file, factsheet_values(aum=75275.0))
    assert result.rows == 1
    assert "Alpha Arbitrage Fund: AUM for 2026-09-30 changed 75274.99 -> 75275.0" in result.notes
    assert db.read_latest_factsheet(conn).iloc[0]["aum_crore"] == pytest.approx(75275.0)


def test_a_reread_lacking_a_value_keeps_the_stored_row(
    conn: db.sqlite3.Connection, config_file: Path
) -> None:
    """Never mix values from two PDFs in one row, never blank a stored value."""
    assert _fetch(conn, config_file, factsheet_values()).ok
    result = _fetch(conn, config_file, factsheet_values(aum=1.0, turnover=None))
    assert result.rows == 0
    assert (
        "Alpha Arbitrage Fund: kept the stored values for 2026-09-30; the new read lacks Turnover"
        in result.notes
    )
    row = db.read_latest_factsheet(conn).iloc[0]
    assert (row["aum_crore"], row["turnover"]) == (75274.99, "Equity 1.60, Gross 11.57")


def test_fetch_never_stores_out_of_range_values(
    conn: db.sqlite3.Connection, config_file: Path
) -> None:
    result = _fetch(conn, config_file, factsheet_values(aum=0.0, ber=39.0))
    assert result.errors == [
        "Alpha Arbitrage Fund: AUM 0.0 is outside 0-1e+06; not stored",
        "Alpha Arbitrage Fund: BER 39.0 is outside 0-2.5; not stored",
    ]
    row = db.read_latest_factsheet(conn).iloc[0]
    assert row.isna()["aum_crore"] and row.isna()["direct_ber"]
    assert row["aaum_crore"] == pytest.approx(75770.41)  # the other values are kept


def test_fetch_reports_missing_values(conn: db.sqlite3.Connection, config_file: Path) -> None:
    result = _fetch(conn, config_file, factsheet_values(turnover=None, aaum=None))
    assert result.ok
    assert "Alpha Arbitrage Fund: not found in the factsheet: AAUM, Turnover" in result.notes


def test_each_month_is_kept_and_future_dates_are_rejected(
    conn: db.sqlite3.Connection, config_file: Path
) -> None:
    assert _fetch(conn, config_file, factsheet_values()).ok
    older = _fetch(conn, config_file, factsheet_values(as_of=date(2026, 8, 31), aum=70000.0))
    assert older.rows == 1  # stored as August; the app still shows September
    assert db.read_latest_factsheet(conn).iloc[0]["aum_crore"] == pytest.approx(75274.99)
    future = _fetch(conn, config_file, factsheet_values(as_of=date(2026, 10, 31)))
    assert future.errors == ["Alpha Arbitrage Fund: as-of date 2026-10-31 is in the future"]
    assert len(db.read_factsheets(conn)) == 2


def test_fetch_flags_ber_mismatch_and_odd_aaum(
    conn: db.sqlite3.Connection, config_file: Path
) -> None:
    config = load_config(config_file)
    ter = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "01/09/2026", 0.25, 2.0)]))
    assert ingest.import_ter(conn, config, ter).ok
    result = _fetch(conn, config_file, factsheet_values(ber=0.34, aaum=10.0))
    assert result.new_flags == 2
    ber = "Alpha Arbitrage Fund: factsheet Direct BER 0.34% vs AMFI BER 0.25% (0.29% with GST)"
    assert ber in set(db.read_flags(conn)["detail"])
    assert f"Flag: {ber}" in result.notes  # listed for the command line and the job log
    assert any("AAUM 10.00 Cr vs AUM 75,274.99 Cr" in n for n in result.notes)


def test_ber_not_checked_is_reported(conn: db.sqlite3.Connection, config_file: Path) -> None:
    result = _fetch(conn, config_file, factsheet_values())
    assert (
        "Alpha Arbitrage Fund: BER not checked against AMFI (no AMFI TER on or before 2026-09-30)"
        in result.notes
    )


def test_check_latest_flags_values_loaded_before_the_ter(
    conn: db.sqlite3.Connection, config: AppConfig, config_file: Path
) -> None:
    """The app loads the values file first and AMFI TER later: the check runs again."""
    assert _fetch(conn, config_file, factsheet_values(ber=0.34)).new_flags == 0
    ter = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "01/09/2026", 0.25, 2.0)]))
    assert ingest.import_ter(conn, config, ter).ok
    assert factsheets.check_latest(conn, config) == 1
    assert factsheets.check_latest(conn, config) == 0  # already flagged


@pytest.mark.parametrize("factsheet_ber", [0.33, 0.34, 0.39])
def test_ber_matching_amfi_with_or_without_gst_is_not_flagged(
    conn: db.sqlite3.Connection, config_file: Path, factsheet_ber: float
) -> None:
    config = load_config(config_file)
    ter = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "01/09/2026", 0.33, 2.0)]))
    assert ingest.import_ter(conn, config, ter).ok
    assert _fetch(conn, config_file, factsheet_values(ber=factsheet_ber)).new_flags == 0


def test_fetch_reports_a_failed_fund_and_downloads_each_file_once(
    conn: db.sqlite3.Connection, tmp_path: Path
) -> None:
    """Beta shares Alpha's factsheet file: one download serves both."""
    # Beta is the last [[funds]] entry, so a table appended to the file belongs to it
    text = CONFIG_TEXT + '\n[funds.factsheet]\npage_title = "Beta Arbitrage Fund"\n' + _ALPHA_LINK
    path = tmp_path / "two.toml"
    path.write_text(text)
    config = load_config(path)
    downloads: list[str] = []

    def download(source: FactsheetSource, today: date) -> Download:
        downloads.append(source.page_title)
        return Download("https://example.com/a.pdf", b"%PDF")

    def read(content: bytes, title: str, turnover: dict[str, str]) -> FactsheetValues:
        if title.startswith("Beta"):
            raise SourceError("no page titled 'Beta Arbitrage Fund'")
        return factsheet_values()

    result = factsheets.fetch_factsheets(conn, config, TODAY, download, read)
    assert downloads == ["Alpha Arbitrage Fund"]
    assert result.rows == 1
    assert result.errors == ["Beta Arbitrage Fund: no page titled 'Beta Arbitrage Fund'"]


def test_a_damaged_pdf_is_reported_for_that_fund(monkeypatch: pytest.MonkeyPatch) -> None:
    import pdfplumber

    def broken(*args: object, **kwargs: object) -> None:
        raise KeyError("/Root")

    monkeypatch.setattr(pdfplumber, "open", broken)
    content = (FIXTURES / "icici_2026-09.pdf").read_bytes()
    with pytest.raises(SourceError, match="the PDF cannot be read: KeyError"):
        read_factsheet(content, "ICICI Prudential Arbitrage Fund", {})


# ---------- values file ----------


def _write_values(conn: db.sqlite3.Connection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(store.render(conn, store.FACTSHEET_FILE))


def _export(conn: db.sqlite3.Connection, config_file: Path, path: Path) -> None:
    assert _fetch(conn, config_file, factsheet_values(turnover=None)).ok
    _write_values(conn, path)


def test_values_file_round_trip(
    conn: db.sqlite3.Connection, config: AppConfig, config_file: Path, tmp_path: Path
) -> None:
    path = tmp_path / "published" / "values.csv"
    _export(conn, config_file, path)
    with db.connect(tmp_path / "fresh.db") as fresh:
        first = factsheets.import_csv(fresh, config, path, TODAY)
        assert first.ok and first.rows == 1
        again = factsheets.import_csv(fresh, config, path, TODAY)
        assert again.ok and again.rows == 0  # loading again changes nothing
        copied = db.read_factsheets(fresh)
    assert copied.astype(str).equals(db.read_factsheets(conn).astype(str))
    assert copied.iloc[0]["turnover"] is None


def test_values_file_does_not_overwrite_a_later_fetch(
    conn: db.sqlite3.Connection, config: AppConfig, config_file: Path, tmp_path: Path
) -> None:
    path = tmp_path / "values.csv"
    _export(conn, config_file, path)
    text = path.read_text().replace("75274.99", "1.0").replace(",2026-10-", ",2020-01-")
    path.write_text(text)  # an older copy of the file with a different value
    assert factsheets.import_csv(conn, config, path, TODAY).rows == 0
    assert db.read_latest_factsheet(conn).iloc[0]["aum_crore"] == pytest.approx(75274.99)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",2026-09-30,", ",30/09/2026,", "as_of '30/09/2026' is not YYYY-MM-DD"),
        (",https://", ",http://", "is not an https link"),
        (",62,", ",,", "page '' is not a positive whole number"),
        ("100001,", "x,", "amfi_code 'x' is not a positive whole number"),
        (",75274.99,", ",lots,", "aum_crore 'lots' is not a number"),
        ("+00:00", "", "is not in UTC"),
        ("+00:00", "nonsense", "is not an ISO timestamp"),
        ("+00:00", "Z", "is not in UTC (+00:00)"),
        (",2026-09-30,", ",2099-09-30,", "as_of 2099-09-30 is in the future"),
    ],
)
def test_values_file_lines_are_checked(
    conn: db.sqlite3.Connection,
    config: AppConfig,
    config_file: Path,
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = tmp_path / "values.csv"
    with db.connect(tmp_path / "source.db") as source:
        assert _fetch(source, config_file, factsheet_values()).ok
        _write_values(source, path)
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    result = factsheets.import_csv(conn, config, path, TODAY)
    assert len(result.errors) == 1 and message in result.errors[0], result.errors
    assert "line 2" in result.errors[0]
    assert db.read_factsheets(conn).empty


def test_values_file_gets_the_same_checks(
    conn: db.sqlite3.Connection, config: AppConfig, config_file: Path, tmp_path: Path
) -> None:
    path = tmp_path / "values.csv"
    with db.connect(tmp_path / "source.db") as source:
        assert _fetch(source, config_file, factsheet_values()).ok
        _write_values(source, path)
    path.write_text(path.read_text().replace(",0.34,", ",9.9,"))
    result = factsheets.import_csv(conn, config, path, TODAY)
    assert result.errors == ["Alpha Arbitrage Fund: BER 9.9 is outside 0-2.5; not stored"]
    assert db.read_latest_factsheet(conn).isna().iloc[0]["direct_ber"]


def test_values_file_errors(conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path) -> None:
    missing = factsheets.import_csv(conn, config, tmp_path / "missing.csv", TODAY)
    assert missing.ok and missing.rows == 0
    bad = tmp_path / "bad.csv"
    bad.write_text("amfi_code,as_of\n1,2026-09-30\n")
    assert "lacks columns" in factsheets.import_csv(conn, config, bad, TODAY).errors[0]
    unknown = tmp_path / "unknown.csv"
    unknown.write_text(
        ",".join(db.FACTSHEET_COLUMNS)
        + "\n999,2026-09-30,1.0,1.0,,0.3,https://x.in/a.pdf,1,2026-10-18T00:00:00+00:00\n"
    )
    result = factsheets.import_csv(conn, config, unknown, TODAY)
    # a fund no longer in config keeps its history (hidden in the app, not deleted)
    assert result.ok and result.rows == 1


# ---------- download ----------


class _Response:
    def __init__(self, status: int, chunks: list[bytes]) -> None:
        self.status_code, self.chunks = status, chunks
        self.headers = {"Content-Type": "application/pdf"}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return self.chunks


def test_http_get(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from arbitrage_analyser.sources import factsheet_links

    responses = {
        "https://a/ok.pdf": _Response(200, [b"%PDF", b"-1"]),
        "https://a/missing.pdf": _Response(404, [b"never read"]),
        "https://a/huge.pdf": _Response(200, [b"x" * 600, b"x" * 600]),
    }

    def get(url: str, **kwargs: object) -> _Response:
        if url == "https://a/down.pdf":
            raise requests.ConnectionError("no route")
        return responses[url]

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(factsheet_links, "MAX_BYTES", 1000)
    assert factsheet_links.http_get("https://a/ok.pdf") == (200, "application/pdf", b"%PDF-1")
    assert factsheet_links.http_get("https://a/missing.pdf")[::2] == (404, b"")
    with pytest.raises(SourceError, match="larger than"):
        factsheet_links.http_get("https://a/huge.pdf")
    with pytest.raises(SourceError, match=r"could not download https://a/down\.pdf"):
        factsheet_links.http_get("https://a/down.pdf")
