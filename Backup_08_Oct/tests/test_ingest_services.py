from datetime import date

import pandas as pd
import pytest

from arbitrage_analyser import db, ingest, metrics, services
from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.mfapi import SchemeHistory
from tests.conftest import (
    aaum_rows,
    benchmark_csv,
    growth_series,
    history_for,
    ter_rows,
    xlsx_bytes,
)

END = "2026-10-06"


def _fetcher(config: AppConfig, rates: dict[int, float]):  # type: ignore[no-untyped-def]
    def fetch(code: int) -> SchemeHistory:
        fund = config.fund_by_code(code)
        series = growth_series("2013-01-01", END, rates[code], noise=0.00002, seed=code)
        return history_for(code, fund.isin, series)

    return fetch


def _load_all(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    result = ingest.refresh_nav(conn, config, _fetcher(config, {100001: 0.072, 100002: 0.066}))
    assert result.ok, result.errors
    bench = growth_series("2019-01-01", "2026-09-30", 0.069, start_value=1000)
    assert ingest.import_benchmark(conn, config, benchmark_csv(bench)).ok
    ter = xlsx_bytes(
        ter_rows(
            [
                ("Alpha Arbitrage Fund", "04/10/2026", 0.33, 2.32),
                ("Beta Arbitrage Fund", "04/10/2026", 0.34, 1.63),
            ]
        )
    )
    assert ingest.import_ter(conn, config, ter).ok
    aaum = xlsx_bytes(aaum_rows([(100001, "Alpha", 7571234.0), (100002, "Beta", 3061752.0)]))
    assert ingest.import_aaum(conn, config, aaum).ok


def test_refresh_nav_stores_from_history_start(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    _load_all(conn, config)
    nav = db.read_nav(conn, 100001)
    assert nav.index[0] >= pd.Timestamp("2019-01-01")
    assert nav.index[-1] == pd.Timestamp(END)
    funds = db.read_funds(conn).set_index("amfi_code")
    assert funds.loc[100001, "launch_date"] == "2013-01-01"  # from full history
    assert db.read_flags(conn).empty


def test_refresh_nav_is_idempotent_and_keeps_reviewed_flags(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    def fetch(code: int) -> SchemeHistory:
        series = growth_series("2019-01-01", END, 0.07, noise=0.00002, seed=code).copy()
        series[pd.Timestamp("2025-06-02")] *= 1.03
        return history_for(code, config.fund_by_code(code).isin, series)

    first = ingest.refresh_nav(conn, config, fetch)
    # a one-day spike flags the jump and the reversal the next trading day, for both funds
    assert first.new_flags == 4
    assert set(db.read_flags(conn)["date"]) == {"2025-06-02", "2025-06-03"}
    flag_id = int(db.read_flags(conn)["id"].iloc[0])
    db.mark_flag_reviewed(conn, flag_id)
    second = ingest.refresh_nav(conn, config, fetch)
    assert second.new_flags == 0
    assert len(db.read_flags(conn)) == 3
    assert conn.execute("SELECT COUNT(*) FROM nav").fetchone()[0] == first.rows


def test_refresh_nav_skips_bad_mapping_and_source_errors(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    def fetch(code: int) -> SchemeHistory:
        if code == 100002:
            raise SourceError("timeout")
        series = growth_series("2019-01-01", END, 0.07)
        return history_for(
            code, config.fund_by_code(code).isin, series, name="Alpha - Regular Plan - Growth"
        )

    result = ingest.refresh_nav(conn, config, fetch)
    assert not result.ok
    assert any("not a Direct Growth plan" in e for e in result.errors)
    assert any("timeout" in e for e in result.errors)
    assert result.rows == 0
    assert db.read_refresh_log(conn).empty


def test_refresh_nav_flags_duplicate_source_dates(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    def fetch(code: int) -> SchemeHistory:
        series = growth_series("2019-01-01", END, 0.07)
        series = pd.concat([series, series.iloc[[-1]]]).sort_index()
        return history_for(code, config.fund_by_code(code).isin, series)

    result = ingest.refresh_nav(conn, config, fetch)
    assert result.ok
    flags = db.read_flags(conn)
    assert set(flags["check_name"]) == {"duplicate_date"}


def test_benchmark_reupload_and_mismatch(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    bench = growth_series("2018-06-01", "2026-09-30", 0.069, start_value=1000)
    first = ingest.import_benchmark(conn, config, benchmark_csv(bench[:"2026-08-31"]))
    assert first.ok and any("before 2019-01-01" in n for n in first.notes)
    second = ingest.import_benchmark(conn, config, benchmark_csv(bench["2026-08-01":]))
    assert second.ok
    assert second.rows == len(bench["2026-09-01":])
    bad = ingest.import_benchmark(conn, config, benchmark_csv(bench["2026-09-01":] * 1.01))
    assert not bad.ok and "different values" in bad.errors[0]
    stored = db.read_benchmark(conn, config.settings.benchmark_name)
    assert stored.iloc[-1] == pytest.approx(bench.iloc[-1], abs=1e-3)  # unchanged


def test_import_errors_are_reported(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    assert not ingest.import_benchmark(conn, config, b"Date,Close\n").ok
    old = growth_series("2018-01-01", "2018-02-01", 0.07, start_value=1000)
    assert (
        "no rows on or after" in ingest.import_benchmark(conn, config, benchmark_csv(old)).errors[0]
    )
    assert not ingest.import_ter(conn, config, b"").ok
    no_match = xlsx_bytes(ter_rows([("Gamma Fund", "04/10/2026", 0.1, 0.2)]))
    assert "no rows" in ingest.import_ter(conn, config, no_match).errors[0]
    assert not ingest.import_aaum(conn, config, xlsx_bytes(aaum_rows([(5, "x", 1.0)]))).ok


def test_ter_unit_flags(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    content = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "04/10/2026", 33.0, 232.0)]))
    result = ingest.import_ter(conn, config, content)
    assert result.ok
    assert set(db.read_flags(conn)["check_name"]) == {"base_ter_range", "total_ter_range"}


def test_fund_comparison(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    _load_all(conn, config)
    view = services.fund_comparison(conn, config, "Arbitrage")
    assert view.years == 3
    assert list(view.table.columns) == [
        "Fund",
        "Launch date",
        "NAV",
        "Fund manager",
        "P2P return % (3Y)",
        "Tracking diff % (3Y)",
        "AUM (Rs Cr)",
        "Base TER %",
        "Total TER %",
        "Exit load",
        "Flag",
    ]
    table = view.table.set_index("Fund")
    alpha = table.loc["Alpha Arbitrage Fund"]
    assert alpha["NAV"] == pytest.approx(db.read_nav(conn, 100001).iloc[-1])
    assert alpha["Fund manager"] == "A. Manager (since 03-Oct-2019)"
    assert table.loc["Beta Arbitrage Fund", "Fund manager"] == (
        "B. Manager, C. Manager (since Dec 2014)"
    )
    assert alpha["AUM (Rs Cr)"] == pytest.approx(75712.34)
    assert alpha["Base TER %"] == pytest.approx(0.33)
    assert alpha["Total TER %"] == pytest.approx(2.32)
    assert alpha["Exit load"] == "0.25% within 15 days"
    assert alpha["P2P return % (3Y)"] == pytest.approx(7.2, abs=0.05)
    # same end date for fund and benchmark: 7.2% - 6.9%
    assert alpha["Tracking diff % (3Y)"] == pytest.approx(0.3, abs=0.05)
    assert alpha["Flag"] == ""
    assert table.loc["Beta Arbitrage Fund", "Flag"] == "Exit load 0.5%"
    # as-of dates move to the info section
    assert view.nav_as_of["Alpha Arbitrage Fund"] == date(2026, 10, 6)
    assert view.td_as_of["Alpha Arbitrage Fund"] == date(2026, 9, 30)
    assert view.ter_as_of and view.aaum_quarter
    assert services.fund_comparison(conn, config, "Liquid").table.empty


@pytest.mark.parametrize("years", [1, 5])
def test_fund_comparison_period_drives_both_columns(
    conn: db.sqlite3.Connection, config: AppConfig, years: int
) -> None:
    _load_all(conn, config)
    table = services.fund_comparison(conn, config, "Arbitrage", years).table.set_index("Fund")
    alpha = table.loc["Alpha Arbitrage Fund"]
    nav = db.read_nav(conn, 100001)
    bench = db.read_benchmark(conn, config.settings.benchmark_name)
    end = bench.index[-1]
    fund_return = metrics.point_to_point(nav, years)
    fund_to_end = metrics.point_to_point(nav, years, end)
    bench_return = metrics.point_to_point(bench, years, end)
    assert fund_return is not None and fund_to_end is not None and bench_return is not None
    assert alpha[f"P2P return % ({years}Y)"] == pytest.approx(fund_return * 100)
    assert alpha[f"Tracking diff % ({years}Y)"] == pytest.approx((fund_to_end - bench_return) * 100)
    assert "P2P return % (3Y)" not in table.columns


def test_fund_comparison_rejects_other_periods(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    with pytest.raises(ValueError, match="years"):
        services.fund_comparison(conn, config, "Arbitrage", 2)


def test_as_of_text() -> None:
    assert services.as_of_text({}) is None
    same = {"A": date(2026, 10, 6), "B": date(2026, 10, 6)}
    assert services.as_of_text(same) == "06-Oct-2026"
    mixed = {"A": date(2026, 10, 6), "B": date(2026, 10, 3)}
    assert services.as_of_text(mixed) == "06-Oct-2026 (except B 03-Oct-2026)"
    assert services.as_of_text({"A": "2026-09-30"}) == "30-Sep-2026"


def test_fund_comparison_without_data(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    view = services.fund_comparison(conn, config, "Arbitrage")
    table = view.table
    assert len(table) == 2
    assert table["NAV"].isna().all()
    assert table["P2P return % (3Y)"].isna().all()
    assert table["AUM (Rs Cr)"].isna().all()
    assert view.nav_as_of == {} and view.td_as_of == {}


def test_rolling_view(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    _load_all(conn, config)
    chart, stats = services.rolling_view(conn, config, [100001, 100002], 3, 7.0)
    assert set(chart["series"]) == {"Alpha Arbitrage Fund", "Beta Arbitrage Fund", "Benchmark"}
    assert chart.loc[chart["series"] == "Benchmark", "is_benchmark"].all()
    assert chart["date"].min() >= pd.Timestamp("2022-01-01")
    stats = stats.set_index("Fund")
    assert stats.loc["Alpha Arbitrage Fund", "% above level"] == pytest.approx(100.0)
    assert stats.loc["Beta Arbitrage Fund", "% above level"] == pytest.approx(0.0)
    assert stats.loc["Alpha Arbitrage Fund", "Avg tracking diff %"] == pytest.approx(0.3, abs=0.05)
    assert pd.isna(stats.loc["Benchmark", "Avg tracking diff %"])


def test_rolling_view_without_benchmark(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    ingest.refresh_nav(conn, config, _fetcher(config, {100001: 0.07, 100002: 0.07}))
    chart, stats = services.rolling_view(conn, config, [100001], 1, 7.0)
    assert set(chart["series"]) == {"Alpha Arbitrage Fund"}
    assert pd.isna(stats.set_index("Fund").loc["Alpha Arbitrage Fund", "Avg tracking diff %"])


def test_data_health_statuses(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    empty = services.data_health(conn, config, date(2026, 10, 7))
    assert set(empty["Status"]) == {"No data"}
    _load_all(conn, config)
    health = services.data_health(conn, config, date(2026, 10, 7)).set_index("Dataset")
    assert health.loc["NAV", "Status"] == "OK"  # 1 day old
    assert health.loc["Benchmark", "Status"] == "OK"  # 7 days old
    assert health.loc["Average AUM", "Data up to"] == "2026-09-30"
    later = services.data_health(conn, config, date(2026, 10, 12)).set_index("Dataset")
    assert later.loc["NAV", "Status"] == "Due"  # 6 days old
    much_later = services.data_health(conn, config, date(2027, 2, 20)).set_index("Dataset")
    assert much_later.loc["Average AUM", "Status"] == "Overdue"


def test_data_health_reports_funds_without_nav(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    def fetch(code: int) -> SchemeHistory:
        if code == 100002:
            raise SourceError("down")
        series = growth_series("2019-01-01", END, 0.07)
        return history_for(code, config.fund_by_code(code).isin, series)

    ingest.refresh_nav(conn, config, fetch)
    health = services.data_health(conn, config, date(2026, 10, 7)).set_index("Dataset")
    assert health.loc["NAV", "Status"] == "Missing for 1 fund(s)"


def test_open_ter_flag_marks_fund(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    content = xlsx_bytes(ter_rows([("Alpha Arbitrage Fund", "04/10/2026", 0.33, 0.2)]))
    assert ingest.import_ter(conn, config, content).ok
    table = services.fund_comparison(conn, config, "Arbitrage").table.set_index("Fund")
    assert table.loc["Alpha Arbitrage Fund", "Flag"] == "Open data flag"
