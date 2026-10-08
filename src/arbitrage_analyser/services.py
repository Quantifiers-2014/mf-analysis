"""Read models for the three screens. The UI calls these; a future API or chatbot can too."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date

import pandas as pd

from arbitrage_analyser import db, metrics
from arbitrage_analyser.config import AppConfig, Fund

BENCHMARK_LABEL = "Benchmark"
TD_YEARS = 3  # tracking difference shown on the Fund comparison screen


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100


def fund_comparison(conn: sqlite3.Connection, config: AppConfig, category: str) -> pd.DataFrame:
    """One row per fund in `category` (spec screen 1). Percent columns are in percent units."""
    settings = config.settings
    funds = config.funds_in(category)
    stored = db.read_funds(conn).set_index("amfi_code")
    ter = db.read_latest_ter(conn).set_index("amfi_code")
    aaum = db.read_latest_aaum(conn).set_index("amfi_code")
    bench = db.read_benchmark(conn, settings.benchmark_name)

    flagged_codes = set(db.read_flags(conn, "open")["subject"])

    rows = []
    for fund in funds:
        nav = db.read_nav(conn, fund.amfi_code)
        returns = {y: metrics.point_to_point(nav, y) for y in metrics.WINDOWS_YEARS}
        td, td_end = _tracking_difference_p2p(nav, bench)
        reasons = []
        if fund.exit_load_pct > settings.exit_load_flag_above_pct:
            reasons.append(f"Exit load {fund.exit_load_pct:g}%")
        if str(fund.amfi_code) in flagged_codes:
            reasons.append("Open data flag")
        # Columns ordered as scanned: returns, flag, size and cost, then reference details.
        row: dict[str, object] = {
            "Fund": fund.name,
            "1Y %": _pct(returns[1]),
            "3Y %": _pct(returns[3]),
            "5Y %": _pct(returns[5]),
            f"TD {TD_YEARS}Y %": _pct(td),
            "Flag": "; ".join(reasons),
        }
        if settings.aaum_enabled:
            row["AUM (Rs Cr)"] = aaum["aaum_crore"].get(fund.amfi_code)
        if settings.ter_enabled:
            row["Base TER %"] = ter["base_ter"].get(fund.amfi_code)
            row["Total TER %"] = ter["total_ter"].get(fund.amfi_code)
        row |= {
            "Exit load": _exit_load_text(fund),
            "Manager": ", ".join(fund.fund_managers),
            "Launch date": stored["launch_date"].get(fund.amfi_code),
            "NAV date": nav.index[-1].date() if not nav.empty else None,
            "TD as of": td_end,
        }
        if settings.ter_enabled:
            row["TER date"] = ter["date"].get(fund.amfi_code)
        if settings.aaum_enabled:
            row["AUM quarter"] = aaum["quarter_end"].get(fund.amfi_code)
        rows.append(row)
    return pd.DataFrame(rows)


def _tracking_difference_p2p(nav: pd.Series, bench: pd.Series) -> tuple[float | None, date | None]:
    """Fund minus benchmark annualised return over TD_YEARS, both ending on the same date:
    the earlier of the two series' last dates (the benchmark is uploaded monthly)."""
    if nav.empty or bench.empty:
        return None, None
    end = min(nav.index[-1], bench.index[-1])
    fund_return = metrics.point_to_point(nav, TD_YEARS, end)
    bench_return = metrics.point_to_point(bench, TD_YEARS, end)
    if fund_return is None or bench_return is None:
        return None, None
    return fund_return - bench_return, end.date()


def _exit_load_text(fund: Fund) -> str:
    if fund.exit_load_pct == 0:
        return "Nil"
    return f"{fund.exit_load_pct:g}% within {fund.exit_load_days} days"


def rolling_view(
    conn: sqlite3.Connection,
    config: AppConfig,
    amfi_codes: Sequence[int],
    years: int,
    level_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spec screen 2. Returns (chart data, stats table).

    Chart data is long format: date, series, return_pct, is_benchmark.
    """
    settings = config.settings
    bench = metrics.rolling_returns(db.read_benchmark(conn, settings.benchmark_name), years)
    frames = []
    stats_rows = []
    for code in amfi_codes:
        fund = config.fund_by_code(code)
        returns = metrics.rolling_returns(db.read_nav(conn, code), years)
        frames.append(_long(returns, fund.name, is_benchmark=False))
        summary = metrics.rolling_stats(returns, level_pct / 100)
        td = metrics.tracking_difference(returns, bench, settings.max_gap_days)
        stats_rows.append(_stats_row(fund.name, summary, td.mean() if not td.empty else None))
    if not bench.empty:
        frames.append(_long(bench, BENCHMARK_LABEL, is_benchmark=True))
        stats_rows.append(
            _stats_row(BENCHMARK_LABEL, metrics.rolling_stats(bench, level_pct / 100), None)
        )
    chart = (
        pd.concat(frames, ignore_index=True) if frames else _long(pd.Series(dtype=float), "", False)
    )
    return chart, pd.DataFrame(stats_rows)


def _long(returns: pd.Series, name: str, is_benchmark: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": returns.index,
            "series": name,
            "return_pct": returns.to_numpy() * 100,
            "is_benchmark": is_benchmark,
        }
    )


def _stats_row(
    name: str, summary: metrics.RollingStats | None, avg_td: float | None
) -> dict[str, object]:
    if summary is None:
        return {"Fund": name, "Windows": 0}
    return {
        "Fund": name,
        "Windows": summary.count,
        "Average %": summary.average * 100,
        "Median %": summary.median * 100,
        "Min %": summary.minimum * 100,
        "Max %": summary.maximum * 100,
        "% above level": summary.pct_above_level * 100,
        "Avg tracking diff %": float("nan") if avg_td is None else avg_td * 100,
    }


def data_health(conn: sqlite3.Connection, config: AppConfig, today: date) -> pd.DataFrame:
    """Spec screen 3: freshness of each dataset, judged by the latest data date it holds.

    For NAV the least up-to-date configured fund decides, so one stale fund is not hidden.
    """
    per_fund = dict(conn.execute("SELECT amfi_code, MAX(date) FROM nav GROUP BY amfi_code"))
    nav_dates = [per_fund[f.amfi_code] for f in config.funds if f.amfi_code in per_fund]
    funds_without_nav = len(config.funds) - len(nav_dates)
    latest = {
        "nav": min(nav_dates) if nav_dates else None,
        "ter": _scalar(conn, "SELECT MAX(date) FROM ter"),
        "aaum": _scalar(conn, "SELECT MAX(quarter_end) FROM aaum"),
        "benchmark": _scalar(
            conn,
            "SELECT MAX(date) FROM benchmark WHERE index_name=?",
            (config.settings.benchmark_name,),
        ),
    }
    runs = db.read_refresh_log(conn).set_index("dataset")["refreshed_at"]
    meta = {
        "nav": ("NAV", "mfapi.in (AMFI data)", "Daily, automatic"),
        "ter": ("TER breakdown", "AMFI TER file", "Monthly upload"),
        "aaum": ("Average AUM", "AMFI AAUM file", "Quarterly upload"),
        "benchmark": ("Benchmark", "niftyindices.com CSV", "Monthly upload"),
    }
    rows = []
    for dataset in config.settings.active_datasets():
        label, source, frequency = meta[dataset]
        latest_date = latest[dataset]
        status = freshness_status(latest_date, today, config, dataset)
        if dataset == "nav" and nav_dates and funds_without_nav:
            status = f"Missing for {funds_without_nav} fund(s)"
        rows.append(
            {
                "Dataset": label,
                "Source": source,
                "Frequency": frequency,
                "Data up to": latest_date,
                "Last loaded": runs.get(dataset),
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def freshness_status(latest: str | None, today: date, config: AppConfig, dataset: str) -> str:
    if latest is None:
        return "No data"
    age = (today - date.fromisoformat(latest)).days
    rule = config.settings.freshness[dataset]
    if age > rule.overdue_after_days:
        return "Overdue"
    if age > rule.due_after_days:
        return "Due"
    return "OK"


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> str | None:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]
