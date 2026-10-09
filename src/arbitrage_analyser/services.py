"""Read models for the three screens. The UI calls these; a future API or chatbot can too."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from arbitrage_analyser import db, metrics
from arbitrage_analyser.config import AppConfig, Fund

BENCHMARK_LABEL = "Benchmark"
DEFAULT_YEARS = 3  # period selected on the Fund comparison screen when it opens


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100


def return_column(years: int) -> str:
    return f"P2P return % ({years}Y)"


def td_column(years: int) -> str:
    return f"Tracking diff % ({years}Y)"


@dataclass(frozen=True)
class ComparisonView:
    """Fund comparison table plus the as-of dates shown above it instead of as columns.

    Each `*_as_of` maps fund name to its date; funds without data are left out.
    """

    table: pd.DataFrame
    years: int
    nav_as_of: dict[str, date]
    td_as_of: dict[str, date]
    ter_as_of: dict[str, str]
    aaum_quarter: dict[str, str]


def fund_comparison(
    conn: sqlite3.Connection, config: AppConfig, category: str, years: int = DEFAULT_YEARS
) -> ComparisonView:
    """One row per fund in `category` (spec screen 1). Percent columns are in percent units.

    Returns and tracking difference are point-to-point over the last `years`, annualised.
    """
    if years not in metrics.WINDOWS_YEARS:
        raise ValueError(f"years must be one of {metrics.WINDOWS_YEARS}")
    settings = config.settings
    stored = db.read_funds(conn).set_index("amfi_code")
    ter = db.read_latest_ter(conn).set_index("amfi_code")
    aaum = db.read_latest_aaum(conn).set_index("amfi_code")
    bench = db.read_benchmark(conn, settings.benchmark_name)
    flagged_codes = set(db.read_flags(conn, "open")["subject"])

    rows = []
    nav_as_of: dict[str, date] = {}
    td_as_of: dict[str, date] = {}
    ter_as_of: dict[str, str] = {}
    aaum_quarter: dict[str, str] = {}
    for fund in config.funds_in(category):
        code = fund.amfi_code
        nav = db.read_nav(conn, code)
        td, td_end = _tracking_difference_p2p(nav, bench, years)
        if not nav.empty:
            nav_as_of[fund.name] = nav.index[-1].date()
        if td_end is not None:
            td_as_of[fund.name] = td_end
        reasons = []
        if fund.exit_load_pct > settings.exit_load_flag_above_pct:
            reasons.append(f"Exit load {fund.exit_load_pct:g}%")
        if str(code) in flagged_codes:
            reasons.append("Open data flag")
        row: dict[str, object] = {
            "Fund": fund.name,
            "Launch date": stored["launch_date"].get(code),
            "NAV": float(nav.iloc[-1]) if not nav.empty else None,
            "Fund manager": ", ".join(m.label() for m in fund.fund_managers),
            return_column(years): _pct(metrics.point_to_point(nav, years)),
            td_column(years): _pct(td),
        }
        if settings.aaum_enabled:
            row["AUM (Rs Cr)"] = aaum["aaum_crore"].get(code)
            if code in aaum.index:
                aaum_quarter[fund.name] = str(aaum.at[code, "quarter_end"])
        if settings.ter_enabled:
            row["BER %"] = ter["base_ter"].get(code)
            row["Total TER %"] = ter["total_ter"].get(code)
            if code in ter.index:
                ter_as_of[fund.name] = str(ter.at[code, "date"])
        row["Exit load"] = _exit_load_text(fund)
        row["Factsheet"] = fund.factsheet_url
        row["Monthly portfolio"] = fund.portfolio_url
        row["Flag"] = "; ".join(reasons)
        rows.append(row)
    return ComparisonView(pd.DataFrame(rows), years, nav_as_of, td_as_of, ter_as_of, aaum_quarter)


def column_help(config: AppConfig, view: ComparisonView) -> dict[str, str]:
    """Definition and as-of date per Fund comparison column (tooltips in the app; the agent
    passes them on so its answers use the same definitions)."""
    y = view.years
    period = "1 year" if y == 1 else f"{y} years"
    nav = as_of_text(view.nav_as_of) or "no NAV data yet"
    td = as_of_text(view.td_as_of) or "no benchmark data yet"
    ter = as_of_text(view.ter_as_of) or "no TER data yet"
    help_text = {
        "NAV": f"Direct Plan - Growth NAV, as of {nav}.",
        return_column(y): (
            f"Point-to-point return over the last {period}: NAV on the as-of date vs NAV "
            f"{period} earlier, annualised (CAGR). Direct plan, growth option, net of expenses. "
            f"NAV as of {nav}."
        ),
        td_column(y): (
            f"Fund P2P return minus {config.settings.benchmark_name} P2P return over the same "
            f"{period}, as of {td}."
        ),
        "BER %": f"Base Expense Ratio, Direct Plan, from AMFI. TER as of {ter}.",
        "Total TER %": (
            "Total expense ratio, Direct Plan: BER + brokerage + transaction cost + statutory "
            f"levies (incl. GST), from AMFI. TER as of {ter}."
        ),
        "Factsheet": "AMC page listing the monthly factsheets.",
        "Monthly portfolio": "AMC page listing the monthly portfolio disclosures.",
        "Flag": (
            f"Exit load above {config.settings.exit_load_flag_above_pct:g}%, or an open data "
            "flag (see Data health)."
        ),
    }
    if view.aaum_quarter:
        quarter = as_of_text(view.aaum_quarter)
        help_text["AUM (Rs Cr)"] = f"Average AUM for the quarter ending {quarter}."
    return help_text


def as_of_text(dates: Mapping[str, object]) -> str | None:
    """'05-Oct-2026', or the latest date plus the funds that are behind it."""
    if not dates:
        return None
    latest = max(dates.values(), key=str)
    behind = [f"{name} {_fmt(d)}" for name, d in dates.items() if d != latest]
    text = _fmt(latest)
    return f"{text} (except {', '.join(behind)})" if behind else text


def _fmt(value: object) -> str:
    if isinstance(value, date):
        return value.strftime("%d-%b-%Y")
    try:
        return date.fromisoformat(str(value)).strftime("%d-%b-%Y")
    except ValueError:
        return str(value)


def _tracking_difference_p2p(
    nav: pd.Series, bench: pd.Series, years: int
) -> tuple[float | None, date | None]:
    """Fund minus benchmark annualised return over `years`, both ending on the same date:
    the earlier of the two series' last dates (the benchmark is uploaded monthly)."""
    if nav.empty or bench.empty:
        return None, None
    end = min(nav.index[-1], bench.index[-1])
    fund_return = metrics.point_to_point(nav, years, end)
    bench_return = metrics.point_to_point(bench, years, end)
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
