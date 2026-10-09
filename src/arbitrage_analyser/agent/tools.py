"""Read-only tools the agent can call. Every number the agent states must come from one of these.

Each tool returns a JSON-ready dict that always carries `source` and `as_of`, so the agent can
cite where a figure came from and how fresh it is. Tools never write to the database.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from arbitrage_analyser import db, metrics, services
from arbitrage_analyser.config import AppConfig

NAV_SOURCE = "mfapi.in (republishes AMFI NAV data), stored in the analyser database"
CONFIG_SOURCE = "config/funds.toml (fund list maintained by the team)"
MAX_CODES = 20
MAX_NAV_POINTS = 400


class ToolError(ValueError):
    """A bad tool request. The message is sent back to the model so it can correct itself."""


# Tool schemas in the format the Claude API expects.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_funds",
        "description": (
            "List the funds in the database with their AMFI code, fund house, category, ISIN, "
            "exit load, fund managers, launch date and latest NAV. Call this to find a fund's "
            "AMFI code or to answer questions about which funds are covered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter, e.g. 'Arbitrage'.",
                }
            },
        },
    },
    {
        "name": "compare_funds",
        "description": (
            "The fund comparison table for a category over one period (1, 3 or 5 years): "
            "annualised point-to-point return, tracking difference against the benchmark, "
            "latest NAV, launch date, fund managers (with start dates), exit load and data "
            "flags. Returns are in percent, ending on each fund's latest NAV date. Call once "
            "per period to compare several periods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Fund category, e.g. 'Arbitrage'."},
                "years": {
                    "type": "integer",
                    "enum": [1, 3, 5],
                    "description": "Period in years. Default 3.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "get_returns",
        "description": (
            "Annualised (CAGR) point-to-point return for chosen funds over a number of years, "
            "ending on a chosen date (default: each fund's latest NAV date). Use for periods or "
            "end dates that compare_funds does not cover."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amfi_codes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "AMFI codes of the funds.",
                },
                "years": {"type": "integer", "minimum": 1, "maximum": 20},
                "end_date": {
                    "type": "string",
                    "description": "Optional end date, YYYY-MM-DD.",
                },
            },
            "required": ["amfi_codes", "years"],
        },
    },
    {
        "name": "get_rolling_return_stats",
        "description": (
            "Rolling-return statistics: for every day, the annualised return over the previous "
            "N years, summarised as count of windows, average, median, minimum, maximum, and "
            "the share of windows above a return level. Shows consistency, not just one period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amfi_codes": {"type": "array", "items": {"type": "integer"}},
                "years": {"type": "integer", "enum": [1, 3, 5]},
                "level_pct": {
                    "type": "number",
                    "description": "Return level in percent for the 'above level' share. "
                    "Default 7.",
                },
            },
            "required": ["amfi_codes", "years"],
        },
    },
    {
        "name": "get_nav_history",
        "description": (
            "NAV values for one fund between two dates. Long ranges are thinned to month-end "
            "values. Use to state a NAV on a date or describe how it moved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amfi_code": {"type": "integer"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["amfi_code"],
        },
    },
    {
        "name": "get_data_status",
        "description": (
            "Freshness of each dataset (latest data date, status), which datasets are switched "
            "off or empty, and open data-quality flags. Check this before relying on data, and "
            "whenever a question needs TER, AUM or benchmark data."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _clean(value: Any) -> Any:
    """Make pandas/numpy values JSON-safe: NaN -> None, dates -> ISO strings, round floats."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()[:10]
    if isinstance(value, float):
        return None if math.isnan(value) else round(float(value), 4)
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    return value


def _out(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = _clean(data)
    return result


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean(row) for row in frame.to_dict(orient="records")]


def _parse_date(text: str | None, field: str) -> pd.Timestamp | None:
    if not text:
        return None
    try:
        return pd.Timestamp(date.fromisoformat(text))
    except ValueError as exc:
        raise ToolError(f"{field} must be YYYY-MM-DD, got {text!r}") from exc


def _codes(config: AppConfig, raw: Any) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ToolError("amfi_codes must be a non-empty list of integers")
    if len(raw) > MAX_CODES:
        raise ToolError(f"At most {MAX_CODES} funds per call")
    known = {f.amfi_code for f in config.funds}
    codes = [int(c) for c in raw]
    unknown = [c for c in codes if c not in known]
    if unknown:
        raise ToolError(f"Unknown AMFI code(s) {unknown}. Call list_funds to see valid codes.")
    return codes


def list_funds(conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category")
    funds = config.funds_in(category) if category else list(config.funds)
    stored = db.read_funds(conn).set_index("amfi_code")
    rows = []
    for fund in funds:
        nav = db.read_nav(conn, fund.amfi_code)
        rows.append(
            {
                "name": fund.name,
                "amfi_code": fund.amfi_code,
                "fund_house": fund.amc,
                "category": fund.category,
                "isin": fund.isin,
                "plan": "Direct Plan - Growth",
                "exit_load_pct": fund.exit_load_pct,
                "exit_load_days": fund.exit_load_days,
                "fund_managers": [m.label() for m in fund.fund_managers],
                "launch_date": stored["launch_date"].get(fund.amfi_code),
                "latest_nav": float(nav.iloc[-1]) if not nav.empty else None,
                "latest_nav_date": nav.index[-1] if not nav.empty else None,
            }
        )
    return _out(
        {
            "funds": rows,
            "categories": config.categories(),
            "source": f"{CONFIG_SOURCE}; latest NAV from {NAV_SOURCE}",
            "as_of": max(
                (r["latest_nav_date"] for r in rows if r["latest_nav_date"]), default=None
            ),
        }
    )


def compare_funds(
    conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]
) -> dict[str, Any]:
    category = str(args.get("category", ""))
    if category not in config.categories():
        raise ToolError(f"Unknown category {category!r}. Known: {config.categories()}")
    years = int(args.get("years", services.DEFAULT_YEARS))
    if years not in metrics.WINDOWS_YEARS:
        raise ToolError(f"years must be one of {list(metrics.WINDOWS_YEARS)}")
    view = services.fund_comparison(conn, config, category, years)
    notes = [
        f"Returns and tracking difference are annualised point-to-point over {years} year(s), "
        "in percent."
    ]
    if not view.td_as_of:
        notes.append("Tracking difference is empty because no benchmark data is loaded.")
    return _out(
        {
            "rows": _records(view.table),
            "years": years,
            "nav_as_of_by_fund": view.nav_as_of,
            "tracking_difference_as_of_by_fund": view.td_as_of,
            "notes": notes,
            "source": NAV_SOURCE,
            "as_of": max(view.nav_as_of.values()) if view.nav_as_of else None,
        }
    )


def get_returns(
    conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]
) -> dict[str, Any]:
    codes = _codes(config, args.get("amfi_codes"))
    years = int(args.get("years", 0))
    if not 1 <= years <= 20:
        raise ToolError("years must be between 1 and 20")
    end = _parse_date(args.get("end_date"), "end_date")
    rows = []
    for code in codes:
        nav = db.read_nav(conn, code)
        end_used = end if end is not None else (nav.index[-1] if not nav.empty else None)
        value = metrics.point_to_point(nav, years, end_used) if end_used is not None else None
        row: dict[str, Any] = {
            "name": config.fund_by_code(code).name,
            "amfi_code": code,
            "years": years,
            "end_date": end_used,
            "annualised_return_pct": None if value is None else value * 100,
        }
        if value is None:
            row["reason"] = "Not enough NAV history for this period, or no NAV stored."
        if end is not None and not nav.empty and end > nav.index[-1]:
            row["warning"] = f"end_date is after the latest NAV ({nav.index[-1].date()})."
        rows.append(row)
    as_of = end if end is not None else _latest_nav_date(conn, codes)
    return _out({"rows": rows, "source": NAV_SOURCE, "as_of": as_of})


def get_rolling_return_stats(
    conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]
) -> dict[str, Any]:
    codes = _codes(config, args.get("amfi_codes"))
    years = int(args.get("years", 0))
    if years not in (1, 3, 5):
        raise ToolError("years must be 1, 3 or 5")
    level = float(args.get("level_pct", 7.0))
    _, stats = services.rolling_view(conn, config, codes, years, level)
    return _out(
        {
            "rows": _records(stats),
            "window_years": years,
            "level_pct": level,
            "notes": ["Percent values. '% above level' is the share of windows above level_pct."],
            "source": NAV_SOURCE,
            "as_of": _latest_nav_date(conn, codes),
        }
    )


def get_nav_history(
    conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]
) -> dict[str, Any]:
    code = _codes(config, [args.get("amfi_code")])[0]
    nav = db.read_nav(conn, code)
    start = _parse_date(args.get("start_date"), "start_date")
    end = _parse_date(args.get("end_date"), "end_date")
    if start is not None:
        nav = nav[nav.index >= start]
    if end is not None:
        nav = nav[nav.index <= end]
    thinned = len(nav) > MAX_NAV_POINTS
    if thinned:
        nav = nav.groupby(nav.index.to_period("M")).tail(1)
    points = [{"date": d, "nav": float(v)} for d, v in nav.items()]
    return _out(
        {
            "name": config.fund_by_code(code).name,
            "amfi_code": code,
            "points": points,
            "frequency": "month-end" if thinned else "daily",
            "source": NAV_SOURCE,
            "as_of": nav.index[-1] if not nav.empty else None,
        }
    )


def get_data_status(
    conn: sqlite3.Connection, config: AppConfig, args: dict[str, Any]
) -> dict[str, Any]:
    health = services.data_health(conn, config, date.today())
    flags = db.read_flags(conn, "open")
    counts = {name: _count(conn, name) for name in ("nav", "benchmark", "ter", "aaum")}
    switched_off = [d for d in ("ter", "aaum") if d not in config.settings.active_datasets()]
    return _out(
        {
            "datasets": _records(health),
            "row_counts": counts,
            "switched_off": switched_off,
            "open_flags": len(flags),
            "open_flag_examples": _records(flags.head(10)[["date", "subject", "detail"]])
            if not flags.empty
            else [],
            "today": date.today(),
            "source": "analyser database (data health checks)",
            "as_of": date.today(),
        }
    )


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _latest_nav_date(conn: sqlite3.Connection, codes: list[int]) -> str | None:
    marks = ",".join("?" * len(codes))
    row = conn.execute(f"SELECT MAX(date) FROM nav WHERE amfi_code IN ({marks})", codes).fetchone()
    return None if row is None else row[0]


Handler = Callable[[sqlite3.Connection, AppConfig, dict[str, Any]], dict[str, Any]]

HANDLERS: dict[str, Handler] = {
    "list_funds": list_funds,
    "compare_funds": compare_funds,
    "get_returns": get_returns,
    "get_rolling_return_stats": get_rolling_return_stats,
    "get_nav_history": get_nav_history,
    "get_data_status": get_data_status,
}


def run_tool(
    conn: sqlite3.Connection, config: AppConfig, name: str, args: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Run a tool by name. Returns (result, is_error). Errors become results the model can read."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool {name!r}"}, True
    try:
        return handler(conn, config, args or {}), False
    except ToolError as exc:
        return {"error": str(exc)}, True
