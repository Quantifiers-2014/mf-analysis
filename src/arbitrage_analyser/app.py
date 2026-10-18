"""Streamlit UI: Fund comparison, Rolling returns, Data health, Ask the analyst (chat).

Run:  python -m arbitrage_analyser app
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

# Import the package from this source tree, not an installed copy. Streamlit Community Cloud
# installs the package once at build time and does not reinstall on a code push, so without this
# a new app.py would run against old modules.
_SRC = str(Path(__file__).resolve().parents[1])
if sys.path[0] != _SRC:
    sys.path.insert(0, _SRC)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from arbitrage_analyser import (  # noqa: E402
    charts,
    chat_ui,
    db,
    ingest,
    metrics,
    publish,
    services,
    store,
)
from arbitrage_analyser.config import AppConfig, ConfigError, load_config  # noqa: E402
from arbitrage_analyser.runtime import (  # noqa: E402
    config_path,
    data_dir,
    db_path,
    load_env_file,
)

MAX_FUNDS = len(charts.FUND_COLOURS)
PCT = st.column_config.NumberColumn(format="%.2f")


def _category_picker(config: AppConfig, key: str) -> str:
    categories = config.categories()
    return str(st.selectbox("Fund category", categories, key=key))


def _csv_button(frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "Export CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        disabled=frame.empty,
    )


RESULT_KEY = "last_load_result"
Notice = publish.Notice


def _finish(label: str, result: ingest.LoadResult | None, notices: list[Notice]) -> None:
    """Keep the outcome for the next run, then rerun so every table shows the new data."""
    st.session_state[RESULT_KEY] = (label, result, notices)
    st.rerun()


def _show_last_result() -> None:
    if RESULT_KEY not in st.session_state:
        return
    label, result, notices = st.session_state.pop(RESULT_KEY)
    st.markdown(f"**Last action: {label}**")
    if result is not None:
        if result.ok:
            st.success(f"Stored {result.rows} row(s). New flags: {result.new_flags}.")
        for error in result.errors:
            st.error(error)
        for note in result.notes:
            st.info(note)
    for level, text in notices:
        getattr(st, level)(text)


def _year_picker(key: str, label: str, default: int = services.DEFAULT_YEARS) -> int:
    options = list(metrics.WINDOWS_YEARS)
    return int(
        st.radio(
            label,
            options,
            index=options.index(default),
            format_func=lambda y: f"{y}Y",
            horizontal=True,
            key=key,
        )
    )


def fund_comparison_tab(config: AppConfig) -> None:
    left, right = st.columns([2, 1])
    with left:
        category = _category_picker(config, "cmp_category")
    with right:
        years = _year_picker("cmp_years", "Period")
    with db.connect(db_path()) as conn:
        view = services.fund_comparison(conn, config, category, years)
    table = view.table
    if table.empty:
        st.info("No funds in this category.")
        return
    if not view.nav_as_of:
        st.info("No NAV data yet: the data refresh job has not run (see Data health).")
    st.caption("Hover over a column name for its definition and as-of date.")
    help_text = services.column_help(config, view)
    column_config = {
        c: st.column_config.NumberColumn(format="%.2f", help=help_text.get(c))
        for c in table.columns
        if "%" in c
    } | {
        "NAV": st.column_config.NumberColumn(format="%.4f", help=help_text["NAV"]),
        "AUM (Rs Cr)": st.column_config.NumberColumn(
            format="%,.0f", help=help_text.get("AUM (Rs Cr)")
        ),
        "AAUM (Rs Cr)": st.column_config.NumberColumn(
            format="%,.0f", help=help_text.get("AAUM (Rs Cr)")
        ),
        "Turnover": st.column_config.TextColumn(width="medium", help=help_text.get("Turnover")),
        "Fund manager": st.column_config.TextColumn(width="medium"),
        "Factsheet": st.column_config.LinkColumn(display_text="Open", help=help_text["Factsheet"]),
        "Monthly portfolio": st.column_config.LinkColumn(
            display_text="Open", help=help_text["Monthly portfolio"]
        ),
        "Flag": st.column_config.TextColumn(width="medium", help=help_text["Flag"]),
    }
    st.dataframe(table, hide_index=True, width="stretch", column_config=column_config)
    _csv_button(table, f"fund_comparison_{category.lower()}_{years}y.csv", "cmp_csv")


START_KEY = "roll_start"


def _start_date_picker(first: date, last: date) -> date:
    """Start of the analysis. Defaults to the first NAV date; kept within the data's range."""
    current = st.session_state.get(START_KEY)
    if not isinstance(current, date) or not first <= current <= last:
        st.session_state[START_KEY] = first
    picked = st.date_input(
        "Start date",
        min_value=first,
        max_value=last,
        format="DD/MM/YYYY",
        key=START_KEY,
        help=f"Only rolling windows that begin on or after this date. Earliest NAV: "
        f"{first:%d-%b-%Y}.",
    )
    return picked if isinstance(picked, date) else first


def rolling_returns_tab(config: AppConfig) -> None:
    category = _category_picker(config, "roll_category")
    funds = config.funds_in(category)
    names = {f.name: f.amfi_code for f in funds}
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        chosen = st.multiselect(
            "Funds",
            list(names),
            default=list(names)[:3],
            max_selections=MAX_FUNDS,
            key="roll_funds",
        )
    with col3:
        years = _year_picker("roll_window", "Rolling Return Period", default=1)
    with col4:
        level = float(st.number_input("Above level (%)", value=7.0, step=0.25, key="roll_level"))
    if not chosen:
        st.info("Pick at least one fund.")
        return
    codes = [names[n] for n in chosen]

    with db.connect(db_path()) as conn:
        first = services.nav_start(conn, codes)
        if first is None:
            st.info("No NAV data yet: the data refresh job has not run (see Data health).")
            return
        with col2:  # left of the period: the analysis starts here
            start = _start_date_picker(first, date.today())
        chart_data, stats = services.rolling_view(conn, config, codes, years, level, start)
    if chart_data.empty:
        st.info(f"Not enough history for {years}Y windows starting on or after {start:%d-%b-%Y}.")
        return

    figure = charts.rolling_returns_figure(
        chart_data, chosen, years, config.settings.benchmark_name
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False}, key="roll_chart")
    st.caption(
        f"Each point is the annualised return of the {years}-year window ending that day, for "
        f"windows starting on or after {start:%d-%b-%Y}. Hover for all values on a date; click a "
        "legend item to hide it, double-click to show only it. Grey dashed line = "
        f"{config.settings.benchmark_name}."
    )
    st.dataframe(
        stats,
        hide_index=True,
        width="stretch",
        column_config={c: PCT for c in stats.columns if c.endswith("%") or c == "% above level"},
    )
    _csv_button(chart_data, f"rolling_{years}y.csv", "roll_csv")


LOAD_KEY = "store_load"


class _NotSaved(Exception):
    """Undo a change in the database that could not be saved to the data store."""


def _change_and_save(
    config: AppConfig,
    change: Callable[[sqlite3.Connection], ingest.LoadResult],
    file: str,
    message: str,
) -> tuple[ingest.LoadResult, list[Notice]]:
    """Apply `change` and save `file`. When the save fails the change is undone, so the app
    never shows data that is not in the data store."""
    damaged = st.session_state[LOAD_KEY].damaged if LOAD_KEY in st.session_state else set()
    result = ingest.LoadResult()
    notices: list[Notice] = []
    try:
        with db.connect(db_path()) as conn:
            result = change(conn)
            if not result.ok:
                raise _NotSaved
            notices = publish.save_store_files(
                conn, config.settings, data_dir(), [file], message, damaged
            )
            if any(level == "error" for level, _ in notices):
                raise _NotSaved
    except _NotSaved:
        if notices:
            notices.append(("info", "Nothing was changed."))
    return result, notices


def _benchmark_upload(config: AppConfig) -> None:
    st.markdown("**Benchmark CSV**")
    st.caption(
        "niftyindices.com > Reports > Historical Data > Equity > Strategy Indices > "
        "NIFTY 50 ARBITRAGE > CSV. Checks: bad values, duplicate dates, gaps, outliers, match "
        "with stored dates. The data is saved to the repository."
    )
    upload = st.file_uploader("Benchmark CSV", type=["csv"], key="up_bench",
                              label_visibility="collapsed")  # fmt: skip
    if upload is None or not st.button("Import benchmark CSV", key="up_bench_go"):
        return
    content = upload.getvalue()

    def change(conn: sqlite3.Connection) -> ingest.LoadResult:
        return ingest.import_benchmark(conn, config, content)

    result, notices = _change_and_save(
        config, change, store.BENCHMARK_FILE, "Benchmark: upload from the app"
    )
    _finish("Import benchmark CSV", result, notices)


def _flag_label(flag: tuple[str, str, str, str], detail: str) -> str:
    return f"{flag[2]} · {flag[0]} {flag[1]} · {detail}"


def _flags_section(flags: pd.DataFrame, config: AppConfig) -> None:
    st.markdown(f"**Open validation flags ({len(flags)})**")
    if flags.empty:
        st.caption("No open flags.")
        return
    st.dataframe(flags[["date", "dataset", "subject", "detail"]], hide_index=True, width="stretch")
    options = {
        _flag_label(key, str(row.detail)): key
        for row in flags.itertuples(index=False)
        for key in [(str(row.dataset), str(row.subject), str(row.date), str(row.check_name))]
    }
    picked = st.multiselect(
        "Mark as reviewed (checked, no action needed)", list(options), key="flag_pick"
    )
    if st.button("Save reviewed flags", disabled=not picked, key="flag_save"):
        keys = [options[p] for p in picked]

        def change(conn: sqlite3.Connection) -> ingest.LoadResult:
            return ingest.LoadResult(rows=db.mark_flags_reviewed(conn, keys))

        _, notices = _change_and_save(
            config, change, store.REVIEWED_FILE, f"Data health: {len(keys)} flag(s) reviewed"
        )
        _finish(f"Mark {len(keys)} flag(s) reviewed", None, notices)


def data_health_tab(config: AppConfig) -> None:
    with db.connect(db_path()) as conn:
        health = services.data_health(conn, config, date.today())
        flags = db.read_flags(conn, "open")
    _show_last_result()
    st.markdown("**Data freshness**")
    st.dataframe(health, hide_index=True, width="stretch")
    job = publish.actions_url(config.settings)
    st.caption(
        "NAV and TER are refreshed by a GitHub job every weekday night, factsheet values on the "
        "12th, 18th and 25th; the app loads them when it starts."
    )
    if job:
        st.link_button("Run the data refresh now (GitHub)", job)
    _flags_section(flags, config)
    _benchmark_upload(config)


def _load_store(config: AppConfig) -> None:
    """Load the data store into the database once per version of its files (the database on
    Streamlit Cloud starts empty after every restart). Problems show on every page view."""
    result = store.sync(db_path(), config, data_dir(), date.today())
    st.session_state[LOAD_KEY] = result
    for error in result.errors:
        st.warning(f"Data store: {error}")


def main() -> None:
    st.set_page_config(page_title="Arbitrage Fund Analyser", layout="wide")
    load_env_file()
    st.title("Arbitrage Fund Analyser")
    try:
        config = load_config(config_path())
    except ConfigError as exc:
        st.error(f"Config error: {exc}")
        st.stop()
    _load_store(config)
    chat_ui.chat_sidebar()
    compare, rolling, health, chat = st.tabs(
        ["Fund comparison", "Rolling returns", "Data health", "Ask the analyst"]
    )
    with compare:
        fund_comparison_tab(config)
    with rolling:
        rolling_returns_tab(config)
    with health:
        data_health_tab(config)
    with chat:
        chat_ui.chat_tab(config)


main()
