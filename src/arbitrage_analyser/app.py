"""Streamlit UI: Fund comparison, Rolling returns, Data health, Ask the analyst (chat).

Run:  python -m arbitrage_analyser app
"""

from __future__ import annotations

import sys
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
    factsheets,
    ingest,
    metrics,
    services,
)
from arbitrage_analyser.config import AppConfig, ConfigError, load_config  # noqa: E402
from arbitrage_analyser.runtime import (  # noqa: E402
    config_path,
    db_path,
    factsheet_csv_path,
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


def _finish_load(label: str, result: ingest.LoadResult) -> None:
    """Keep the outcome for the next run, then rerun so every table shows the new data."""
    st.session_state[RESULT_KEY] = (label, result)
    st.rerun()


def _show_last_result() -> None:
    if RESULT_KEY not in st.session_state:
        return
    label, result = st.session_state.pop(RESULT_KEY)
    st.markdown(f"**Last action: {label}**")
    if result.ok:
        st.success(f"Stored {result.rows} row(s). New flags: {result.new_flags}.")
    for error in result.errors:
        st.error(error)
    for note in result.notes:
        st.info(note)


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
        st.info("No NAV data yet. Run a refresh from the Data health tab.")
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
    with col2:
        years = _year_picker("roll_window", "Window", default=1)
    with col3:
        level = float(st.number_input("Above level (%)", value=7.0, step=0.25, key="roll_level"))
    if not chosen:
        st.info("Pick at least one fund.")
        return
    codes = [names[n] for n in chosen]

    with db.connect(db_path()) as conn:
        first = services.nav_start(conn, codes)
        if first is None:
            st.info("No NAV data yet. Run a refresh from the Data health tab.")
            return
        with col4:
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


def _upload_section(
    config: AppConfig,
    label: str,
    help_text: str,
    types: list[str],
    importer: ingest.FileImporter,
    key: str,
) -> None:
    st.markdown(f"**{label}**")
    st.caption(help_text)
    upload = st.file_uploader(label, type=types, key=key, label_visibility="collapsed")
    if upload is not None and st.button(f"Import {label.lower()}", key=f"{key}_go"):
        with db.connect(db_path()) as conn:
            result = importer(conn, config, upload.getvalue())
        _finish_load(f"Import {label.lower()}", result)


def _fetch_ter_section(config: AppConfig) -> None:
    st.markdown("**TER from AMFI**")
    st.caption(
        "Downloads the Direct Plan BER and Total TER for one month from the AMFI TER page. "
        "This uses the page's own data request, not a published API, so it can stop working "
        "if AMFI changes the page; upload the Excel below if it does."
    )
    left, right = st.columns([1, 3])
    month = left.text_input(
        "Month (MM-YYYY)", value=ingest.previous_month(date.today()), key="ter_month"
    )
    right.write("")  # aligns the button with the input box
    if right.button("Fetch TER from AMFI", key="ter_fetch"):
        with st.spinner(f"Downloading TER for {month}..."), db.connect(db_path()) as conn:
            result = ingest.fetch_ter(conn, config, month.strip())
        _finish_load(f"Fetch TER for {month.strip()}", result)


def _fetch_factsheets_section(config: AppConfig) -> None:
    st.markdown("**Factsheet values (AUM, AAUM, turnover, Direct BER)**")
    st.caption(
        "Downloads each fund's latest factsheet PDF from the AMC site and reads the values from "
        "the fund's page. A monthly GitHub job does the same and saves them in the repository. "
        "Values fetched with this button are kept only until the app restarts."
    )
    if st.button("Fetch factsheets now", key="factsheet_fetch"):
        with st.spinner("Downloading factsheets (1-2 minutes)..."), db.connect(db_path()) as conn:
            result = factsheets.fetch_factsheets(conn, config, date.today())
        _finish_load("Fetch factsheets", result)


FACTSHEET_FILE_KEY = "factsheet_file_loaded"


def _sync_factsheet_file(config: AppConfig) -> None:
    """Copy the repository's factsheet values into the database, once per session and file
    version (the database on Streamlit Cloud starts empty after a restart)."""
    path = factsheet_csv_path()
    if not config.settings.factsheet_enabled or not path.exists():
        return
    version = path.stat().st_mtime
    if st.session_state.get(FACTSHEET_FILE_KEY) == version:
        return
    with db.connect(db_path()) as conn:
        result = factsheets.import_csv(conn, config, path, date.today())
    for error in result.errors:
        st.warning(f"Factsheet values file: {error}")
    st.session_state[FACTSHEET_FILE_KEY] = version


def _check_factsheets(config: AppConfig) -> None:
    """Flags for the latest factsheet values against the TER loaded since (no-op when the flags
    already exist)."""
    if config.settings.factsheet_enabled:
        with db.connect(db_path()) as conn:
            factsheets.check_latest(conn, config)


def data_health_tab(config: AppConfig) -> None:
    with db.connect(db_path()) as conn:
        health = services.data_health(conn, config, date.today())
        flags = db.read_flags(conn, "open")
    _show_last_result()
    st.markdown("**Data freshness**")
    st.dataframe(health, hide_index=True, width="stretch")

    if st.button("Refresh NAV now"):
        with st.spinner("Downloading NAV history..."), db.connect(db_path()) as conn:
            result = ingest.refresh_nav(conn, config)
        _finish_load("Refresh NAV", result)

    st.markdown(f"**Open validation flags ({len(flags)})**")
    if flags.empty:
        st.caption("No open flags.")
    for flag in flags.head(100).itertuples(index=False):
        left, right = st.columns([5, 1])
        left.write(f"{flag.date} · {flag.dataset} · {flag.subject} · {flag.detail}")
        if right.button("Mark reviewed", key=f"flag_{flag.id}"):
            with db.connect(db_path()) as conn:
                db.mark_flag_reviewed(conn, int(flag.id))
            st.rerun()
    if len(flags) > 100:
        st.caption(f"Showing the latest 100 of {len(flags)} open flags.")

    _upload_section(
        config,
        "Benchmark CSV",
        "niftyindices.com > Reports > Historical Data > Equity > Strategy Indices > "
        "NIFTY 50 ARBITRAGE. Checks: bad values, duplicate dates, gaps, outliers, "
        "overlap match with stored values.",
        ["csv"],
        ingest.import_benchmark,
        "up_bench",
    )
    if config.settings.ter_enabled:
        _fetch_ter_section(config)
        _upload_section(
            config,
            "TER file",
            "If the fetch fails: amfiindia.com/ter-of-mf-schemes > month, Category 'Hybrid "
            "Scheme', Sub Category 'Arbitrage Fund' > GO > Download Excel.",
            ["xlsx", "xls", "csv", "html"],
            ingest.import_ter,
            "up_ter",
        )
    if config.settings.factsheet_enabled:
        _fetch_factsheets_section(config)


def main() -> None:
    st.set_page_config(page_title="Arbitrage Fund Analyser", layout="wide")
    load_env_file()
    st.title("Arbitrage Fund Analyser")
    try:
        config = load_config(config_path())
    except ConfigError as exc:
        st.error(f"Config error: {exc}")
        st.stop()
    _sync_factsheet_file(config)
    _check_factsheets(config)
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
