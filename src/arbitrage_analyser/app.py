"""Streamlit UI: Fund comparison, Rolling returns, Data health.

Run:  streamlit run src/arbitrage_analyser/app.py
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from arbitrage_analyser import db, ingest, services
from arbitrage_analyser.config import AppConfig, ConfigError, load_config
from arbitrage_analyser.runtime import config_path, db_path

MAX_FUNDS = 8
PCT = st.column_config.NumberColumn(format="%.2f")
# Tableau 10, one colour per selectable fund (MAX_FUNDS); grey is kept for the benchmark.
FUND_COLOURS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]
BENCHMARK_COLOUR = "#555555"


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


def fund_comparison_tab(config: AppConfig) -> None:
    category = _category_picker(config, "cmp_category")
    with db.connect(db_path()) as conn:
        table = services.fund_comparison(conn, config, category)
    if table.empty:
        st.info("No funds in this category.")
        return
    latest = table["NAV date"].dropna()
    st.caption(
        f"NAV as of {max(latest)}"
        if not latest.empty
        else "No NAV data yet. Run a refresh from the Data health tab."
    )
    st.caption(
        f"Flag = exit load above {config.settings.exit_load_flag_above_pct:g}% or an open "
        f"data flag. TD = tracking difference vs {config.settings.benchmark_name}, "
        f"{services.TD_YEARS}Y, both ending on the 'TD as of' date."
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={c: PCT for c in table.columns if c.endswith("%")}
        | {
            "AUM (Rs Cr)": st.column_config.NumberColumn(format="%,.0f"),
            "Flag": st.column_config.TextColumn(width="medium"),
        },
    )
    _csv_button(table, f"fund_comparison_{category.lower()}.csv", "cmp_csv")


def rolling_returns_tab(config: AppConfig) -> None:
    category = _category_picker(config, "roll_category")
    funds = config.funds_in(category)
    names = {f.name: f.amfi_code for f in funds}
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        chosen = st.multiselect(
            "Funds",
            list(names),
            default=list(names)[:3],
            max_selections=MAX_FUNDS,
            key="roll_funds",
        )
    with col2:
        years = int(
            st.radio(
                "Window",
                [1, 3, 5],
                format_func=lambda y: f"{y}Y",
                horizontal=True,
                key="roll_window",
            )
        )
    with col3:
        level = float(st.number_input("Above level (%)", value=7.0, step=0.25, key="roll_level"))
    if not chosen:
        st.info("Pick at least one fund.")
        return

    with db.connect(db_path()) as conn:
        chart_data, stats = services.rolling_view(
            conn, config, [names[n] for n in chosen], years, level
        )
    if chart_data.empty:
        st.info(f"Not enough history for {years}Y windows yet.")
        return

    # Funds get distinct colours in selection order; the benchmark is always dark grey.
    fund_series = [n for n in chosen if n in set(chart_data["series"])]
    has_benchmark = bool(chart_data["is_benchmark"].any())
    domain = fund_series + ([services.BENCHMARK_LABEL] if has_benchmark else [])
    colours = FUND_COLOURS[: len(fund_series)] + ([BENCHMARK_COLOUR] if has_benchmark else [])

    chart = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(
                "return_pct:Q", title=f"Rolling {years}Y return (%)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=colours)),
            strokeDash=alt.StrokeDash(
                "is_benchmark:N",
                legend=None,
                scale=alt.Scale(domain=[False, True], range=[[1, 0], [5, 4]]),
            ),
            tooltip=[
                alt.Tooltip("date:T", format="%d-%b-%Y"),
                "series:N",
                alt.Tooltip("return_pct:Q", title="Return %", format=".2f"),
            ],
        )
        .properties(height=380)
        .interactive(bind_y=False)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Grey dashed line = {config.settings.benchmark_name}. Each point is the annualised "
        f"return of the {years}-year window ending that day."
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
        "NIFTY 50 ARBITRAGE. Checks: bad values, duplicate/weekend dates, gaps, outliers, "
        "overlap match with stored values.",
        ["csv"],
        ingest.import_benchmark,
        "up_bench",
    )
    if config.settings.ter_enabled:
        _upload_section(
            config,
            "TER file",
            "AMFI > Research & Information > TER of MF Schemes, export the month.",
            ["xlsx", "xls", "csv", "html"],
            ingest.import_ter,
            "up_ter",
        )
    if config.settings.aaum_enabled:
        _upload_section(
            config,
            "AAUM file",
            "AMFI > Average AUM: Schemewise, Categorywise, one fund house, one quarter, "
            "Excel export.",
            ["xlsx", "xls", "csv", "html"],
            ingest.import_aaum,
            "up_aaum",
        )


def main() -> None:
    st.set_page_config(page_title="Arbitrage Fund Analyser", layout="wide")
    st.title("Arbitrage Fund Analyser")
    try:
        config = load_config(config_path())
    except ConfigError as exc:
        st.error(f"Config error: {exc}")
        st.stop()
    compare, rolling, health = st.tabs(["Fund comparison", "Rolling returns", "Data health"])
    with compare:
        fund_comparison_tab(config)
    with rolling:
        rolling_returns_tab(config)
    with health:
        data_health_tab(config)


main()
