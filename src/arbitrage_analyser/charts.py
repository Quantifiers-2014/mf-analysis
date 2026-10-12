"""Plotly figures for the Streamlit screens.

Kept out of app.py so the figures can be built and tested without running the app.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

# Tableau 10, one colour per selectable fund (up to 8); grey is kept for the benchmark.
FUND_COLOURS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
)
BENCHMARK_COLOUR = "#555555"


def rolling_returns_figure(
    chart_data: pd.DataFrame, fund_order: Sequence[str], years: int, benchmark_label: str
) -> go.Figure:
    """Rolling returns: one line per fund in `fund_order` (colours follow that order, so a fund
    keeps its colour while others are added or removed), the benchmark as a dashed grey line.

    `chart_data` is the long frame from services.rolling_view: date, series, return_pct,
    is_benchmark. Hovering a date shows every series' value for that day.
    """
    fig = go.Figure()
    for i, name in enumerate(fund_order):
        rows = chart_data[(chart_data["series"] == name) & ~chart_data["is_benchmark"]]
        if rows.empty:
            continue
        fig.add_scatter(
            x=rows["date"],
            y=rows["return_pct"],
            name=name,
            mode="lines",
            line={"color": FUND_COLOURS[i % len(FUND_COLOURS)], "width": 2},
            hovertemplate="%{y:.2f}%",
        )
    bench = chart_data[chart_data["is_benchmark"]]
    if not bench.empty:
        fig.add_scatter(
            x=bench["date"],
            y=bench["return_pct"],
            name=f"{benchmark_label} (benchmark)",
            mode="lines",
            line={"color": BENCHMARK_COLOUR, "width": 2, "dash": "dash"},
            hovertemplate="%{y:.2f}%",
        )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font_size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font_size": 12},
        yaxis={
            "title": f"Rolling {years}Y return (annualised)",
            "ticksuffix": "%",
            "gridcolor": "#eeeeee",
        },
        xaxis={
            "hoverformat": "%d-%b-%Y",
            "showspikes": True,
            "spikemode": "across",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "#999999",
            "rangeselector": {
                "x": 0,
                "y": 1.14,
                "buttons": [
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                    {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
            },
            "rangeslider": {"visible": True, "thickness": 0.07},
        },
    )
    return fig
