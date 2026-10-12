import pandas as pd

from arbitrage_analyser import charts


def _data() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=5)
    rows = [
        {"date": d, "series": name, "return_pct": value, "is_benchmark": bench}
        for name, value, bench in [
            ("Beta", 6.5, False),
            ("Alpha", 7.0, False),
            ("Benchmark", 6.8, True),
        ]
        for d in dates
    ]
    return pd.DataFrame(rows)


def test_rolling_figure_lines_colours_and_hover() -> None:
    fig = charts.rolling_returns_figure(
        _data(), ["Alpha", "Beta", "Gamma"], 3, "NIFTY 50 Arbitrage"
    )
    names = [t.name for t in fig.data]
    # selection order, missing fund skipped, benchmark last
    assert names == ["Alpha", "Beta", "NIFTY 50 Arbitrage (benchmark)"]
    assert fig.data[0].line.color == charts.FUND_COLOURS[0]
    assert fig.data[1].line.color == charts.FUND_COLOURS[1]
    assert fig.data[2].line.color == charts.BENCHMARK_COLOUR and fig.data[2].line.dash == "dash"
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.yaxis.ticksuffix == "%"
    assert fig.layout.yaxis.title.text == "Rolling 3Y return (annualised)"
    assert [b.label for b in fig.layout.xaxis.rangeselector.buttons] == ["1Y", "3Y", "5Y", "All"]


def test_rolling_figure_without_benchmark() -> None:
    data = _data()
    fig = charts.rolling_returns_figure(data[~data["is_benchmark"]], ["Alpha"], 1, "X")
    assert [t.name for t in fig.data] == ["Alpha"]
