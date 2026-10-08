import pandas as pd
import pytest

from arbitrage_analyser import metrics
from tests.conftest import growth_series


def test_point_to_point_recovers_constant_growth() -> None:
    series = growth_series("2018-01-01", "2026-10-06", 0.07)
    for years in (1, 3, 5):
        assert metrics.point_to_point(series, years) == pytest.approx(0.07, abs=2e-4)


def test_point_to_point_needs_full_history() -> None:
    series = growth_series("2024-01-01", "2026-10-06", 0.07)
    assert metrics.point_to_point(series, 3) is None
    assert metrics.point_to_point(pd.Series(dtype=float), 1) is None


def test_point_to_point_uses_nearest_earlier_value_on_holidays() -> None:
    series = growth_series("2024-01-01", "2026-10-06", 0.07)
    end = pd.Timestamp("2026-10-04")  # a Sunday: Friday's value is used
    assert metrics.point_to_point(series, 1, end) == pytest.approx(
        series.asof(end) / series.asof(end - pd.DateOffset(years=1)) - 1
    )


def test_rolling_returns_start_after_one_full_window() -> None:
    series = growth_series("2019-01-01", "2026-10-06", 0.065)
    for years in (1, 3, 5):
        rolled = metrics.rolling_returns(series, years)
        first_allowed = series.index[0] + pd.DateOffset(years=years)
        assert rolled.index[0] >= first_allowed
        assert rolled.index[-1] == series.index[-1]
        assert rolled.min() == pytest.approx(0.065, abs=3e-3)
        assert rolled.max() == pytest.approx(0.065, abs=3e-3)


def test_rolling_returns_value_matches_point_to_point() -> None:
    series = growth_series("2019-01-01", "2026-10-06", 0.07, noise=0.0005)
    rolled = metrics.rolling_returns(series, 3)
    for ts in rolled.index[[0, len(rolled) // 2, -1]]:
        assert rolled[ts] == pytest.approx(metrics.point_to_point(series, 3, ts))


def test_rolling_returns_empty_and_short() -> None:
    assert metrics.rolling_returns(pd.Series(dtype=float), 1).empty
    short = growth_series("2026-01-01", "2026-10-06", 0.07)
    assert metrics.rolling_returns(short, 1).empty


def test_tracking_difference_aligns_dates_within_gap() -> None:
    fund = pd.Series(
        [0.07, 0.072, 0.071], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-20"])
    )
    bench = pd.Series([0.075, 0.074], index=pd.to_datetime(["2026-01-02", "2026-01-06"]))
    td = metrics.tracking_difference(fund, bench, max_gap_days=5)
    # 01-05 uses 01-02 (3 days back), 01-06 exact, 01-20 has no benchmark within 5 days
    assert list(td.index) == list(pd.to_datetime(["2026-01-05", "2026-01-06"]))
    assert td.tolist() == pytest.approx([-0.005, -0.002])


def test_rolling_stats() -> None:
    returns = pd.Series([0.05, 0.06, 0.07, 0.08])
    stats = metrics.rolling_stats(returns, 0.065)
    assert stats is not None
    assert (stats.count, stats.minimum, stats.maximum) == (4, 0.05, 0.08)
    assert stats.average == pytest.approx(0.065)
    assert stats.median == pytest.approx(0.065)
    assert stats.pct_above_level == 0.5
    assert metrics.rolling_stats(pd.Series(dtype=float), 0.07) is None
