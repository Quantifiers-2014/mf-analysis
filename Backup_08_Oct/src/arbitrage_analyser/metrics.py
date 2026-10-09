"""Return calculations (spec: Calculations). Pure functions on date-indexed series.

All returns are annualised (CAGR) and expressed as fractions (0.07 = 7%).
A missing date (holiday) uses the nearest earlier value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS_YEARS = (1, 3, 5)


@dataclass(frozen=True)
class RollingStats:
    count: int
    average: float
    median: float
    minimum: float
    maximum: float
    pct_above_level: float


def _window_start(end: pd.Timestamp, years: int) -> pd.Timestamp:
    return pd.Timestamp(end) - pd.DateOffset(years=years)


def _annualise(growth: float | np.ndarray, years: int) -> float | np.ndarray:
    return np.power(growth, 1.0 / years) - 1.0


def point_to_point(series: pd.Series, years: int, end: pd.Timestamp | None = None) -> float | None:
    """Annualised return over `years` ending at `end` (default: last date). None if history
    does not reach back far enough."""
    if series.empty:
        return None
    end_ts = series.index[-1] if end is None else pd.Timestamp(end)
    start = _window_start(end_ts, years)
    if start < series.index[0]:
        return None
    start_value = series.asof(start)
    end_value = series.asof(end_ts)
    if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0:
        return None
    return float(_annualise(end_value / start_value, years))


def rolling_returns(series: pd.Series, years: int) -> pd.Series:
    """Annualised return for the window ending on each date, one value per date.

    Only dates whose full window lies inside the data are returned.
    """
    if series.empty:
        return pd.Series(dtype=float, name=f"rolling_{years}y")
    index = series.index
    starts = index - pd.DateOffset(years=years)
    valid = starts >= index[0]
    # position of the last date on or before each window start
    pos = np.searchsorted(index.values, starts[valid].values, side="right") - 1
    start_values = series.to_numpy()[pos]
    end_values = series.to_numpy()[valid]
    result = _annualise(end_values / start_values, years)
    return pd.Series(result, index=index[valid], name=f"rolling_{years}y")


def align_to(reference: pd.Series, other: pd.Series, max_gap_days: int) -> pd.Series:
    """Values of `other` on `reference`'s dates, using the nearest earlier date within
    `max_gap_days`. Dates without such a value are dropped."""
    if reference.empty or other.empty:
        return pd.Series(dtype=float)
    left = pd.DataFrame({"date": reference.index})
    right = pd.DataFrame({"date": other.index, "value": other.to_numpy()})
    merged = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_gap_days),
    )
    merged = merged.dropna(subset=["value"])
    return pd.Series(merged["value"].to_numpy(), index=pd.DatetimeIndex(merged["date"]))


def tracking_difference(
    fund_returns: pd.Series, benchmark_returns: pd.Series, max_gap_days: int
) -> pd.Series:
    """Fund return minus benchmark return for the same window end date."""
    bench = align_to(fund_returns, benchmark_returns, max_gap_days)
    return (fund_returns.reindex(bench.index) - bench).dropna()


def rolling_stats(returns: pd.Series, level: float) -> RollingStats | None:
    """Summary of rolling returns; `level` is a fraction (0.07 = 7%)."""
    clean = returns.dropna()
    if clean.empty:
        return None
    return RollingStats(
        count=int(clean.size),
        average=float(clean.mean()),
        median=float(clean.median()),
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        pct_above_level=float((clean > level).mean()),
    )
