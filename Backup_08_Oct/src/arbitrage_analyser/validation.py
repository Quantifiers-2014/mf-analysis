"""Data-quality checks run on every load (spec: Data validation).

Two outcomes:
- a *rejection* stops the load (the data is clearly wrong, e.g. a mismatched benchmark file);
- a *flag* stores the data and lists the finding on the Data health screen for review.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arbitrage_analyser.config import Fund, Settings
from arbitrage_analyser.db import Flag
from arbitrage_analyser.sources.mfapi import SchemeHistory


@dataclass(frozen=True)
class UploadCheck:
    rejections: list[str]
    flags: list[Flag]

    @property
    def accepted(self) -> bool:
        return not self.rejections


def _day(ts: pd.Timestamp) -> str:
    return str(ts.date().isoformat())


def per_day_changes(series: pd.Series) -> pd.Series:
    """Change from the previous value divided by the calendar days between the two dates.

    Arbitrage NAVs accrue over weekends and holidays, so a raw Monday change is larger than a
    Tuesday one; dividing by elapsed days puts every day on the same footing.
    """
    days = series.index.to_series().diff().dt.days
    return (series / series.shift(1) - 1) / days


def check_series(
    series: pd.Series, dataset: str, subject: str, settings: Settings, label: str
) -> list[Flag]:
    """Flags for gaps, zero/negative values and unusual daily moves in a dated series.

    `series` must be sorted by date with unique dates. `label` names it in flag details.
    """
    flags: list[Flag] = []
    if series.empty:
        return flags

    for ts, value in series[series <= 0].items():
        flags.append(
            Flag(
                dataset,
                subject,
                _day(ts),
                "bad_value",
                f"{label}: value {value} is zero or negative",
            )
        )
    positive = series[series > 0]

    gaps = positive.index.to_series().diff().dt.days
    for ts, gap in gaps[gaps > settings.max_gap_days].items():
        flags.append(
            Flag(
                dataset,
                subject,
                _day(ts),
                "date_gap",
                f"{label}: {int(gap)} calendar days since the previous value",
            )
        )

    change = per_day_changes(positive)
    typical = (
        change.abs()
        .shift(1)
        .rolling(settings.outlier_lookback_days, min_periods=settings.outlier_min_history)
        .median()
    )
    limit = typical * settings.outlier_multiplier
    unusual = change[(typical > 0) & (change.abs() > limit)]
    for ts, value in unusual.items():
        flags.append(
            Flag(
                dataset,
                subject,
                _day(ts),
                "unusual_move",
                f"{label}: daily move {value:+.4%} is above {settings.outlier_multiplier:g}x the "
                f"median daily move ({typical[ts]:.4%})",
            )
        )
    return flags


def check_fund_mapping(fund: Fund, history: SchemeHistory) -> list[str]:
    """Hard errors when the NAV source does not describe the configured direct growth plan."""
    errors: list[str] = []
    name = history.scheme_name.lower()
    if "direct" not in name or "growth" not in name:
        errors.append(
            f"{fund.name}: scheme {fund.amfi_code} is '{history.scheme_name}', "
            "not a Direct Growth plan"
        )
    if history.isin_growth and history.isin_growth.upper() != fund.isin:
        errors.append(
            f"{fund.name}: ISIN in config is {fund.isin} but the source says {history.isin_growth}"
        )
    return errors


def check_duplicate_dates(dates: pd.Series) -> list[str]:
    duplicated = dates[dates.duplicated(keep=False)]
    return sorted({_day(d) for d in duplicated})


def check_benchmark_upload(
    upload: pd.DataFrame, stored: pd.Series, settings: Settings
) -> UploadCheck:
    """Rules for a benchmark CSV (spec: Data validation, benchmark rows).

    Weekend dates are allowed: NSE holds special Saturday/Sunday sessions (e.g. Budget day
    01-Feb-2026, disaster-recovery drills in 2024), and the index has values on those days.

    Reject the file on bad values or duplicate dates, or when dates already stored
    carry different values (a wrong file or index version). Otherwise flag gaps and outliers
    on the combined series.
    """
    rejections: list[str] = []
    bad = upload[upload["value"] <= 0]
    if not bad.empty:
        rejections.append(
            f"Zero or negative values on {len(bad)} date(s), e.g. {_day(bad['date'].iloc[0])}"
        )
    dupes = check_duplicate_dates(upload["date"])
    if dupes:
        rejections.append(f"Duplicate dates: {', '.join(dupes[:5])}")
    if rejections:
        return UploadCheck(rejections, [])

    new = pd.Series(upload["value"].to_numpy(), index=pd.DatetimeIndex(upload["date"])).sort_index()
    overlap = new.index.intersection(stored.index)
    if len(overlap):
        diff = (new[overlap] - stored[overlap]).abs()
        mismatched = diff[diff > settings.overlap_tolerance]
        if not mismatched.empty:
            first = mismatched.index[0]
            rejections.append(
                f"{len(mismatched)} date(s) already stored have different values, e.g. "
                f"{_day(first)}: stored {stored[first]}, file {new[first]}. "
                "Check the file is the NIFTY 50 Arbitrage index."
            )
            return UploadCheck(rejections, [])

    combined = pd.concat([stored, new[~new.index.isin(stored.index)]]).sort_index()
    flags = check_series(
        combined, "benchmark", settings.benchmark_name, settings, settings.benchmark_name
    )
    return UploadCheck([], flags)
