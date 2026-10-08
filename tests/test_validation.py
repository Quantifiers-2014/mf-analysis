import pandas as pd
import pytest

from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.validation import (
    check_benchmark_upload,
    check_fund_mapping,
    check_series,
    per_day_changes,
)
from tests.conftest import growth_series, history_for


def _checks(flags: list) -> list[str]:  # type: ignore[type-arg]
    return [f.check_name for f in flags]


def test_clean_series_has_no_flags(config: AppConfig) -> None:
    series = growth_series("2019-01-01", "2026-10-06", 0.07, noise=0.00002)
    assert check_series(series, "nav", "1", config.settings, "Fund") == []


def test_weekend_moves_are_not_outliers(config: AppConfig) -> None:
    # Monday's raw change is ~3x a weekday's; per-day normalisation must absorb it.
    series = growth_series("2019-01-01", "2026-10-06", 0.07)
    changes = per_day_changes(series).dropna()
    # equal up to the small compounding difference of a 3-day vs 1-day change
    assert changes.max() == pytest.approx(changes.min(), rel=1e-3)
    assert check_series(series, "nav", "1", config.settings, "Fund") == []


def test_gap_zero_and_outlier_are_flagged(config: AppConfig) -> None:
    series = growth_series("2019-01-01", "2026-10-06", 0.07, noise=0.00002).copy()
    series = series.drop(pd.bdate_range("2024-03-01", "2024-03-12"))  # gap
    series[pd.Timestamp("2025-06-02")] *= 1.02  # +2% in a day
    series[pd.Timestamp("2025-09-01")] = 0.0
    flags = check_series(series, "nav", "1", config.settings, "Fund")
    by_check = {f.check_name: f for f in flags}
    assert by_check["date_gap"].date == "2024-03-13"
    assert by_check["bad_value"].date == "2025-09-01"
    assert "2025-06-02" in [f.date for f in flags if f.check_name == "unusual_move"]


def test_outlier_check_waits_for_min_history(config: AppConfig) -> None:
    series = growth_series("2026-01-01", "2026-02-15", 0.07, noise=0.00002).copy()
    series.iloc[-1] *= 1.05
    assert "unusual_move" not in _checks(check_series(series, "nav", "1", config.settings, "F"))


def test_fund_mapping(config: AppConfig) -> None:
    fund = config.fund_by_code(100001)
    series = growth_series("2026-01-01", "2026-02-01", 0.07)
    assert check_fund_mapping(fund, history_for(100001, fund.isin, series)) == []
    regular = history_for(100001, fund.isin, series, name="Alpha - Regular Plan - Growth")
    assert "not a Direct Growth plan" in check_fund_mapping(fund, regular)[0]
    wrong_isin = history_for(100001, "INF999Z01ZZ9", series)
    assert "ISIN" in check_fund_mapping(fund, wrong_isin)[0]


def _upload(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": series.index, "value": series.to_numpy()})


def test_benchmark_upload_accepts_matching_overlap(config: AppConfig) -> None:
    full = growth_series("2019-01-01", "2026-09-30", 0.07, start_value=1000)
    stored = full[:"2026-06-30"]
    check = check_benchmark_upload(_upload(full["2026-06-01":]), stored, config.settings)
    assert check.accepted
    assert check.flags == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.assign(value=f["value"].where(f.index != 3, -1.0)), "negative"),
        (lambda f: pd.concat([f, f.iloc[[2]]]), "Duplicate dates"),
        (lambda f: f.assign(value=f["value"] + 1.0), "different values"),
    ],
)
def test_benchmark_upload_rejections(config: AppConfig, mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    full = growth_series("2019-01-01", "2026-09-30", 0.07, start_value=1000)
    stored = full[:"2026-06-30"]
    upload = mutate(_upload(full["2026-06-01":"2026-07-31"]).reset_index(drop=True))
    check = check_benchmark_upload(upload, stored, config.settings)
    assert not check.accepted
    assert message in " ".join(check.rejections)


def test_benchmark_upload_accepts_special_weekend_sessions(config: AppConfig) -> None:
    # NSE traded on Sunday 01-Feb-2026 (Budget day); the index has a value that day.
    days = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-01", "2026-02-02"])
    upload = pd.DataFrame({"date": days, "value": [1000.0, 1000.2, 1000.3, 1000.4]})
    check = check_benchmark_upload(upload, pd.Series(dtype=float), config.settings)
    assert check.accepted
