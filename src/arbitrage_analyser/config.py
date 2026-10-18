"""Load and validate the fund configuration file (config/funds.toml)."""

from __future__ import annotations

import re
import string
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "funds.toml"

DATASETS = ("nav", "ter", "factsheet", "benchmark")

# Placeholders allowed in a factsheet link, filled per factsheet edition month (e.g. Sep-2026):
# month=september Month=September mm=09 year=2026, next_mm/next_year = the month after (10, 2026).
LINK_FIELDS = frozenset({"month", "Month", "mm", "year", "next_mm", "next_year"})


class ConfigError(ValueError):
    """Raised when the configuration file is missing fields or has invalid values."""


@dataclass(frozen=True)
class FundManager:
    """A fund manager and when they started on the fund.

    `since` is the first day of the month when only the month is known (`day_known` False).
    """

    name: str
    since: date | None = None
    day_known: bool = False

    def label(self) -> str:
        """'Name (since 03-Oct-2019)', 'Name (since Dec 2014)' or 'Name'."""
        if self.since is None:
            return self.name
        when = self.since.strftime("%d-%b-%Y" if self.day_known else "%b %Y")
        return f"{self.name} (since {when})"


@dataclass(frozen=True)
class FactsheetSource:
    """Where a fund's monthly factsheet is and how to find the fund in it.

    Either `link` (the PDF link, with LINK_FIELDS placeholders when it changes each month), or
    `listing` (a page listing the factsheets) plus `link_text` (text in the PDF link to pick,
    with placeholders). `page_title`: the fund's page heading in the PDF. `turnover`: (name shown
    in the app, label printed in the factsheet) pairs.
    """

    page_title: str
    link: str | None = None
    listing: str | None = None
    link_text: str | None = None
    turnover: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Fund:
    name: str
    amc: str
    category: str
    amfi_code: int
    isin: str
    ter_match: str
    exit_load_pct: float
    exit_load_days: int
    fund_managers: tuple[FundManager, ...]
    amfi_mf_id: int | None = None  # AMFI fund house id (MF_ID), used to fetch TER
    factsheet_url: str | None = None  # AMC page listing monthly factsheets
    portfolio_url: str | None = None  # AMC page listing monthly portfolio disclosures
    factsheet: FactsheetSource | None = None  # monthly factsheet PDF, read by the data job


@dataclass(frozen=True)
class Freshness:
    """Calendar-day thresholds after which a dataset is Due, then Overdue."""

    due_after_days: int
    overdue_after_days: int


@dataclass(frozen=True)
class Settings:
    history_start: date
    benchmark_name: str
    exit_load_flag_above_pct: float
    outlier_multiplier: float
    outlier_lookback_days: int
    outlier_min_history: int
    max_gap_days: int
    overlap_tolerance: float
    freshness: dict[str, Freshness]
    ter_enabled: bool
    factsheet_enabled: bool
    # Fund category -> AMFI TER page "Sub Category" id (strCat), used to fetch TER from AMFI.
    amfi_ter_category_ids: dict[str, int] = field(default_factory=dict)
    # GitHub repository ("owner/name") and branch that hold the data store. The app saves
    # benchmark uploads and reviewed flags there, and links to the data refresh job.
    github_repository: str | None = None
    github_branch: str = "main"

    def active_datasets(self) -> tuple[str, ...]:
        """Datasets in use. TER and factsheet values can be switched off in
        [settings.features]."""
        off = {"ter": not self.ter_enabled, "factsheet": not self.factsheet_enabled}
        return tuple(d for d in DATASETS if not off.get(d, False))


@dataclass(frozen=True)
class AppConfig:
    settings: Settings
    funds: tuple[Fund, ...]

    def categories(self) -> list[str]:
        return sorted({f.category for f in self.funds})

    def funds_in(self, category: str) -> list[Fund]:
        return [f for f in self.funds if f.category == category]

    def fund_by_code(self, amfi_code: int) -> Fund:
        for fund in self.funds:
            if fund.amfi_code == amfi_code:
                return fund
        raise KeyError(amfi_code)


def _require(table: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{where}: missing '{key}'")
    value = table[key]
    # bool is a subclass of int; reject it where a number is expected.
    if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise ConfigError(f"{where}: '{key}' must not be true/false")
    if not isinstance(value, kind):
        raise ConfigError(f"{where}: '{key}' has the wrong type ({type(value).__name__})")
    return value


def _parse_settings(raw: dict[str, Any]) -> Settings:
    where = "[settings]"
    history_start = _require(raw, "history_start", str, where)
    try:
        start = date.fromisoformat(history_start)
    except ValueError as exc:
        raise ConfigError(f"{where}: history_start must be YYYY-MM-DD") from exc

    fresh_raw = _require(raw, "freshness", dict, where)
    freshness: dict[str, Freshness] = {}
    for dataset in DATASETS:
        pair = _require(fresh_raw, dataset, list, f"{where}.freshness")
        if len(pair) != 2 or not all(isinstance(v, int) and v > 0 for v in pair):
            raise ConfigError(f"{where}.freshness: '{dataset}' must be two positive integers")
        if pair[0] >= pair[1]:
            raise ConfigError(f"{where}.freshness: '{dataset}' Due must be below Overdue")
        freshness[dataset] = Freshness(pair[0], pair[1])

    features = _require(raw, "features", dict, where)
    settings = Settings(
        history_start=start,
        benchmark_name=_require(raw, "benchmark_name", str, where),
        exit_load_flag_above_pct=float(
            _require(raw, "exit_load_flag_above_pct", (int, float), where)
        ),
        outlier_multiplier=float(_require(raw, "outlier_multiplier", (int, float), where)),
        outlier_lookback_days=_require(raw, "outlier_lookback_days", int, where),
        outlier_min_history=_require(raw, "outlier_min_history", int, where),
        max_gap_days=_require(raw, "max_gap_days", int, where),
        overlap_tolerance=float(_require(raw, "overlap_tolerance", (int, float), where)),
        freshness=freshness,
        ter_enabled=_require(features, "ter", bool, f"{where}.features"),
        factsheet_enabled=_require(features, "factsheet", bool, f"{where}.features"),
        amfi_ter_category_ids=_category_ids(raw.get("amfi_ter_category_ids", {}), where),
        github_repository=_repository(raw, where),
        github_branch=_require(raw, "github_branch", str, where).strip()
        if "github_branch" in raw
        else "main",
    )
    if not settings.github_branch:
        raise ConfigError(f"{where}: github_branch must not be empty")
    if settings.outlier_multiplier <= 0 or settings.outlier_lookback_days <= 0:
        raise ConfigError(f"{where}: outlier settings must be positive")
    if settings.outlier_min_history <= 0 or settings.max_gap_days <= 0:
        raise ConfigError(f"{where}: outlier_min_history and max_gap_days must be positive")
    if settings.overlap_tolerance < 0:
        raise ConfigError(f"{where}: overlap_tolerance must not be negative")
    return settings


def _repository(raw: dict[str, Any], where: str) -> str | None:
    if "github_repository" not in raw:
        return None
    name: str = _require(raw, "github_repository", str, where).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
        raise ConfigError(f"{where}: github_repository must be owner/name, got '{name}'")
    return name


def _category_ids(raw: object, where: str) -> dict[str, int]:
    where = f"{where}.amfi_ter_category_ids"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table of category = id")
    for category, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{where}: '{category}' must be a positive whole number")
    return dict(raw)


def _url(raw: dict[str, Any], key: str, where: str) -> str | None:
    if key not in raw:
        return None
    url: str = _require(raw, key, str, where).strip()
    if not url.startswith("https://") or " " in url:
        raise ConfigError(f"{where}: {key} must be an https:// link")
    return url


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _template(raw: dict[str, Any], key: str, where: str, https: bool) -> str | None:
    if key not in raw:
        return None
    text: str = _require(raw, key, str, where).strip()
    if not text:
        raise ConfigError(f"{where}: {key} must not be empty")
    if https and (not text.startswith("https://") or " " in text):
        raise ConfigError(f"{where}: {key} must be an https:// link (write spaces as %20)")
    try:
        unknown = _fields(text) - LINK_FIELDS
    except ValueError as exc:
        raise ConfigError(f"{where}: {key} has a broken {{placeholder}}: {exc}") from exc
    if unknown:
        raise ConfigError(
            f"{where}: {key} has unknown placeholders {sorted(unknown)}; "
            f"allowed: {sorted(LINK_FIELDS)}"
        )
    return text


def _parse_factsheet(raw: object, where: str) -> FactsheetSource:
    where = f"{where} factsheet"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: must be a table ([funds.factsheet])")
    title = _require(raw, "page_title", str, where).strip()
    if not re.search(r"[A-Za-z]", title):
        raise ConfigError(f"{where}: page_title must not be empty")
    link = _template(raw, "link", where, https=True)
    listing = _template(raw, "listing", where, https=True)
    link_text = _template(raw, "link_text", where, https=False)
    if (link is None) == (listing is None):
        raise ConfigError(f"{where}: set either link or listing (with link_text)")
    if listing is not None and (link_text is None or _fields(listing)):
        raise ConfigError(f"{where}: listing needs link_text, and takes no placeholders")
    if link is not None and link_text is not None:
        raise ConfigError(f"{where}: link_text is used only with listing")
    turnover_raw = raw.get("turnover", {})
    if not isinstance(turnover_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
        for k, v in turnover_raw.items()
    ):
        raise ConfigError(f'{where}: turnover must be a table of name = "printed label"')
    return FactsheetSource(
        page_title=title,
        link=link,
        listing=listing,
        link_text=link_text,
        turnover=tuple((k.strip(), v.strip()) for k, v in turnover_raw.items()),
    )


def _parse_manager(raw: object, where: str) -> FundManager:
    """A manager is a plain name, or {name = "...", since = "YYYY-MM" or "YYYY-MM-DD"}."""
    if isinstance(raw, str):
        if not raw.strip():
            raise ConfigError(f"{where}: fund manager name must not be empty")
        return FundManager(raw.strip())
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: fund_managers entries must be names or {{name, since}}")
    name = _require(raw, "name", str, where).strip()
    if not name:
        raise ConfigError(f"{where}: fund manager name must not be empty")
    if "since" not in raw:
        return FundManager(name)
    since = _require(raw, "since", str, where).strip()
    error = f"{where}: since for {name} must be YYYY-MM or YYYY-MM-DD"
    if len(since) not in (7, 10):  # rules out other forms fromisoformat accepts, e.g. 20191003
        raise ConfigError(error)
    day_known = len(since) == 10
    try:
        start = date.fromisoformat(since if day_known else f"{since}-01")
    except ValueError as exc:
        raise ConfigError(error) from exc
    return FundManager(name, start, day_known)


def _parse_fund(raw: dict[str, Any], index: int) -> Fund:
    where = f"[[funds]] #{index + 1}"
    managers = _require(raw, "fund_managers", list, where)
    if not managers:
        raise ConfigError(f"{where}: fund_managers must list at least one manager")
    fund = Fund(
        name=_require(raw, "name", str, where).strip(),
        amc=_require(raw, "amc", str, where).strip(),
        category=_require(raw, "category", str, where).strip(),
        amfi_code=_require(raw, "amfi_code", int, where),
        isin=_require(raw, "isin", str, where).strip().upper(),
        ter_match=_require(raw, "ter_match", str, where).strip(),
        exit_load_pct=float(_require(raw, "exit_load_pct", (int, float), where)),
        exit_load_days=_require(raw, "exit_load_days", int, where),
        fund_managers=tuple(_parse_manager(m, where) for m in managers),
        amfi_mf_id=_require(raw, "amfi_mf_id", int, where) if "amfi_mf_id" in raw else None,
        factsheet_url=_url(raw, "factsheet_url", where),
        portfolio_url=_url(raw, "portfolio_url", where),
        factsheet=_parse_factsheet(raw["factsheet"], where) if "factsheet" in raw else None,
    )
    for field_name in ("name", "amc", "category", "ter_match"):
        if not getattr(fund, field_name):
            raise ConfigError(f"{where}: '{field_name}' must not be empty")
    if len(fund.isin) != 12 or not fund.isin.startswith("INF"):
        raise ConfigError(f"{where}: isin '{fund.isin}' is not a mutual fund ISIN")
    if fund.amfi_code <= 0:
        raise ConfigError(f"{where}: amfi_code must be positive")
    if fund.amfi_mf_id is not None and fund.amfi_mf_id <= 0:
        raise ConfigError(f"{where}: amfi_mf_id must be positive")
    if fund.exit_load_pct < 0 or fund.exit_load_days < 0:
        raise ConfigError(f"{where}: exit load values must not be negative")
    if (fund.exit_load_pct == 0) != (fund.exit_load_days == 0):
        raise ConfigError(f"{where}: exit_load_pct and exit_load_days must both be 0 or both set")
    return fund


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read and validate the configuration file."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML: {exc}") from exc

    settings = _parse_settings(_require(raw, "settings", dict, "config"))
    funds_raw = _require(raw, "funds", list, "config")
    if not funds_raw:
        raise ConfigError("config: at least one [[funds]] entry is required")
    funds = tuple(_parse_fund(f, i) for i, f in enumerate(funds_raw))

    for key in ("amfi_code", "isin", "name"):
        values = [getattr(f, key) for f in funds]
        duplicates = {v for v in values if values.count(v) > 1}
        if duplicates:
            raise ConfigError(f"config: duplicate {key}: {sorted(map(str, duplicates))}")
    return AppConfig(settings=settings, funds=funds)
