from datetime import date
from pathlib import Path

import pytest

from arbitrage_analyser.config import DEFAULT_CONFIG_PATH, ConfigError, FundManager, load_config
from tests.conftest import CONFIG_TEXT


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.toml"
    path.write_text(text)
    return path


def test_shipped_config_is_valid() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert len(config.funds) == 8
    assert config.categories() == ["Arbitrage"]
    assert all(f.isin.startswith("INF") for f in config.funds)
    # every fund can be fetched from AMFI's TER page, one distinct fund house id each
    assert len({f.amfi_mf_id for f in config.funds if f.amfi_mf_id}) == len(config.funds)


def test_fixture_config_loads(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.fund_by_code(100002).fund_managers == (
        FundManager("B. Manager"),
        FundManager("C. Manager", date(2014, 12, 1), day_known=False),
    )
    assert [f.name for f in config.funds_in("Arbitrage")] == [
        "Alpha Arbitrage Fund",
        "Beta Arbitrage Fund",
    ]
    with pytest.raises(KeyError):
        config.fund_by_code(1)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('isin = "INF000A01AA1"', 'isin = "US0000000001"', "not a mutual fund ISIN"),
        ("amfi_code = 100002", "amfi_code = 100001", "duplicate amfi_code"),
        (
            "exit_load_days = 15\nfund_managers = [{",
            "exit_load_days = 0\nfund_managers = [{",
            "both be 0",
        ),
        ("nav = [4, 7]", "nav = [7, 4]", "Due must be below Overdue"),
        ('history_start = "2019-01-01"', 'history_start = "01-01-2019"', "YYYY-MM-DD"),
        ("amfi_code = 100001", 'amfi_code = "100001"', "wrong type"),
        ("outlier_multiplier = 10.0", "outlier_multiplier = true", "true/false"),
        ('ter_match = "Alpha Arbitrage Fund"', 'ter_match = " "', "must not be empty"),
        ('"https://example.com/alpha/factsheets"', '"www.example.com"', "https:// link"),
        ('since = "2019-10-03"', 'since = "03-10-2019"', "YYYY-MM or YYYY-MM-DD"),
        ('since = "2014-12"', 'since = "2014-13"', "YYYY-MM or YYYY-MM-DD"),
        ('since = "2019-10-03"', 'since = "20191003"', "YYYY-MM or YYYY-MM-DD"),
        ('["B. Manager", {', '[" ", {', "name must not be empty"),
        ('["B. Manager", {', "[5, {", "names or"),
        (
            'fund_managers = [{ name = "A. Manager", since = "2019-10-03" }]',
            "fund_managers = []",
            "at least one manager",
        ),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in CONFIG_TEXT
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, CONFIG_TEXT.replace(old, new, 1)))


def test_missing_file_and_bad_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(_write(tmp_path, "[settings\n"))
    with pytest.raises(ConfigError, match="missing 'funds'"):
        load_config(_write(tmp_path, CONFIG_TEXT.split("[[funds]]")[0]))


def test_manager_labels() -> None:
    assert FundManager("Hiten Shah", date(2019, 10, 3), True).label() == (
        "Hiten Shah (since 03-Oct-2019)"
    )
    assert FundManager("L. Solanki", date(2014, 12, 1)).label() == "L. Solanki (since Dec 2014)"
    assert FundManager("No Date").label() == "No Date"
