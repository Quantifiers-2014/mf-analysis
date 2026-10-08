from pathlib import Path

import pytest

from arbitrage_analyser.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
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


def test_fixture_config_loads(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.fund_by_code(100002).fund_managers == ("B. Manager", "C. Manager")
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
            'exit_load_days = 15\nfund_managers = ["A',
            'exit_load_days = 0\nfund_managers = ["A',
            "both be 0",
        ),
        ("nav = [4, 7]", "nav = [7, 4]", "Due must be below Overdue"),
        ('history_start = "2019-01-01"', 'history_start = "01-01-2019"', "YYYY-MM-DD"),
        ("amfi_code = 100001", 'amfi_code = "100001"', "wrong type"),
        ("outlier_multiplier = 10.0", "outlier_multiplier = true", "true/false"),
        ('ter_match = "Alpha Arbitrage Fund"', 'ter_match = " "', "must not be empty"),
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
