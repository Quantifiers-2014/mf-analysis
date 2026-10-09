"""TER and AAUM switched off in [settings.features]."""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arbitrage_analyser import cli, db, services
from arbitrage_analyser.config import ConfigError, load_config
from arbitrage_analyser.runtime import CONFIG_ENV, DB_ENV
from tests.conftest import CONFIG_TEXT, CONFIG_TEXT_OFF
from tests.test_cli_app import APP_PATH


def test_shipped_config_has_ter_on_and_aaum_off() -> None:
    settings = load_config().settings
    assert (settings.ter_enabled, settings.aaum_enabled) == (True, False)
    assert settings.active_datasets() == ("nav", "ter", "benchmark")


def test_active_datasets(config_file: Path, config_file_off: Path) -> None:
    assert load_config(config_file).settings.active_datasets() == (
        "nav",
        "ter",
        "aaum",
        "benchmark",
    )
    assert load_config(config_file_off).settings.active_datasets() == ("nav", "benchmark")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            CONFIG_TEXT.replace("[settings.features]\nter = true\naaum = true\n", ""),
            "missing 'features'",
        ),
        (CONFIG_TEXT.replace("ter = true", "ter = 1"), "wrong type"),
        (CONFIG_TEXT.replace("aaum = true\n", ""), "missing 'aaum'"),
    ],
)
def test_features_validation(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_services_hide_ter_and_aaum(config_file_off: Path, db_file: Path) -> None:
    config = load_config(config_file_off)
    with db.connect(db_file) as conn:
        table = services.fund_comparison(conn, config, "Arbitrage").table
        health = services.data_health(conn, config, date(2026, 10, 8))
    assert list(table.columns) == [
        "Fund",
        "Launch date",
        "NAV",
        "Fund manager",
        "P2P return % (3Y)",
        "Tracking diff % (3Y)",
        "Exit load",
        "Factsheet",
        "Monthly portfolio",
        "Flag",
    ]
    assert list(health["Dataset"]) == ["NAV", "Benchmark"]


def test_cli_refuses_switched_off_imports(
    monkeypatch: pytest.MonkeyPatch,
    config_file_off: Path,
    db_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(config_file_off))
    monkeypatch.setenv(DB_ENV, str(db_file))
    file = tmp_path / "ter.xlsx"
    file.write_bytes(b"x")
    assert cli.main(["import-ter", str(file)]) == 2
    assert cli.main(["import-aaum", str(file)]) == 2
    assert "switched off" in capsys.readouterr().err


def test_app_hides_ter_and_aaum_uploads(
    monkeypatch: pytest.MonkeyPatch, config_file_off: Path, db_file: Path
) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(config_file_off))
    monkeypatch.setenv(DB_ENV, str(db_file))
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    headings = [m.value for m in app.markdown]
    assert "**Benchmark CSV**" in headings
    assert "**TER file**" not in headings
    assert "**AAUM file**" not in headings


def test_text_fixtures_differ_only_in_features() -> None:
    assert CONFIG_TEXT_OFF != CONFIG_TEXT
    assert CONFIG_TEXT_OFF.replace("false", "true") == CONFIG_TEXT
