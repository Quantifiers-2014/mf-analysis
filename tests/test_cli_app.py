from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arbitrage_analyser import cli, db, ingest
from arbitrage_analyser.agent.store import CHAT_DB_ENV
from arbitrage_analyser.config import load_config
from arbitrage_analyser.runtime import CONFIG_ENV, DB_ENV, ENV_FILE_ENV
from tests.conftest import benchmark_csv, growth_series, history_for

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "arbitrage_analyser" / "app.py"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, config_file: Path, db_file: Path) -> Path:
    monkeypatch.setenv(CONFIG_ENV, str(config_file))
    monkeypatch.setenv(DB_ENV, str(db_file))
    monkeypatch.setenv(CHAT_DB_ENV, str(db_file.with_name("chat.db")))
    monkeypatch.setenv(ENV_FILE_ENV, str(db_file.with_name("no.env")))
    for name in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_BASE_URL",
        "ARBITRAGE_AGENT_PROVIDER",
        "ARBITRAGE_AGENT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    return db_file


def test_cli_check_config(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check-config"]) == 0
    assert "2 funds" in capsys.readouterr().out


def test_cli_bad_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "none.toml"))
    assert cli.main(["check-config"]) == 2


def test_cli_import_benchmark(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "bench.csv"
    csv.write_bytes(
        benchmark_csv(growth_series("2025-01-01", "2025-03-31", 0.07, start_value=1000))
    )
    assert cli.main(["import-benchmark", str(csv)]) == 0
    assert "Rows stored: 64" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"Date,Close\n")
    assert cli.main(["import-benchmark", str(bad)]) == 1
    assert cli.main(["import-benchmark", str(tmp_path / "missing.csv")]) == 2


def test_cli_refresh_reports_errors(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(code: int):  # type: ignore[no-untyped-def]
        raise ingest.SourceError("network down")

    monkeypatch.setattr(ingest, "fetch_history", broken)
    assert cli.main(["refresh"]) == 1


def _seed(db_file: Path, config_file: Path) -> None:
    config = load_config(config_file)
    with db.connect(db_file) as conn:

        def fetch(code: int):  # type: ignore[no-untyped-def]
            series = growth_series("2019-01-01", "2026-10-06", 0.07, noise=0.00002, seed=code)
            return history_for(code, config.fund_by_code(code).isin, series)

        assert ingest.refresh_nav(conn, config, fetch).ok
        bench = growth_series("2019-01-01", "2026-09-30", 0.069, start_value=1000)
        assert ingest.import_benchmark(conn, config, benchmark_csv(bench)).ok


def test_app_renders_all_tabs(env: Path, config_file: Path) -> None:
    _seed(env, config_file)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Fund comparison",
        "Rolling returns",
        "Data health",
        "Ask the analyst",
    ]
    assert len(app.dataframe) == 3  # comparison, rolling stats, freshness
    assert app.multiselect(key="roll_funds").value == [
        "Alpha Arbitrage Fund",
        "Beta Arbitrage Fund",
    ]

    app.radio(key="roll_window").set_value(5).run()
    assert not app.exception
    app.multiselect(key="roll_funds").set_value([]).run()
    assert any("Pick at least one fund" in i.value for i in app.info)


def test_app_with_empty_database(env: Path) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert any("No NAV data yet" in c.value for c in app.caption)


def test_app_shows_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "none.toml"))
    monkeypatch.setenv(DB_ENV, str(tmp_path / "x.db"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert any("Config error" in e.value for e in app.error)


def test_app_mark_flag_reviewed(env: Path, config_file: Path) -> None:
    with db.connect(env) as conn:
        db.record_flags(conn, [db.Flag("nav", "100001", "2026-01-05", "unusual_move", "spike")])
        flag_id = int(db.read_flags(conn)["id"].iloc[0])
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key=f"flag_{flag_id}").click().run()
    assert not app.exception
    with db.connect(env) as conn:
        assert db.read_flags(conn).empty


def test_app_refresh_button_shows_result(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(code: int):  # type: ignore[no-untyped-def]
        raise ingest.SourceError("network down")

    monkeypatch.setattr(ingest, "fetch_history", broken)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    next(b for b in app.button if b.label == "Refresh NAV now").click().run()
    assert not app.exception
    assert any("network down" in e.value for e in app.error)
