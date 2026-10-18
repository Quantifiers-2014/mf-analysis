"""Data store (published/*.csv), the data refresh job, and saving app changes to GitHub."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from arbitrage_analyser import cli, db, factsheets, ingest, jobs, publish, store
from arbitrage_analyser.config import AppConfig, ConfigError, load_config
from arbitrage_analyser.runtime import CONFIG_ENV, DATA_DIR_ENV, DB_ENV
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.mfapi import SchemeHistory
from tests.conftest import (
    CONFIG_TEXT,
    benchmark_csv,
    factsheet_values,
    fake_factsheets,
    growth_series,
    history_for,
    ter_rows,
    xlsx_bytes,
)
from tests.test_cli_app import APP_PATH

TODAY = date(2026, 10, 18)
END = "2026-10-16"


def _nav_fetcher(config: AppConfig, spike: str | None = None):  # type: ignore[no-untyped-def]
    def fetch(code: int) -> SchemeHistory:
        series = growth_series("2019-01-01", END, 0.07, noise=0.00002, seed=code).copy()
        if spike:
            series[pd.Timestamp(spike)] *= 1.03
        return history_for(code, config.fund_by_code(code).isin, series)

    return fetch


def _seed(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    assert ingest.refresh_nav(conn, config, _nav_fetcher(config)).ok
    bench = growth_series("2019-01-01", "2026-09-30", 0.069, start_value=1000)
    assert ingest.import_benchmark(conn, config, benchmark_csv(bench)).ok
    ter = ter_rows([("Alpha Arbitrage Fund", "01/09/2026", 0.33, 2.32)])
    assert ingest.import_ter(conn, config, xlsx_bytes(ter)).ok
    download, read = fake_factsheets(factsheet_values())
    assert factsheets.fetch_factsheets(conn, config, TODAY, download, read).ok


def _tables(conn: db.sqlite3.Connection) -> dict[str, pd.DataFrame]:
    found = {t.name: pd.read_sql_query(t.select(), conn) for t in store.TABLES}
    return found | {"factsheet": db.read_factsheets(conn)}


# ---------- save and load ----------


def test_save_and_load_round_trip(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    _seed(conn, config)
    folder = tmp_path / "store"
    written = store.save(conn, folder)
    assert written == [f for f in store.FILES if f != store.REVIEWED_FILE]  # no reviews yet
    assert (folder / "nav.csv").read_text().startswith("amfi_code,date,nav\n100001,2019-01-01,")
    assert store.save(conn, folder) == []  # nothing changed: nothing written

    with db.connect(tmp_path / "fresh.db") as fresh:
        result = store.load(fresh, config, folder, TODAY)
        assert result.ok, result.errors
        for name, frame in _tables(conn).items():
            assert _tables(fresh)[name].astype(str).equals(frame.astype(str)), name
        assert store.save(fresh, folder) == []  # loading and saving changes no file
        again = store.load(fresh, config, folder, TODAY)
        assert again.rows == result.rows and again.new_flags == 0 and not again.damaged


def test_flags_are_worked_out_again_and_reviews_kept(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    """Flags need no file: loading recomputes them. Review decisions are kept in the store."""
    assert ingest.refresh_nav(conn, config, _nav_fetcher(config, spike="2025-06-02")).ok
    flags = db.read_flags(conn)
    assert len(flags) == 4
    first = flags.iloc[0]
    key = (str(first["dataset"]), str(first["subject"]), str(first["date"]), "unusual_move")
    assert db.mark_flags_reviewed(conn, [key]) == 1
    folder = tmp_path / "store"
    store.save(conn, folder)
    assert (folder / store.REVIEWED_FILE).read_text().splitlines()[1] == ",".join(key)

    with db.connect(tmp_path / "fresh.db") as fresh:
        result = store.load(fresh, config, folder, TODAY)
        assert result.new_flags == 3  # the reviewed one is not new
        assert len(db.read_flags(fresh, "open")) == 3
        assert db.reviewed_flag_keys(fresh) == [key]


def test_a_review_is_kept_while_its_flag_is_gone(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    """A review decision survives loads and saves even when its flag is not recreated."""
    key = ("factsheet", "100001", "2026-09-30", "ber_vs_amfi")
    db.mark_flags_reviewed(conn, [key])
    folder = tmp_path / "store"
    assert store.save(conn, folder, [store.REVIEWED_FILE]) == [store.REVIEWED_FILE]
    with db.connect(tmp_path / "fresh.db") as fresh:
        store.load(fresh, config, folder, TODAY)
        assert store.save(fresh, folder) == []
        assert db.reviewed_flag_keys(fresh) == [key]
        # and applies when the flag comes back
        db.record_flags(fresh, [db.Flag(*key[:3], key[3], "BER differs")])
        assert db.read_flags(fresh, "open").empty


@pytest.mark.parametrize(
    ("file", "header", "line", "message"),
    [
        ("nav.csv", "amfi_code,date,nav", "100001,06/10/2026,10.5", "date '06/10/2026' is not"),
        ("nav.csv", "amfi_code,date,nav", "100001,2026-10-06,", "nav '' is not a number"),
        ("nav.csv", "amfi_code,date,nav", "100001,2026-10-06,inf", "nav 'inf' is not a number"),
        ("nav.csv", "amfi_code,date,nav", "x,2026-10-06,10.5", "amfi_code 'x' is not"),
        (
            "ter.csv",
            "amfi_code,date,base_ter,brokerage,transaction_cost,statutory_levies,total_ter",
            "100001,2026-09-01,,0.1,0.1,1.7,2.3",
            "base_ter '' is not a number",
        ),
        ("benchmark.csv", "index_name,date,value", ",2026-10-06,2400.5", "index_name is empty"),
        (
            "funds.csv",
            "amfi_code,scheme_name,isin,category,launch_date",
            "100001,Alpha,INF000A01AA1,,2013-31-01",
            "launch_date '2013-31-01'",
        ),
    ],
)
def test_bad_lines_are_skipped_and_reported(
    conn: db.sqlite3.Connection,
    config: AppConfig,
    tmp_path: Path,
    file: str,
    header: str,
    line: str,
    message: str,
) -> None:
    folder = tmp_path / "store"
    folder.mkdir()
    good = {
        "nav.csv": "100001,2026-10-05,10.4",
        "ter.csv": "100001,2026-09-02,0.33,0.1,0.1,1.7,2.3",
        "benchmark.csv": "NIFTY 50 Arbitrage,2026-10-05,2400.1",
        "funds.csv": "100002,Beta,INF000B01BB2,,",
    }[file]
    (folder / file).write_text(f"{header}\n{line}\n{good}\n")
    result = store.load(conn, config, folder, TODAY)
    assert len(result.errors) == 1, result.errors
    assert result.errors[0].startswith(f"{file} line 2: ") and message in result.errors[0]
    assert result.rows == 1  # the good line still loads


def test_funds_not_in_config_are_kept_and_missing_columns_reported(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    folder = tmp_path / "store"
    folder.mkdir()
    nav = "amfi_code,date,nav\n999,2026-10-05,10.4\n"
    (folder / "nav.csv").write_text(nav)
    (folder / "ter.csv").write_text("amfi_code,date\n100001,2026-09-01\n")
    result = store.load(conn, config, folder, TODAY)
    assert result.errors[0].startswith("ter.csv: missing columns ['base_ter'")
    assert result.damaged == {"ter.csv"}
    assert store.save(conn, folder, ["nav.csv"]) == []  # fund 999 is not dropped from the file
    assert (folder / "nav.csv").read_text() == nav


def test_fixed_data_clears_its_flag(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    folder = tmp_path / "store"
    folder.mkdir()
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,-1\n")
    store.load(conn, config, folder, TODAY)
    assert list(db.read_flags(conn)["check_name"]) == ["bad_value"]
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,10.4\n")
    store.load(conn, config, folder, TODAY)
    assert db.read_flags(conn).empty


def test_app_undoes_a_change_it_could_not_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _app_env(monkeypatch, _with_repo(tmp_path), tmp_path / "app.db")
    folder.mkdir()
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-01-05,-1.0\n")
    monkeypatch.setenv(publish.TOKEN_ENV, "secret")

    def refused(*args: Any) -> bool:
        raise SourceError("HTTP 401")

    monkeypatch.setattr(publish, "save_file", refused)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    label = "2026-01-05 · nav 100001 · Alpha Arbitrage Fund: value -1.0 is zero or negative"
    app.multiselect(key="flag_pick").set_value([label]).run()
    app.button(key="flag_save").click().run()
    assert not app.exception
    assert any("not saved to GitHub: HTTP 401" in e.value for e in app.error)
    with db.connect(tmp_path / "app.db") as conn:
        assert len(db.read_flags(conn)) == 1  # still open
        assert db.reviewed_flag_keys(conn) == []


def test_rows_removed_from_a_file_do_not_come_back(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    folder = tmp_path / "store"
    folder.mkdir()
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,10.4\n"
                                    "100001,2026-10-06,10.5\n")  # fmt: skip
    store.load(conn, config, folder, TODAY)
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,10.4\n")
    store.load(conn, config, folder, TODAY)  # the same database, as on a laptop
    assert len(db.read_nav(conn, 100001)) == 1


def test_sync_loads_once_per_version(
    monkeypatch: pytest.MonkeyPatch,
    conn: db.sqlite3.Connection,
    config: AppConfig,
    tmp_path: Path,
    db_file: Path,
) -> None:
    folder = tmp_path / "store"
    folder.mkdir()
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,10.4\n")
    calls: list[int] = []
    real_load = store.load

    def counting_load(*args: Any) -> store.StoreLoad:
        calls.append(1)
        return real_load(*args)

    monkeypatch.setattr(store, "load", counting_load)
    first = store.sync(db_file, config, folder, TODAY)
    assert first.rows == 1 and len(calls) == 1
    assert store.sync(db_file, config, folder, TODAY) is first and len(calls) == 1
    (folder / "nav.csv").write_text("amfi_code,date,nav\n100001,2026-10-05,10.5\n")
    assert store.sync(db_file, config, folder, TODAY).rows == 1 and len(calls) == 2
    assert float(db.read_nav(conn, 100001).iloc[0]) == 10.5
    db_file.unlink()  # a new or deleted database loads again
    assert store.sync(db_file, config, folder, TODAY).rows == 1 and len(calls) == 3


# ---------- the data refresh job ----------


@pytest.fixture
def job_env(monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path) -> Path:
    """A fresh runner: an empty database and the store folder, as on GitHub."""
    monkeypatch.setenv(CONFIG_ENV, str(config_file))
    monkeypatch.setenv(DB_ENV, str(tmp_path / "runner.db"))
    folder = tmp_path / "published"
    monkeypatch.setenv(DATA_DIR_ENV, str(folder))
    config = load_config(config_file)
    monkeypatch.setattr(ingest, "fetch_history", _nav_fetcher(config))
    monkeypatch.setattr(ingest, "fetch_ter_records", lambda m, c, f: [])
    download, read = fake_factsheets(factsheet_values())
    monkeypatch.setattr(factsheets, "download_latest", download)
    monkeypatch.setattr(factsheets, "read_factsheet", read)
    return folder


def test_update_data_writes_the_store(
    job_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert cli.main(["update-data", "--datasets", "nav,factsheets"]) == 0
    assert {p.name for p in job_env.iterdir()} == {"funds.csv", "nav.csv", "factsheet_values.csv"}
    nav = (job_env / "nav.csv").read_text()
    # a later run on a new runner adds only what is new
    monkeypatch.setenv(DB_ENV, str(tmp_path / "runner2.db"))
    assert cli.main(["update-data", "--datasets", "nav"]) == 0
    assert (job_env / "nav.csv").read_text() == nav


def test_update_data_fails_on_errors_and_new_flags_but_saves(
    job_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_file: Path
) -> None:
    assert cli.main(["update-data", "--datasets", "nav"]) == 0
    monkeypatch.setenv(DB_ENV, str(tmp_path / "runner2.db"))
    config = load_config(config_file)
    monkeypatch.setattr(ingest, "fetch_history", _nav_fetcher(config, spike="2026-10-15"))
    assert cli.main(["update-data", "--datasets", "nav"]) == 1  # a new unusual move
    assert "2026-10-15" in (job_env / "nav.csv").read_text()

    def down(code: int) -> SchemeHistory:
        raise SourceError("mfapi.in is down")

    monkeypatch.setattr(ingest, "fetch_history", down)
    monkeypatch.setenv(DB_ENV, str(tmp_path / "runner3.db"))
    assert cli.main(["update-data", "--datasets", "nav"]) == 1
    assert (job_env / "nav.csv").read_text().count("\n") > 1000  # kept


def test_update_data_does_not_overwrite_a_damaged_store(
    job_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_env.mkdir()
    (job_env / "nav.csv").write_text("amfi_code,date,nav\n100001,bad,1\n")
    assert cli.main(["update-data", "--datasets", "nav"]) == 2
    assert "Fix the data store files" in capsys.readouterr().err
    assert (job_env / "nav.csv").read_text() == "amfi_code,date,nav\n100001,bad,1\n"


def test_job_checks_stored_factsheet_ber_against_new_ter(
    monkeypatch: pytest.MonkeyPatch, conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    download, read = fake_factsheets(factsheet_values(ber=0.34))  # as of 30-Sep-2026
    assert factsheets.fetch_factsheets(conn, config, TODAY, download, read).new_flags == 0

    def ter(month: str, category_id: int, mf_id: int) -> list[dict[str, Any]]:
        if month != "09-2026" or mf_id != 17:
            return []
        return [{"Scheme_Name": "Alpha Arbitrage Fund", "TER_Date": "2026-09-01T00:00:00.000Z",
                 "D_BER": "0.20", "D_TER": "2.0"}]  # fmt: skip

    monkeypatch.setattr(ingest, "fetch_ter_records", ter)
    results = jobs.fetch(conn, config, TODAY, ["ter"])
    assert list(results) == ["ter 09-2026", "ter 10-2026", "factsheet BER check"]
    assert results["factsheet BER check"].new_flags == 1


def test_update_data_rejects_unknown_datasets(job_env: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["update-data", "--datasets", "nav,prices"])


# ---------- saving to GitHub ----------


def _with_repo(tmp_path: Path, repo: str = "owner/repo") -> Path:
    path = tmp_path / "c.toml"
    setting = f'overlap_tolerance = 0.01\ngithub_repository = "{repo}"\n'
    path.write_text(CONFIG_TEXT.replace("overlap_tolerance = 0.01\n", setting))
    return path


class _Reply:
    def __init__(self, status: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code, self.payload = status, payload or {}
        self.text = str(self.payload)

    def json(self) -> dict[str, Any]:
        return self.payload


class _GitHub:
    """Fake requests.Session for the contents API: records PUTs."""

    def __init__(self, status: int = 201) -> None:
        self.status = status
        self.puts: list[dict[str, Any]] = []

    def put(self, url: str, **kwargs: Any) -> _Reply:
        assert url.endswith("/repos/owner/repo/contents/published/benchmark.csv")
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        self.puts.append(kwargs["json"])
        return _Reply(self.status, {"message": "sha does not match"})


REPO = publish.Repository("owner/repo", "main", "secret")


def test_blob_sha_is_gits() -> None:
    # git hash-object of a file containing "a\n"
    assert publish.blob_sha(b"a\n") == "78981922613b2afb6025042ff6bd878ac1994e85"


def test_save_file_creates_updates_and_skips_unchanged() -> None:
    new = _GitHub()
    assert publish.save_file(REPO, "benchmark.csv", b"a\n", "msg", None, new)  # type: ignore[arg-type]
    assert new.puts == [{"message": "msg", "content": "YQo=", "branch": "main"}]
    changed = _GitHub()
    assert publish.save_file(REPO, "benchmark.csv", b"b\n", "msg", b"a\n", changed)  # type: ignore[arg-type]
    assert changed.puts[0]["sha"] == publish.blob_sha(b"a\n")  # replaces only what was loaded
    same = _GitHub()
    assert not publish.save_file(REPO, "benchmark.csv", b"a\n", "msg", b"a\n", same)  # type: ignore[arg-type]
    assert same.puts == []


def test_save_file_never_overwrites_a_newer_version() -> None:
    newer = _GitHub(status=409)
    with pytest.raises(SourceError, match=r"HTTP 409.*changed on GitHub since the app loaded"):
        publish.save_file(REPO, "benchmark.csv", b"b\n", "msg", b"a\n", newer)  # type: ignore[arg-type]
    refused = _GitHub(status=403)
    with pytest.raises(SourceError, match="lacks 'Contents: Read and write'"):
        publish.save_file(REPO, "benchmark.csv", b"b\n", "msg", None, refused)  # type: ignore[arg-type]

    class Offline:
        def put(self, url: str, **kwargs: Any) -> None:
            raise requests.ConnectionError("no network")

    with pytest.raises(SourceError, match="could not reach GitHub"):
        publish.save_file(REPO, "benchmark.csv", b"b\n", "msg", None, Offline())  # type: ignore[arg-type]


def test_repository_needs_token_and_config(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig, tmp_path: Path
) -> None:
    monkeypatch.delenv(publish.TOKEN_ENV, raising=False)
    path = _with_repo(tmp_path)
    settings = load_config(path).settings
    assert publish.repository(settings) is None
    assert publish.actions_url(settings) == (
        "https://github.com/owner/repo/actions/workflows/refresh-data.yml"
    )
    monkeypatch.setenv(publish.TOKEN_ENV, "secret")
    assert publish.repository(settings) == REPO
    assert publish.repository(config.settings) is None  # no repository in config
    with pytest.raises(ConfigError, match="owner/name"):
        load_config(_with_repo(tmp_path, "x"))


# ---------- app ----------


def _app_env(monkeypatch: pytest.MonkeyPatch, config_path: Path, db_file: Path) -> Path:
    monkeypatch.setenv(CONFIG_ENV, str(config_path))
    monkeypatch.setenv(DB_ENV, str(db_file))
    monkeypatch.delenv(publish.TOKEN_ENV, raising=False)
    folder = db_file.parent / "published"
    monkeypatch.setenv(DATA_DIR_ENV, str(folder))
    return folder


def test_app_shows_data_from_the_store(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig, config_file: Path, tmp_path: Path
) -> None:
    with db.connect(tmp_path / "source.db") as conn:
        _seed(conn, config)
        store.save(conn, tmp_path / "published")
    _app_env(monkeypatch, config_file, tmp_path / "app.db")  # an empty database, as on Cloud
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception and not app.warning
    table = app.dataframe[0].value.set_index("Fund")
    assert table.loc["Alpha Arbitrage Fund", "NAV"] > 0
    assert table.loc["Alpha Arbitrage Fund", "BER % (AMFI)"] == pytest.approx(0.33)
    assert table.loc["Alpha Arbitrage Fund", "AUM (Rs Cr)"] == pytest.approx(75274.99)
    assert app.radio(key="roll_window").label == "Rolling Return Period"


def test_app_warns_about_a_bad_store_file(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    folder = _app_env(monkeypatch, config_file, tmp_path / "app.db")
    folder.mkdir()
    (folder / "nav.csv").write_text("amfi_code\n1\n")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert "nav.csv: missing columns" in app.warning[0].value


def test_app_links_to_the_refresh_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _with_repo(tmp_path)
    _app_env(monkeypatch, path, tmp_path / "app.db")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    buttons = app.get("link_button")
    assert [b.proto.label for b in buttons] == ["Run the data refresh now (GitHub)"]
    assert buttons[0].proto.url.endswith("/owner/repo/actions/workflows/refresh-data.yml")


def test_save_store_files(
    monkeypatch: pytest.MonkeyPatch, conn: db.sqlite3.Connection, tmp_path: Path
) -> None:
    """The step behind the app's benchmark upload and flag reviews."""
    config = load_config(_with_repo(tmp_path))
    folder = tmp_path / "published"
    bench = growth_series("2025-01-01", "2025-03-31", 0.07, start_value=1000)
    assert ingest.import_benchmark(conn, config, benchmark_csv(bench)).ok
    files = [store.BENCHMARK_FILE]

    monkeypatch.delenv(publish.TOKEN_ENV, raising=False)
    local = publish.save_store_files(conn, config.settings, folder, files, "msg")
    assert local[0][0] == "warning" and "on this machine only" in local[0][1]
    loaded = (folder / "benchmark.csv").read_bytes()

    monkeypatch.setenv(publish.TOKEN_ENV, "secret")
    sent: list[tuple[bytes, bytes | None]] = []

    def fake_save(
        repo: publish.Repository, file: str, content: bytes, message: str, based_on: bytes | None
    ) -> bool:
        if sent:
            raise SourceError("HTTP 409")
        sent.append((content, based_on))
        return True

    monkeypatch.setattr(publish, "save_file", fake_save)
    more = growth_series("2025-04-01", "2025-04-30", 0.07, start_value=1017.2)
    assert ingest.import_benchmark(conn, config, benchmark_csv(more)).ok
    ok = publish.save_store_files(conn, config.settings, folder, files, "msg")
    assert ok[0][0] == "success" and "Saved benchmark.csv to GitHub" in ok[0][1]
    assert sent[0][1] == loaded  # based on the version the app loaded
    assert (folder / "benchmark.csv").read_bytes() == loaded  # the checkout is left as loaded
    failed = publish.save_store_files(conn, config.settings, folder, files, "msg")
    assert failed == [("error", "benchmark.csv not saved to GitHub: HTTP 409")]
    blocked = publish.save_store_files(
        conn, config.settings, folder, files, "msg", {"benchmark.csv"}
    )
    assert blocked[0][0] == "error" and "did not load cleanly" in blocked[0][1]
    assert publish.save_store_files(conn, config.settings, folder, [], "msg") == []
