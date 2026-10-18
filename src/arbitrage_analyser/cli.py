"""Command line: daily NAV refresh, file imports, config check.

python -m arbitrage_analyser refresh
python -m arbitrage_analyser import-benchmark FILE.csv
python -m arbitrage_analyser fetch-ter [--month MM-YYYY]
python -m arbitrage_analyser import-ter FILE
python -m arbitrage_analyser fetch-factsheets
python -m arbitrage_analyser check-config
python -m arbitrage_analyser app
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from arbitrage_analyser import db, factsheets, ingest
from arbitrage_analyser.config import AppConfig, ConfigError, load_config
from arbitrage_analyser.runtime import config_path, db_path, factsheet_csv_path

APP_FILE = Path(__file__).resolve().parent / "app.py"


def run_app() -> int:
    """Start the Streamlit app with this same Python, so the app always finds this package.

    A bare `streamlit run` can pick up a Streamlit installed outside the virtual environment,
    which then fails with "No module named 'arbitrage_analyser'".
    """
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_FILE)])


_IMPORTERS: dict[str, ingest.FileImporter] = {
    "import-benchmark": ingest.import_benchmark,
    "import-ter": ingest.import_ter,
}
# Commands that run only when switched on in [settings.features].
_FEATURE_OF = {"fetch-ter": "ter", "import-ter": "ter", "fetch-factsheets": "factsheet"}


def _report(result: ingest.LoadResult) -> int:
    print(f"Rows stored: {result.rows}. New flags: {result.new_flags}.")
    for note in result.notes:
        print(f"Note: {note}")
    for error in result.errors:
        print(f"Error: {error}", file=sys.stderr)
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arbitrage-analyser")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("refresh", help="download NAV history for all configured funds")
    commands.add_parser("check-config", help="validate config/funds.toml")
    commands.add_parser("app", help="open the app in your browser")
    fetch_ter = commands.add_parser("fetch-ter", help="download one month of TER from AMFI")
    fetch_ter.add_argument(
        "--month", help="MM-YYYY (default: last month)", default=None, metavar="MM-YYYY"
    )
    commands.add_parser(
        "fetch-factsheets",
        help="read AUM, AAUM, turnover and Direct BER from the latest AMC factsheets "
        "and update the values file (published/factsheet_values.csv)",
    )
    for name in _IMPORTERS:
        sub = commands.add_parser(name, help=f"{name.replace('-', ' ')} from a downloaded file")
        sub.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "app":
        return run_app()  # the app reports config errors on screen

    try:
        config = load_config(config_path())
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.command == "check-config":
        print(f"Config OK: {len(config.funds)} funds in {config.categories()}")
        return 0

    feature = _FEATURE_OF.get(args.command)
    if feature and feature not in config.settings.active_datasets():
        print(
            f"{args.command} is switched off. Set {feature} = true under "
            "[settings.features] in config/funds.toml to use it.",
            file=sys.stderr,
        )
        return 2

    with db.connect(db_path()) as conn:
        if args.command == "refresh":
            return _report(ingest.refresh_nav(conn, config))
        if args.command == "fetch-factsheets":
            return _fetch_factsheets(conn, config)
        if args.command == "fetch-ter":
            month = args.month or ingest.previous_month(date.today())
            print(f"Fetching TER for {month} from AMFI...")
            return _report(ingest.fetch_ter(conn, config, month))
        try:
            content = args.file.read_bytes()
        except OSError as exc:
            print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
            return 2
        return _report(_IMPORTERS[args.command](conn, config, content))


def _fetch_factsheets(conn: sqlite3.Connection, config: AppConfig) -> int:
    """Values file -> database, fetch, database -> values file, so the file keeps every month.
    Exit code 1 when a fund failed or a value was flagged (the values read are still saved)."""
    path = factsheet_csv_path()
    loaded = factsheets.import_csv(conn, config, path, date.today())
    if not loaded.ok:  # a damaged values file: stop before writing over it
        _report(loaded)
        return 2
    print("Fetching the latest factsheets...")
    result = factsheets.fetch_factsheets(conn, config, date.today())
    code = _report(result)
    rows = factsheets.export_csv(conn, path)
    print(f"Wrote {rows} row(s) to {path}")
    return 1 if result.new_flags else code
