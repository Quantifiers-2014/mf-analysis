"""Command line: daily NAV refresh, file imports, config check.

python -m arbitrage_analyser refresh
python -m arbitrage_analyser import-benchmark FILE.csv
python -m arbitrage_analyser import-ter FILE
python -m arbitrage_analyser import-aaum FILE
python -m arbitrage_analyser check-config
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from arbitrage_analyser import db, ingest
from arbitrage_analyser.config import ConfigError, load_config
from arbitrage_analyser.runtime import config_path, db_path

_IMPORTERS: dict[str, ingest.FileImporter] = {
    "import-benchmark": ingest.import_benchmark,
    "import-ter": ingest.import_ter,
    "import-aaum": ingest.import_aaum,
}
# Importers that run only when switched on in [settings.features].
_FEATURE_OF = {"import-ter": "ter", "import-aaum": "aaum"}


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
    for name in _IMPORTERS:
        sub = commands.add_parser(name, help=f"{name.replace('-', ' ')} from a downloaded file")
        sub.add_argument("file", type=Path)
    args = parser.parse_args(argv)

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
        try:
            content = args.file.read_bytes()
        except OSError as exc:
            print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
            return 2
        return _report(_IMPORTERS[args.command](conn, config, content))
