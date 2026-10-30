"""Command line: data refresh job, file imports, config check, app.

python -m arbitrage_analyser update-data [--datasets nav,ter,factsheets]
python -m arbitrage_analyser import-benchmark FILE.csv
python -m arbitrage_analyser import-ter FILE
python -m arbitrage_analyser check-config
python -m arbitrage_analyser eval-search [--modes keyword,semantic,hybrid] [--answers]
python -m arbitrage_analyser app

Every data command loads the data store (published/), changes it, and writes it back.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from arbitrage_analyser import db, ingest, jobs, store
from arbitrage_analyser.config import AppConfig, ConfigError, load_config
from arbitrage_analyser.runtime import config_path, data_dir, db_path

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
_FEATURE_OF = {"import-ter": "ter"}


def _report(label: str, result: ingest.LoadResult) -> int:
    print(f"{label}: {result.rows} row(s) stored, {result.new_flags} new flag(s).")
    for note in result.notes:
        print(f"  Note: {note}")
    for error in result.errors:
        print(f"  Error: {error}", file=sys.stderr)
    return 0 if result.ok else 1


def _datasets(text: str) -> list[str]:
    chosen = [d.strip() for d in text.split(",") if d.strip()]
    unknown = sorted(set(chosen) - set(jobs.DATASETS))
    if unknown or not chosen:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(jobs.DATASETS)}")
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arbitrage-analyser")
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser(
        "update-data", help="fetch new NAV, TER and factsheet data into the data store"
    )
    update.add_argument(
        "--datasets",
        type=_datasets,
        default=list(jobs.DAILY),
        help=f"comma-separated, from {','.join(jobs.DATASETS)} (default: {','.join(jobs.DAILY)})",
    )
    commands.add_parser("check-config", help="validate config/funds.toml")
    evals_cmd = commands.add_parser(
        "eval-search", help="score the chatbot's factsheet search on the test set (evals/)"
    )
    evals_cmd.add_argument(
        "--modes",
        default="keyword",
        help="comma-separated, from keyword,semantic,hybrid (semantic and hybrid need "
        "MISTRAL_API_KEY)",
    )
    evals_cmd.add_argument(
        "--answers", action="store_true", help="also ask the chatbot each question (model key)"
    )
    evals_cmd.add_argument(
        "--pause", type=float, default=10.0, help="seconds between chatbot questions"
    )
    commands.add_parser("app", help="open the app in your browser")
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
    if args.command == "eval-search":
        return run_evals(config, args.modes, args.answers, args.pause)

    feature = _FEATURE_OF.get(args.command)
    if feature and feature not in config.settings.active_datasets():
        print(
            f"{args.command} is switched off. Set {feature} = true under "
            "[settings.features] in config/funds.toml to use it.",
            file=sys.stderr,
        )
        return 2

    if args.command == "update-data":
        datasets: list[str] = args.datasets
        return _with_store(
            config,
            lambda conn: jobs.fetch(conn, config, date.today(), datasets),
            store.JOB_FILES,
        )
    try:
        content = args.file.read_bytes()
    except OSError as exc:
        print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    importer = _IMPORTERS[args.command]
    files = [store.BENCHMARK_FILE] if args.command == "import-benchmark" else ["ter.csv"]
    return _with_store(config, lambda conn: {args.command: importer(conn, config, content)}, files)


def run_evals(config: AppConfig, modes_text: str, answers: bool, pause: float) -> int:
    """Score search (and answers) on the test set, in a scratch database: the fund data from
    the data store, the factsheet chunks from the test PDFs. Writes a report to evals/results/."""
    import tempfile

    from arbitrage_analyser import evals
    from arbitrage_analyser.config import SEARCH_MODES

    modes = [m.strip() for m in modes_text.split(",") if m.strip()]
    if not modes or set(modes) - set(SEARCH_MODES):
        print(f"--modes: choose from {', '.join(SEARCH_MODES)}", file=sys.stderr)
        return 2
    try:
        eval_set = evals.load_set(config)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Test set error: {exc}", file=sys.stderr)
        return 2
    today = date.today()
    with tempfile.TemporaryDirectory() as scratch, db.connect(Path(scratch) / "evals.db") as conn:
        store.load(conn, config, data_dir(), today)
        notes = evals.build_index(conn, eval_set)
        if set(modes) & {"semantic", "hybrid"}:
            notes += evals.embed_index(conn, config)
        reports = [evals.score_search(conn, config, eval_set, mode, today) for mode in modes]
        if answers:
            evals.score_answers(reports[0], eval_set, evals.chatbot_asker(conn, config), pause)
    for report in reports:
        print("\n".join(evals.summary(report)))
    path = evals.write_report(reports, notes, today)
    print(f"Report: {path}")
    return 0


def _with_store(
    config: AppConfig,
    change: Callable[[sqlite3.Connection], dict[str, ingest.LoadResult]],
    files: Sequence[str],
) -> int:
    """Load the data store, apply `change`, write `files` back.

    Exit codes: 2 when the store cannot be loaded cleanly (nothing is written over it); 1 when a
    step failed or raised a new flag (what was fetched is still saved); 0 otherwise.
    """
    folder = data_dir()
    with db.connect(db_path()) as conn:
        loaded = store.load(conn, config, folder, date.today())
        if not loaded.ok:
            _report("Data store", loaded)
            print("Fix the data store files before running again.", file=sys.stderr)
            return 2
        results = change(conn)
        written = store.save(conn, folder, files)
    code = 0
    for label, result in results.items():
        if _report(label, result) or result.new_flags:
            code = 1
    print(f"Data store {folder}: " + (f"updated {', '.join(written)}" if written else "no changes"))
    return code
