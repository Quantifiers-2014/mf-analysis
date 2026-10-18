"""Where the config file and database live. Override with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from arbitrage_analyser.config import DEFAULT_CONFIG_PATH
from arbitrage_analyser.db import DEFAULT_DB_PATH

CONFIG_ENV = "ARBITRAGE_CONFIG"
DB_ENV = "ARBITRAGE_DB"
ENV_FILE_ENV = "ARBITRAGE_ENV_FILE"
ENV_FILE = DEFAULT_CONFIG_PATH.parents[1] / ".env"
DATA_DIR_ENV = "ARBITRAGE_DATA_DIR"
# The data store: CSV files kept in the repository (NAV, TER, benchmark, factsheet values). The
# daily GitHub job writes them; the app loads them into the database on start.
DEFAULT_DATA_DIR = DEFAULT_CONFIG_PATH.parents[1] / "published"


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH))


def db_path() -> Path:
    return Path(os.environ.get(DB_ENV, DEFAULT_DB_PATH))


def data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


def load_env_file() -> None:
    """Load secrets (API keys) from the project's .env file, if present. Variables already set
    in the environment win. Needs python-dotenv (a project dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(os.environ.get(ENV_FILE_ENV, ENV_FILE)), override=False)
