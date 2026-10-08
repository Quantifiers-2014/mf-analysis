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


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH))


def db_path() -> Path:
    return Path(os.environ.get(DB_ENV, DEFAULT_DB_PATH))


def load_env_file() -> None:
    """Load secrets (API keys) from the project's .env file, if present. Variables already set
    in the environment win. Needs python-dotenv, installed with the [agent] extras."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(os.environ.get(ENV_FILE_ENV, ENV_FILE)), override=False)
