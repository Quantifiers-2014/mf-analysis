"""Where the config file and database live. Override with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from arbitrage_analyser.config import DEFAULT_CONFIG_PATH
from arbitrage_analyser.db import DEFAULT_DB_PATH

CONFIG_ENV = "ARBITRAGE_CONFIG"
DB_ENV = "ARBITRAGE_DB"


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH))


def db_path() -> Path:
    return Path(os.environ.get(DB_ENV, DEFAULT_DB_PATH))
