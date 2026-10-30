"""Embeddings (vectors of numbers that capture meaning) for factsheet chunks, from the Mistral
embeddings API. Kept now so search by meaning can be switched on later (settings.search_mode).

The data job embeds every chunk that has no embedding for the configured model yet. A chunk is
embedded again only when its text changes (its text_hash). Needs MISTRAL_API_KEY; without it the
step is skipped with a note. Vectors are stored as base64 of float32 values, to keep the file
small (a mistral-embed vector is 1,024 numbers: about 5.5 KB per chunk).
"""

from __future__ import annotations

import base64
import os
import sqlite3
import time
from collections.abc import Callable, Sequence

import numpy as np
import requests

from arbitrage_analyser.config import AppConfig
from arbitrage_analyser.ingest import LoadResult
from arbitrage_analyser.sources import SourceError

KEY_ENV = "MISTRAL_API_KEY"
URL = "https://api.mistral.ai/v1/embeddings"
TIMEOUT_SECONDS = 60
BATCH = 32  # texts per request: 32 chunks of <= 250 tokens stay well under the request limit
PAUSE_SECONDS = 1.5  # between requests: the free tier allows about 1 request per second
RETRIES = 3

# embed(texts, model) -> one vector per text; tests pass a fake.
Embedder = Callable[[Sequence[str], str], list[list[float]]]


def api_key() -> str | None:
    return os.environ.get(KEY_ENV, "").strip() or None


def mistral_embed(texts: Sequence[str], model: str) -> list[list[float]]:
    """Vectors for `texts`, in order. Raises SourceError when Mistral refuses or is unreachable;
    rate limits (HTTP 429) and server errors are retried a few times first."""
    key = api_key()
    if key is None:
        raise SourceError(f"{KEY_ENV} is not set")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        if start:
            time.sleep(PAUSE_SECONDS)
        vectors += _request(key, list(texts[start : start + BATCH]), model)
    return vectors


def _request(key: str, texts: list[str], model: str) -> list[list[float]]:
    for attempt in range(RETRIES):
        try:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "input": texts},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise SourceError(f"could not reach Mistral: {exc}") from exc
        retry = response.status_code == 429 or response.status_code >= 500
        if retry and attempt < RETRIES - 1:
            time.sleep(PAUSE_SECONDS * 2 ** (attempt + 1))
            continue
        if response.status_code != 200:
            hint = {401: " (the key is wrong or expired)", 429: " (rate limit)"}.get(
                response.status_code, ""
            )
            raise SourceError(
                f"Mistral returned HTTP {response.status_code}{hint}: {response.text[:200]}"
            )
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise SourceError(f"Mistral returned {len(data)} vectors for {len(texts)} texts")
        return [list(map(float, d["embedding"])) for d in data]
    raise SourceError("Mistral did not answer")  # not reached: the last attempt returns or raises


def encode(vector: Sequence[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()


def decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


def update(
    conn: sqlite3.Connection, config: AppConfig, embed: Embedder | None = None
) -> LoadResult:
    """Embed the chunks that have no embedding for the configured model; drop embeddings no
    chunk uses any more. Skipped (with a note) when MISTRAL_API_KEY is not set."""
    from arbitrage_analyser import factsheet_search  # imports this module

    result = LoadResult()
    model = config.settings.embedding_model
    stale = conn.execute(
        "DELETE FROM factsheet_embeddings WHERE text_hash NOT IN "
        "(SELECT text_hash FROM factsheet_chunks)"
    ).rowcount
    missing = factsheet_search.chunks_without_embedding(conn, config, model)
    if not missing:
        result.rows = stale
        return result
    if embed is None:
        if api_key() is None:
            result.notes.append(
                f"{len(missing)} factsheet chunk(s) not embedded: {KEY_ENV} is not set"
            )
            return result
        embed = mistral_embed
    try:
        vectors = embed([text for _, text in missing], model)
    except SourceError as exc:
        result.errors.append(f"{len(missing)} factsheet chunk(s) not embedded: {exc}")
        return result
    conn.executemany(
        "INSERT OR REPLACE INTO factsheet_embeddings (text_hash, model, vector) VALUES (?, ?, ?)",
        [(h, model, encode(v)) for (h, _), v in zip(missing, vectors, strict=True)],
    )
    result.rows = len(missing) + stale
    return result
