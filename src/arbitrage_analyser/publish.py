"""Save a data store file to GitHub from the app (benchmark uploads, reviewed flags).

The app on Streamlit Cloud cannot keep files between restarts, so a change made in the app is
committed to the repository with the GitHub contents API. It needs a token in the GITHUB_TOKEN
environment variable (a Streamlit secret on Cloud): a fine-grained token for this repository
only, with "Contents: Read and write". Without a token the file is only saved locally.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

from arbitrage_analyser import store
from arbitrage_analyser.config import Settings
from arbitrage_analyser.sources import SourceError

Notice = tuple[str, str]  # (level: "success" / "warning" / "error", text) for the app
TOKEN_ENV = "GITHUB_TOKEN"
API = "https://api.github.com"
TIMEOUT_SECONDS = 30
STORE_DIR = "published"  # the data store folder, relative to the repository root


# Content this process committed, per repository file: the base of the next save until the app
# restarts with the new version (the local checkout is left as loaded).
_committed: dict[str, bytes] = {}


@dataclass(frozen=True)
class Repository:
    name: str  # owner/name
    branch: str
    token: str


def repository(settings: Settings) -> Repository | None:
    """The repository to save to, or None when no token or repository is set."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token or not settings.github_repository:
        return None
    return Repository(settings.github_repository, settings.github_branch, token)


def actions_url(settings: Settings, workflow: str = "refresh-data.yml") -> str | None:
    """Page of the data refresh job, where it can be run by hand."""
    if not settings.github_repository:
        return None
    return f"https://github.com/{settings.github_repository}/actions/workflows/{workflow}"


def blob_sha(content: bytes) -> str:
    """Git's id of a file's content (what GitHub calls the file's sha)."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def save_file(
    repo: Repository,
    file: str,
    content: bytes,
    message: str,
    based_on: bytes | None,
    session: requests.Session | None = None,
) -> bool:
    """Commit `content` as published/<file>, replacing the version `based_on` (the file as this
    app loaded it; None for a new file). GitHub refuses when the file changed there meanwhile,
    so a newer version is never overwritten. Returns False when nothing changed. Raises
    SourceError when GitHub refuses or cannot be reached."""
    if based_on == content:
        return False
    http = session or requests.Session()
    url = f"{API}/repos/{repo.name}/contents/{STORE_DIR}/{file}"
    headers = {
        "Authorization": f"Bearer {repo.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "message": message,
        "content": base64.b64encode(content).decode(),
        "branch": repo.branch,
    } | ({"sha": blob_sha(based_on)} if based_on is not None else {})
    try:
        saved = http.put(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise SourceError(f"could not reach GitHub: {exc}") from exc
    if saved.status_code not in (200, 201):
        raise SourceError(_refusal(saved))
    return True


def _refusal(response: requests.Response) -> str:
    try:
        detail = str(response.json().get("message", ""))
    except ValueError:
        detail = response.text[:200]
    hint = {
        401: "the token is wrong or expired",
        403: "the token lacks 'Contents: Read and write' for this repository",
        404: "the repository or branch is not found, or the token cannot see it",
        409: "the file changed on GitHub since the app loaded it; wait for the app to restart "
        "with the new version, then try again",
        422: "the file changed on GitHub since the app loaded it; wait for the app to restart "
        "with the new version, then try again",
    }.get(response.status_code, "")
    return f"GitHub did not save the file (HTTP {response.status_code}: {detail}) {hint}".strip()


def save_store_files(
    conn: sqlite3.Connection,
    settings: Settings,
    folder: Path,
    files: Sequence[str],
    message: str,
    damaged: Iterable[str] = (),
) -> list[Notice]:
    """Save store files from the database: to GitHub when a token is set (the local copy then
    stays the version the app loaded, so the app's checkout matches the repository), else to
    `folder` on this machine. A file that did not load cleanly is not written over.
    Returns what to tell the user."""
    blocked = [f for f in files if f in set(damaged)]
    notices: list[Notice] = [
        (
            "error",
            f"{f} did not load cleanly (see the warning above); fix it in the repository "
            "first, so its rows are not lost",
        )
        for f in blocked
    ]
    files = [f for f in files if f not in blocked]
    repo = repository(settings)
    if repo is None:
        written = store.save(conn, folder, files)
        if written:
            notices.append(
                (
                    "warning",
                    f"Saved {', '.join(written)} on this machine only (no {TOKEN_ENV} set): "
                    "commit and push it, or set the token (README), or it is lost when the app "
                    "restarts.",
                )
            )
        return notices
    for file in files:
        local = folder / file
        key = f"{repo.name}@{repo.branch}:{file}"
        base = _committed.get(key, local.read_bytes() if local.exists() else None)
        content = store.render(conn, file)
        try:
            changed = save_file(repo, file, content, message, base)
        except SourceError as exc:
            notices.append(("error", f"{file} not saved to GitHub: {exc}"))
            continue
        _committed[key] = content
        if changed:
            notices.append(
                (
                    "success",
                    f"Saved {file} to GitHub ({repo.name}, {repo.branch}); the app "
                    "restarts with it in a minute.",
                )
            )
    return notices
