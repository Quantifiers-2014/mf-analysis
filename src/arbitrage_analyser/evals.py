"""Evals for the chatbot's factsheet search: a fixed set of questions with known answers, read
from real factsheet pages, to measure search (and answers) before and after a change.

python -m arbitrage_analyser eval-search                      # keyword search
python -m arbitrage_analyser eval-search --modes keyword,semantic,hybrid   # needs MISTRAL_API_KEY
python -m arbitrage_analyser eval-search --answers            # also ask the chatbot (model key)

The test set (evals/factsheet_search.toml) lists the factsheet PDFs to search and the
questions. Each question names the fund and data month holding the answer, words the answer
chunk contains (`chunk_has`) and, optionally, words a correct answer contains (`answer_has`).
`not_stated = true` marks a question the factsheet does not answer.

Scores:
- search: the answer chunk is the first result (top 1), or among the first 3 (top 3); for a
  not-stated question, that nothing is returned for its fund.
- answers: the chatbot's answer contains one of `answer_has`; for a not-stated question, that
  it says "not stated".
A report is written to evals/results/.
"""

from __future__ import annotations

import re
import sqlite3
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from arbitrage_analyser import embeddings, factsheet_search
from arbitrage_analyser.config import AppConfig, Fund
from arbitrage_analyser.sources.factsheet_pdf import read_factsheet
from arbitrage_analyser.sources.factsheet_text import read_chunks

ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = ROOT / "evals" / "factsheet_search.toml"
RESULTS_DIR = ROOT / "evals" / "results"
NOT_STATED = ("not stated", "does not state", "doesn't state", "not mentioned", "not available")


class EvalError(ValueError):
    """The test set file is wrong."""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    fund: Fund
    as_of: str
    kind: str
    chunk_has: tuple[str, ...] = ()
    answer_has: tuple[str, ...] = ()
    not_stated: bool = False


@dataclass(frozen=True)
class EvalSet:
    pdfs: tuple[tuple[Fund, Path], ...]
    questions: tuple[Question, ...]


@dataclass
class SearchScore:
    question: Question
    rank: int | None  # 1-based rank of the answer chunk among the results; None = not found
    returned: int  # extracts returned
    top: str  # first extract: fund, month, section
    problem: str = ""  # e.g. the answer chunk is missing from the index

    @property
    def passed(self) -> bool:
        if self.question.not_stated:
            return self.returned == 0
        return self.rank is not None and self.rank <= 3


@dataclass
class AnswerScore:
    question: Question
    answer: str
    passed: bool
    tokens: int
    error: str = ""


@dataclass
class Report:
    mode: str
    search: list[SearchScore] = field(default_factory=list)
    answers: list[AnswerScore] = field(default_factory=list)
    note: str = ""


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).lower()


def _fund(config: AppConfig, name: str, where: str) -> Fund:
    for fund in config.funds:
        if fund.name == name:
            return fund
    raise EvalError(f"{where}: unknown fund {name!r}")


def load_set(config: AppConfig, path: Path = EVAL_FILE) -> EvalSet:
    raw = tomllib.loads(path.read_text())
    pdfs = tuple(
        (_fund(config, p["fund"], f"pdf {i}"), ROOT / p["file"])
        for i, p in enumerate(raw.get("pdfs", []), start=1)
    )
    questions = []
    for q in raw.get("questions", []):
        where = f"question {q.get('id')}"
        question = Question(
            id=q["id"],
            question=q["question"],
            fund=_fund(config, q["fund"], where),
            as_of=q["as_of"],
            kind=q["kind"],
            chunk_has=tuple(q.get("chunk_has", ())),
            answer_has=tuple(q.get("answer_has", ())),
            not_stated=bool(q.get("not_stated", False)),
        )
        if not question.not_stated and not question.chunk_has:
            raise EvalError(f"{where}: needs chunk_has, or not_stated = true")
        questions.append(question)
    if len({q.id for q in questions}) != len(questions):
        raise EvalError("question ids must be unique")
    return EvalSet(pdfs, tuple(questions))


def build_index(conn: sqlite3.Connection, eval_set: EvalSet) -> list[str]:
    """Replace the stored factsheet chunks with chunks of the test PDFs. Returns notes."""
    conn.execute("DELETE FROM factsheet_chunks")
    notes = []
    for fund, path in eval_set.pdfs:
        content = path.read_bytes()
        source = fund.factsheet
        if source is None:
            raise EvalError(f"{fund.name} has no [funds.factsheet] in config")
        values = read_factsheet(content, source.page_title, dict(source.turnover))
        chunks = read_chunks(content, values.page)
        url = f"https://example.invalid/{path.name}"  # a test PDF has no public link
        factsheet_search.store_chunks(conn, fund, values.as_of, values.page, url, chunks)
        notes.append(f"{path.name}: {fund.name}, data as of {values.as_of}, {len(chunks)} chunks")
    return notes


def answer_chunks(conn: sqlite3.Connection, q: Question) -> set[int]:
    """chunk_no of each chunk holding the answer (all of chunk_has)."""
    return {
        int(chunk_no)
        for chunk_no, text in conn.execute(
            "SELECT chunk_no, text FROM factsheet_chunks WHERE amfi_code=? AND as_of=?",
            (q.fund.amfi_code, q.as_of),
        )
        if all(_squash(part) in _squash(text) for part in q.chunk_has)
    }


def says(answer: str, expected: str) -> bool:
    """The answer contains `expected` as whole words (any case; commas in numbers ignored)."""
    pattern = r"(?<![\w.])" + re.escape(_squash(expected)) + r"(?![\w])"
    return re.search(pattern, _squash(answer)) is not None


def score_search(
    conn: sqlite3.Connection,
    config: AppConfig,
    eval_set: EvalSet,
    mode: str,
    today: date,
    embed_query: factsheet_search.QueryEmbedder | None = None,
) -> Report:
    report = Report(mode)
    for q in eval_set.questions:
        found = factsheet_search.search(
            conn, config, q.question, mode=mode, today=today, embed_query=embed_query
        )
        if found.note:
            report.note = found.note
        rank, problem = None, ""
        if not q.not_stated:
            wanted = answer_chunks(conn, q)
            if not wanted:
                problem = "no chunk holds the answer words (chunking split them?)"
            for n, hit in enumerate(found.hits, start=1):
                same = hit.amfi_code == q.fund.amfi_code and hit.as_of == q.as_of
                if same and hit.chunk_no in wanted:
                    rank = n
                    break
        returned = sum(1 for h in found.hits if h.amfi_code == q.fund.amfi_code)
        top = (
            f"{found.hits[0].fund} {found.hits[0].as_of[:7]}: {found.hits[0].section[:40]}"
            if found.hits
            else "(nothing)"
        )
        report.search.append(
            SearchScore(q, rank, returned if q.not_stated else len(found.hits), top, problem)
        )
    return report


# ask(question) -> (answer, tokens); runs the chatbot.
Asker = Callable[[str], tuple[str, int]]


def score_answers(report: Report, eval_set: EvalSet, ask: Asker, pause: float) -> None:
    for i, q in enumerate(eval_set.questions):
        if i:
            time.sleep(pause)  # free tiers allow only a few thousand tokens per minute
        try:
            answer, used = ask(q.question)
        except Exception as exc:  # report and go on: one failed call should not end the run
            report.answers.append(AnswerScore(q, "", False, 0, f"{type(exc).__name__}: {exc}"))
            continue
        if q.not_stated:
            passed = any(says(answer, p) for p in NOT_STATED)
        else:
            passed = any(says(answer, a) for a in q.answer_has) if q.answer_has else True
        report.answers.append(AnswerScore(q, answer, passed, used))


def _rate(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "-"


def summary(report: Report) -> list[str]:
    stated = [s for s in report.search if not s.question.not_stated]
    absent = [s for s in report.search if s.question.not_stated]
    lines = [
        f"Mode: {report.mode}" + (f" ({report.note})" if report.note else ""),
        "Search, answer chunk first: " + _rate(sum(1 for s in stated if s.rank == 1), len(stated)),
        "Search, answer chunk in top 3: "
        + _rate(sum(1 for s in stated if s.rank is not None and s.rank <= 3), len(stated)),
        "Search, nothing returned when not stated: "
        + _rate(sum(1 for s in absent if s.passed), len(absent)),
    ]
    if report.answers:
        lines.append(
            "Answers correct: "
            + _rate(sum(1 for a in report.answers if a.passed), len(report.answers))
        )
    return lines


def _cell(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def write_report(reports: Sequence[Report], notes: Sequence[str], today: date) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = "-".join(r.mode for r in reports)
    path = RESULTS_DIR / f"{today.isoformat()}-{modes}.md"
    out: list[str] = [f"# Factsheet search evals, {today:%d-%b-%Y}", "", "Index:", ""]
    out += [f"- {n}" for n in notes]
    for report in reports:
        out += ["", f"## {report.mode}", ""] + [f"- {line}" for line in summary(report)]
        out += [
            "",
            "| Id | Kind | Question | Rank | Top result | Pass |",
            "|---|---|---|---|---|---|",
        ]
        for s in report.search:
            rank = "-" if s.question.not_stated else (str(s.rank) if s.rank else "miss")
            ok = "yes" if s.passed else "no"
            top = s.top + (f" ({s.problem})" if s.problem else "")
            out.append(
                f"| {s.question.id} | {s.question.kind} | {_cell(s.question.question)} | {rank} "
                f"| {_cell(top, 80)} | {ok} |"
            )
        if report.answers:
            out += ["", "| Id | Expected | Answer | Pass |", "|---|---|---|---|"]
            for a in report.answers:
                expected = (
                    "not stated" if a.question.not_stated else ", ".join(a.question.answer_has)
                )
                shown = a.error or a.answer
                out.append(
                    f"| {a.question.id} | {_cell(expected, 60)} | {_cell(shown)} "
                    f"| {'yes' if a.passed else 'no'} |"
                )
    path.write_text("\n".join(out) + "\n")
    return path


def embed_index(conn: sqlite3.Connection, config: AppConfig) -> list[str]:
    """Embeddings for the test chunks (search by meaning). Returns problems."""
    result = embeddings.update(conn, config)
    return result.errors + result.notes


def chatbot_asker(conn: sqlite3.Connection, config: AppConfig) -> Asker:
    """Ask the chatbot (the configured provider), each question in a new conversation."""
    from arbitrage_analyser.agent import agent  # the agent needs the provider packages

    provider = agent.make_provider()

    def ask(question: str) -> tuple[str, int]:
        result: Any = agent.answer(
            provider, conn, config, [], question, user_id="evals", conversation_id="evals"
        )
        return str(result.answer), int(result.input_tokens + result.output_tokens)

    return ask
