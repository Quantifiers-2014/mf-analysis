"""Factsheet text for the chatbot: chunking, storing, search, embeddings, the agent's search
before its first call, and the evals."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from arbitrage_analyser import db, embeddings, evals, factsheet_search, factsheets, store
from arbitrage_analyser.agent import agent
from arbitrage_analyser.agent.providers import AnthropicProvider
from arbitrage_analyser.config import AppConfig, ConfigError, load_config
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources import factsheet_text as ft
from arbitrage_analyser.sources.factsheet_pdf import FactsheetValues, read_factsheet
from arbitrage_analyser.sources.factsheet_text import Chunk, read_chunks
from tests.conftest import CONFIG_TEXT, FAKE_CHUNKS, fake_factsheets
from tests.test_agent import FakeClient, text_reply

PDFS = Path(__file__).parent / "data" / "factsheets"
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "funds.toml"
SEP, AUG = date(2026, 9, 30), date(2026, 8, 31)
URL = "https://example.com/alpha/factsheet.pdf"


def _page_chunks(file: str, fund_name: str) -> list[Chunk]:
    config = load_config(REPO_CONFIG)
    fund = next(f for f in config.funds if f.name == fund_name)
    assert fund.factsheet is not None
    content = (PDFS / file).read_bytes()
    values = read_factsheet(content, fund.factsheet.page_title, dict(fund.factsheet.turnover))
    return read_chunks(content, values.page)


def _lines(chunks: Sequence[Chunk]) -> list[str]:
    return [line for c in chunks for line in c.text.split("\n")]


# ---------- chunking ----------


@pytest.mark.parametrize(
    ("file", "fund", "rows"),
    [
        # a label stays on the line of its value (table rows are read across)
        ("sbi_2026-09.pdf", "SBI Arbitrage Opportunities Fund", ["Direct | 1.48% | 0.34%"]),
        ("nippon_2026-09.pdf", "Nippon India Arbitrage Fund",
         ["AUM: | ` 17,215.96 Cr", "a) Equity Turnover | NA", "Direct | 0.33%"]),
        ("tata_2026-09.pdf", "Tata Arbitrage Fund",
         ["Direct plan : | 0.26%", "1 HDFC Bank Ltd. | 7.89%"]),
        ("edelweiss_2026-09.pdf", "Edelweiss Arbitrage Fund", ["Direct Plan: | 0.33%"]),
    ],
)  # fmt: skip
def test_chunks_keep_labels_with_values(file: str, fund: str, rows: list[str]) -> None:
    chunks = _page_chunks(file, fund)
    for row in rows:
        assert row in _lines(chunks)
    assert all(c.tokens <= ft.MAX_TOKENS for c in chunks)


def test_page_columns_are_read_one_after_the_other() -> None:
    """SBI's left column (fund details) is not mixed with the middle one (performance)."""
    text = "\n".join(
        c.text for c in _page_chunks("sbi_2026-09.pdf", "SBI Arbitrage Opportunities Fund")
    )
    objective = text.index("objective of the Scheme is to provide")
    achieved = text.index("will be achieved.")
    assert "1 Year" not in text[objective:achieved]


def test_chart_labels_and_page_numbers_are_dropped() -> None:
    lines = _lines(_page_chunks("nippon_2026-09.pdf", "Nippon India Arbitrage Fund"))
    assert not any(ft._CHART_DIGITS.fullmatch(line) for line in lines)
    assert "7 4 2" not in lines and "120" in lines  # "120" (a page number) stays inside a chunk


def test_a_big_block_is_split_with_overlap() -> None:
    lines = [f"line {i} " + "word " * 20 for i in range(40)]
    chunks = ft._pack([lines], max_tokens=100)
    assert len(chunks) > 2 and all(c.tokens <= 100 for c in chunks)
    for before, after in itertools.pairwise(chunks):
        assert (
            before.text.split("\n")[-ft.OVERLAP_LINES :]
            == after.text.split("\n")[: ft.OVERLAP_LINES]
        )


def test_a_heading_moves_to_the_chunk_of_its_block() -> None:
    first = ["x " * 150]  # 75 tokens
    chunks = ft._pack([first, ["Expense Ratio"], ["Direct | 0.33% " * 10]], max_tokens=100)
    assert chunks[1].section == "Expense Ratio" and not chunks[0].text.endswith("Expense Ratio")


def test_a_page_that_is_not_there_is_an_error() -> None:
    with pytest.raises(SourceError, match="no page 9"):
        read_chunks((PDFS / "sbi_2026-09.pdf").read_bytes(), 9)
    with pytest.raises(SourceError, match="cannot be read"):
        read_chunks(b"%PDF-1.7 fake", 1)


# ---------- storing ----------


def test_store_chunks_replaces_the_month_and_skips_no_change(
    conn: db.sqlite3.Connection, config: AppConfig
) -> None:
    alpha = config.fund_by_code(100001)
    assert factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, FAKE_CHUNKS) == 2
    assert factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, FAKE_CHUNKS) == 0
    assert factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, []) == 0  # kept
    assert factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, FAKE_CHUNKS[:1]) == 1
    assert factsheet_search.store_chunks(conn, alpha, AUG, 3, URL, FAKE_CHUNKS) == 2
    rows = conn.execute("SELECT as_of, COUNT(*) FROM factsheet_chunks GROUP BY as_of").fetchall()
    assert rows == [("2026-08-31", 2), ("2026-09-30", 1)]  # each month kept


def test_fetch_stores_the_fund_page_text(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    values = FactsheetValues(4, SEP, 100.0, 90.0, "Equity 1.00", 0.33)
    download, read = fake_factsheets(values)
    result = factsheets.fetch_factsheets(conn, config, date(2026, 10, 18), download, read)
    assert result.ok and any("2 chunk(s) stored" in n for n in result.notes)
    page, url = conn.execute("SELECT DISTINCT page, source_url FROM factsheet_chunks").fetchone()
    assert page == 4 and url.endswith(".pdf")

    def broken(content: bytes, page: int) -> list[Chunk]:
        raise SourceError("the PDF cannot be read")

    again = factsheets.fetch_factsheets(conn, config, date(2026, 10, 18), download, read, broken)
    assert any("factsheet text not stored for search" in e for e in again.errors)
    assert conn.execute("SELECT COUNT(*) FROM factsheet_chunks").fetchone()[0] == 2  # kept


def test_chunk_and_embedding_files_round_trip(
    conn: db.sqlite3.Connection, config: AppConfig, tmp_path: Path
) -> None:
    alpha = config.fund_by_code(100001)
    factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, FAKE_CHUNKS)
    embeddings.update(conn, config, embed=lambda texts, model: [[0.5, -1.0]] * len(texts))
    folder = tmp_path / "store"
    assert store.save(conn, folder, [store.CHUNKS_FILE, store.EMBEDDINGS_FILE]) == [
        store.CHUNKS_FILE, store.EMBEDDINGS_FILE
    ]  # fmt: skip
    with db.connect(tmp_path / "other.db") as other:
        loaded = store.load(other, config, folder, date(2026, 10, 18))
        assert loaded.ok, loaded.errors
        for table in ("factsheet_chunks", "factsheet_embeddings"):
            query = f"SELECT * FROM {table} ORDER BY 1, 2, 3"
            assert other.execute(query).fetchall() == conn.execute(query).fetchall()


# ---------- search ----------


@pytest.fixture
def indexed(conn: db.sqlite3.Connection, config: AppConfig) -> db.sqlite3.Connection:
    alpha, beta = config.fund_by_code(100001), config.fund_by_code(100002)
    factsheet_search.store_chunks(conn, alpha, SEP, 3, URL, FAKE_CHUNKS)
    factsheet_search.store_chunks(
        conn, alpha, AUG, 3, URL, [Chunk("Expense Ratio", "Expense Ratio\nDirect | 0.31%")]
    )
    factsheet_search.store_chunks(
        conn,
        beta,
        SEP,
        7,
        "https://example.com/beta.pdf",
        [
            Chunk("Fund Managers", "Fund Managers\nB. Manager, managing since 2014"),
            Chunk("Top Holdings", "Top 10 Holdings\nHDFC Bank Ltd. | 7.89%"),
            Chunk("Load Structure", "Load Structure\nExit Load: Nil"),
        ],
    )
    return conn


def test_search_finds_the_fund_and_latest_month(
    indexed: db.sqlite3.Connection, config: AppConfig
) -> None:
    found = factsheet_search.search(indexed, config, "What is Alpha's direct expense ratio?")
    assert found.funds == ["Alpha Arbitrage Fund"] and found.month is None
    assert [h.as_of for h in found.hits] == ["2026-09-30"]
    assert "0.33%" in found.hits[0].text and found.hits[0].link == URL + "#page=3"


def test_search_by_month_and_by_synonym(indexed: db.sqlite3.Connection, config: AppConfig) -> None:
    august = factsheet_search.search(indexed, config, "Alpha expense ratio in August?")
    assert august.month == "2026-08" and "0.31%" in august.hits[0].text
    docs = [c.text for c in FAKE_CHUNKS]
    hedge = factsheet_search.keyword_scores("Does it hedge?", docs)  # hedge ~ derivatives
    assert hedge[0] > 0 and hedge[1] == 0


def test_search_returns_nothing_below_the_threshold(
    indexed: db.sqlite3.Connection, config: AppConfig
) -> None:
    found = factsheet_search.search(indexed, config, "Who is Beta's auditor?")
    assert found.funds == ["Beta Arbitrage Fund"] and found.hits == []


def test_fund_names_do_not_count_as_matches(
    indexed: db.sqlite3.Connection, config: AppConfig
) -> None:
    scores = factsheet_search.keyword_scores(
        "Beta Arbitrage Fund", ["Beta Arbitrage Fund"], factsheet_search.name_words(config)
    )
    assert scores == [0.0]


def test_search_caps_results_and_tokens(indexed: db.sqlite3.Connection, config: AppConfig) -> None:
    found = factsheet_search.search(indexed, config, "manager holdings exit load", max_results=2)
    assert len(found.hits) == 2
    tight = factsheet_search.search(indexed, config, "manager holdings exit load", max_tokens=12)
    assert sum(ft.estimate_tokens(h.text) for h in tight.hits) <= 12


def test_month_in_questions() -> None:
    stored = ["2025-08", "2026-08", "2026-09"]
    today = date(2026, 10, 30)
    assert factsheet_search.month_in("what did it say in August", stored, today) == "2026-08"
    assert factsheet_search.month_in("as of 30-Sep-2026", stored, today) == "2026-09"
    assert factsheet_search.month_in("market risk", stored, today) is None
    assert factsheet_search.month_in("may it hedge?", stored, today) is None
    assert factsheet_search.month_in("in December", stored, today) == "2025-12"


def _with_vectors(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    """Fake embeddings: 'Expense' chunks point one way, the others another."""

    def embed(texts: Sequence[str], model: str) -> list[list[float]]:
        return [[1.0, 0.0] if "Expense" in t else [0.0, 1.0] for t in texts]

    assert embeddings.update(conn, config, embed=embed).ok


def test_semantic_and_hybrid_search(indexed: db.sqlite3.Connection, config: AppConfig) -> None:
    _with_vectors(indexed, config)
    costs = factsheet_search.search(
        indexed, config, "what does Alpha charge", mode="semantic",
        embed_query=lambda text, model: [1.0, 0.1],
    )  # fmt: skip
    assert costs.mode == "semantic" and costs.hits[0].section == "Expense Ratio"
    hybrid = factsheet_search.search(
        indexed, config, "Alpha derivatives", mode="hybrid",
        embed_query=lambda text, model: [1.0, 0.0],
    )  # fmt: skip
    assert {h.section for h in hybrid.hits} == {"Expense Ratio", "Investment Objective"}


def test_semantic_falls_back_to_keyword(indexed: db.sqlite3.Connection, config: AppConfig) -> None:
    def down(text: str, model: str) -> list[float]:
        raise SourceError("could not reach Mistral")

    _with_vectors(indexed, config)
    found = factsheet_search.search(
        indexed, config, "Alpha derivatives", mode="semantic", embed_query=down
    )
    assert found.mode == "keyword" and "unavailable" in (found.note or "") and found.hits
    indexed.execute("DELETE FROM factsheet_embeddings")
    missing = factsheet_search.search(
        indexed, config, "Alpha derivatives", mode="hybrid", embed_query=lambda t, m: [1.0, 0.0]
    )
    assert missing.mode == "keyword" and "no embedding" in (missing.note or "")


# ---------- embeddings ----------


def test_embeddings_need_a_key_and_catch_up(
    indexed: db.sqlite3.Connection, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(embeddings.KEY_ENV, raising=False)
    skipped = embeddings.update(indexed, config)
    assert skipped.ok and "6 factsheet chunk(s) not embedded" in skipped.notes[0]
    calls: list[int] = []

    def embed(texts: Sequence[str], model: str) -> list[list[float]]:
        calls.append(len(texts))
        return [[0.1, 0.2, 0.3]] * len(texts)

    assert embeddings.update(indexed, config, embed=embed).rows == 6
    assert embeddings.update(indexed, config, embed=embed).rows == 0 and calls == [6]
    # a changed chunk is embedded again; the old vector is dropped
    alpha = config.fund_by_code(100001)
    factsheet_search.store_chunks(indexed, alpha, SEP, 3, URL, [Chunk("New", "New text")])
    assert embeddings.update(indexed, config, embed=embed).rows == 3  # 1 new + 2 dropped
    assert indexed.execute("SELECT COUNT(*) FROM factsheet_embeddings").fetchone()[0] == 5


def test_embedding_errors_are_reported(indexed: db.sqlite3.Connection, config: AppConfig) -> None:
    def down(texts: Sequence[str], model: str) -> list[list[float]]:
        raise SourceError("Mistral returned HTTP 401 (the key is wrong or expired)")

    result = embeddings.update(indexed, config, embed=down)
    assert not result.ok and "HTTP 401" in result.errors[0]


def test_vectors_round_trip() -> None:
    vector = [0.25, -1.5, 3.0]
    assert embeddings.decode(embeddings.encode(vector)).tolist() == vector


def test_mistral_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, object]] = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, n: int) -> None:
            self.n = n

        def json(self) -> dict[str, object]:
            return {
                "data": [{"index": i, "embedding": [float(i)]} for i in reversed(range(self.n))]
            }

    def post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> Response:
        sent.append(json)
        return Response(len(json["input"]))  # type: ignore[arg-type]

    monkeypatch.setenv(embeddings.KEY_ENV, "k")
    monkeypatch.setattr(embeddings.requests, "post", post)
    monkeypatch.setattr(embeddings, "PAUSE_SECONDS", 0)
    vectors = embeddings.mistral_embed([f"t{i}" for i in range(embeddings.BATCH + 1)], "m")
    assert len(sent) == 2 and sent[0]["model"] == "m"
    assert vectors[:2] == [[0.0], [1.0]]  # in input order


# ---------- agent ----------


def test_agent_sends_extracts_with_the_question(
    indexed: db.sqlite3.Connection, config: AppConfig
) -> None:
    client = FakeClient([text_reply("Alpha's Direct BER is 0.33%.")])
    result = agent.answer(
        AnthropicProvider(client, "claude-test"), indexed, config, [],
        "What is Alpha's direct expense ratio?", user_id="u", conversation_id="c",
    )  # fmt: skip
    sent = client.requests[0]["messages"][-1]["content"]
    assert "<extract 1> Alpha Arbitrage Fund, factsheet data as of 30-Sep-2026, page 3" in sent
    assert "Direct | 0.33%" in sent and URL + "#page=3" in sent
    assert result.tool_calls[0]["name"] == "search_factsheets"
    assert result.tool_calls[0]["output"]["extracts"][0]["as_of"] == "2026-09-30"


def test_agent_adds_nothing_when_no_extract_matches(
    indexed: db.sqlite3.Connection, config: AppConfig
) -> None:
    client = FakeClient([text_reply("ok")])
    agent.answer(
        AnthropicProvider(client, "claude-test"), indexed, config, [],
        "Which fund did best over 3 years?", user_id="u", conversation_id="c",
    )  # fmt: skip
    assert client.requests[0]["messages"][-1]["content"] == "Which fund did best over 3 years?"


# ---------- config and evals ----------


def test_search_settings(tmp_path: Path) -> None:
    path = tmp_path / "funds.toml"
    path.write_text(CONFIG_TEXT)
    assert load_config(path).settings.search_mode == "keyword"
    text = CONFIG_TEXT.replace(
        "[settings.features]", '[settings.factsheet_search]\nmode = "hybrid"\n\n[settings.features]'
    )
    path.write_text(text)
    assert load_config(path).settings.search_mode == "hybrid"
    path.write_text(text.replace('"hybrid"', '"magic"'))
    with pytest.raises(ConfigError, match="mode must be one of"):
        load_config(path)


def test_shipped_eval_set_passes_with_keyword_search(conn: db.sqlite3.Connection) -> None:
    """Guards search quality: every answer chunk of the test set is in the top 3."""
    config = load_config(REPO_CONFIG)
    eval_set = evals.load_set(config)
    assert len(eval_set.questions) >= 20
    evals.build_index(conn, eval_set)
    report = evals.score_search(conn, config, eval_set, "keyword", date(2026, 10, 30))
    stated = [s for s in report.search if not s.question.not_stated]
    assert not [s.question.id for s in stated if not s.passed]
    assert all(not s.problem for s in report.search)


def test_answer_scoring() -> None:
    assert evals.says("Turnover is NA (not applicable).", "NA")
    assert not evals.says("The name is long", "NA")
    assert evals.says("AUM was Rs 35,576.34 crore", "35576.34")
    assert not evals.says("0.335%", "0.33")
