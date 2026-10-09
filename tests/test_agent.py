"""Agent tests. A scripted fake Claude client stands in for the API, so no key is needed.
Gemini/OpenAI-compatible translation is tested in test_providers.py."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from arbitrage_analyser import db, ingest
from arbitrage_analyser.agent import agent, store, tools
from arbitrage_analyser.agent.prompt import build_system_prompt
from arbitrage_analyser.agent.providers import AnthropicProvider
from arbitrage_analyser.config import AppConfig
from tests.conftest import growth_series, history_for
from tests.test_cli_app import APP_PATH, env  # noqa: F401  (env is a fixture)

# ---------- fakes ----------


def text_reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def tool_reply(name: str, tool_input: dict[str, Any], tool_id: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=80, output_tokens=10),
    )


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


# ---------- fixtures ----------


def seed(conn: db.sqlite3.Connection, config: AppConfig) -> None:
    def fetch(code: int):  # type: ignore[no-untyped-def]
        series = growth_series("2019-01-01", "2026-10-06", 0.07, noise=0.00002, seed=code)
        return history_for(code, config.fund_by_code(code).isin, series)

    assert ingest.refresh_nav(conn, config, fetch).ok


@pytest.fixture
def seeded(conn: db.sqlite3.Connection, config: AppConfig) -> Iterator[db.sqlite3.Connection]:
    seed(conn, config)
    yield conn


def ask(
    client: FakeClient,
    conn: db.sqlite3.Connection,
    config: AppConfig,
    question: str = "Which fund did best over 3 years?",
    history: list[store.StoredMessage] | None = None,
) -> agent.TurnResult:
    return agent.answer(
        AnthropicProvider(client, "claude-test"),
        conn,
        config,
        history or [],
        question,
        user_id="u",
        conversation_id="c",
    )


# ---------- tools ----------


def test_every_tool_cites_source_and_date(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    calls = {
        "list_funds": {},
        "compare_funds": {"category": "Arbitrage"},
        "get_returns": {"amfi_codes": [100001, 100002], "years": 3},
        "get_rolling_return_stats": {"amfi_codes": [100001], "years": 1},
        "get_nav_history": {"amfi_code": 100001, "start_date": "2026-09-01"},
        "get_data_status": {},
    }
    assert set(calls) == {spec["name"] for spec in tools.TOOL_SPECS} == set(tools.HANDLERS)
    for name, args in calls.items():
        result, is_error = tools.run_tool(seeded, config, name, args)
        assert not is_error, (name, result)
        assert result["source"] and result["as_of"], name
        json.dumps(result)  # JSON-safe


def test_returns_match_metrics(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    result, _ = tools.run_tool(seeded, config, "get_returns", {"amfi_codes": [100001], "years": 3})
    assert result["rows"][0]["annualised_return_pct"] == pytest.approx(7.0, abs=0.05)
    too_long, _ = tools.run_tool(
        seeded, config, "get_returns", {"amfi_codes": [100001], "years": 15}
    )
    assert too_long["rows"][0]["annualised_return_pct"] is None
    assert "Not enough" in too_long["rows"][0]["reason"]


def test_nav_history_thins_long_ranges(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    result, _ = tools.run_tool(seeded, config, "get_nav_history", {"amfi_code": 100001})
    assert result["frequency"] == "month-end"
    assert len(result["points"]) <= tools.MAX_NAV_POINTS


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("get_returns", {"amfi_codes": [999], "years": 1}, "Unknown AMFI code"),
        ("get_returns", {"amfi_codes": [100001], "years": 1, "end_date": "07/10"}, "YYYY-MM-DD"),
        ("compare_funds", {"category": "Gold"}, "Unknown category"),
        ("get_rolling_return_stats", {"amfi_codes": [100001], "years": 2}, "1, 3 or 5"),
        ("drop_tables", {}, "Unknown tool"),
    ],
)
def test_bad_tool_requests_return_errors(
    seeded: db.sqlite3.Connection, config: AppConfig, name: str, args: dict[str, Any], message: str
) -> None:
    result, is_error = tools.run_tool(seeded, config, name, args)
    assert is_error
    assert message in result["error"]


def test_data_status_reports_empty_and_switched_off(
    seeded: db.sqlite3.Connection, config: AppConfig
) -> None:
    result, _ = tools.run_tool(seeded, config, "get_data_status", {})
    assert result["row_counts"]["benchmark"] == 0
    assert result["row_counts"]["nav"] > 0


# ---------- prompt ----------


def test_prompt_lists_funds_and_scope_rules(config: AppConfig) -> None:
    prompt = build_system_prompt(config)
    assert "Alpha Arbitrage Fund (AMFI code 100001" in prompt
    assert "Out of scope" in prompt
    assert "Sources" in prompt


# ---------- agent loop ----------


def test_agent_runs_tools_then_answers(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    client = FakeClient(
        [tool_reply("compare_funds", {"category": "Arbitrage"}), text_reply("Alpha did best.")]
    )
    result = ask(client, seeded, config)

    assert result.answer == "Alpha did best."
    assert [c["name"] for c in result.tool_calls] == ["compare_funds"]
    assert result.input_tokens == 180 and result.output_tokens == 30
    # The tool result went back to the model on the second call.
    second = client.requests[1]["messages"]
    tool_result = second[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert "Alpha Arbitrage Fund" in tool_result["content"]
    assert client.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_tool_errors_go_back_to_the_model(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    client = FakeClient(
        [tool_reply("get_returns", {"amfi_codes": [1], "years": 1}), text_reply("Unknown fund.")]
    )
    result = ask(client, seeded, config)
    assert result.tool_calls[0]["is_error"]
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_agent_stops_a_runaway_tool_loop(seeded: db.sqlite3.Connection, config: AppConfig) -> None:
    client = FakeClient([tool_reply("list_funds", {})])
    result = ask(client, seeded, config)
    assert result.stopped_early
    assert len(client.requests) == agent.MAX_STEPS
    assert "narrow the question" in result.answer


def test_history_is_sent_with_the_new_question(
    seeded: db.sqlite3.Connection, config: AppConfig
) -> None:
    history = [
        store.StoredMessage(1, "user", "Tell me about Alpha", [], None),
        store.StoredMessage(2, "assistant", "Alpha is an arbitrage fund.", [], None),
    ]
    client = FakeClient([text_reply("Its 1Y return is ...")])
    ask(client, seeded, config, "And its 1-year return?", history)
    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[-1]["content"] == "And its 1-year return?"


def test_history_merges_unanswered_questions_and_starts_with_user() -> None:
    stored = [
        store.StoredMessage(1, "assistant", "orphan", [], None),
        store.StoredMessage(2, "user", "first try", [], None),
        store.StoredMessage(3, "user", "second try", [], None),
    ]
    messages = agent.history_messages(stored)
    assert messages == [{"role": "user", "content": "first try\n\nsecond try"}]


# ---------- storage ----------


def test_store_round_trip_and_feedback(tmp_path: Path) -> None:
    with store.connect(tmp_path / "chat.db") as conn:
        cid = store.create_conversation(conn, "asha", "  Which fund   is cheapest? ")
        store.add_message(conn, cid, "user", "Which fund is cheapest?")
        mid = store.add_message(
            conn, cid, "assistant", "Alpha.", tool_calls=[{"name": "list_funds"}], trace_id="tr1"
        )
        store.set_feedback(conn, mid, 1)
        store.set_feedback(conn, mid, 0)  # changing your mind overwrites

        assert store.list_conversations(conn, "asha")[0]["title"] == "Which fund is cheapest?"
        assert store.list_conversations(conn, "someone else") == []
        messages = store.read_messages(conn, cid)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].tool_calls == [{"name": "list_funds"}] and messages[1].trace_id == "tr1"
        assert store.read_feedback(conn, mid) == 0


# ---------- chat screen ----------


def test_chat_tab_asks_for_setup_without_a_key(env: Path) -> None:  # noqa: F811
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert any("GEMINI_API_KEY" in i.value for i in app.info)


def test_chat_tab_answers_and_saves_the_conversation(
    env: Path,  # noqa: F811
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arbitrage_analyser.config import load_config

    with db.connect(env) as conn:
        seed(conn, load_config(config_file))
    monkeypatch.setenv("ARBITRAGE_AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = FakeClient(
        [tool_reply("compare_funds", {"category": "Arbitrage"}), text_reply("Alpha leads.")]
    )
    monkeypatch.setattr(agent, "make_provider", lambda: AnthropicProvider(client, "claude-test"))

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.chat_input(key="chat_input").set_value("Who leads on 3Y?").run()
    assert not app.exception

    with store.connect(env.with_name("chat.db")) as conn:
        conversation = store.list_conversations(conn, "local")[0]
        messages = store.read_messages(conn, conversation["id"])
    assert [m.content for m in messages] == ["Who leads on 3Y?", "Alpha leads."]
    assert messages[1].tool_calls[0]["name"] == "compare_funds"
    assert any("Alpha leads." in m.value for m in app.markdown)
