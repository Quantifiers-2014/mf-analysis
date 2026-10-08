"""Provider tests: choosing a provider from .env settings, and the Gemini (OpenAI-compatible)
translation, using a fake client shaped like the openai package's responses."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from arbitrage_analyser import db
from arbitrage_analyser.agent import agent, providers
from arbitrage_analyser.agent.providers import OpenAICompatProvider, openai_tools
from arbitrage_analyser.agent.tools import TOOL_SPECS
from arbitrage_analyser.config import AppConfig
from tests.test_agent import seeded  # noqa: F401  (fixture)

KEYS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_COMPAT_BASE_URL",
    "ARBITRAGE_AGENT_PROVIDER",
    "ARBITRAGE_AGENT_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


# ---------- choosing a provider ----------


def test_nothing_set_asks_for_a_gemini_key() -> None:
    assert providers.selected_provider() is None
    assert "GEMINI_API_KEY" in providers.setup_problems()[0]


def test_gemini_key_selects_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    info = providers.selected_provider()
    assert info is not None and info.name == "gemini"
    assert providers.setup_problems() == []
    provider = providers.make_provider()
    assert provider.model == "gemini-3.8-flash"
    assert str(provider.client.base_url) == providers.GEMINI_BASE_URL  # type: ignore[attr-defined]


def test_explicit_provider_and_model_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("ARBITRAGE_AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("ARBITRAGE_AGENT_MODEL", "claude-x")
    provider = providers.make_provider()
    assert provider.info.name == "anthropic" and provider.model == "claude-x"


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"ARBITRAGE_AGENT_PROVIDER": "chatbot9000"}, "not known"),
        ({"ARBITRAGE_AGENT_PROVIDER": "anthropic"}, "ANTHROPIC_API_KEY"),
        (
            {"ARBITRAGE_AGENT_PROVIDER": "openai_compatible", "OPENAI_COMPAT_API_KEY": "k"},
            "OPENAI_COMPAT_BASE_URL",
        ),
    ],
)
def test_setup_problems_explain_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, str], problem: str
) -> None:
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    assert any(problem in p for p in providers.setup_problems())
    with pytest.raises(RuntimeError):
        providers.make_provider()


# ---------- tool schemas Gemini accepts ----------


def test_openai_tools_use_the_gemini_safe_schema_subset() -> None:
    converted = {t["function"]["name"]: t["function"] for t in openai_tools(TOOL_SPECS)}
    assert set(converted) == {s["name"] for s in TOOL_SPECS}
    assert "parameters" not in converted["get_data_status"]  # no empty objects
    returns = converted["get_returns"]["parameters"]["properties"]["years"]
    assert "minimum" not in returns and "maximum" not in returns
    rolling = converted["get_rolling_return_stats"]["parameters"]["properties"]["years"]
    assert "enum" not in rolling and "One of: 1, 3, 5" in rolling["description"]
    text = json.dumps(converted)
    assert '"minimum":' not in text and '"maximum":' not in text


# ---------- the Gemini loop with a fake client ----------


def _call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    dump = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        "extra_content": {"google": {"thought_signature": "sig-" + call_id}},
    }
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        model_dump=lambda exclude_none=True: dump,
    )


def _reply(content: str | None, calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5),
    )


class FakeOpenAI:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self.replies.pop(0)


def test_gemini_loop_runs_tools_and_returns_signatures(
    seeded: db.sqlite3.Connection,  # noqa: F811
    config: AppConfig,
) -> None:
    client = FakeOpenAI(
        [
            _reply(None, [_call("c1", "compare_funds", '{"category": "Arbitrage"}')]),
            _reply("Alpha leads on 3Y returns."),
        ]
    )
    provider = OpenAICompatProvider(client, "gemini-3.8-flash")
    result = agent.answer(
        provider, seeded, config, [], "Who leads?", user_id="u", conversation_id="c"
    )

    assert result.answer == "Alpha leads on 3Y returns."
    assert [c["name"] for c in result.tool_calls] == ["compare_funds"]
    assert (result.input_tokens, result.output_tokens) == (100, 10)
    first, second = client.requests
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][0]["type"] == "function"
    sent_back = second["messages"]
    assistant = sent_back[-2]
    assert assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig-c1"
    tool_msg = sent_back[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert "Alpha Arbitrage Fund" in tool_msg["content"]


def test_gemini_bad_tool_arguments_are_reported_back(
    seeded: db.sqlite3.Connection,  # noqa: F811
    config: AppConfig,
) -> None:
    client = FakeOpenAI(
        [_reply(None, [_call("c1", "get_returns", "{not json")]), _reply("Sorry, retrying.")]
    )
    result = agent.answer(
        OpenAICompatProvider(client, "m"), seeded, config, [], "q", user_id="u", conversation_id="c"
    )
    assert result.tool_calls[0]["is_error"]
    assert "not valid JSON" in client.requests[1]["messages"][-1]["content"]
