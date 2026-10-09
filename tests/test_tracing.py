"""Langfuse tracing tests. Spans are captured in memory with the real Langfuse SDK and OpenAI
integration (model replies come from a mock HTTP transport), so nothing is sent anywhere."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langfuse")

import httpx
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from arbitrage_analyser import db
from arbitrage_analyser.agent import agent, providers, tracing
from arbitrage_analyser.config import AppConfig
from tests.test_agent import FakeClient, seeded, text_reply, tool_reply  # noqa: F401

# Langfuse keeps one span pipeline per public key for the whole process, so all tests share
# one in-memory exporter and client; each test starts from an empty exporter.
_EXPORTER = InMemorySpanExporter()
_LANGFUSE: Langfuse | None = None


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    global _LANGFUSE
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    if _LANGFUSE is None:
        _LANGFUSE = Langfuse(
            mask=tracing.mask_sensitive, environment="test", span_exporter=_EXPORTER
        )
    _LANGFUSE.flush()
    _EXPORTER.clear()
    tracing._client_instance = _LANGFUSE
    yield _EXPORTER
    tracing.reset_client()


def _tree(exporter: InMemorySpanExporter) -> list[tuple[str, str | None, dict[str, Any]]]:
    tracing._client_instance.flush()  # type: ignore[union-attr]
    finished = sorted(exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    names = {s.context.span_id: s.name for s in finished}
    return [
        (s.name, names.get(s.parent.span_id) if s.parent else None, dict(s.attributes or {}))
        for s in finished
    ]


def _gemini_client(replies: list[dict[str, Any]], seen: list[dict[str, Any]]) -> Any:
    from langfuse.openai import OpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        message = replies[len(seen) - 1]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    return OpenAI(
        api_key="k",
        base_url=providers.GEMINI_BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_gemini_turn_trace_follows_best_practices(
    spans: InMemorySpanExporter,
    seeded: db.sqlite3.Connection,  # noqa: F811
    config: AppConfig,
) -> None:
    seen: list[dict[str, Any]] = []
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "compare_funds", "arguments": json.dumps({"category": "Arbitrage"})},
    }
    client = _gemini_client(
        [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "Alpha leads."},
        ],
        seen,
    )
    provider = providers.OpenAICompatProvider(client, "gemini-3.8-flash", traced=True)
    result = agent.answer(
        provider,
        seeded,
        config,
        [],
        "Who leads? Mail me at a@b.com",
        user_id="asha",
        conversation_id="conv-9",
    )

    tree = _tree(spans)
    assert [(name, parent) for name, parent, _ in tree] == [
        ("answer-fund-question", None),
        ("generate-response", "answer-fund-question"),
        ("compare_funds", "answer-fund-question"),  # sibling of the generation, not a child
        ("generate-response", "answer-fund-question"),
    ]
    root, gen, tool, _ = (attrs for _, _, attrs in tree)
    assert root["langfuse.observation.type"] == "agent"
    assert root["langfuse.observation.input"] == "Who leads? Mail me at [EMAIL]"  # masked
    assert root["langfuse.observation.output"] == "Alpha leads."
    assert root["session.id"] == "conv-9" and root["user.id"] == "asha"
    assert root["langfuse.environment"] == "test"
    assert root["langfuse.trace.tags"] == ("ask-the-analyst",)
    assert root["langfuse.observation.metadata.provider"] == "gemini"
    assert gen["langfuse.observation.type"] == "generation"
    assert gen["langfuse.observation.model.name"] == "gemini-3.8-flash"
    assert "prompt_tokens" in gen["langfuse.observation.usage_details"]
    assert "[EMAIL]" in gen["langfuse.observation.input"]
    assert tool["langfuse.observation.type"] == "tool"
    assert "Alpha Arbitrage Fund" in tool["langfuse.observation.output"]
    assert result.trace_id == format(spans.get_finished_spans()[0].context.trace_id, "032x")
    assert all("name" not in request for request in seen)  # Langfuse args not sent to Gemini


def _claude(replies: list[Any]) -> providers.AnthropicProvider:
    return providers.AnthropicProvider(FakeClient(replies), "claude-test")


def test_claude_generations_use_openai_message_format(
    spans: InMemorySpanExporter,
    seeded: db.sqlite3.Connection,  # noqa: F811
    config: AppConfig,
) -> None:
    provider = _claude(
        [tool_reply("get_returns", {"amfi_codes": [1], "years": 1}), text_reply("Unknown fund.")]
    )
    agent.answer(provider, seeded, config, [], "q", user_id="u", conversation_id="c")

    tree = _tree(spans)
    gens = [attrs for name, _, attrs in tree if name == "generate-response"]
    tool = next(attrs for name, _, attrs in tree if name == "get_returns")
    first_in = json.loads(gens[0]["langfuse.observation.input"])
    assert first_in[0]["role"] == "system" and first_in[-1] == {"role": "user", "content": "q"}
    first_out = json.loads(gens[0]["langfuse.observation.output"])
    assert first_out["tool_calls"][0]["function"]["name"] == "get_returns"
    assert isinstance(first_out["tool_calls"][0]["function"]["arguments"], str)
    second_in = json.loads(gens[1]["langfuse.observation.input"])
    assert second_in[-1]["role"] == "tool" and second_in[-1]["tool_call_id"] == "t1"
    assert gens[0]["langfuse.observation.model.name"] == "claude-test"
    assert tool["langfuse.observation.level"] == "WARNING"
    assert "Unknown AMFI code" in tool["langfuse.observation.status_message"]


def test_failed_model_call_marks_trace_as_error(
    spans: InMemorySpanExporter,
    seeded: db.sqlite3.Connection,  # noqa: F811
    config: AppConfig,
) -> None:
    class Broken:
        messages = SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("429")))

    provider = providers.AnthropicProvider(Broken(), "claude-test")
    with pytest.raises(RuntimeError):
        agent.answer(provider, seeded, config, [], "q", user_id="u", conversation_id="c")
    tree = _tree(spans)
    root = next(attrs for name, parent, attrs in tree if parent is None)
    assert root["langfuse.observation.level"] == "ERROR"
    assert "429" in root["langfuse.observation.status_message"]


def test_feedback_is_a_boolean_score_with_stable_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tracing, "client", lambda: SimpleNamespace(create_score=lambda **kw: calls.append(kw))
    )
    tracing.score_feedback("abc123", 0)
    tracing.score_feedback(None, 1)  # no trace: nothing sent
    assert calls == [
        {
            "name": "response_rating",
            "value": 0,
            "data_type": "BOOLEAN",
            "trace_id": "abc123",
            "score_id": "response_rating-abc123",
        }
    ]


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("mail a.b+x@y.co.in", "mail [EMAIL]"),
        ("PAN ABCDE1234F", "PAN [PAN]"),
        ("aadhaar 1234 5678 9012", "aadhaar [AADHAAR]"),
        ("call +91 98765 43210 or 9876543210", "call [PHONE] or [PHONE]"),
        # fund data must stay intact
        ("INF204K01XZ7 code 119771 on 2026-10-07: 43.4513, 1670 windows", None),
    ],
)
def test_masking(text: str, masked: str | None) -> None:
    assert tracing.mask_sensitive(data={"k": [text]}) == {"k": [masked or text]}


def test_tracing_off_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing.reset_client()
    assert tracing.client() is None
    with tracing.turn("u", "c", "q", "v1") as trace:
        assert trace.trace_id is None


def test_gemini_uses_langfuse_openai_integration_when_tracing_is_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    for name in ("ARBITRAGE_AGENT_PROVIDER", "ANTHROPIC_API_KEY", "ARBITRAGE_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    provider = providers.make_provider()
    assert provider.auto_traced
    assert type(provider.client).__module__.startswith(("langfuse", "openai"))  # type: ignore[attr-defined]
    import langfuse.openai

    assert isinstance(provider.client, langfuse.openai.OpenAI)  # type: ignore[attr-defined]
