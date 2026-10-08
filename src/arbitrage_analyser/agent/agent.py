"""The agent loop: send the conversation to the AI model, run the tools it asks for, repeat
until it answers. One call to `answer` = one chat turn. Works with any Provider (Gemini, Claude,
other OpenAI-compatible APIs); see providers.py.

    question ─► model ─► wants tools? ─yes─► run tools on the fund DB ─► back to the model
                              │no
                              ▼
                           answer (+ the tool calls behind it, tokens, timing)
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from arbitrage_analyser.agent import tracing
from arbitrage_analyser.agent.prompt import PROMPT_VERSION, build_system_prompt
from arbitrage_analyser.agent.providers import Provider, ToolCall, make_provider
from arbitrage_analyser.agent.store import StoredMessage
from arbitrage_analyser.agent.tools import run_tool
from arbitrage_analyser.config import AppConfig

MAX_STEPS = 8  # model calls per turn; stops a runaway tool loop
# Earlier messages sent with each question. Older ones are dropped for now; Phase 3 replaces
# this with a running summary so long conversations keep their early context.
MAX_HISTORY_MESSAGES = 30

__all__ = ["TurnResult", "answer", "history_messages", "make_provider"]


@dataclass
class TurnResult:
    answer: str
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    trace_id: str | None = None
    stopped_early: bool = False


def history_messages(stored: Sequence[StoredMessage]) -> list[dict[str, Any]]:
    """Earlier turns as API messages: recent ones only, starting with a user message, and
    with any same-role neighbours merged (left behind if an earlier answer failed)."""
    messages: list[dict[str, Any]] = []
    for item in stored[-MAX_HISTORY_MESSAGES:]:
        if messages and messages[-1]["role"] == item.role:
            messages[-1]["content"] += "\n\n" + item.content
        else:
            messages.append({"role": item.role, "content": item.content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def _run(
    fund_conn: sqlite3.Connection, config: AppConfig, call: ToolCall
) -> tuple[dict[str, Any], bool]:
    if call.bad_arguments is not None:
        return {"error": f"Tool arguments were not valid JSON: {call.bad_arguments!r}"}, True
    return run_tool(fund_conn, config, call.name, call.input)


def answer(
    provider: Provider,
    fund_conn: sqlite3.Connection,
    config: AppConfig,
    history: Sequence[StoredMessage],
    question: str,
    *,
    user_id: str,
    conversation_id: str,
) -> TurnResult:
    model = provider.model
    started = time.monotonic()
    system = build_system_prompt(config)
    conversation = history_messages(history)
    if conversation and conversation[-1]["role"] == "user":
        conversation[-1]["content"] += "\n\n" + question
    else:
        conversation.append({"role": "user", "content": question})
    messages = [*provider.first_messages(system), *conversation]
    result = TurnResult(answer="", model=model)

    with tracing.turn(user_id, conversation_id, question, PROMPT_VERSION) as trace:
        result.trace_id = trace.trace_id
        for _ in range(MAX_STEPS):
            with trace.generation(model, messages) as gen:
                step = provider.call(system, messages)
                gen.record(step.log_output, step.input_tokens, step.output_tokens)
            result.input_tokens += step.input_tokens
            result.output_tokens += step.output_tokens
            messages.append(step.assistant_message)

            if not step.tool_calls:
                result.answer = step.text
                if step.truncated:
                    result.answer += "\n\n*(Answer cut short: it reached the length limit.)*"
                break

            outputs = []
            for call in step.tool_calls:
                output, is_error = _run(fund_conn, config, call)
                trace.tool(call.name, call.input, output, is_error)
                result.tool_calls.append(
                    {"name": call.name, "input": call.input, "output": output, "is_error": is_error}
                )
                outputs.append((call, output, is_error))
            messages.extend(provider.tool_results(outputs))
        else:
            result.stopped_early = True
            result.answer = (
                "I wasn't able to finish answering this: it needed more steps than I'm allowed. "
                "Could you narrow the question, for example to fewer funds or one time period?"
            )

        result.latency_ms = int((time.monotonic() - started) * 1000)
        trace.finish(
            result.answer,
            {
                "provider": provider.info.name,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "tool_calls": [c["name"] for c in result.tool_calls],
                "stopped_early": result.stopped_early,
            },
        )
    return result
