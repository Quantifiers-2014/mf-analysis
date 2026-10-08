"""Optional Langfuse tracing. On when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(plus LANGFUSE_BASE_URL for a self-hosted or non-EU server); otherwise every call is a no-op.

One chat turn = one Langfuse trace, grouped into a session per conversation:
    trace "chat-turn"  (input: question, output: answer, user and session ids)
      ├─ generation "claude"   one per model call, with tokens
      ├─ tool "compare_funds"  one per tool call, with input and output
      └─ ...
Tracing failures are logged and swallowed: they must never break the chat.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _client() -> Any | None:
    if not enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:  # missing package or bad settings
        log.warning("Langfuse tracing is configured but could not start", exc_info=True)
        return None


class Generation:
    """Handle for one model call; `record` attaches the output and token usage."""

    def __init__(self, observation: Any | None) -> None:
        self._obs = observation

    def record(self, output: Any, input_tokens: int, output_tokens: int) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(
                output=output, usage_details={"input": input_tokens, "output": output_tokens}
            )
        except Exception:
            log.warning("Langfuse update failed", exc_info=True)


class TurnTrace:
    def __init__(self, client: Any | None, root: Any | None) -> None:
        self._client = client
        self._root = root
        self.trace_id: str | None = None
        if client is not None:
            try:
                self.trace_id = client.get_current_trace_id()
            except Exception:
                log.warning("Langfuse trace id unavailable", exc_info=True)

    @contextmanager
    def generation(self, model: str, messages: list[dict[str, Any]]) -> Iterator[Generation]:
        if self._client is None:
            yield Generation(None)
            return
        try:
            cm = self._client.start_as_current_observation(
                name="claude", as_type="generation", model=model, input=messages
            )
        except Exception:
            log.warning("Langfuse generation failed", exc_info=True)
            cm = nullcontext(None)
        with cm as obs:
            yield Generation(obs)

    def tool(self, name: str, tool_input: Any, output: Any, is_error: bool) -> None:
        if self._client is None:
            return
        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type="tool",
                input=tool_input,
                output=output,
                level="WARNING" if is_error else "DEFAULT",
            ):
                pass
        except Exception:
            log.warning("Langfuse tool span failed", exc_info=True)

    def finish(self, answer: str, metadata: dict[str, Any]) -> None:
        if self._root is None:
            return
        try:
            self._root.update(output=answer, metadata=metadata)
        except Exception:
            log.warning("Langfuse trace update failed", exc_info=True)


@contextmanager
def turn(
    user_id: str, conversation_id: str, question: str, prompt_version: str
) -> Iterator[TurnTrace]:
    client = _client()
    if client is None:
        yield TurnTrace(None, None)
        return
    try:
        from langfuse import propagate_attributes

        root_cm = client.start_as_current_observation(
            name="chat-turn", as_type="agent", input=question
        )
        attrs_cm = propagate_attributes(
            user_id=user_id,
            session_id=conversation_id,
            trace_name="chat-turn",
            version=prompt_version,
            tags=["fund-agent"],
        )
    except Exception:
        log.warning("Langfuse trace failed to start", exc_info=True)
        yield TurnTrace(None, None)
        return
    # Spans are sent in the background, so a slow or unreachable Langfuse never delays answers.
    with root_cm as root, attrs_cm:
        yield TurnTrace(client, root)


def score_feedback(trace_id: str | None, rating: int) -> None:
    """Send a thumbs up (1) or down (0) to Langfuse as a score on the answer's trace."""
    client = _client()
    if client is None or not trace_id:
        return
    try:
        client.create_score(
            name="user_feedback", value=rating, trace_id=trace_id, data_type="BOOLEAN"
        )
    except Exception:
        log.warning("Langfuse score failed", exc_info=True)
