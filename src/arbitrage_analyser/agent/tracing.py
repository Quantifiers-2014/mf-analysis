"""Optional Langfuse tracing, following Langfuse's trace best practices
(https://langfuse.com/docs/observability/best-practices).

On when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (plus LANGFUSE_BASE_URL for the US
cloud or a self-hosted server); otherwise every call here is a no-op.

One chat turn = one trace; one conversation = one session:

    agent "answer-fund-question"   input: the user's question, output: the answer
      ├─ generation "generate-response"   one per model call, OpenAI-format messages,
      │                                    model and token usage (Gemini: via Langfuse's
      │                                    OpenAI integration; Claude: recorded here)
      ├─ tool "compare_funds"              one per tool call: arguments in, result out,
      ├─ generation "generate-response"    sibling of the generation that requested it
      └─ ...

Trace attributes: user id, session id (= conversation), environment, version (= prompt
version), release (= app version), tag "ask-the-analyst". Thumbs up/down become the boolean
score "response_rating" on the trace. Personal identifiers (emails, phone, PAN, Aadhaar
numbers) are masked before anything leaves the machine.

Tracing failures are logged and swallowed: they must never break the chat. Spans are sent in
the background, so a slow or unreachable Langfuse never delays an answer.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib import metadata
from typing import Any

log = logging.getLogger(__name__)

# Observation and score names are used by filters, dashboards and evaluators in Langfuse:
# treat them as an API and do not rename them casually.
TRACE_NAME = "answer-fund-question"
GENERATION_NAME = "generate-response"
FEATURE_TAG = "ask-the-analyst"
FEEDBACK_SCORE = "response_rating"
ENVIRONMENT_ENV = "LANGFUSE_TRACING_ENVIRONMENT"
DEFAULT_ENVIRONMENT = "development"

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN]"),  # Indian PAN, e.g. ABCDE1234F
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR]"),  # 12-digit Aadhaar
    (re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), "[PHONE]"),  # mobile
]

_client_instance: Any | None = None


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def environment() -> str:
    return os.environ.get(ENVIRONMENT_ENV, DEFAULT_ENVIRONMENT)


def mask_sensitive(*, data: Any, **kwargs: Any) -> Any:
    """Langfuse mask hook: replace personal identifiers in any traced text."""
    if isinstance(data, str):
        for pattern, label in _PATTERNS:
            data = pattern.sub(label, data)
        return data
    if isinstance(data, dict):
        return {key: mask_sensitive(data=value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [mask_sensitive(data=item) for item in data]
    return data


def _app_version() -> str | None:
    try:
        return metadata.version("arbitrage-analyser")
    except metadata.PackageNotFoundError:
        return None


def client() -> Any | None:
    """The configured Langfuse client, created once (with masking) before any other Langfuse
    use, so the OpenAI integration reuses the same settings. None when tracing is off."""
    global _client_instance
    if not enabled():
        return None
    if _client_instance is None:
        try:
            from langfuse import Langfuse

            _client_instance = Langfuse(
                mask=mask_sensitive, environment=environment(), release=_app_version()
            )
        except Exception:  # missing package or bad settings
            log.warning("Langfuse tracing is configured but could not start", exc_info=True)
            return None
    return _client_instance


def reset_client() -> None:
    """Forget the client (tests, or after changing Langfuse settings)."""
    global _client_instance
    _client_instance = None


class Generation:
    """Handle for one manually recorded model call (used for providers without a Langfuse
    integration, i.e. Claude). `record` attaches the output and token usage."""

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


def _fail(observation: Any | None, exc: BaseException) -> None:
    if observation is None:
        return
    try:
        observation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
    except Exception:
        log.warning("Langfuse error update failed", exc_info=True)


class TurnTrace:
    def __init__(self, langfuse: Any | None, root: Any | None) -> None:
        self._langfuse = langfuse
        self._root = root
        self.trace_id: str | None = None
        if langfuse is not None:
            try:
                self.trace_id = langfuse.get_current_trace_id()
            except Exception:
                log.warning("Langfuse trace id unavailable", exc_info=True)

    @contextmanager
    def generation(self, model: str, messages: list[dict[str, Any]]) -> Iterator[Generation]:
        """A manually recorded generation. `messages` should be OpenAI-format messages."""
        if self._langfuse is None:
            yield Generation(None)
            return
        try:
            cm = self._langfuse.start_as_current_observation(
                name=GENERATION_NAME, as_type="generation", model=model, input=messages
            )
        except Exception:
            log.warning("Langfuse generation failed", exc_info=True)
            cm = nullcontext(None)
        with cm as obs:
            try:
                yield Generation(obs)
            except Exception as exc:
                _fail(obs, exc)
                raise

    def tool(self, name: str, tool_input: Any, output: Any, is_error: bool) -> None:
        if self._langfuse is None:
            return
        error = output.get("error") if is_error and isinstance(output, dict) else None
        try:
            with self._langfuse.start_as_current_observation(
                name=name,
                as_type="tool",
                input=tool_input,
                output=output,
                level="WARNING" if is_error else "DEFAULT",
                status_message=str(error) if error else None,
            ):
                pass
        except Exception:
            log.warning("Langfuse tool span failed", exc_info=True)

    def finish(self, answer: str, details: dict[str, Any]) -> None:
        if self._root is None:
            return
        try:
            self._root.update(output=answer, metadata=details)
        except Exception:
            log.warning("Langfuse trace update failed", exc_info=True)


@contextmanager
def turn(
    user_id: str, conversation_id: str, question: str, prompt_version: str
) -> Iterator[TurnTrace]:
    """One trace for one chat turn. Errors raised inside are recorded on the trace."""
    langfuse = client()
    if langfuse is None:
        yield TurnTrace(None, None)
        return
    try:
        from langfuse import propagate_attributes

        attrs_cm = propagate_attributes(
            user_id=user_id,
            session_id=conversation_id,
            trace_name=TRACE_NAME,
            version=prompt_version,
            tags=[FEATURE_TAG],
        )
        root_cm = langfuse.start_as_current_observation(
            name=TRACE_NAME, as_type="agent", input=question
        )
    except Exception:
        log.warning("Langfuse trace failed to start", exc_info=True)
        yield TurnTrace(None, None)
        return
    with attrs_cm, root_cm as root:
        trace = TurnTrace(langfuse, root)
        try:
            yield trace
        except Exception as exc:
            _fail(root, exc)
            raise


def score_feedback(trace_id: str | None, rating: int) -> None:
    """Thumbs up (1) or down (0) as the boolean score `response_rating` on the answer's trace.
    A stable score id means changing your mind updates the score instead of adding one."""
    langfuse = client()
    if langfuse is None or not trace_id:
        return
    try:
        langfuse.create_score(
            name=FEEDBACK_SCORE,
            value=rating,
            data_type="BOOLEAN",
            trace_id=trace_id,
            score_id=f"{FEEDBACK_SCORE}-{trace_id}",
        )
    except Exception:
        log.warning("Langfuse score failed", exc_info=True)
