"""AI model providers. The agent loop talks to a Provider; each Provider translates to one API.

- "gemini":    Google Gemini through its OpenAI-compatible endpoint (free tier available).
- "anthropic": Claude through the Anthropic API.
- "openai_compatible": any other OpenAI-compatible API (Groq, OpenRouter, Ollama, OpenAI),
                       set with OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY.

Pick one with ARBITRAGE_AGENT_PROVIDER in .env. If it is not set, the first provider with a key
wins: GEMINI_API_KEY, then ANTHROPIC_API_KEY. ARBITRAGE_AGENT_MODEL overrides the model.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

PROVIDER_ENV = "ARBITRAGE_AGENT_PROVIDER"
MODEL_ENV = "ARBITRAGE_AGENT_MODEL"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    key_env: str
    default_model: str
    package: str  # Python package the provider needs


PROVIDERS = {
    "gemini": ProviderInfo(
        "gemini", "Google Gemini", "GEMINI_API_KEY", "gemini-3.8-flash", "openai"
    ),
    "anthropic": ProviderInfo(
        "anthropic", "Claude", "ANTHROPIC_API_KEY", "claude-sonnet-5-5", "anthropic"
    ),
    "openai_compatible": ProviderInfo(
        "openai_compatible", "OpenAI-compatible API", "OPENAI_COMPAT_API_KEY", "", "openai"
    ),
}


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    bad_arguments: str | None = None  # set when the model sent arguments that are not JSON


@dataclass
class Step:
    """One model reply, in a provider-neutral shape."""

    text: str
    tool_calls: list[ToolCall]
    truncated: bool
    input_tokens: int
    output_tokens: int
    assistant_message: dict[str, Any]  # the reply in the provider's own format, to send back
    log_output: Any = field(default=None)  # what Langfuse shows as the model's output


class Provider(Protocol):
    info: ProviderInfo
    model: str

    def first_messages(self, system: str) -> list[dict[str, Any]]: ...

    def call(self, system: str, messages: list[dict[str, Any]]) -> Step: ...

    def tool_results(
        self, results: list[tuple[ToolCall, dict[str, Any], bool]]
    ) -> list[dict[str, Any]]: ...


def selected_provider() -> ProviderInfo | None:
    chosen = os.environ.get(PROVIDER_ENV, "").strip().lower()
    if chosen:
        return PROVIDERS.get(chosen)
    for name in ("gemini", "anthropic"):
        if os.environ.get(PROVIDERS[name].key_env):
            return PROVIDERS[name]
    return None


def setup_problems() -> list[str]:
    """What is missing before the chat can run, in plain words. Empty list = ready."""
    chosen = os.environ.get(PROVIDER_ENV, "").strip().lower()
    if chosen and chosen not in PROVIDERS:
        return [f"`{PROVIDER_ENV}={chosen}` is not known. Use one of: {', '.join(PROVIDERS)}."]
    info = selected_provider()
    if info is None:
        return [
            "No AI model key found. Copy `.env.example` to `.env` in the project folder, put "
            "your Gemini key after `GEMINI_API_KEY=`, then restart the app."
        ]
    problems = []
    if importlib.util.find_spec(info.package) is None:
        problems.append('The agent packages are not installed. Run: `pip install -e ".[agent]"`')
    if not os.environ.get(info.key_env):
        problems.append(f"{info.label} is selected but `{info.key_env}` is empty in `.env`.")
    if info.name == "openai_compatible":
        if not os.environ.get("OPENAI_COMPAT_BASE_URL"):
            problems.append("Set `OPENAI_COMPAT_BASE_URL` in `.env` for the OpenAI-compatible API.")
        if not os.environ.get(MODEL_ENV):
            problems.append("Set `ARBITRAGE_AGENT_MODEL` in `.env` for the OpenAI-compatible API.")
    return problems


def make_provider() -> Provider:
    info = selected_provider()
    if info is None or setup_problems():
        raise RuntimeError("The AI model is not set up: " + " ".join(setup_problems()))
    model = os.environ.get(MODEL_ENV) or info.default_model
    if info.name == "anthropic":
        import anthropic

        return AnthropicProvider(anthropic.Anthropic(), model)
    import openai

    base_url = GEMINI_BASE_URL if info.name == "gemini" else os.environ["OPENAI_COMPAT_BASE_URL"]
    client = openai.OpenAI(api_key=os.environ[info.key_env], base_url=base_url)
    return OpenAICompatProvider(client, model, info)


# ---------- Claude ----------


class AnthropicProvider:
    max_tokens = 4096

    def __init__(self, client: Any, model: str, tools: list[dict[str, Any]] | None = None) -> None:
        from arbitrage_analyser.agent.tools import TOOL_SPECS

        self.info = PROVIDERS["anthropic"]
        self.client = client
        self.model = model
        self.tools = tools if tools is not None else TOOL_SPECS

    def first_messages(self, system: str) -> list[dict[str, Any]]:
        return []  # Claude takes the system prompt as a separate field

    def call(self, system: str, messages: list[dict[str, Any]]) -> Step:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # cache_control: reuse the prompt across calls, which makes them cheaper
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=self.tools,
            messages=messages,
        )
        blocks: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
                calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        usage = getattr(response, "usage", None)
        cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        return Step(
            text="\n".join(b["text"] for b in blocks if b["type"] == "text").strip(),
            tool_calls=calls if response.stop_reason == "tool_use" else [],
            truncated=response.stop_reason == "max_tokens",
            input_tokens=int((getattr(usage, "input_tokens", 0) or 0) + cached),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            assistant_message={"role": "assistant", "content": blocks},
            log_output=blocks,
        )

    def tool_results(
        self, results: list[tuple[ToolCall, dict[str, Any], bool]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(output, default=str),
                        "is_error": is_error,
                    }
                    for call, output, is_error in results
                ],
            }
        ]


# ---------- Gemini and other OpenAI-compatible APIs ----------


def openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Claude-style tool specs -> OpenAI 'function' tools, in the plain JSON schema subset that
    Gemini accepts: no numeric bounds, enums only on strings, no empty parameter objects."""
    tools = []
    for spec in specs:
        function: dict[str, Any] = {"name": spec["name"], "description": spec["description"]}
        schema = _simplify(spec["input_schema"])
        if schema.get("properties"):
            function["parameters"] = schema
        tools.append({"type": "function", "function": function})
    return tools


def _simplify(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("minimum", "maximum"):
            continue
        if key == "enum" and schema.get("type") != "string":
            allowed = ", ".join(str(v) for v in value)
            out["description"] = (schema.get("description", "") + f" One of: {allowed}.").strip()
            continue
        if key == "properties":
            out[key] = {name: _simplify(sub) for name, sub in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _simplify(value)
        elif key == "description" and "description" in out:
            continue  # already extended by an enum note
        else:
            out[key] = value
    return out


class OpenAICompatProvider:
    def __init__(
        self,
        client: Any,
        model: str,
        info: ProviderInfo | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        from arbitrage_analyser.agent.tools import TOOL_SPECS

        self.info = info or PROVIDERS["gemini"]
        self.client = client
        self.model = model
        self.tools = openai_tools(tools if tools is not None else TOOL_SPECS)

    def first_messages(self, system: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system}]

    def call(self, system: str, messages: list[dict[str, Any]]) -> Step:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=self.tools
        )
        choice = response.choices[0]
        message = choice.message
        raw_calls = list(message.tool_calls or [])
        calls = []
        for raw in raw_calls:
            try:
                args = json.loads(raw.function.arguments or "{}")
                calls.append(
                    ToolCall(raw.id, raw.function.name, args if isinstance(args, dict) else {})
                )
            except json.JSONDecodeError:
                calls.append(ToolCall(raw.id, raw.function.name, {}, raw.function.arguments))
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content}
        if raw_calls:
            # model_dump keeps provider extras (e.g. Gemini's thought signatures), which must be
            # sent back unchanged on the next call.
            assistant["tool_calls"] = [raw.model_dump(exclude_none=True) for raw in raw_calls]
        usage = response.usage
        return Step(
            text=(message.content or "").strip(),
            tool_calls=calls,
            truncated=choice.finish_reason == "length",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            assistant_message=assistant,
            log_output=assistant,
        )

    def tool_results(
        self, results: list[tuple[ToolCall, dict[str, Any], bool]]
    ) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output, default=str)}
            for call, output, _ in results
        ]
