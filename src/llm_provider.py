"""
src/llm_provider.py

Thin provider abstraction for the Dynamic Pricing Agent's LLM layer.

Why this is its own module:
  * agent.py depends on the abstract LLMProvider interface only, so "the LLM is
    unavailable" is an ordinary, well-typed branch rather than a special case.
  * the `anthropic` package stays a SOFT dependency - the deterministic
    ML + policy pipeline (and its tests) run with it uninstalled.
  * tests inject a scripted provider instead of monkeypatching an SDK.

Credentials come from the environment ONLY (ANTHROPIC_API_KEY). The model name
is configurable via AGENT_MODEL. Nothing here is ever hardcoded or logged.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 20.0


class LLMError(Exception):
    """Any failure in the LLM layer. Always non-fatal to a checkout."""


class LLMUnavailableError(LLMError):
    """No provider is configured / reachable (missing key, missing SDK, network)."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Provider-neutral view of one assistant turn."""
    text: str = ""
    tool_calls: list = field(default_factory=list)      # list[ToolCall]
    stop_reason: Optional[str] = None
    # Raw assistant content, appended verbatim to the running message list so
    # multi-turn tool use round-trips correctly for whichever provider produced it.
    assistant_content: Any = None


class LLMProvider(ABC):
    """Minimal surface the orchestrator needs: one tool-enabled turn."""

    name: str = "abstract"
    model: str = "unknown"

    @abstractmethod
    def create_message(self, *, system: str, messages: list, tools: list,
                       max_tokens: int = DEFAULT_MAX_TOKENS,
                       timeout: float = DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        ...


class AnthropicProvider(LLMProvider):
    """Claude via the official SDK. Constructed lazily so import never explodes."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.model = model or os.getenv("AGENT_MODEL") or DEFAULT_MODEL
        self._timeout = timeout
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise LLMUnavailableError(
                "ANTHROPIC_API_KEY is not set; the agent will run in deterministic mode."
            )
        try:
            import anthropic  # noqa: F401  (soft dependency)
        except ImportError as exc:
            raise LLMUnavailableError(
                "the 'anthropic' package is not installed; the agent will run in "
                "deterministic mode."
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=self._api_key, timeout=timeout)

    def create_message(self, *, system, messages, tools,
                       max_tokens=DEFAULT_MAX_TOKENS,
                       timeout=DEFAULT_TIMEOUT_SECONDS) -> LLMResponse:
        if _simulate_llm_failure():
            raise LLMUnavailableError("SIMULATE_LLM_FAILURE is set (injected failure)")
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                timeout=timeout,
            )
        except Exception as exc:
            # Never leak the key or a full stack into the caller's audit trail.
            raise LLMError(f"{type(exc).__name__} while calling the LLM API") from exc

        text_parts, tool_calls = [], []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name,
                                           input=dict(block.input or {})))
        return LLMResponse(
            text="\n".join(p for p in text_parts if p).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(resp, "stop_reason", None),
            assistant_content=resp.content,
        )


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _simulate_llm_failure() -> bool:
    return _env_flag("SIMULATE_LLM_FAILURE")


def get_llm_provider(model: Optional[str] = None,
                     timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[LLMProvider]:
    """Best-effort provider construction.

    Returns None (never raises) when no LLM is available, because an absent LLM
    must degrade the *narration*, not the checkout.
    """
    if _env_flag("AGENT_DISABLE_LLM"):
        return None
    try:
        return AnthropicProvider(model=model, timeout=timeout)
    except LLMError:
        return None
