"""What a provider's stream can show of a model's reasoning: thinking text, summaries or nothing.

Whether a model *reasons* is the ``reasoning`` capability (:class:`~orkexa.ModelCapabilities`) and
how much it reasons is the request's effort or budget. This module answers a separate question
for interfaces that display reasoning live: what the provider's API transmits of it while it streams.

Providers differ, and Orkexa never fills the gap with generated text:

- ``thinking``: the model's reasoning text as it was generated (open-weight models behind Ollama,
  vLLM and other OpenAI-compatible servers; DeepSeek and gpt-oss models on Bedrock).
- ``summary``: a summary the provider writes of reasoning it keeps to itself (OpenAI reasoning
  summaries, Gemini thought summaries, Claude's summarized thinking).
- ``none``: the API transmits no reasoning text for this model (it does not reason, or the
  provider does not expose it); ``reason`` says which.
- ``unknown``: depends on the server and the model (OpenAI-compatible endpoints without declared
  capabilities); text is shown if it arrives, never assumed.

Usage::

    info = agent.adapter.reasoning_stream("gpt-5.5")
    info.text            # "summary"
    info.request_option  # "reasoning_summary": OpenAI sends summaries only when asked
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ReasoningStream", "ReasoningText"]

ReasoningText = Literal["thinking", "summary", "none", "unknown"]


@dataclass(frozen=True)
class ReasoningStream:
    """What a model's stream transmits of its reasoning (see the module documentation).

    Attributes:
        text: ``thinking``, ``summary``, ``none`` or ``unknown``.
        status: Whether the stream marks a reasoning phase even when no text is transmitted
            (:class:`~orkexa.ReasoningStatus`): Anthropic thinking blocks, OpenAI reasoning items and
            Bedrock reasoning blocks do; Gemini and Chat Completions servers only send text.
        request_option: The provider option that asks for the text, when the provider sends it
            only on request: ``reasoning_summary`` (``OpenAIOptions``), ``include_thoughts``
            (``GeminiOptions``), ``thinking_display`` (``AnthropicOptions``; the default depends
            on the model). ``None``: transmitted without an option (or not at all).
        reason: Why no text is transmitted (``text == "none"``), in English; ``""`` otherwise.
    """

    text: ReasoningText = "unknown"
    status: bool = False
    request_option: str | None = None
    reason: str = ""

    @classmethod
    def none(cls, reason: str) -> ReasoningStream:
        return cls(text="none", reason=reason)
