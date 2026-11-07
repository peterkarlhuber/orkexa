"""Optional live smoke tests against OpenAI, Gemini and a vLLM server (ORK-012; never part of a normal run).

Each runs only with ``pytest -m live`` and its variables set; a paid provider costs a few cents at most:

* OpenAI: ``OPENAI_API_KEY`` and ``ORKEXA_OPENAI_LIVE_MODEL`` (a small model, e.g. ``gpt-5.5-mini``);
* Gemini: ``GEMINI_API_KEY`` and ``ORKEXA_GEMINI_LIVE_MODEL`` (e.g. ``gemini-3.5-flash-lite``);
* vLLM: ``ORKEXA_VLLM_LIVE_URL`` (e.g. ``http://127.0.0.1:8000/v1``) and ``ORKEXA_VLLM_LIVE_MODEL``, a model
  served with tool calling enabled (``--enable-auto-tool-choice`` and its ``--tool-call-parser``).

The model names are not built in, so a test never silently runs against a model the owner did not choose.
Each test sends three small requests: a text answer, one tool call and a structured answer. The Anthropic and
Bedrock smoke tests are ``tests/anthropic/test_live.py`` and ``tests/test_bedrock_live.py``; Ollama has
``tests/test_ollama_live.py`` (weekly in ``ollama-live.yml``).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from pydantic import BaseModel
import pytest

from orkexa import Agent, Tool
from orkexa.core.types import ToolCallPart

pytestmark = pytest.mark.live


@dataclass(frozen=True)
class Live:
    provider: str
    needs: tuple[str, ...]
    model_variable: str
    url_variable: str | None = None
    #: What a self-hosted model is declared to support (Orkexa assumes nothing for unknown models).
    capabilities: tuple[tuple[str, str], ...] = ()

    def missing(self) -> list[str]:
        return [name for name in (*self.needs, self.model_variable) if not os.environ.get(name)]

    def agent(self, **kwargs: Any) -> Agent:
        options: dict[str, Any] = {"max_tokens": 1024, **kwargs}
        if self.url_variable is not None:
            options["base_url"] = os.environ[self.url_variable]
        if self.capabilities:
            options["capabilities"] = dict(self.capabilities)
        return Agent(provider=self.provider, model=os.environ[self.model_variable], **options)


PROVIDERS = {
    "openai": Live("openai", ("OPENAI_API_KEY",), "ORKEXA_OPENAI_LIVE_MODEL"),
    "gemini": Live("gemini", ("GEMINI_API_KEY",), "ORKEXA_GEMINI_LIVE_MODEL"),
    "vllm": Live(
        "vllm",
        ("ORKEXA_VLLM_LIVE_URL",),
        "ORKEXA_VLLM_LIVE_MODEL",
        "ORKEXA_VLLM_LIVE_URL",
        (("tools", "supported"), ("structured_output", "supported")),
    ),
}


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum.
    """
    return a + b


class Capital(BaseModel):
    country: str
    city: str


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_text_tool_and_structured_output(name: str) -> None:
    live = PROVIDERS[name]
    missing = live.missing()
    if missing:
        pytest.skip(f"needs {' and '.join(missing)}")

    chat = live.agent(system_prompt="Answer in one word.")
    assert "paris" in str(chat.ask("What is the capital of France?")).lower()

    calculator = live.agent(tools=[add])
    answer = str(calculator.ask("Use the add tool to compute 1234 + 2999. Reply with the result."))
    assert "4233" in answer.replace(",", "").replace(".", "")
    history = calculator.history()
    parts = [part for message in history if isinstance(message.content, list) for part in message.content]
    assert [part.name for part in parts if isinstance(part, ToolCallPart)] == ["add"], history
    usage = calculator.last_usage
    assert usage is not None and usage.provider == name and usage.input_tokens > 0

    extractor = live.agent(output_format=Capital)
    capital = extractor.ask("Italy")
    assert isinstance(capital, Capital) and capital.city.lower() in ("rome", "roma")
