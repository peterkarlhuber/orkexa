"""DX-01: a missing API key is one Orkexa error (MissingCredentialsError) for every provider.

Before, each SDK raised its own exception (anthropic ``TypeError`` at request time,
``openai.OpenAIError`` and a ``ValueError`` from google-genai when the client was built), which
``except OrkexaError`` did not catch. No request reaches the network in these tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    AuthenticationError,
    ClaudeAgent,
    MissingCredentialsError,
    OrkexaError,
)
from orkexa.providers import ProviderConfig, create_provider

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_PROFILE",
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
)

CASES = [
    ("anthropic:claude-haiku-5-5", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")),
    ("openai:gpt-5.5", ("OPENAI_API_KEY",)),
    ("gemini:gemini-3-pro", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
]


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no credential files of the machine
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def _check(error: BaseException, provider: str, variables: tuple[str, ...]) -> None:
    assert isinstance(error, MissingCredentialsError)
    assert isinstance(error, AuthenticationError) and isinstance(error, OrkexaError)
    assert error.provider == provider and error.variables == variables
    assert all(name in str(error) for name in variables)
    assert error.__cause__ is None and error.__context__ is None  # no link to the SDK exception


@pytest.mark.parametrize(("model", "variables"), CASES, ids=[c[0].split(":")[0] for c in CASES])
@pytest.mark.parametrize("stream", [False, True])
def test_ask_without_a_key(model: str, variables: tuple[str, ...], stream: bool) -> None:
    agent = Agent(model=model, stream=stream)
    with pytest.raises(OrkexaError) as caught:
        agent.ask("hi")
    _check(caught.value, model.split(":")[0], variables)
    assert agent.messages == []


@pytest.mark.parametrize(("model", "variables"), CASES, ids=[c[0].split(":")[0] for c in CASES])
def test_async_and_token_counting_without_a_key(model: str, variables: tuple[str, ...]) -> None:
    with pytest.raises(OrkexaError) as caught:
        asyncio.run(AsyncAgent(model=model).ask("hi"))
    _check(caught.value, model.split(":")[0], variables)
    with pytest.raises(OrkexaError) as caught:
        Agent(model=model).count_tokens("hi")
    _check(caught.value, model.split(":")[0], variables)


def test_claude_agent_without_a_key() -> None:
    # ClaudeAgent raises SDK exceptions unchanged, except this one.
    with pytest.raises(MissingCredentialsError):
        ClaudeAgent().ask("hi")


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_an_explicit_key_builds_the_client(provider: str) -> None:
    adapter = create_provider(provider, ProviderConfig(api_key="sk-test-0123456789"))
    assert adapter.client is not None  # no request is sent
