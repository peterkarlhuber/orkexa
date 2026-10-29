"""Untrusted input at security boundaries: registry paths, error redaction, tool loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orkexa import Agent, ProviderServerError
from orkexa.dynamic import RegistryError, ToolRegistry
from orkexa.providers.base import ProviderConfig
from orkexa.providers.fake import FakeModel, text
from orkexa.providers.openai import OpenAIProvider

SECRET = "sk-MOCKTOKEN1234567890"


def _add(registry: ToolRegistry, name: str) -> None:
    registry.add_generated(
        name=name,
        function_name="f",
        description="test",
        code="def f(): return 1",
        tests="def test_f(): assert f() == 1",
        input_schema={},
        output_schema=None,
    )


# ---- ToolRegistry paths ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../outside", "..", ".", "", "/tmp/abs", "a/b", "a\\b", "x" * 129, "tööl", "name\n", "a b", "C:evil"],
)
def test_registry_rejects_unsafe_names_without_writing(tmp_path: Path, name: str) -> None:
    registry = ToolRegistry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="Invalid tool name"):
        _add(registry, name)
    assert name not in registry
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_registry_rejects_non_string_names(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        _add(ToolRegistry(tmp_path), 7)  # type: ignore[arg-type]


def test_registry_stores_valid_names_inside_its_root(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "registry")
    _add(registry, "safe_tool-1")
    _add(registry, "safe_tool-1")
    registry.approve("safe_tool-1", 2)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert files == ["registry/safe_tool-1/v1.json", "registry/safe_tool-1/v2.json"]
    reloaded = ToolRegistry(tmp_path / "registry")
    assert reloaded.record("safe_tool-1").version == 2


def test_registry_skips_stored_records_with_unsafe_names(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    _add(registry, "good")
    stored = json.loads((tmp_path / "good" / "v1.json").read_text(encoding="utf-8"))
    for directory, name in (("bad", "../escape"), ("other", "good2")):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "v1.json").write_text(json.dumps({**stored, "name": name}), encoding="utf-8")
    reloaded = ToolRegistry(tmp_path)
    assert len(reloaded.versions("good")) == 1
    assert reloaded.versions("../escape") == () and reloaded.versions("good2") == ()
    with pytest.raises(RegistryError):
        reloaded.approve("../escape", 1)


# ---- redaction of provider error texts ---------------------------------------------------


def _failed(message: str) -> SimpleNamespace:
    return SimpleNamespace(status="failed", error=SimpleNamespace(code="server_error", message=message))


@pytest.mark.parametrize(
    ("api_key", "leaked"),
    [
        (SECRET, SECRET),  # the configured key
        ("configured-key-without-pattern", "configured-key-without-pattern"),
        (None, "sk-proj-OtherKeyLooking1234567890"),  # a key-shaped string
        (None, "AIzaSyA1234567890abcdefghijklmnopqr"),
        (None, "Bearer abcdefghijklmnop.qrstuv"),
    ],
)
def test_openai_failed_response_is_redacted(api_key: str | None, leaked: str) -> None:
    adapter = OpenAIProvider(ProviderConfig(api_key=api_key, client=object()))
    with pytest.raises(ProviderServerError) as caught:
        adapter.normalize(_failed(f"upstream echoed {leaked}"), None)  # type: ignore[arg-type]
    assert leaked.split()[-1] not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert caught.value.provider == "openai" and caught.value.retryable


def test_runtime_redacts_provider_errors_raised_by_adapters() -> None:
    """Any ProviderError an adapter builds from native text is redacted centrally."""
    fake = FakeModel([ProviderServerError(f"proxy said key={SECRET} and {SECRET}", provider="fake")])
    agent = Agent(model="fake:m", client=fake, api_key=SECRET)
    with pytest.raises(ProviderServerError) as caught:
        agent.ask("hi")
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value.args)


# ---- policy for tools a model loads ------------------------------------------------------


def test_load_tool_respects_allowed_agent_tools() -> None:
    from orkexa import Tool
    from orkexa.dynamic import AutoApprover, CodingAgent, DynamicPolicy, ToolBuilder

    @Tool
    def harmless(x: int) -> int:
        """Return x.

        Args:
            x: A number.

        Returns:
            x.
        """
        return x

    @Tool
    def dangerous(path: str) -> str:
        """Pretend to delete a file.

        Args:
            path: A path.

        Returns:
            The path.
        """
        return path

    registry = ToolRegistry()
    registry.register_trusted(harmless)
    registry.register_trusted(dangerous)
    builder = ToolBuilder(
        CodingAgent(client=FakeModel([]), model="fake:coder"),
        registry,
        approver=AutoApprover(acknowledge_unsafe=True),
        policy=DynamicPolicy(allowed_agent_tools=["harmless"]),
    )
    agent = Agent(model="fake:m", client=FakeModel([text("ok")]))
    _, load_tool, _ = builder.as_tools(agent)
    assert "not allowed" in load_tool.execute({"name": "dangerous"})
    assert "dangerous" not in agent.tools
    assert "now available" in load_tool.execute({"name": "harmless"})
    assert "harmless" in agent.tools
    # The Python API is the developer's decision and is not restricted.
    assert builder.attach(agent, "dangerous")
