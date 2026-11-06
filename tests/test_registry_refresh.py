"""Registries that share a directory with other processes (ORK-002, #24).

Two registry objects on one directory stand in for two processes: each keeps its own
snapshot until ``refresh()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from orkexa import Tool
from orkexa.dynamic import AgentRegistry, AgentSpec, RegistryError, SubprocessSandbox, ToolRegistry

CODE = "def double(x: int) -> int:\n    return 2 * x\n"


@Tool
def lookup(query: str) -> str:
    """Look something up.

    Args:
        query: What to look up.

    Returns:
        The answer.
    """
    return query


def _add(registry: ToolRegistry, name: str = "double", code: str = CODE) -> Any:
    return registry.add_generated(
        name=name,
        function_name="double",
        description="Doubles a number.",
        code=code,
        tests="",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        output_schema=None,
    )


def _tools(path: Path) -> ToolRegistry:
    return ToolRegistry(path, sandbox=SubprocessSandbox(acknowledge_unsafe=True))


def _spec(name: str = "Analyst", prompt: str = "You analyze.") -> AgentSpec:
    return AgentSpec(
        name=name,
        description="Analyzes things.",
        system_prompt=prompt,
        model="claude-sonnet-5-5",
        effort="medium",
        tools=[],
    )


def test_tool_registry_sees_other_writers_only_after_refresh(tmp_path: Path) -> None:
    reader, writer = _tools(tmp_path), _tools(tmp_path)
    record = _add(writer)
    writer.approve(record.name, record.version)
    assert "double" not in reader  # a snapshot until refresh
    reader.refresh()
    assert reader.record("double").version == 1
    writer.revoke("double")
    assert "double" in reader
    reader.refresh()
    assert "double" not in reader
    assert [r.status for r in reader.versions("double")] == ["revoked"]


def test_tool_refresh_keeps_trusted_tools_and_hides_stored_tools_of_their_name(tmp_path: Path) -> None:
    reader, writer = _tools(tmp_path), _tools(tmp_path)
    reader.register_trusted(lookup)
    for name in ("lookup", "double"):
        record = _add(writer, name=name)
        writer.approve(record.name, record.version)
    reader.refresh()
    assert reader.record("lookup").kind == "trusted"
    assert reader.get_tool("lookup") is lookup
    assert reader.names() == ["double", "lookup"]


def test_tool_refresh_skips_tampered_records(tmp_path: Path) -> None:
    reader, writer = _tools(tmp_path), _tools(tmp_path)
    record = _add(writer)
    writer.approve(record.name, record.version)
    file = tmp_path / "double" / "v1.json"
    file.write_text(file.read_text(encoding="utf-8").replace("2 * x", "3 * x"), encoding="utf-8")
    reader.refresh()
    assert "double" not in reader
    assert reader.versions("double") == ()


def test_new_tool_version_never_overwrites_one_stored_by_another_process(tmp_path: Path) -> None:
    first, second = _tools(tmp_path), _tools(tmp_path)
    assert _add(first).version == 1
    # ``second`` has not refreshed: it must not write its version over v1.
    assert _add(second, code=CODE + "# second\n").version == 2
    reloaded = _tools(tmp_path)
    assert [r.version for r in reloaded.versions("double")] == [1, 2]
    assert "# second" not in (reloaded.versions("double")[0].code or "")


def test_refresh_without_path_is_a_no_op() -> None:
    tools = ToolRegistry(sandbox=SubprocessSandbox(acknowledge_unsafe=True))
    tools.register_trusted(lookup)
    tools.refresh()
    assert "lookup" in tools
    agents = AgentRegistry()
    agents.register(_spec())
    agents.refresh()
    assert "Analyst" in agents


def test_agent_registry_sees_other_writers_only_after_refresh(tmp_path: Path) -> None:
    reader, writer = AgentRegistry(tmp_path), AgentRegistry(tmp_path)
    writer.register(_spec())
    assert "Analyst" not in reader
    reader.refresh()
    assert reader.get("Analyst").version == 1
    writer.revoke("Analyst")
    reader.refresh()
    with pytest.raises(RegistryError):
        reader.get("Analyst")


def test_new_agent_version_never_overwrites_one_stored_by_another_process(tmp_path: Path) -> None:
    first, second = AgentRegistry(tmp_path), AgentRegistry(tmp_path)
    assert first.register(_spec()).version == 1
    assert second.register(_spec(prompt="You analyze twice.")).version == 2
    reloaded = AgentRegistry(tmp_path)
    prompts = [r.spec.system_prompt for r in reloaded.versions("Analyst")]
    assert prompts == ["You analyze.", "You analyze twice."]
