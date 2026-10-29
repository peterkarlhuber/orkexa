"""Generated code runs only in an isolated sandbox unless the caller explicitly opts out.

Covers the secure default (Docker, failing closed), the acknowledged opt-in for
``SubprocessSandbox``, timeout and cleanup, trusted tools and the static pre-filter.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import stat
import sys
import tempfile
import types
from typing import Any
import warnings

import pytest

from orkexa import Tool, ToolExecutionError
from orkexa.dynamic import (
    DockerSandbox,
    SandboxError,
    SubprocessSandbox,
    ToolRegistry,
    UnsafeOptInWarning,
    check_tool_source,
)
from orkexa.dynamic.static_check import ALLOWED_IMPORTS, FORBIDDEN_MODULE_ATTRIBUTES

CODE = '''def square(x: int) -> str:
    """Square a number.

    Args:
        x: The number.

    Returns:
        The square as text.
    """
    return str(x * x)
'''
TESTS = "from tool import square\n\ndef test_square():\n    assert square(3) == '9'\n"


def _registry(**kwargs: Any) -> ToolRegistry:
    registry = ToolRegistry(**kwargs)
    registry.add_generated(
        name="square",
        function_name="square",
        description="Square a number.",
        code=CODE,
        tests=TESTS,
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        output_schema=None,
    )
    registry.approve("square", 1)
    return registry


@pytest.fixture
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sandbox run directories are created here, so the test can see what is left behind."""
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


# ---- the default fails closed ----------------------------------------------------------------


def test_registry_default_is_the_isolated_docker_sandbox() -> None:
    registry = ToolRegistry()
    assert isinstance(registry.sandbox, DockerSandbox) and registry.sandbox.isolated
    assert not SubprocessSandbox.isolated


def test_default_registry_executes_nothing_without_docker(private_tempdir: Path, tmp_path: Path) -> None:
    canary = tmp_path / "executed"
    registry = ToolRegistry(sandbox=DockerSandbox(docker=str(tmp_path / "no-docker-here")))
    registry.add_generated(
        name="touch",
        function_name="touch",
        description="Touch a file.",
        code=f"def touch() -> str:\n    open({str(canary)!r}, 'w').close()\n    return 'done'\n",
        tests="",
        input_schema={"type": "object", "properties": {}},
        output_schema=None,
    )
    registry.approve("touch", 1)
    with pytest.raises(ToolExecutionError, match="Cannot start docker") as caught:
        registry.get_tool("touch").execute({})
    assert "acknowledge_unsafe=True" in str(caught.value)  # the message names the opt-in
    assert not canary.exists()
    assert list(private_tempdir.iterdir()) == []  # run directory removed after the failure


def test_validation_fails_closed_without_docker(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="isolated sandbox"):
        DockerSandbox(docker=str(tmp_path / "no-docker-here")).validate(CODE, TESTS, "square")


# ---- explicit opt-in for the subprocess sandbox ----------------------------------------------


def test_subprocess_sandbox_warns_unless_acknowledged() -> None:
    with pytest.warns(UnsafeOptInWarning, match="not a security boundary"):
        SubprocessSandbox()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SubprocessSandbox(acknowledge_unsafe=True)


def test_acknowledged_subprocess_sandbox_runs_approved_tools(private_tempdir: Path) -> None:
    sandbox = SubprocessSandbox(acknowledge_unsafe=True)
    assert sandbox.validate(CODE, TESTS, "square").ok
    registry = _registry(sandbox=sandbox)
    assert registry.get_tool("square").execute({"x": 7}) == "49"
    assert list(private_tempdir.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_subprocess_run_directory_stays_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = SubprocessSandbox(acknowledge_unsafe=True)
    modes: list[int] = []
    execute = sandbox._execute

    def watching(workdir: Path, request: str) -> str:
        modes.append(stat.S_IMODE(workdir.stat().st_mode))
        return execute(workdir, request)

    monkeypatch.setattr(sandbox, "_execute", watching)
    sandbox.call(CODE, "square", {"x": 2})
    assert modes == [0o700]  # not readable for other local users while the code runs


def test_timeout_is_enforced_and_the_run_directory_removed(private_tempdir: Path) -> None:
    sandbox = SubprocessSandbox(timeout=1, acknowledge_unsafe=True)
    endless = _function("str(sum(iter(int, 1)))", "")  # sums zeros forever
    with pytest.raises(ToolExecutionError, match="Timed out after 1 seconds"):
        sandbox.call(endless, "f", {})
    assert list(private_tempdir.iterdir()) == []


def test_trusted_tools_still_run_in_the_host_process() -> None:
    @Tool
    def greet(name: str) -> str:
        """Greet someone.

        Args:
            name: Who to greet.

        Returns:
            The greeting.
        """
        return f"Hello {name}"

    registry = ToolRegistry()  # Docker default, but trusted tools never use the sandbox
    registry.register_trusted(greet)
    assert registry.get_tool("greet")(name="Ada") == "Hello Ada"


# ---- static pre-filter -----------------------------------------------------------------------


def _function(body: str, imports: str) -> str:
    docstring = '    """Doc.\n\n    Returns:\n        Text.\n    """\n'
    return f"{imports}\n\ndef f() -> str:\n{docstring}    return {body}\n"


@pytest.mark.parametrize(
    ("imports", "body"),
    [
        ("import calendar", 'calendar.sys.modules["pathlib"].Path("/etc/hostname").read_text()'),
        ("import dataclasses", "dataclasses.inspect.getsource(f)"),
        ("import typing", 'typing.sys.modules["os"].getcwd()'),
        ("import json", 'json.codecs.open("/etc/hostname").read()'),
        ("import enum", 'enum.bltns.open("/etc/hostname").read()'),
        ("import typing_extensions", "str(typing_extensions._socket)"),
        ("import typing_extensions", "str(typing_extensions._types.FunctionType)"),
        ("import collections", "str(collections._sys.modules)"),
        ("import functools", "str(functools.reduce.__module__)"),
        ("", '"".__class__.__mro__[1].__subclasses__()'),
        ("", '__import__("os").getcwd()'),
        ("", 'getattr(str, "mro")'),
    ],
)
def test_static_check_rejects_indirect_module_access(imports: str, body: str) -> None:
    assert not check_tool_source(_function(body, imports), "f").ok


#: Modules that allowed modules expose as attributes and that were reviewed as harmless
#: (no I/O, no process, network, import or code-object access).
REVIEWED_MODULE_ATTRIBUTES = frozenset(
    {
        "_random",
        "_sre",
        "_string",
        "abc",
        "contextlib",
        "copy",
        "core_schema",
        "keyword",
        "random",
        "sre_compile",
        "sre_parse",
        "typing_objects",
    }
)


def test_every_module_reachable_from_allowed_imports_is_reviewed() -> None:
    """Fails when a new Python or dependency version exposes an unreviewed module attribute."""
    unreviewed: dict[str, str] = {}
    seen: set[str] = set()

    def walk(module: types.ModuleType, path: str, depth: int) -> None:
        if module.__name__ in seen or depth > 5:
            return
        seen.add(module.__name__)
        for name in dir(module):
            if name in FORBIDDEN_MODULE_ATTRIBUTES or (name.startswith("__") and name.endswith("__")):
                continue
            try:
                value = getattr(module, name)
            except Exception:
                continue
            if not isinstance(value, types.ModuleType):
                continue
            if value.__name__.split(".")[0] in ALLOWED_IMPORTS:
                walk(value, f"{path}.{name}", depth + 1)
            elif name not in REVIEWED_MODULE_ATTRIBUTES:
                unreviewed.setdefault(name, f"{path}.{name} -> {value.__name__}")

    for name in sorted(ALLOWED_IMPORTS - {"__future__"}):
        walk(importlib.import_module(name), name, 0)
    assert unreviewed == {}


def test_static_check_still_accepts_ordinary_code() -> None:
    assert check_tool_source(CODE, "square").ok
    code = _function(
        'json.dumps(dict(a=math.sqrt(4), b=re.sub("a", "b", "aa")))', "import json\nimport math\nimport re"
    )
    assert check_tool_source(code, "f").ok
