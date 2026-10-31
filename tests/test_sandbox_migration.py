"""The sandbox mounts the ``orkexa`` package; no provider secret enters it."""

from __future__ import annotations

from pathlib import Path

import pytest

import orkexa
from orkexa.dynamic import DockerSandbox, SubprocessSandbox

_DOC = '"""Probe.\\n\\n    Returns:\\n        A value.\\n    """'


def test_no_provider_key_reaches_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ORKEXA_TOKEN"):
        monkeypatch.setenv(name, "secret-value")
    code = f"import os\n\ndef env() -> str:\n    {_DOC}\n    return ','.join(sorted(os.environ))\n"
    names = SubprocessSandbox(acknowledge_unsafe=True).call(code, "env", {}).split(",")
    assert not [n for n in names if "KEY" in n or "TOKEN" in n]


def test_docker_mounts_the_orkexa_package(tmp_path: Path) -> None:
    sandbox = DockerSandbox()
    assert sandbox.image == "orkexa-sandbox"
    command = sandbox.command(tmp_path, "sb-1")
    package = Path(orkexa.__file__).resolve().parent
    assert f"{package}:/opt/orkexa/orkexa:ro" in command
    assert command[-1] == "/opt/orkexa/orkexa/dynamic/_sandbox_runner.py"
    assert not any("claude" in part for part in command)


def test_runner_loads_tools_without_the_package_init() -> None:
    """The runner registers ``orkexa`` as a bare namespace: no SDK or subsystem import."""
    code = (
        "import sys\n\ndef modules() -> str:\n"
        f"    {_DOC}\n"
        "    return ','.join(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('orkexa', 'anthropic', 'openai', 'google')))\n"
    )
    loaded = SubprocessSandbox(acknowledge_unsafe=True).call(code, "modules", {}).split(",")
    assert "orkexa.tools" in loaded
    assert not [m for m in loaded if m.startswith(("anthropic", "openai", "google"))]
    assert "orkexa.agent" not in loaded
