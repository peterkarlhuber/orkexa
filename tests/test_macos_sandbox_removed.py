"""The macOS Seatbelt sandbox was removed (Docker-first, ADR 0012); its deprecated name never runs code.

``MacOSSandbox`` stays importable so that status reports (nodestro-runtime) keep working until they drop it.
These tests pin that it always reports itself unavailable, claims no isolation and refuses every run, on every
operating system, so it can never be selected for untrusted code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from orkexa import ToolExecutionError
import orkexa.dynamic as dynamic
from orkexa.dynamic import DockerSandbox, MacOSSandbox, Sandbox, SandboxError

CODE = '''def ping(text: str) -> str:
    """Echo."""
    return text
'''
TESTS = """from tool import ping


def test_ping():
    assert ping("x") == "x"
"""
DETAIL = "removed; use DockerSandbox (Docker-first)"


def _sandbox(**kwargs: object) -> MacOSSandbox:
    with pytest.warns(DeprecationWarning, match="Docker-first"):
        return MacOSSandbox(**kwargs)


def test_the_name_is_still_exported_but_deprecated() -> None:
    assert "MacOSSandbox" in dynamic.__all__
    assert issubclass(MacOSSandbox, Sandbox)
    # Arguments of the former backend are accepted and ignored.
    sandbox = _sandbox(timeout=30.0, memory_mb=256, tmp_mb=8, read=("/tmp",))
    assert isinstance(sandbox, MacOSSandbox)


def test_probe_always_reports_removed() -> None:
    for probe in (MacOSSandbox.probe(), MacOSSandbox.probe(refresh=True)):
        assert not probe.available
        assert probe.reason == "removed"
        assert probe.detail == DETAIL
        assert probe.as_dict()["available"] is False


def test_capabilities_claim_no_isolation() -> None:
    caps = _sandbox().capabilities()
    assert caps.backend == "macos-seatbelt"
    assert not caps.isolating and caps.isolation == "none" and caps.network == "host"
    assert caps.maturity == "experimental" and caps.memory == "none" and not caps.limits
    assert any("Docker-first" in note for note in caps.notes)
    assert MacOSSandbox.isolated is False


def test_it_refuses_to_run_before_writing_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _sandbox()

    def no_tempdir(*args: object, **kwargs: object) -> None:
        raise AssertionError("the removed sandbox must not create a run directory")

    monkeypatch.setattr("tempfile.TemporaryDirectory", no_tempdir)
    with pytest.raises(SandboxError, match="Docker-first"):
        sandbox.validate(CODE, TESTS, "ping")
    with pytest.raises(ToolExecutionError, match="Docker-first"):
        sandbox.call(CODE, "ping", {"text": "ok"})
    with pytest.raises(SandboxError, match="Docker-first"):
        sandbox._execute(Path("."), "{}")


def test_selection_never_picks_it() -> None:
    # Any selection that requires an available, isolating backend rejects it, and Docker stays the default.
    sandbox = _sandbox()
    assert not (MacOSSandbox.probe().available and sandbox.capabilities().isolating)
    assert DockerSandbox().capabilities().isolating
    assert dynamic.ToolRegistry().sandbox.__class__ is DockerSandbox


def test_the_seatbelt_implementation_is_gone() -> None:
    assert importlib.util.find_spec("orkexa.dynamic._macos_confine") is None
    source = Path(dynamic.__file__).with_name("macos.py").read_text(encoding="utf-8")
    assert "ctypes" not in source
