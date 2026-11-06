"""The managed Ollama server on a host that several users share (ORK-003). No network."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import time

import pytest

from orkexa.local import ManagedOllama, OllamaError
import orkexa.local.server as server_module

from .audit3.test_ork3_04_managed_server import _fake_ollama, _start

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes; the fake server is a script")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_server_files_are_private_to_the_user(tmp_path: Path) -> None:
    home = tmp_path / "home"
    server = ManagedOllama(home, binary=_fake_ollama(tmp_path))
    old = os.umask(0o002)  # a permissive umask must not make them readable by others
    try:
        _start(server)
    finally:
        os.umask(old)
    try:
        assert _mode(home / "ollama-home") == 0o700
        assert _mode(home / "logs") == 0o700
        assert _mode(server.log_file) == 0o600
        assert _mode(home / "ollama.state.json") == 0o600
    finally:
        server.stop()


@posix_only
def test_large_log_is_rotated_at_the_next_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "LOG_ROTATE_BYTES", 100)
    home = tmp_path / "home"
    server = ManagedOllama(home, binary=_fake_ollama(tmp_path))
    server.log_file.parent.mkdir(parents=True)
    server.log_file.write_bytes(b"old line\n" * 20)
    (server.log_file.parent / "ollama.log.1").write_bytes(b"oldest\n")
    _start(server)
    server.stop()
    rotated = server.log_file.with_name("ollama.log.1")
    assert rotated.read_bytes() == b"old line\n" * 20  # the oldest log is dropped
    assert b"old line" not in server.log_file.read_bytes()


@posix_only
def test_small_log_is_appended_to(tmp_path: Path) -> None:
    home = tmp_path / "home"
    server = ManagedOllama(home, binary=_fake_ollama(tmp_path))
    server.log_file.parent.mkdir(parents=True)
    server.log_file.write_bytes(b"earlier start\n")
    _start(server)
    server.stop()
    assert server.log_file.read_bytes().startswith(b"earlier start\n")
    assert not server.log_file.with_name("ollama.log.1").exists()


@posix_only
def test_binary_that_cannot_be_executed_is_an_ollama_error(tmp_path: Path) -> None:
    binary = _fake_ollama(tmp_path)
    binary.chmod(0o644)
    server = ManagedOllama(tmp_path / "home", binary=binary)
    with pytest.raises(OllamaError) as caught:
        server.start(timeout=5)
    assert caught.value.code == "start_failed"
    assert "install-runtime" in str(caught.value)
    assert caught.value.__cause__ is None


@posix_only
def test_unusable_home_is_an_ollama_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "logs").write_text("a file where the log folder belongs")
    server = ManagedOllama(home, binary=_fake_ollama(tmp_path))
    with pytest.raises(OllamaError) as caught:
        server.start(timeout=5)
    assert caught.value.code == "start_failed"


@posix_only
def test_another_server_answering_on_the_port_is_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Our server exits (its port was taken) while something else answers there."""
    binary = tmp_path / "bin" / "ollama"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable} -I\nimport time\ntime.sleep(0.3)\nraise SystemExit(1)\n")
    binary.chmod(0o755)

    class ForeignAnswers:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def is_running(self) -> bool:
            time.sleep(2.0)  # the child exits meanwhile; the foreign server answers
            return True

    monkeypatch.setattr(server_module, "OllamaClient", ForeignAnswers)
    server = ManagedOllama(tmp_path / "home", binary=binary)
    with pytest.raises(OllamaError) as caught:
        server.start(timeout=10)
    assert caught.value.code == "port_in_use"
    assert server.base_url is None


@posix_only
def test_runtime_folders_are_private_to_the_user(tmp_path: Path) -> None:
    from orkexa.local.install import install_runtime

    old = os.umask(0o002)
    try:
        with pytest.raises(OllamaError):  # no archive there: only the folders are created
            install_runtime(tmp_path / "home", archive=tmp_path / "missing.tgz")
    finally:
        os.umask(old)
    assert _mode(tmp_path / "home" / "runtime") == 0o700
