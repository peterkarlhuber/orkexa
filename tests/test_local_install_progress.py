"""The runtime installation's phases, a download that continues, a full disk, the install lock, and the error
codes and cancellation of streamed Ollama answers (no network: urlopen and the server are fakes)."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import http.server
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time
from types import SimpleNamespace
from typing import Any
import urllib.error

import pytest

from orkexa.local import OllamaClient, OllamaError, install_runtime
from orkexa.local._http import stream_error_code
from orkexa.local.install import RuntimeArtifact


def _archive(tmp_path: Path) -> tuple[bytes, RuntimeArtifact]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in {
            "bin/ollama": b"#!/bin/sh\n" + b"x" * 5000,
            "lib/ollama/libx.so": os.urandom(20000),
        }.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            bundle.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    return data, RuntimeArtifact("darwin", "any", "ollama-darwin.tgz", hashlib.sha256(data).hexdigest())


class _Response:
    def __init__(self, data: bytes, status: int, *, fail_after: int | None = None) -> None:
        self._data, self.status, self._fail_after, self._sent = data, status, fail_after, 0
        self.length = len(data)

    def geturl(self) -> str:
        return "https://example.invalid/ollama-darwin.tgz"

    def read(self, size: int) -> bytes:
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise ConnectionResetError("connection reset by peer")
        block = self._data[self._sent : self._sent + min(size, 4096)]
        self._sent += len(block)
        return block

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A release page: serves ``data`` (honoring ranges), optionally breaking after ``fail_after`` bytes."""
    state = SimpleNamespace(data=b"", fail_after=None, ranges=[], refuse=False)

    def urlopen(request: Any, timeout: float = 0) -> _Response:
        if state.refuse:
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        header = request.get_header("Range")
        state.ranges.append(header)
        if header:
            start = int(header.removeprefix("bytes=").rstrip("-"))
            return _Response(state.data[start:], 206, fail_after=state.fail_after)
        return _Response(state.data, 200, fail_after=state.fail_after)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    yield state


def test_a_broken_download_continues_where_it_stopped(tmp_path: Path, server: SimpleNamespace) -> None:
    data, artifact = _archive(tmp_path)
    server.data, server.fail_after = data, 8192
    seen: list[tuple[int, int | None]] = []
    phases: list[str] = []
    with pytest.raises(OllamaError) as error:
        install_runtime(
            tmp_path / "home", download=True, artifact=artifact, progress=lambda d, t: seen.append((d, t))
        )
    assert error.value.code == "network"
    partial = tmp_path / "home/runtime/.downloads/ollama-darwin.tgz.partial"
    assert partial.stat().st_size == 8192  # kept for the next attempt
    server.fail_after, seen[:] = None, []
    binary = install_runtime(
        tmp_path / "home",
        download=True,
        artifact=artifact,
        progress=lambda d, t: seen.append((d, t)),
        phase=phases.append,
    )
    assert binary.is_file()
    assert server.ranges[-1] == "bytes=8192-"  # only the rest was fetched
    assert phases == ["download", "verify", "unpack"]
    # The download starts at the bytes it had, of the whole archive; the checksum counts the archive too.
    assert seen[0] == (8192, len(data))
    assert (len(data), len(data)) in seen
    assert not partial.exists()  # removed after a successful installation


def test_a_server_without_ranges_starts_over(
    tmp_path: Path, server: SimpleNamespace, monkeypatch: Any
) -> None:
    data, artifact = _archive(tmp_path)
    server.data = data
    partial = tmp_path / "home/runtime/.downloads/ollama-darwin.tgz.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"garbage")

    def whole(request: Any, timeout: float = 0) -> _Response:
        return _Response(data, 200)  # ignores the Range header

    monkeypatch.setattr("urllib.request.urlopen", whole)
    assert install_runtime(tmp_path / "home", download=True, artifact=artifact).is_file()


def test_a_full_disk_is_named_before_the_download(
    tmp_path: Path, server: SimpleNamespace, monkeypatch: Any
) -> None:
    data, artifact = _archive(tmp_path)
    server.data = data
    full = SimpleNamespace(free=100, total=0, used=0)
    monkeypatch.setattr("shutil.disk_usage", lambda path: full)
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", download=True, artifact=artifact)
    assert error.value.code == "disk_full"
    assert "100 bytes free" in str(error.value)


def test_a_full_disk_while_writing_is_named(
    tmp_path: Path, server: SimpleNamespace, monkeypatch: Any
) -> None:
    data, artifact = _archive(tmp_path)
    server.data = data
    real_open = Path.open

    class Full(io.BytesIO):
        def write(self, block: Any) -> int:
            raise OSError(28, "No space left on device")

    def opener(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.name.endswith(".partial") and "b" in mode and ("a" in mode or "w" in mode):
            return Full()
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opener)
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", download=True, artifact=artifact)
    assert error.value.code == "disk_full"


def test_unreachable_release_page_and_a_bad_download(tmp_path: Path, server: SimpleNamespace) -> None:
    _data, artifact = _archive(tmp_path)
    server.refuse = True
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", download=True, artifact=artifact)
    assert error.value.code == "network"
    server.refuse, server.data = False, b"not the archive"
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", download=True, artifact=artifact)
    assert error.value.code == "checksum_mismatch"
    # A download that does not match is not continued next time.
    assert not (tmp_path / "home/runtime/.downloads/ollama-darwin.tgz.partial").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_one_installation_at_a_time(tmp_path: Path, server: SimpleNamespace) -> None:
    import fcntl

    data, artifact = _archive(tmp_path)
    server.data = data
    runtime = tmp_path / "home/runtime"
    runtime.mkdir(parents=True)
    with (runtime / ".install.lock").open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # another process installs
        with pytest.raises(OllamaError) as error:
            install_runtime(tmp_path / "home", download=True, artifact=artifact)
        assert error.value.code == "install_running"
    assert install_runtime(tmp_path / "home", download=True, artifact=artifact).is_file()


def test_cancel_stops_the_checksum(tmp_path: Path) -> None:
    data, artifact = _archive(tmp_path)
    archive = tmp_path / artifact.name
    archive.write_bytes(data)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", archive=archive, artifact=artifact, cancel=cancel)
    assert error.value.code == "cancelled"
    assert not (tmp_path / "home/runtime/0.40.14").exists()
    shutil.rmtree(tmp_path / "home")


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("write /models/blobs/sha256-x-partial: no space left on device", "disk_full"),
        ('max retries exceeded: Get "https://x": proxyconnect tcp: dial tcp 127.0.0.1:9: connect', "network"),
        ("pull model manifest: file does not exist", "model_missing"),
        ("something else", "http_error"),
    ],
)
def test_error_lines_of_a_download_get_codes(message: str, code: str) -> None:
    assert stream_error_code(message) == code


class _Silent(http.server.BaseHTTPRequestHandler):
    """Sends one progress line, then nothing (a download that hangs)."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(b'{"status": "pulling manifest"}\n')
        self.wfile.flush()
        time.sleep(5)


def test_cancel_ends_a_silent_download_at_once() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Silent)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cancel = threading.Event()
        client = OllamaClient(f"http://127.0.0.1:{server.server_address[1]}")
        pull = client.pull("qwen3.5:4b", cancel=cancel)
        assert next(pull).status == "pulling manifest"
        threading.Timer(0.2, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(OllamaError) as error:
            next(pull)
        assert error.value.code == "cancelled"
        assert time.monotonic() - started < 2.0  # not the 60 s line timeout
    finally:
        server.shutdown()
