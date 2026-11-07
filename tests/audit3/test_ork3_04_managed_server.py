"""ORK3-04: lifecycle of the managed Ollama server (runners, SIGTERM, IPv6 loopback).

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from orkexa.local import ManagedOllama, OllamaError

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX links and process groups")


# ---- ORK3-04: lifecycle of the managed server ------------------------------------------

_FAKE = textwrap.dedent(
    """\
    #!{python} -I
    import http.server, json, os, signal, subprocess
    host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
    runner = subprocess.Popen(["sleep", "120"])
    open(os.path.join(os.environ["HOME"], "runner.pid"), "w").write(str(runner.pid))
    if os.path.exists(os.path.join(os.environ["HOME"], "ignore-term")):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({{"version": "0.40.14"}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    class Server(http.server.HTTPServer):
        address_family = __import__("socket").AF_INET6 if ":" in host else __import__("socket").AF_INET
        def server_bind(self):
            # HTTPServer.server_bind asks socket.getfqdn(), a reverse DNS lookup that can take
            # longer than a start timeout on CI machines (macOS); the fake needs no server name.
            __import__("socketserver").TCPServer.server_bind(self)
            self.server_name, self.server_port = "localhost", self.server_address[1]
    Server((host.strip("[]"), int(port)), Handler).serve_forever()
    """
)


def _fake_ollama(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "ollama"
    binary.parent.mkdir()
    binary.write_text(_FAKE.format(python=sys.executable))
    binary.chmod(0o755)
    return binary


#: A slow CI machine (macOS on Intel) needs several seconds to start a Python process.
START_TIMEOUT = 60.0


def _start(server: ManagedOllama) -> str:
    """``server.start()`` with a generous bound; a failure shows the fake server's log."""
    try:
        return server.start(timeout=START_TIMEOUT)
    except OllamaError as exc:
        log = server.log_file.read_text(errors="replace") if server.log_file.exists() else "(no log)"
        raise AssertionError(f"{exc}\n--- {server.log_file} ---\n{log}") from exc


#: Linux; macOS has no /proc.
_HAS_PROC = Path("/proc/self/stat").exists()


def _alive(pid: int) -> bool:
    """Whether ``pid`` runs; a killed process that its parent has not reaped yet (a zombie) does not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if _HAS_PROC:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            # Reaped after os.kill: an orphaned runner is reaped by init whenever init gets to it, so its
            # /proc entry can vanish at any moment, also between a check that it exists and the read.
            return False
        # The state follows the command name, which is in parentheses and may contain spaces.
        return stat.rpartition(")")[2].split()[0] != "Z"
    # No /proc (macOS): a killed child that its parent has not reaped yet is a zombie, not alive.
    ps = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = ps.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not _HAS_PROC, reason="reads /proc")
def test_alive_counts_a_process_reaped_during_the_check_as_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    # The interleaving from CI: os.kill(pid, 0) still finds the killed runner, then init reaps it before
    # /proc/<pid>/stat is read.
    child = subprocess.Popen(["sleep", "60"])
    stat = Path(f"/proc/{child.pid}/stat")
    read_text = Path.read_text

    def reaped_first(path: Path, *args: object, **kwargs: object) -> str:
        if path == stat and child.poll() is None:
            child.kill()
            child.wait()
        return read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    try:
        assert _alive(child.pid)
        monkeypatch.setattr(Path, "read_text", reaped_first)
        assert not _alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@posix_only
def test_stop_also_ends_the_runners_of_a_server_that_had_to_be_killed(tmp_path: Path) -> None:
    server = ManagedOllama(tmp_path / "home", binary=_fake_ollama(tmp_path))
    (tmp_path / "home" / "ollama-home").mkdir(parents=True)
    (tmp_path / "home" / "ollama-home" / "ignore-term").write_text("")
    _start(server)
    runner = int((tmp_path / "home" / "ollama-home" / "runner.pid").read_text())
    server.stop(timeout=0.5)
    assert _gone(runner)


@posix_only
def test_cli_serve_stops_the_server_on_sigterm(tmp_path: Path) -> None:
    home = tmp_path / "home"
    binary = _fake_ollama(tmp_path)
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import orkexa.local.server as server
        server.installed_runtime = lambda home, *args, **kwargs: Path({str(binary)!r})
        from orkexa.local.__main__ import main
        sys.exit(main(["--home", {str(home)!r}, "serve"]))
        """
    )
    cli = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert cli.stdout is not None and "runs at" in cli.stdout.readline()
        pid = json.loads((home / "ollama.state.json").read_text())["pid"]
        cli.send_signal(signal.SIGTERM)
        assert cli.wait(20) == 0
        assert _gone(pid)
        assert not (home / "ollama.state.json").exists()
    finally:
        if cli.poll() is None:
            cli.kill()


@posix_only
@pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6")
def test_managed_server_on_ipv6_loopback(tmp_path: Path) -> None:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("no IPv6 loopback")
    server = ManagedOllama(tmp_path / "home", binary=_fake_ollama(tmp_path), host="::1")
    try:
        url = _start(server)
        assert url.startswith("http://[::1]:")
        assert server.status().running
    finally:
        server.stop()
