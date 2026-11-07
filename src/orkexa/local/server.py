"""A managed Ollama server: a child process this application starts, watches and stops.

:class:`ManagedOllama` runs the ``ollama`` binary of :mod:`orkexa.local.install` (or one
you name) on a free loopback port with its own model directory, so it neither needs
an installed Ollama nor touches one that already runs: an Ollama on port 11434 is
only *detected* (:func:`detect_ollama`), never stopped or reconfigured.

The child gets a minimal environment: ``OLLAMA_HOST`` (loopback), ``OLLAMA_MODELS``,
``OLLAMA_NO_CLOUD=1`` (no Ollama cloud models), no history file, no browser origins.

The server does not outlive this process: a watcher process (POSIX) or a job object (Windows) stops it when
this process ends without stopping it (:mod:`orkexa.local._lifeline`). A server that survived anyway (the
watcher was killed too, or the machine went down) is reclaimed at the next start from the state file
(POSIX).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import threading
import time
from typing import Any

from .._storage import PRIVATE_FILE_MODE, make_private_dirs, write_private_text
from . import _lifeline
from ._http import OllamaError
from ._lifeline import alive as _alive
from ._lifeline import is_managed_server as _is_managed_server
from ._lifeline import stop_group as _stop_group
from .client import OllamaClient
from .install import OLLAMA_VERSION, installed_runtime

__all__ = ["DEFAULT_PORT", "ManagedOllama", "ServerState", "detect_ollama", "free_port"]

DEFAULT_PORT = 11434
#: The server log is rotated at the next start once it is larger than this (one older log is kept).
LOG_ROTATE_BYTES = 10 * 1024 * 1024
_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "SystemRoot",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
)


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port nobody listens on now (the OS picks it)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def detect_ollama(url: str = f"http://127.0.0.1:{DEFAULT_PORT}") -> str | None:
    """The version of an Ollama server at ``url`` (e.g. one the user installed), or ``None``."""
    try:
        return OllamaClient(url, timeout=2).version(timeout=1.5)
    except OllamaError:
        return None


@dataclass(frozen=True)
class ServerState:
    """What :meth:`ManagedOllama.status` reports."""

    running: bool
    base_url: str | None = None
    version: str | None = None
    pid: int | None = None
    binary: str | None = None
    models_dir: str | None = None
    error: str | None = None


class ManagedOllama:
    """Start, check and stop an Ollama server owned by this application.

    Args:
        home: Directory for the runtime (``runtime/``), logs and state; models go to
            ``home/models`` unless ``models_dir`` is given.
        binary: The ``ollama`` executable; default: the runtime installed under ``home``.
        models_dir: Where models are stored (any local path, e.g. an external SSD).
        host: Loopback address to listen on.
        port: Fixed port; ``None``: a free port is chosen at each start.
        offline: Proxy variables are not passed to the child, so it has no route through
            a configured proxy (downloads fail instead of leaving the machine).
        environment: Extra ``OLLAMA_*`` settings, e.g. ``{"OLLAMA_CONTEXT_LENGTH": "8192"}``.
    """

    def __init__(
        self,
        home: str | os.PathLike[str],
        *,
        binary: str | os.PathLike[str] | None = None,
        models_dir: str | os.PathLike[str] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        offline: bool = False,
        environment: dict[str, str] | None = None,
    ) -> None:
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("A managed Ollama listens on loopback only (it has no authentication)")
        for key in environment or {}:
            if not key.startswith("OLLAMA_") or key in ("OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_ORIGINS"):
                raise ValueError(f"Not an allowed Ollama setting: {key}")
        self.home = Path(home)
        self._binary = Path(binary) if binary is not None else None
        self.models_dir = Path(models_dir) if models_dir is not None else self.home / "models"
        self.host = host
        self.port = port
        self.offline = offline
        self.environment = dict(environment or {})
        self._process: subprocess.Popen[bytes] | None = None
        #: What ends the server with this process: the watcher (POSIX) or the job handle (Windows).
        self._lifeline: subprocess.Popen[bytes] | None = None
        self._job: int | None = None
        self._base_url: str | None = None
        self._lock = threading.Lock()

    @property
    def binary(self) -> Path | None:
        return self._binary if self._binary is not None else installed_runtime(self.home)

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def log_file(self) -> Path:
        return self.home / "logs" / "ollama.log"

    def client(self) -> OllamaClient:
        if self._base_url is None:
            raise OllamaError(
                "not_running",
                "The managed Ollama server is not running; start() it (or `orkexa-local serve`)",
            )
        return OllamaClient(self._base_url)

    @property
    def _authority(self) -> str:
        """The host as it is written before ``:port`` (an IPv6 address in brackets)."""
        return f"[{self.host}]" if ":" in self.host else self.host

    def _env(self, port: int) -> dict[str, str]:
        env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
        if self.offline:
            for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
                env.pop(key, None)
        env["HOME"] = env["USERPROFILE"] = str(self.home / "ollama-home")  # keys and caches stay here
        env.update(self.environment)
        env.update(
            {
                "OLLAMA_HOST": f"{self._authority}:{port}",
                "OLLAMA_MODELS": str(self.models_dir),
                "OLLAMA_NO_CLOUD": "1",
                "OLLAMA_NOHISTORY": "1",
                "OLLAMA_ORIGINS": "",
            }
        )
        return env

    def start(self, *, timeout: float = 60.0) -> str:
        """Start the server (if it does not run yet) and return its base URL."""
        with self._lock:
            if self._process is not None and self._process.poll() is None and self._base_url:
                return self._base_url
            binary = self.binary
            if binary is None or not binary.is_file():
                raise OllamaError(
                    "runtime_missing",
                    "The Ollama runtime is not installed; install it with install_runtime() "
                    "(`orkexa-local install-runtime --download`, or `--archive FILE` offline)",
                )
            self._reclaim_orphan()
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OllamaError(
                    "models_dir", f"The model folder {self.models_dir} cannot be used: {exc}"
                ) from exc
            if not os.access(self.models_dir, os.W_OK):
                raise OllamaError("models_dir", f"The model folder {self.models_dir} is not writable")
            try:
                # The server's home holds its key pair, the logs may show prompts and model names:
                # both are private to this user (on a host that several users share).
                make_private_dirs(self.home / "ollama-home")
                make_private_dirs(self.log_file.parent)
            except OSError as exc:
                raise OllamaError(
                    "start_failed", f"{self.home} cannot be used for the server's files: {exc}"
                ) from exc
            port = self.port or free_port(self.host)
            if self.port is not None and detect_ollama(f"http://{self._authority}:{port}") is not None:
                raise OllamaError("port_in_use", f"Another Ollama already listens on port {port}")
            try:
                log_fd = self._open_log()
            except OSError as exc:
                raise OllamaError(
                    "start_failed", f"The log {self.log_file} cannot be written: {exc}"
                ) from exc
            kwargs: dict[str, Any] = {}
            if platform.system() == "Windows":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                kwargs["start_new_session"] = True
            try:
                self._process = subprocess.Popen(
                    [str(binary), "serve"],
                    env=self._env(port),
                    stdin=subprocess.DEVNULL,
                    stdout=log_fd,
                    stderr=subprocess.STDOUT,
                    cwd=self.home,
                    **kwargs,
                )
            except OSError as exc:
                raise OllamaError(
                    "start_failed",
                    f"{binary} cannot be started: {exc}; reinstall the runtime "
                    "(`orkexa-local install-runtime --download`)",
                ) from None
            finally:
                os.close(log_fd)
            self._bind_lifetime(binary)
            url = f"http://{self._authority}:{port}"
            client = OllamaClient(url, timeout=3)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise OllamaError(
                        "start_failed", f"Ollama exited with {self._process.returncode}; see {self.log_file}"
                    )
                if client.is_running():
                    if self._process.poll() is not None:
                        # Something else answered on the port (taken between choosing it and the
                        # start), and our server gave up: never hand out another server's address.
                        raise OllamaError(
                            "port_in_use",
                            f"Another server answered on port {port} and Ollama exited "
                            f"with {self._process.returncode}; start again (see {self.log_file})",
                        )
                    self._base_url = url
                    self._write_state(port, binary)
                    return url
                time.sleep(0.2)
            self._terminate()
            raise OllamaError(
                "start_failed", f"Ollama did not answer within {timeout:.0f} s; see {self.log_file}"
            )

    def _bind_lifetime(self, binary: Path) -> None:
        """Make sure the server just started ends with this process (see the module docstring)."""
        self._release_lifetime()  # of a server that exited by itself before (a failed start)
        process = self._process
        if process is None:
            return
        if platform.system() == "Windows":
            self._job = _lifeline.kill_on_close_job(int(process._handle))  # type: ignore[attr-defined]
        else:
            self._lifeline = _lifeline.start_watcher(process.pid, str(binary))

    def _release_lifetime(self) -> None:
        """After the server stopped: end the watcher, close the job (which ends what is left of it)."""
        watcher, self._lifeline = self._lifeline, None
        _lifeline.release_watcher(watcher)
        job, self._job = self._job, None
        _lifeline.close_job(job)

    def _open_log(self) -> int:
        """The log file, opened for appending (``0600`` when new); rotated first if it grew too large."""
        log = self.log_file
        with contextlib.suppress(FileNotFoundError):
            if log.stat().st_size > LOG_ROTATE_BYTES:
                os.replace(log, log.with_name(log.name + ".1"))
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        return os.open(log, flags, PRIVATE_FILE_MODE)

    @property
    def _state_file(self) -> Path:
        return self.home / "ollama.state.json"

    def _write_state(self, port: int, binary: Path) -> None:
        state = {
            "pid": self._process.pid if self._process else None,
            "port": port,
            "version": OLLAMA_VERSION,
            "binary": str(binary),
            "owner": os.getpid(),
        }
        write_private_text(self._state_file, json.dumps(state))

    def _reclaim_orphan(self) -> None:
        """Stop a server that an earlier, crashed process of this application left running.

        The state file names the server, its executable and the process that owned it. The
        server is stopped only if that owner is gone and the process is still ``<binary> serve``
        leading its own process group (POSIX); otherwise it is left alone. A state file that
        names no such server is removed."""
        path = self._state_file
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            state = None
        if isinstance(state, dict):
            pid, binary, owner = state.get("pid"), state.get("binary"), state.get("owner")
            if isinstance(owner, int) and owner != os.getpid() and _alive(owner):
                return  # another running process of this application owns that server
            if isinstance(pid, int) and isinstance(binary, str) and _is_managed_server(pid, binary):
                _stop_group(pid)
        path.unlink(missing_ok=True)

    def _terminate(self, timeout: float = 10.0) -> None:
        process = self._process
        if process is None:
            return
        running = process.poll() is None
        if running:
            try:
                if platform.system() == "Windows":
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    process.terminate()
                process.wait(timeout)
            except (subprocess.TimeoutExpired, OSError):
                process.kill()
                process.wait(5)
        if running and platform.system() != "Windows":
            # The server runs in a session of its own: its model runners are in its process group
            # and must not outlive it (they would keep the loaded model's memory). Only right after
            # stopping it, while the group id cannot belong to anything else yet.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        self._release_lifetime()
        self._process = None
        self._base_url = None
        self._state_file.unlink(missing_ok=True)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the server this object started (never another process)."""
        with self._lock:
            self._terminate(timeout)

    def restart(self, **changes: Any) -> str:
        """Stop, apply ``models_dir``/``offline``/``environment`` changes and start again."""
        self.stop()
        if "models_dir" in changes:
            self.models_dir = Path(changes.pop("models_dir"))
        if "offline" in changes:
            self.offline = bool(changes.pop("offline"))
        if "environment" in changes:
            self.environment = dict(changes.pop("environment") or {})
        if changes:
            raise TypeError(f"Unknown settings: {', '.join(changes)}")
        return self.start()

    def status(self) -> ServerState:
        binary = self.binary
        running = self._process is not None and self._process.poll() is None and self._base_url is not None
        version = None
        error = None
        if running:
            try:
                version = self.client().version()
            except OllamaError as exc:
                error = str(exc)
                running = False
        elif self._process is not None and self._process.poll() is not None:
            error = f"Ollama exited with {self._process.returncode}; see {self.log_file}"
        return ServerState(
            running=running,
            base_url=self._base_url if running else None,
            version=version,
            pid=self._process.pid if running and self._process else None,
            binary=str(binary) if binary else None,
            models_dir=str(self.models_dir),
            error=error,
        )

    def __enter__(self) -> ManagedOllama:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
