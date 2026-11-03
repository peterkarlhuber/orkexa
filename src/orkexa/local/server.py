"""A managed Ollama server: a child process this application starts, watches and stops.

:class:`ManagedOllama` runs the ``ollama`` binary of :mod:`orkexa.local.install` (or one
you name) on a free loopback port with its own model directory, so it neither needs
an installed Ollama nor touches one that already runs: an Ollama on port 11434 is
only *detected* (:func:`detect_ollama`), never stopped or reconfigured.

The child gets a minimal environment: ``OLLAMA_HOST`` (loopback), ``OLLAMA_MODELS``,
``OLLAMA_NO_CLOUD=1`` (no Ollama cloud models), no history file, no browser origins.
"""

from __future__ import annotations

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

from ._http import OllamaError
from .client import OllamaClient
from .install import OLLAMA_VERSION, installed_runtime

__all__ = ["DEFAULT_PORT", "ManagedOllama", "ServerState", "detect_ollama", "free_port"]

DEFAULT_PORT = 11434
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
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
            raise OllamaError("not_running", "The managed Ollama server is not running")
        return OllamaClient(self._base_url)

    def _env(self, port: int) -> dict[str, str]:
        env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
        if self.offline:
            for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
                env.pop(key, None)
        env["HOME"] = env["USERPROFILE"] = str(self.home / "ollama-home")  # keys and caches stay here
        env.update(self.environment)
        env.update(
            {
                "OLLAMA_HOST": f"{self.host}:{port}",
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
                raise OllamaError("runtime_missing", "The Ollama runtime is not installed")
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OllamaError(
                    "models_dir", f"The model folder {self.models_dir} cannot be used: {exc}"
                ) from exc
            (self.home / "ollama-home").mkdir(parents=True, exist_ok=True)
            if not os.access(self.models_dir, os.W_OK):
                raise OllamaError("models_dir", f"The model folder {self.models_dir} is not writable")
            port = self.port or free_port(self.host)
            if self.port is not None and detect_ollama(f"http://{self.host}:{port}") is not None:
                raise OllamaError("port_in_use", f"Another Ollama already listens on port {port}")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_file.open("ab")
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
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.home,
                    **kwargs,
                )
            finally:
                log.close()
            url = f"http://{self.host}:{port}"
            client = OllamaClient(url, timeout=3)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise OllamaError(
                        "start_failed", f"Ollama exited with {self._process.returncode}; see {self.log_file}"
                    )
                if client.is_running():
                    self._base_url = url
                    self._write_state(port)
                    return url
                time.sleep(0.2)
            self._terminate()
            raise OllamaError(
                "start_failed", f"Ollama did not answer within {timeout:.0f} s; see {self.log_file}"
            )

    def _write_state(self, port: int) -> None:
        state = {"pid": self._process.pid if self._process else None, "port": port, "version": OLLAMA_VERSION}
        (self.home / "ollama.state.json").write_text(json.dumps(state), encoding="utf-8")

    def _terminate(self, timeout: float = 10.0) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                if platform.system() == "Windows":
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    process.terminate()
                process.wait(timeout)
            except (subprocess.TimeoutExpired, OSError):
                process.kill()
                process.wait(5)
        self._process = None
        self._base_url = None
        (self.home / "ollama.state.json").unlink(missing_ok=True)

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
