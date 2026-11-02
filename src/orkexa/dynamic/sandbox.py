"""Execution of generated tool code outside the host process.

Two backends share one interface:

* :class:`DockerSandbox` runs the code in a throwaway container without network,
  with a read-only file system, no capabilities, an unprivileged user and
  memory/CPU/process limits. It is the isolation boundary for untrusted
  (model-written) code and the default of :class:`~orkexa.dynamic.ToolRegistry`.
  Without Docker or the image it fails closed: nothing runs.
* :class:`SubprocessSandbox` runs the code in a separate Python process with an
  empty environment (no API keys), a temporary working directory, a timeout
  and, on Linux/macOS, CPU/memory/file/process limits. It is **not** a security
  boundary: the process can read every file the user can read and reach the
  network. Use it only for code you trust or review before it runs (development,
  tests, demos); creating one without ``acknowledge_unsafe=True`` emits an
  :class:`~orkexa.dynamic.UnsafeOptInWarning`.

Other backends (gVisor, Firecracker, a hosted sandbox) only need to implement
:meth:`Sandbox._execute`; they should also describe what they enforce
(:meth:`Sandbox.capabilities`) and whether they can run here (:meth:`Sandbox.probe`).
The operating system sandbox :class:`~orkexa.dynamic.MacOSSandbox` is experimental.

Every call can be cancelled: by its caller (:func:`orkexa.dynamic.cancel_scope`)
or for all callers at once (:meth:`Sandbox.cancel`). The call's process (and
container) is killed, its directory removed, and :class:`SandboxCancelled` raised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import IO, Any, ClassVar
import uuid
import warnings

from ..errors import ToolExecutionError
from .capabilities import SandboxCapabilities, SandboxProbe
from .errors import SandboxError, UnsafeOptInWarning
from .lifecycle import CancelCheck, SandboxCancelled, cancel_requested, current_cancel
from .static_check import TOOL_MODULE

__all__ = ["DockerSandbox", "Sandbox", "SandboxTestResult", "SubprocessSandbox", "ValidationResult"]

_RUNNER = Path(__file__).with_name("_sandbox_runner.py")
_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # .../orkexa


@dataclass(frozen=True)
class SandboxTestResult:
    name: str
    ok: bool
    error: str | None


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of running generated code and its tests in the sandbox.

    On success, ``name``, ``description``, ``input_schema`` and
    ``output_schema`` hold the tool definition as derived by ``@Tool`` inside
    the sandbox.
    """

    ok: bool
    error: str | None
    tests: tuple[SandboxTestResult, ...]
    output: str
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def feedback(self) -> str:
        """A compact error report suitable for the Coding Agent."""
        lines = []
        if self.error:
            lines.append(f"Error: {self.error}")
        if not self.tests and not self.error:
            lines.append("No tests were run.")
        lines += [f"{t.name} failed: {t.error}" for t in self.tests if not t.ok]
        if self.output.strip():
            lines.append(f"Captured output:\n{self.output.strip()}")
        return "\n".join(lines)


class Sandbox(ABC):
    """Runs generated tool code outside the host process.

    Args:
        timeout: Wall-clock limit per run in seconds.
        memory_mb: Memory limit per run.
        max_output_chars: Maximum length of a tool result.
    """

    def __init__(
        self, *, timeout: float = 10.0, memory_mb: int = 1024, max_output_chars: int = 65_536
    ) -> None:
        if timeout <= 0 or memory_mb < 64 or max_output_chars < 1:
            raise ValueError("timeout must be positive, memory_mb >= 64, max_output_chars >= 1")
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.max_output_chars = max_output_chars
        #: Stop events of the calls running now (:meth:`cancel`).
        self._running: set[threading.Event] = set()
        self._running_lock = threading.Lock()

    #: Whether the backend is meant as a security boundary for untrusted code. What it actually enforces is
    #: :meth:`capabilities`; whether that suffices for a kind of code is the application's decision.
    isolated: ClassVar[bool] = False
    #: Whether the run directory must be readable for another (container) user.
    _shared_workdir: ClassVar[bool] = False

    def capabilities(self) -> SandboxCapabilities:
        """What this backend enforces. Subclasses describe themselves; this default claims nothing."""
        return SandboxCapabilities(
            backend=type(self).__name__,
            isolation="none",
            filesystem="host",
            network="host",
            processes="allowed",
            clean_environment=False,
            limits=frozenset({"output"}),
            memory="none",
            cancellable=False,
            maturity="experimental",
            notes=("A custom backend that does not describe its guarantees.",),
        )

    @classmethod
    def probe(cls) -> SandboxProbe:
        """Whether this backend can run on this computer now (cheap checks; no tool code is run)."""
        return SandboxProbe(backend=cls.__name__, available=True)

    def cancel(self) -> None:
        """Stop every call this sandbox is running now, of every caller.

        Each stopped call raises :class:`SandboxCancelled`.

        To stop only one caller's call, bind a cancel check with :func:`orkexa.dynamic.cancel_scope`."""
        with self._running_lock:
            for stop in self._running:
                stop.set()

    @contextlib.contextmanager
    def _call_scope(self) -> Iterator[threading.Event]:
        """A stop event for one call, set by :meth:`cancel`; backends pass it to :func:`run_bounded`."""
        stop = threading.Event()
        with self._running_lock:
            self._running.add(stop)
        try:
            yield stop
        finally:
            with self._running_lock:
                self._running.discard(stop)

    @abstractmethod
    def _execute(self, workdir: Path, request: str) -> str:
        """Run the sandbox runner in ``workdir`` with ``request`` on stdin; return its stdout."""

    def _run(self, files: dict[str, str], request: dict[str, Any]) -> dict[str, Any]:
        # CPU limit slightly above the wall-clock timeout, so the timeout normally fires first.
        request["limits"] = {"cpu_seconds": int(self.timeout) + 1, "memory_mb": self.memory_mb}
        with tempfile.TemporaryDirectory(prefix="orkexa-sandbox-") as directory:
            workdir = Path(directory)  # mode 0700
            for name, content in files.items():
                path = workdir / name
                path.write_text(content, encoding="utf-8")
                if self._shared_workdir:
                    path.chmod(0o644)
            if self._shared_workdir:
                workdir.chmod(0o755)  # readable for the unprivileged container user
            stdout = self._execute(workdir, json.dumps(request))
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise SandboxError(f"Malformed sandbox output: {stdout[-500:]!r}") from exc
        if not isinstance(response, dict):
            raise SandboxError("Malformed sandbox output: expected a JSON object")
        return response

    def validate(self, code: str, tests: str, function_name: str) -> ValidationResult:
        """Derive the tool schema from ``code`` and run ``tests`` against it."""
        response = self._run(
            {f"{TOOL_MODULE}.py": code, f"test_{TOOL_MODULE}.py": tests},
            {"mode": "validate", "function_name": function_name},
        )
        tests_run = tuple(
            SandboxTestResult(name=str(t.get("name")), ok=bool(t.get("ok")), error=t.get("error"))
            for t in response.get("tests") or []
            if isinstance(t, dict)
        )
        input_schema = response.get("input_schema")
        output_schema = response.get("output_schema")
        return ValidationResult(
            ok=bool(response.get("ok")),
            error=response.get("error"),
            tests=tests_run,
            output=str(response.get("output") or ""),
            name=response.get("name"),
            description=response.get("description"),
            input_schema=input_schema if isinstance(input_schema, dict) else None,
            output_schema=output_schema if isinstance(output_schema, dict) else None,
        )

    def call(self, code: str, function_name: str, payload: dict[str, Any]) -> str:
        """Execute the tool with a JSON payload; raise ToolExecutionError on failure."""
        try:
            response = self._run(
                {f"{TOOL_MODULE}.py": code},
                {"mode": "call", "function_name": function_name, "payload": payload},
            )
        except SandboxError as exc:
            raise ToolExecutionError(f"{function_name}: {exc}") from exc
        if not response.get("ok"):
            raise ToolExecutionError(f"{function_name}: {response.get('error') or 'unknown sandbox error'}")
        result = response.get("result")
        if not isinstance(result, str):
            raise ToolExecutionError(f"{function_name}: sandbox returned no result")
        if len(result) > self.max_output_chars:
            raise ToolExecutionError(
                f"{function_name}: result has {len(result)} characters, limit is {self.max_output_chars}"
            )
        return result


#: Bytes kept of a sandbox's diagnostic output (stderr): only its end is shown in errors.
_STDERR_TAIL = 64 * 1024


@dataclass(frozen=True)
class BoundedOutput:
    """What :func:`communicate_bounded` collected from a process."""

    stdout: bytes
    #: The last bytes of stderr (at most ``max_stderr``).
    stderr: bytes
    #: Whether stdout passed ``max_stdout``; the process was then killed and ``stdout`` is cut.
    overflow: bool


def response_limit(max_output_chars: int) -> int:
    """The largest runner response (bytes) for a tool result of ``max_output_chars`` characters.

    The response is JSON: a result of n characters takes at most 12n bytes (``ensure_ascii`` escapes
    astral characters as two ``\\uXXXX`` sequences), plus test reports and captured output.
    """
    return max(12 * max_output_chars, 0) + 1024 * 1024


def communicate_bounded(
    process: subprocess.Popen[bytes],
    data: bytes,
    *,
    max_stdout: int,
    max_stderr: int = _STDERR_TAIL,
    on_overflow: Any = None,
) -> BoundedOutput:
    """Like :meth:`subprocess.Popen.communicate`, but the host never holds more than ``max_stdout`` bytes of
    stdout and ``max_stderr`` bytes of stderr, whatever the process writes.

    Code in a sandbox can write to the inherited stdout and stderr directly (``os.write(1, ...)``), past
    the runner's own output limit. The sandbox's memory limit applies to the child, not to what the host
    reads, so an unbounded read would let one tool call exhaust the memory of the API or worker process.
    When stdout passes the limit, ``on_overflow()`` is called (default: ``process.kill()``) and reading
    stops; stderr keeps only its end. Blocks until both pipes are closed.
    """
    stdout: list[bytes] = []
    tail = bytearray()
    state = {"size": 0, "overflow": False}
    lock = threading.Lock()

    def stop() -> None:
        with contextlib.suppress(OSError):
            (on_overflow or process.kill)()

    def read_stdout(pipe: IO[bytes]) -> None:
        while True:
            chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
            if not chunk:
                return
            with lock:
                if state["overflow"]:
                    continue  # drain, so the process is never blocked on a full pipe before it dies
                if state["size"] + len(chunk) > max_stdout:
                    state["overflow"] = True
                    stdout.append(chunk[: max_stdout - state["size"]])
                    state["size"] = max_stdout
                    overflow = True
                else:
                    stdout.append(chunk)
                    state["size"] += len(chunk)
                    overflow = False
            if overflow:
                stop()

    def read_stderr(pipe: IO[bytes]) -> None:
        while True:
            chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
            if not chunk:
                return
            tail.extend(chunk)
            if len(tail) > max_stderr:
                del tail[: len(tail) - max_stderr]

    def write_stdin(pipe: IO[bytes]) -> None:
        try:
            pipe.write(data)
        except (BrokenPipeError, OSError):
            pass
        finally:
            with contextlib.suppress(OSError):
                pipe.close()

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=read_stdout, args=(process.stdout,), daemon=True),
        threading.Thread(target=read_stderr, args=(process.stderr,), daemon=True),
    ]
    if process.stdin is not None:
        threads.append(threading.Thread(target=write_stdin, args=(process.stdin,), daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    process.wait()
    return BoundedOutput(stdout=b"".join(stdout), stderr=bytes(tail), overflow=bool(state["overflow"]))


def run_bounded(
    args: list[str],
    data: bytes,
    *,
    timeout: float,
    max_stdout: int,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    on_abort: Any = None,
    cancel: CancelCheck | None = None,
    stop: threading.Event | None = None,
    new_session: bool = False,
    popen: Callable[..., Any] | None = None,
    watch: Callable[[Any], str | None] | None = None,
) -> tuple[int, BoundedOutput]:
    """Run ``args`` with ``data`` on stdin under a wall-clock ``timeout`` and :func:`communicate_bounded`.

    Returns the exit code and the output. Raises :class:`SandboxError` on a timeout or when stdout passes
    ``max_stdout``, and :class:`SandboxCancelled` when ``cancel()`` returns ``True`` (default: the check bound
    with :func:`orkexa.dynamic.cancel_scope`) or ``stop`` is set; in each case the process is killed and
    ``on_abort()`` runs (e.g. removing a container that the killed client started). ``new_session`` starts the
    process in its own session on POSIX, so killing it also kills every process it started. ``watch(process)``
    is called every 50 ms; a message it returns stops the run with that message (e.g. a memory watchdog).
    ``popen`` replaces :class:`subprocess.Popen` (a backend that starts processes its own way).
    Raises :class:`OSError` when the program cannot be started.
    """
    check = cancel if cancel is not None else current_cancel()
    kwargs: dict[str, Any] = {}
    if new_session and os.name == "posix":
        kwargs["start_new_session"] = True
    process = (popen or subprocess.Popen)(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=cwd,
        **kwargs,
    )

    def kill() -> None:
        _kill_tree(process, group=bool(kwargs))

    result: dict[str, BoundedOutput] = {}
    reader = threading.Thread(
        target=lambda: result.setdefault(
            "output", communicate_bounded(process, data, max_stdout=max_stdout, on_overflow=kill)
        ),
        daemon=True,
    )
    reader.start()
    deadline = time.monotonic() + timeout
    problem: SandboxError | None = None
    try:
        while reader.is_alive():
            reader.join(0.05)
            if not reader.is_alive():
                break
            if (stop is not None and stop.is_set()) or cancel_requested(check):
                problem = SandboxCancelled("The call was cancelled; its process was stopped")
            elif time.monotonic() > deadline:
                problem = SandboxError(f"Timed out after {timeout:g} seconds")
            elif watch is not None:
                message = watch(process)
                if message:
                    problem = SandboxError(message)
            if problem is not None:
                kill()
                reader.join(10)
                break
    finally:
        if problem is not None or reader.is_alive():
            kill()
    if problem is not None:
        if on_abort is not None:
            on_abort()
        raise problem
    output = result["output"]
    if output.overflow:
        if on_abort is not None:
            on_abort()
        raise SandboxError(f"The sandbox wrote more than {max_stdout} bytes of output; the run was stopped")
    return process.returncode if process.returncode is not None else -1, output


def _kill_tree(process: Any, *, group: bool) -> None:
    """Kill ``process`` and, with ``group``, its whole session (POSIX); wait for it.

    The group is only signalled while the leader is alive (not yet reaped), so its id cannot have been
    reused."""
    if group and process.poll() is None:
        killpg = getattr(os, "killpg", None)
        if killpg is not None:
            with contextlib.suppress(OSError):
                killpg(process.pid, 9)
    if process.poll() is None:
        with contextlib.suppress(OSError):
            process.kill()
    with contextlib.suppress(Exception):
        process.wait(10)


def _minimal_environment() -> dict[str, str]:
    """No inherited variables, so no API keys or tokens reach the code."""
    if sys.platform == "win32":  # pragma: no cover - Python needs SYSTEMROOT on Windows
        return {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows")}
    return {}


class SubprocessSandbox(Sandbox):
    """Separate Python process: empty environment, temp dir, timeout, resource limits.

    **Not a security boundary** (see the module docstring): only for code you trust
    or review before it runs. Untrusted, model-written code belongs in
    :class:`DockerSandbox` or a stronger isolation. Creating one emits an
    :class:`~orkexa.dynamic.UnsafeOptInWarning`; pass ``acknowledge_unsafe=True``
    to state the choice explicitly and silence it.

    Args:
        python: Interpreter to use; defaults to the current one. It needs
            ``pydantic`` and ``typing-extensions``; ``orkexa`` itself is
            loaded from this package. A virtual environment's interpreter makes
            the packages installed there importable for the generated code.
        env: Variables for the sandbox process, e.g. ``{"OPENBLAS_NUM_THREADS": "1"}``
            for numerical libraries under the process limit. Nothing else is
            inherited from the host, so never put secrets here.
        acknowledge_unsafe: Confirms that the code it runs is trusted or reviewed.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        memory_mb: int = 1024,
        max_output_chars: int = 65_536,
        python: str | None = None,
        env: Mapping[str, str] | None = None,
        acknowledge_unsafe: bool = False,
    ) -> None:
        super().__init__(timeout=timeout, memory_mb=memory_mb, max_output_chars=max_output_chars)
        if not acknowledge_unsafe:
            warnings.warn(
                "SubprocessSandbox is not a security boundary: the code it runs can read the user's "
                "files and reach the network. Use DockerSandbox for untrusted code, or pass "
                "SubprocessSandbox(acknowledge_unsafe=True) for trusted or reviewed code",
                UnsafeOptInWarning,
                stacklevel=2,
            )
        self.python = python or sys.executable
        self.env = dict(env or {})

    def capabilities(self) -> SandboxCapabilities:
        limits = {"wall_clock", "output"}
        if os.name == "posix":
            limits |= {"cpu", "file_size"}
        if sys.platform.startswith("linux"):
            limits.add("memory")
        return SandboxCapabilities(
            backend="subprocess",
            isolation="none",
            filesystem="host",
            network="host",
            processes="allowed",
            clean_environment=True,
            limits=frozenset(limits),
            memory="kernel" if sys.platform.startswith("linux") else "none",
            cancellable=True,
            maturity="stable",
            notes=(
                "Not a security boundary: the code runs as the user and can read the user's files and reach "
                "the network, including localhost.",
            ),
        )

    def _execute(self, workdir: Path, request: str) -> str:
        with self._call_scope() as stop:
            code, output = run_bounded(
                [self.python, "-I", "-B", str(_RUNNER)],
                request.encode("utf-8"),
                timeout=self.timeout,
                max_stdout=response_limit(self.max_output_chars),
                env={**self.env, **_minimal_environment()},
                cwd=workdir,
                stop=stop,
                new_session=True,
            )
        stdout = output.stdout.decode("utf-8", "replace")
        if not stdout.strip():
            stderr = output.stderr.decode("utf-8", "replace")
            raise SandboxError(f"Sandbox process exited with code {code}: {stderr[-1000:]}")
        return stdout


_DOCKER_HINT = (
    "Generated code runs only in an isolated sandbox: install Docker and build the image "
    "(docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .), or, for trusted or "
    "reviewed code only, pass sandbox=SubprocessSandbox(acknowledge_unsafe=True)."
)


class DockerSandbox(Sandbox):
    """Throwaway Docker container: no network, read-only FS, no capabilities, nobody user.

    The image needs Python 3.10+ with ``pydantic`` and ``typing-extensions`` installed;
    ``docker/sandbox.Dockerfile`` in the repository builds a suitable one::

        docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .

    The installed ``orkexa`` package is mounted read-only into the container. If
    Docker or the image is missing, every run fails with a :class:`SandboxError`
    and no code is executed.
    """

    isolated: ClassVar[bool] = True
    _shared_workdir: ClassVar[bool] = True

    def __init__(
        self,
        *,
        image: str = "orkexa-sandbox",
        timeout: float = 20.0,
        memory_mb: int = 512,
        max_output_chars: int = 65_536,
        cpus: float = 1.0,
        pids_limit: int = 64,
        docker: str = "docker",
    ) -> None:
        super().__init__(timeout=timeout, memory_mb=memory_mb, max_output_chars=max_output_chars)
        self.image = image
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker = docker

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            backend="docker",
            isolation="container",
            filesystem="private",
            network="none",
            processes="limited",
            clean_environment=True,
            limits=frozenset(
                {"wall_clock", "cpu", "memory", "processes", "output", "temp_size", "file_size"}
            ),
            memory="kernel",
            cancellable=True,
            maturity="stable",
            notes=(
                "Shares the kernel with the host (or with Docker Desktop's virtual machine).",
                "Only as strong as the Docker installation: a misconfigured daemon may drop limits.",
            ),
        )

    @classmethod
    def probe(cls, docker: str = "docker", image: str = "orkexa-sandbox") -> SandboxProbe:
        """Whether the Docker CLI, a reachable daemon and the sandbox image are present."""
        if shutil.which(docker) is None:
            return SandboxProbe("docker", False, "missing", "The docker command is not installed")
        try:
            info = subprocess.run([docker, "info"], capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SandboxProbe("docker", False, "unreachable", f"docker info failed: {exc}")
        if info.returncode != 0:
            return SandboxProbe("docker", False, "unreachable", "The Docker daemon is not reachable")
        try:
            found = subprocess.run(
                [docker, "image", "inspect", image], capture_output=True, timeout=10, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SandboxProbe("docker", False, "image", f"Cannot inspect the image {image}: {exc}")
        if found.returncode != 0:
            return SandboxProbe("docker", False, "image", f"The sandbox image {image} is not built")
        return SandboxProbe("docker", True)

    @staticmethod
    def available(docker: str = "docker") -> bool:
        """Whether a Docker daemon is reachable."""
        if shutil.which(docker) is None:
            return False
        try:
            return (
                subprocess.run([docker, "info"], capture_output=True, timeout=10, check=False).returncode == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

    def command(self, workdir: Path, container_name: str) -> list[str]:
        """The ``docker run`` command line for one sandbox run."""
        return [
            self.docker,
            "run",
            "--rm",
            "-i",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            f"{self.memory_mb}m",
            "--memory-swap",
            f"{self.memory_mb}m",
            "--cpus",
            str(self.cpus),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "-v",
            f"{workdir}:/work:ro",
            "-v",
            f"{_PACKAGE_DIR}:/opt/orkexa/orkexa:ro",
            "-w",
            "/work",
            self.image,
            "python",
            "-I",
            "-B",
            "/opt/orkexa/orkexa/dynamic/_sandbox_runner.py",
        ]

    def _execute(self, workdir: Path, request: str) -> str:
        name = f"orkexa-sandbox-{uuid.uuid4().hex[:12]}"

        def remove() -> None:
            # Killing the docker client does not stop the container.
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, check=False)

        try:
            with self._call_scope() as stop:
                code, output = run_bounded(
                    self.command(workdir, name),
                    request.encode("utf-8"),
                    timeout=self.timeout,
                    max_stdout=response_limit(self.max_output_chars),
                    env=None,  # the docker CLI's environment is never passed into the container
                    on_abort=remove,
                    stop=stop,
                )
        except OSError as exc:
            raise SandboxError(f"Cannot start docker: {exc}. {_DOCKER_HINT}") from exc
        stdout = output.stdout.decode("utf-8", "replace")
        if not stdout.strip():
            stderr = output.stderr.decode("utf-8", "replace")
            raise SandboxError(
                f"Container exited with code {code}: {stderr[-1000:]}"
                + (f" {_DOCKER_HINT}" if code == 125 else "")
            )
        return stdout
