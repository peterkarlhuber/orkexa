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
The former macOS sandbox :class:`~orkexa.dynamic.MacOSSandbox` was removed (Docker-first, ADR 0012); the
name remains as a deprecated placeholder that never runs code.

Every call can be cancelled: by its caller (:func:`orkexa.dynamic.cancel_scope`)
or for all callers at once (:meth:`Sandbox.cancel`). The call's process (and
container) is killed, its directory removed, and :class:`SandboxCancelled` raised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import IO, Any, ClassVar, Literal
import uuid
import warnings

from ..errors import ToolExecutionError
from .capabilities import SandboxCapabilities, SandboxProbe
from .docker import (
    BOOTSTRAP,
    SELF_TEST_SCRIPT,
    ContainerLimits,
    DockerHost,
    DockerProbeError,
    ImageInfo,
    envelope,
    evaluate_self_test,
    host_fingerprint,
    inspect_host,
    inspect_image,
    preflight,
    reap,
    run_args,
    stop_container,
)
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

    def _workdir_shared(self) -> bool:
        """Whether this sandbox's run directory must be readable for another (container) user."""
        return self._shared_workdir

    @abstractmethod
    def _execute(self, workdir: Path, request: str) -> str:
        """Run the sandbox runner in ``workdir`` with ``request`` on stdin; return its stdout."""

    def _run(self, files: dict[str, str], request: dict[str, Any]) -> dict[str, Any]:
        # CPU limit slightly above the wall-clock timeout, so the timeout normally fires first.
        request["limits"] = {"cpu_seconds": int(self.timeout) + 1, "memory_mb": self.memory_mb}
        with tempfile.TemporaryDirectory(prefix="orkexa-sandbox-") as directory:
            workdir = Path(directory)  # mode 0700
            shared = self._workdir_shared()
            for name, content in files.items():
                path = workdir / name
                path.write_text(content, encoding="utf-8")
                if shared:
                    path.chmod(0o644)
            if shared:
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
#: Seconds a container may outlive its call's timeout before :func:`~orkexa.dynamic.docker.reap` removes it.
_DEADLINE_GRACE = 60.0
#: Seconds a passed preflight and self-test are trusted before a sandbox inspects the daemon again.
_READY_SECONDS = 60.0
#: Least time a self-test container gets (the first start of an image can be slow).
_SELF_TEST_TIMEOUT = 60.0
#: Passed self-tests by (host fingerprint, image id, shape hash).
_SELF_TESTS: dict[tuple[str, str, str], SandboxProbe] = {}
_SELF_TESTS_LOCK = threading.Lock()
#: Where the deprecated bind transport mounts the orkexa package.
_BIND_RUNNER = "/opt/orkexa/orkexa/dynamic/_sandbox_runner.py"


class DockerSandbox(Sandbox):
    """Throwaway Docker container: no network, read-only FS, no capabilities, nobody user.

    The image needs Python 3.10+ with ``pydantic`` and ``typing-extensions`` installed;
    ``docker/sandbox.Dockerfile`` in the repository builds a suitable one::

        docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .

    Each call runs in a new container of the shape :func:`orkexa.dynamic.docker.run_args` (no network,
    read-only root, no capabilities, ``nobody``, memory without swap, CPU, process, file and open-file limits,
    ``noexec`` temporary file systems, never pulled). The tool code, the payload and the runner modules travel
    on stdin into a ``tmpfs`` work directory (``transport="stdin"``); nothing of the host's file system is
    mounted, so the daemon may run on another host or in a VM. The containers are labelled with ``owner`` and
    a deadline, so :func:`orkexa.dynamic.docker.reap` can remove what a crashed process left behind.

    A run starts only when the daemon passes :func:`orkexa.dynamic.docker.preflight`, the image exists and
    :meth:`self_test` passed for this daemon, image and shape; otherwise it fails with a
    :class:`SandboxError` and no code is executed. Calls run the image by its id.

    Args:
        transport: ``"stdin"`` (default) or the deprecated ``"bind"``, which mounts the run directory and the
            installed ``orkexa`` package read-only from the host (the daemon must see the host's files).
        owner: The value of the ``org.orkexa.sandbox`` label of this sandbox's containers.
        environ: The environment of every ``docker`` CLI process this sandbox starts (preflight, image
            inspection, self-test, runs, stopping and reaping containers), e.g. only ``PATH``, ``HOME`` and
            ``DOCKER_HOST``; ``None`` (default) inherits this process's. It never reaches the container:
            :func:`~orkexa.dynamic.docker.run_args` passes no ``-e``.
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
        transport: Literal["stdin", "bind"] = "stdin",
        owner: str = "orkexa",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, memory_mb=memory_mb, max_output_chars=max_output_chars)
        if transport not in ("stdin", "bind"):
            raise ValueError("transport must be 'stdin' or 'bind'")
        if transport == "bind":
            warnings.warn(
                "DockerSandbox(transport='bind') is deprecated: it mounts host directories into the "
                "container and needs a daemon that sees the host's files; transport='stdin' does neither",
                DeprecationWarning,
                stacklevel=2,
            )
        self.image = image
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker = docker
        self.transport: Literal["stdin", "bind"] = transport
        self.owner = owner
        self.environ: dict[str, str] | None = dict(environ) if environ is not None else None
        #: (monotonic time, image id) of the last passed preflight and self-test.
        self._ready: tuple[float, str] | None = None
        self._ready_lock = threading.Lock()
        # Validates the limits and the owner now, not at the first call.
        run_args(docker, "orkexa-check", image or "unset", self.limits(), owner=owner)

    def _workdir_shared(self) -> bool:
        return self.transport == "bind"

    def limits(self) -> ContainerLimits:
        """The limits of this sandbox's containers."""
        return ContainerLimits(
            memory_mb=self.memory_mb,
            cpus=self.cpus,
            pids=self.pids_limit,
            lifetime_s=self.timeout + _DEADLINE_GRACE,
        )

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            backend="docker",
            isolation="container",
            filesystem="private",
            network="none",
            processes="limited",
            clean_environment=True,
            limits=frozenset(
                {
                    "wall_clock",
                    "cpu",
                    "memory",
                    "processes",
                    "output",
                    "temp_size",
                    "file_size",
                    "open_files",
                }
            ),
            memory="kernel",
            cancellable=True,
            maturity="stable",
            notes=(
                "Shares the kernel with the host (or with Docker Desktop's virtual machine).",
                "Refuses to run on a daemon that does not report memory, process and CPU limits and seccomp.",
            ),
        )

    @classmethod
    def probe(cls, docker: str = "docker", image: str = "orkexa-sandbox") -> SandboxProbe:
        """Whether this sandbox can run here: the Docker CLI, a reachable daemon that passes
        :func:`~orkexa.dynamic.docker.preflight`, the image, and a passed :meth:`self_test` (a container of
        exactly the sandbox's shape, run once per daemon, image and shape). The daemon is judged by what it
        enforces, never by its name; no tool code runs."""
        return cls(docker=docker, image=image)._readiness()

    @staticmethod
    def available(docker: str = "docker", image: str = "orkexa-sandbox") -> bool:
        """Whether :meth:`probe` passes: preflight and self-test included."""
        return DockerSandbox.probe(docker, image).available

    @classmethod
    def clear_self_test_cache(cls) -> None:
        """Forget every passed self-test, e.g. after the daemon's configuration was changed in place."""
        with _SELF_TESTS_LOCK:
            _SELF_TESTS.clear()

    def _host_and_image(self, image: str | None) -> tuple[SandboxProbe, DockerHost | None, ImageInfo | None]:
        """CLI, daemon, preflight and image: the checks before any container starts."""
        ref = image or self.image
        found = (
            shutil.which(self.docker)
            if self.environ is None or "PATH" not in self.environ
            else shutil.which(self.docker, path=self.environ["PATH"])
        )
        if found is None:
            return (
                SandboxProbe("docker", False, "missing", f"The {self.docker} command is not installed"),
                None,
                None,
            )
        try:
            host = inspect_host(self.docker, env=self.environ)
        except DockerProbeError as exc:
            return SandboxProbe("docker", False, exc.reason, exc.detail), None, None
        checked = preflight(host)
        if not checked.available:
            return checked, host, None
        try:
            info = inspect_image(self.docker, ref, env=self.environ)
        except (SandboxError, ValueError) as exc:
            return (
                SandboxProbe(
                    "docker", False, "image", f"Cannot inspect the image {ref}: {exc}", checked.facts
                ),
                host,
                None,
            )
        if info is None:
            return (
                SandboxProbe(
                    "docker", False, "image", f"The sandbox image {ref} is not built", checked.facts
                ),
                host,
                None,
            )
        return checked, host, info

    def self_test(self, *, image: str | None = None) -> SandboxProbe:
        """Run one container with exactly this sandbox's :func:`~orkexa.dynamic.docker.run_args` and check
        inside it what the daemon applied: user 65534, no capabilities, ``NoNewPrivs``, a seccomp filter,
        only the loopback interface, a read-only root, ``noexec`` ``/tmp`` and ``/work``, the cgroup v2
        memory, swap, process and CPU limits, the open-file, file-size and core limits, and no environment
        variable but the engine's and the image's own (:func:`~orkexa.dynamic.docker.evaluate_self_test`).

        ``image`` defaults to :attr:`image`; the container runs the image by its id. A passed self-test is
        remembered per daemon (:func:`~orkexa.dynamic.docker.host_fingerprint`), image id and shape. Reason
        ``self-test`` when the container shows a problem (cgroup v1 included)."""
        checked, host, info = self._host_and_image(image)
        if host is None or info is None:
            return checked
        limits = self.limits()
        shape = run_args("docker", "orkexa-self-test", info.id, limits, owner=self.owner, deadline=0)
        key = (
            host_fingerprint(host),
            info.id,
            hashlib.sha256(json.dumps([*shape, info.env_names]).encode("utf-8")).hexdigest(),
        )
        facts = (*checked.facts, ("image_id", info.id), ("host_fingerprint", key[0]))
        with _SELF_TESTS_LOCK:
            remembered = _SELF_TESTS.get(key)
        if remembered is not None:
            return remembered
        name = f"orkexa-selftest-{uuid.uuid4().hex[:12]}"
        args = [*run_args(self.docker, name, info.id, limits, owner=self.owner), "python", "-I", "-B", "-c"]

        def refuse(detail: str) -> SandboxProbe:
            return SandboxProbe("docker", False, "self-test", detail, facts)

        try:
            with self._call_scope() as stop:
                code, output = run_bounded(
                    [*args, SELF_TEST_SCRIPT],
                    b"",
                    timeout=max(self.timeout, _SELF_TEST_TIMEOUT),
                    max_stdout=1024 * 1024,
                    env=self.environ,
                    on_abort=lambda: stop_container(self.docker, name, env=self.environ),
                    stop=stop,
                )
        except SandboxCancelled:
            raise
        except (OSError, SandboxError) as exc:
            return refuse(f"The self-test container did not run: {exc}")
        lines = output.stdout.decode("utf-8", "replace").strip().splitlines()
        try:
            report: Any = json.loads(lines[-1]) if code == 0 and lines else None
        except ValueError:
            report = None
        if not isinstance(report, dict):
            stderr = output.stderr.decode("utf-8", "replace")[-500:]
            return refuse(f"The self-test container exited with code {code} without a report: {stderr}")
        problems = evaluate_self_test(report, limits, info.env_names)
        if problems:
            return refuse("; ".join(problems))
        passed = SandboxProbe("docker", True, facts=(*facts, ("self_test", "passed")))
        with _SELF_TESTS_LOCK:
            _SELF_TESTS[key] = passed
        return passed

    def _readiness(self) -> SandboxProbe:
        """Preflight and self-test (the self-test includes the CLI, daemon and image checks)."""
        return self.self_test()

    def _require_ready(self) -> str:
        """The id of the image to run; raises :class:`SandboxError` unless preflight and self-test pass.

        The result is reused for :data:`_READY_SECONDS`; after that the daemon is inspected again."""
        with self._ready_lock:
            if self._ready is not None and time.monotonic() - self._ready[0] < _READY_SECONDS:
                return self._ready[1]
        ready = self._readiness()
        if ready.available:
            image_id = dict(ready.facts)["image_id"]
            with self._ready_lock:
                self._ready = (time.monotonic(), image_id)
            return image_id
        if ready.reason == "missing":
            raise SandboxError(f"Cannot start docker: {ready.detail}. {_DOCKER_HINT}")
        raise SandboxError(
            f"The Docker sandbox is not available ({ready.reason}): {ready.detail}. {_DOCKER_HINT}"
        )

    def command(self, workdir: Path, container_name: str, *, image: str | None = None) -> list[str]:
        """The ``docker run`` command line for one sandbox run (``image`` defaults to :attr:`image`)."""
        args = run_args(self.docker, container_name, image or self.image, self.limits(), owner=self.owner)
        if self.transport == "stdin":
            return [*args, "python", "-I", "-B", "-c", BOOTSTRAP]
        # Deprecated: the run directory and the package are mounted from the host instead of a tmpfs /work.
        index = args.index("--tmpfs", args.index("--tmpfs") + 1)
        del args[index : index + 2]
        mounts = ["-v", f"{workdir}:/work:ro", "-v", f"{_PACKAGE_DIR}:/opt/orkexa/orkexa:ro"]
        return [*args[:-1], *mounts, args[-1], "python", "-I", "-B", _BIND_RUNNER]

    def _execute(self, workdir: Path, request: str) -> str:
        image = self._require_ready()
        name = f"orkexa-sandbox-{uuid.uuid4().hex[:12]}"
        if self.transport == "stdin":
            files = {p.name: p.read_text(encoding="utf-8") for p in sorted(workdir.iterdir()) if p.is_file()}
            data = envelope(files, request)
        else:
            data = request.encode("utf-8")

        def remove() -> None:
            # Killing the docker client does not stop the container.
            stop_container(self.docker, name, env=self.environ)

        try:
            with self._call_scope() as stop:
                code, output = run_bounded(
                    self.command(workdir, name, image=image),
                    data,
                    timeout=self.timeout,
                    max_stdout=response_limit(self.max_output_chars),
                    env=self.environ,  # the docker CLI's environment; it never reaches the container
                    on_abort=remove,
                    stop=stop,
                )
        except OSError as exc:
            raise SandboxError(f"Cannot start docker: {exc}. {_DOCKER_HINT}") from exc
        stdout = output.stdout.decode("utf-8", "replace")
        if not stdout.strip():
            with self._ready_lock:
                self._ready = None  # check the daemon and the image again before the next call
            stderr = output.stderr.decode("utf-8", "replace")
            raise SandboxError(
                f"Container exited with code {code}: {stderr[-1000:]}"
                + (f" {_DOCKER_HINT}" if code == 125 else "")
            )
        return stdout

    def reap(self, *, remove_all: bool = False) -> int:
        """Remove this owner's containers past their deadline (:func:`orkexa.dynamic.docker.reap`)."""
        return reap(self.docker, self.owner, remove_all=remove_all, env=self.environ)
