"""Isolated execution of generated tool code.

Two backends share one interface:

* :class:`SubprocessSandbox` runs the code in a separate Python process with an
  empty environment (no API keys), a temporary working directory, a timeout
  and, on Linux/macOS, CPU/memory/file/process limits. It is a layer of
  defense for development, **not** a security boundary: the process can still
  reach the network and read files the user can read. The static check blocks
  the obvious ways to do so.
* :class:`DockerSandbox` runs the same code in a throwaway container without
  network, with a read-only file system, no capabilities and an unprivileged
  user. Use it in production.

Other backends (gVisor, Firecracker, a hosted sandbox) only need to implement
:meth:`Sandbox._execute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import uuid

from ..errors import ToolExecutionError
from .errors import SandboxError
from .static_check import TOOL_MODULE

__all__ = ["DockerSandbox", "Sandbox", "SandboxTestResult", "SubprocessSandbox", "ValidationResult"]

_RUNNER = Path(__file__).with_name("_sandbox_runner.py")
_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # .../claude_agent


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
    """Runs generated tool code in isolation.

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

    @abstractmethod
    def _execute(self, workdir: Path, request: str) -> str:
        """Run the sandbox runner in ``workdir`` with ``request`` on stdin; return its stdout."""

    def _run(self, files: dict[str, str], request: dict[str, Any]) -> dict[str, Any]:
        # CPU limit slightly above the wall-clock timeout, so the timeout normally fires first.
        request["limits"] = {"cpu_seconds": int(self.timeout) + 1, "memory_mb": self.memory_mb}
        with tempfile.TemporaryDirectory(prefix="claude-agent-sandbox-") as directory:
            workdir = Path(directory)
            workdir.chmod(0o755)  # readable for an unprivileged container user
            for name, content in files.items():
                path = workdir / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o644)
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


def _minimal_environment() -> dict[str, str]:
    """No inherited variables, so no API keys or tokens reach the code."""
    if sys.platform == "win32":  # pragma: no cover - Python needs SYSTEMROOT on Windows
        return {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows")}
    return {}


class SubprocessSandbox(Sandbox):
    """Separate Python process: empty environment, temp dir, timeout, resource limits.

    A defense-in-depth layer for development, not a security boundary (see the
    module docstring). Prefer :class:`DockerSandbox` in production.

    Args:
        python: Interpreter to use; defaults to the current one. It needs
            ``pydantic`` and ``typing-extensions``; ``claude_agent`` itself is
            loaded from this package. A virtual environment's interpreter makes
            the packages installed there importable for the generated code.
        env: Variables for the sandbox process, e.g. ``{"OPENBLAS_NUM_THREADS": "1"}``
            for numerical libraries under the process limit. Nothing else is
            inherited from the host, so never put secrets here.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        memory_mb: int = 1024,
        max_output_chars: int = 65_536,
        python: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, memory_mb=memory_mb, max_output_chars=max_output_chars)
        self.python = python or sys.executable
        self.env = dict(env or {})

    def _execute(self, workdir: Path, request: str) -> str:
        try:
            completed = subprocess.run(
                [self.python, "-I", "-B", str(_RUNNER)],
                input=request,
                capture_output=True,
                text=True,
                cwd=workdir,
                env={**self.env, **_minimal_environment()},
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"Timed out after {self.timeout:g} seconds") from exc
        if not completed.stdout.strip():
            raise SandboxError(
                f"Sandbox process exited with code {completed.returncode}: {completed.stderr[-1000:]}"
            )
        return completed.stdout


class DockerSandbox(Sandbox):
    """Throwaway Docker container: no network, read-only FS, no capabilities, nobody user.

    The image needs Python 3.10+ with ``pydantic`` and ``typing-extensions`` installed;
    ``docker/sandbox.Dockerfile`` in the repository builds a suitable one::

        docker build -t claude-agent-sandbox -f docker/sandbox.Dockerfile .

    The installed ``claude_agent`` package is mounted read-only into the container.
    """

    def __init__(
        self,
        *,
        image: str = "claude-agent-sandbox",
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
            f"{_PACKAGE_DIR}:/opt/claude-agent/claude_agent:ro",
            "-w",
            "/work",
            self.image,
            "python",
            "-I",
            "-B",
            "/opt/claude-agent/claude_agent/dynamic/_sandbox_runner.py",
        ]

    def _execute(self, workdir: Path, request: str) -> str:
        name = f"claude-agent-sandbox-{uuid.uuid4().hex[:12]}"
        try:
            completed = subprocess.run(
                self.command(workdir, name),
                input=request,
                capture_output=True,
                text=True,
                env=None,  # the docker CLI's environment is never passed into the container
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, check=False)
            raise SandboxError(f"Timed out after {self.timeout:g} seconds") from exc
        except OSError as exc:
            raise SandboxError(f"Cannot start docker: {exc}") from exc
        if not completed.stdout.strip():
            raise SandboxError(
                f"Container exited with code {completed.returncode}: {completed.stderr[-1000:]}"
            )
        return completed.stdout
