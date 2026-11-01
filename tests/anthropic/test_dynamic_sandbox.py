from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from orkexa import ClaudeAgent, ToolDefinitionError, ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxedTool, SandboxError, SubprocessSandbox
from orkexa.dynamic.sandbox import communicate_bounded

from .conftest import FakeClient
from .dynamic_helpers import MARGIN_CODE, MARGIN_TESTS, text, tool_result, tool_use

_DOC = '"""Do it.\n\n    Returns:\n        Something.\n    """'


@pytest.fixture(scope="module")
def sandbox() -> SubprocessSandbox:
    return SubprocessSandbox(timeout=10)


def test_validate_derives_schema_and_runs_tests(sandbox: SubprocessSandbox) -> None:
    result = sandbox.validate(MARGIN_CODE, MARGIN_TESTS, "calculate_operating_margin")
    assert result.ok, result.feedback()
    assert result.name == "calculate_operating_margin"
    assert result.description is not None and result.description.startswith("Calculate the operating margin.")
    assert result.input_schema is not None
    assert result.input_schema["required"] == ["revenue", "operating_income"]
    assert result.input_schema["additionalProperties"] is False
    assert result.output_schema == {"type": "number"}
    assert [t.name for t in result.tests] == ["test_basic", "test_invalid_revenue"]


def test_failing_tests_are_reported(sandbox: SubprocessSandbox) -> None:
    tests = (
        MARGIN_TESTS
        + "\n\ndef test_wrong():\n    print('debug output')\n    assert 1 == 2, 'one is not two'\n"
    )
    result = sandbox.validate(MARGIN_CODE, tests, "calculate_operating_margin")
    assert not result.ok
    assert "test_wrong failed: AssertionError: one is not two" in result.feedback()
    assert "debug output" in result.output


def test_invalid_tool_definition_is_reported(sandbox: SubprocessSandbox) -> None:
    code = "def f(x):\n    return x\n"
    result = sandbox.validate(code, "def test_a():\n    assert True\n", "f")
    assert not result.ok
    assert result.error is not None and "missing docstring" in result.error


def test_environment_is_empty(sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    # The sandbox itself does not run the static check, so os is importable here.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    code = f"import os\n\ndef env() -> str:\n    {_DOC}\n    return ','.join(sorted(os.environ))\n"
    assert "ANTHROPIC_API_KEY" not in sandbox.call(code, "env", {})


def test_extra_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    configured = SubprocessSandbox(env={"OPENBLAS_NUM_THREADS": "1"})
    code = f"import os\n\ndef env() -> str:\n    {_DOC}\n    return ','.join(sorted(os.environ))\n"
    names = configured.call(code, "env", {}).split(",")
    assert "OPENBLAS_NUM_THREADS" in names and "ANTHROPIC_API_KEY" not in names


def test_working_directory_is_temporary(sandbox: SubprocessSandbox) -> None:
    code = f"import os\n\ndef cwd() -> str:\n    {_DOC}\n    return os.getcwd()\n"
    cwd = sandbox.call(code, "cwd", {})
    assert "orkexa-sandbox-" in cwd
    assert not Path(cwd).exists()


def test_timeout(sandbox: SubprocessSandbox) -> None:
    fast = SubprocessSandbox(timeout=1)
    code = f"def spin() -> str:\n    {_DOC}\n    while True:\n        pass\n"
    with pytest.raises(SandboxError, match="Timed out"):
        fast.validate(code, "from tool import spin\n\ndef test_a():\n    spin()\n", "spin")
    with pytest.raises(ToolExecutionError, match="Timed out"):
        fast.call(code, "spin", {})


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only reliable on Linux")
def test_memory_limit() -> None:
    small = SubprocessSandbox(memory_mb=256)
    code = f"def hog() -> str:\n    {_DOC}\n    return str(len(bytearray(1024 * 1024 * 1024)))\n"
    with pytest.raises(ToolExecutionError, match="MemoryError"):
        small.call(code, "hog", {})


def test_output_limit() -> None:
    tiny = SubprocessSandbox(max_output_chars=10)
    code = f"def long() -> str:\n    {_DOC}\n    return 'x' * 100\n"
    with pytest.raises(ToolExecutionError, match="limit is 10"):
        tiny.call(code, "long", {})


def test_raw_output_is_bounded_in_the_host() -> None:
    # Code that writes past the runner straight to the inherited stdout must not fill the host's memory:
    # reading stops at the response limit and the process is killed.
    tiny = SubprocessSandbox(max_output_chars=10, timeout=20)
    code = (
        "import os\n\n"
        f"def flood() -> str:\n    {_DOC}\n"
        "    block = b'x' * (1024 * 1024)\n"
        "    for _ in range(64):\n        os.write(1, block)\n"
        "    return 'done'\n"
    )
    with pytest.raises(ToolExecutionError, match="more than 1048696 bytes of output"):
        tiny.call(code, "flood", {})


def test_communicate_bounded_keeps_the_stderr_tail() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stderr.write('a' * 200000 + 'END'); print('ok')"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = communicate_bounded(process, b"", max_stdout=100, max_stderr=10)
    assert output.stdout.strip() == b"ok" and not output.overflow
    assert output.stderr == b"aaaaaaaEND"


def test_tool_errors_become_tool_execution_errors(sandbox: SubprocessSandbox) -> None:
    with pytest.raises(ToolExecutionError, match="revenue must be positive"):
        sandbox.call(MARGIN_CODE, "calculate_operating_margin", {"revenue": 0, "operating_income": 1})
    with pytest.raises(ToolExecutionError, match="invalid tool inputs"):
        sandbox.call(MARGIN_CODE, "calculate_operating_margin", {"revenue": "a lot", "operating_income": 1})


def _margin_tool(sandbox: SubprocessSandbox) -> SandboxedTool:
    result = sandbox.validate(MARGIN_CODE, MARGIN_TESTS, "calculate_operating_margin")
    assert result.name and result.description and result.input_schema
    return SandboxedTool(
        name=result.name,
        description=result.description,
        input_schema=result.input_schema,
        output_schema=result.output_schema,
        code=MARGIN_CODE,
        function_name="calculate_operating_margin",
        sandbox=sandbox,
    )


def test_sandboxed_tool_schema_execute_and_direct_call(sandbox: SubprocessSandbox) -> None:
    tool = _margin_tool(sandbox)
    schema = tool.schema()
    assert schema["name"] == "calculate_operating_margin"
    assert schema["strict"] is True
    assert schema["input_schema"]["properties"]["revenue"]["type"] == "number"
    assert tool.execute({"revenue": 200, "operating_income": 50}) == "0.25"
    assert tool(revenue=200, operating_income=50) == "0.25"
    assert "SubprocessSandbox" in repr(tool)


def test_sandboxed_tool_checks_payload_in_host(sandbox: SubprocessSandbox) -> None:
    tool = _margin_tool(sandbox)
    with pytest.raises(ToolExecutionError, match="missing: \\['operating_income'\\]"):
        tool.execute({"revenue": 200})
    with pytest.raises(ToolExecutionError, match="unknown: \\['extra'\\]"):
        tool.execute({"revenue": 200, "operating_income": 1, "extra": 1})


def test_sandboxed_tool_rejects_bad_schema(sandbox: SubprocessSandbox) -> None:
    with pytest.raises(ToolDefinitionError):
        SandboxedTool(
            name="x",
            description="d",
            input_schema={"type": "object", "additionalProperties": True},
            output_schema=None,
            code="",
            function_name="x",
            sandbox=sandbox,
        )
    with pytest.raises(ToolDefinitionError):
        SandboxedTool(
            name="bad name",
            description="d",
            input_schema={"type": "object"},
            output_schema=None,
            code="",
            function_name="x",
            sandbox=sandbox,
        )


def test_agent_uses_sandboxed_tool(sandbox: SubprocessSandbox) -> None:
    client = FakeClient(
        [tool_use("calculate_operating_margin", {"revenue": 200, "operating_income": 50}), text("25 %")]
    )
    agent = ClaudeAgent(model="claude-opus-5-5", client=client)  # type: ignore[arg-type]
    agent.add_tool(_margin_tool(sandbox))
    assert agent.ask("Margin?") == "25 %"
    assert tool_result(client.messages.create_calls[1])["content"] == "0.25"
    assert agent.remove_tool("calculate_operating_margin").name == "calculate_operating_margin"
    assert agent.tools == {}


def test_docker_command_is_locked_down(tmp_path: Path) -> None:
    command = DockerSandbox(memory_mb=256).command(tmp_path, "sb-1")
    joined = " ".join(command)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--memory 256m",
        f"{tmp_path}:/work:ro",
    ):
        assert flag in joined
    assert "-e" not in command and "--env" not in command


def _docker_image_ready(image: str = "orkexa-sandbox") -> bool:  # Orkexa: renamed image
    if not DockerSandbox.available():
        return False
    completed = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False)
    return completed.returncode == 0


@pytest.mark.skipif(
    not _docker_image_ready(),
    reason="needs Docker and the image from docker/sandbox.Dockerfile",
)
def test_docker_sandbox_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:  # pragma: no cover
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    sandbox = DockerSandbox()
    result = sandbox.validate(MARGIN_CODE, MARGIN_TESTS, "calculate_operating_margin")
    assert result.ok, result.feedback()
    payload = {"revenue": 4, "operating_income": 1}
    assert sandbox.call(MARGIN_CODE, "calculate_operating_margin", payload) == "0.25"

    probe = (
        "import os, socket\n\n"
        f"def probe() -> str:\n    {_DOC}\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "        net = 'online'\n"
        "    except OSError:\n"
        "        net = 'offline'\n"
        "    try:\n"
        "        open('/work/x', 'w')\n"
        "        fs = 'writable'\n"
        "    except OSError:\n"
        "        fs = 'read-only'\n"
        "    return f\"{net} {fs} {os.getuid()} {'ANTHROPIC_API_KEY' in os.environ}\"\n"
    )
    assert sandbox.call(probe, "probe", {}) == "offline read-only 65534 False"
