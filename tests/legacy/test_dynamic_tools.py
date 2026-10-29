from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_agent import ClaudeAgent, Tool
from claude_agent.dynamic import (
    ApprovalDeniedError,
    ApprovalRequest,
    AutoApprover,
    CallbackApprover,
    CodingAgent,
    ConsoleApprover,
    DenyApprover,
    DynamicPolicy,
    PolicyViolationError,
    RegistryError,
    SandboxedTool,
    SubprocessSandbox,
    ToolBuilder,
    ToolProposal,
    ToolRegistry,
    ToolValidationError,
)

from .conftest import FakeClient
from .dynamic_helpers import MARGIN, MARGIN_CODE, MARGIN_TESTS, parsed, text, tool_result, tool_use


@Tool
def web_search(query: str) -> str:
    """Search the web.

    Args:
        query: Search terms.

    Returns:
        Search results.
    """
    return f"results for {query}"


@pytest.fixture(scope="module")
def sandbox() -> SubprocessSandbox:
    return SubprocessSandbox()


def _builder(
    client: FakeClient, sandbox: SubprocessSandbox, *, approver: object = None, **kwargs: object
) -> ToolBuilder:
    coding_agent = CodingAgent(client=client)  # type: ignore[arg-type]
    return ToolBuilder(
        coding_agent,
        ToolRegistry(sandbox=sandbox),
        approver=approver or AutoApprover(),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ---- registry ------------------------------------------------------------------


def _add_margin(registry: ToolRegistry) -> None:
    result = registry.sandbox.validate(MARGIN_CODE, MARGIN_TESTS, "calculate_operating_margin")
    assert result.name and result.description and result.input_schema
    record = registry.add_generated(
        name=result.name,
        function_name="calculate_operating_margin",
        description=result.description,
        code=MARGIN_CODE,
        tests=MARGIN_TESTS,
        input_schema=result.input_schema,
        output_schema=result.output_schema,
        requirement="operating margin",
    )
    registry.approve(record.name, record.version)


def test_trusted_tools_run_in_host(sandbox: SubprocessSandbox) -> None:
    registry = ToolRegistry(sandbox=sandbox)
    record = registry.register_trusted(web_search)
    assert record.kind == "trusted" and record.status == "approved"
    assert registry.get_tool("web_search") is web_search
    assert "web_search" in registry and len(registry) == 1
    with pytest.raises(RegistryError):
        registry.register_trusted(web_search)
    with pytest.raises(RegistryError, match="trusted"):
        registry.set_status("web_search", 1, "revoked")
    registry.revoke("web_search")
    assert "web_search" not in registry


def test_generated_tool_lifecycle_and_versions(sandbox: SubprocessSandbox) -> None:
    registry = ToolRegistry(sandbox=sandbox)
    _add_margin(registry)
    tool = registry.get_tool("calculate_operating_margin")
    assert isinstance(tool, SandboxedTool)
    assert tool.execute({"revenue": 10, "operating_income": 1}) == "0.1"
    _add_margin(registry)
    assert [r.version for r in registry.versions("calculate_operating_margin")] == [1, 2]
    assert registry.record("calculate_operating_margin").version == 2
    assert registry.catalog() == [
        {
            "name": "calculate_operating_margin",
            "description": registry.record("calculate_operating_margin").description,
        }
    ]
    registry.revoke("calculate_operating_margin")
    assert "calculate_operating_margin" not in registry
    assert {r.status for r in registry.versions("calculate_operating_margin")} == {"revoked"}
    with pytest.raises(RegistryError):
        registry.get_tool("calculate_operating_margin")
    with pytest.raises(RegistryError):
        registry.revoke("calculate_operating_margin")


def test_pending_tools_are_not_usable(sandbox: SubprocessSandbox) -> None:
    registry = ToolRegistry(sandbox=sandbox)
    registry.add_generated(
        name="f", function_name="f", description="d", code="", tests="", input_schema={}, output_schema=None
    )
    assert "f" not in registry
    with pytest.raises(RegistryError):
        registry.get_tool("f")


def test_search(sandbox: SubprocessSandbox) -> None:
    registry = ToolRegistry(sandbox=sandbox)
    registry.register_trusted(web_search)
    _add_margin(registry)
    assert [r.name for r in registry.search("operating margin")] == ["calculate_operating_margin"]
    assert [r.name for r in registry.search("search the web")] == ["web_search"]
    assert registry.search("unrelated words") == []
    assert len(registry.search("")) == 2


def test_persistence_and_tamper_detection(tmp_path: Path, sandbox: SubprocessSandbox) -> None:
    registry = ToolRegistry(tmp_path, sandbox=sandbox)
    _add_margin(registry)
    file = tmp_path / "calculate_operating_margin" / "v1.json"
    assert json.loads(file.read_text())["status"] == "approved"

    reloaded = ToolRegistry(tmp_path, sandbox=sandbox)
    assert (
        reloaded.get_tool("calculate_operating_margin").execute({"revenue": 2, "operating_income": 1})
        == "0.5"
    )

    data = json.loads(file.read_text())
    data["code"] = data["code"].replace("operating_income / revenue", "revenue")
    file.write_text(json.dumps(data))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "v1.json").write_text("not json")
    assert "calculate_operating_margin" not in ToolRegistry(tmp_path, sandbox=sandbox)


# ---- approvers -------------------------------------------------------------------


def test_approvers() -> None:
    request = ApprovalRequest(kind="tool", name="t", summary="s", details={"code": "x"})
    assert AutoApprover().approve(request)
    assert not DenyApprover().approve(request)
    assert CallbackApprover(lambda r: r.name == "t").approve(request)
    printed: list[str] = []
    console = ConsoleApprover(input_func=lambda prompt: "j", print_func=printed.append)
    assert console.approve(request)
    assert any("Approval required" in line for line in printed)
    assert not ConsoleApprover(input_func=lambda prompt: "", print_func=printed.append).approve(request)


# ---- builder -------------------------------------------------------------------


def test_build_registers_and_returns_sandboxed_tool(sandbox: SubprocessSandbox) -> None:
    client = FakeClient([parsed(MARGIN)])
    seen: list[ApprovalRequest] = []
    builder = _builder(client, sandbox, approver=CallbackApprover(lambda r: seen.append(r) is None))
    tool = builder.build("Compute the operating margin.")
    assert isinstance(tool, SandboxedTool)
    assert tool.execute({"revenue": 200, "operating_income": 50}) == "0.25"
    assert (
        builder.registry.record("calculate_operating_margin").requirement == "Compute the operating margin."
    )
    assert seen[0].kind == "tool" and "2 tests passed" in seen[0].summary
    assert seen[0].details["code"] == MARGIN_CODE
    assert (
        "Requirement:\nCompute the operating margin."
        in client.messages.parse_calls[0]["messages"][0]["content"]
    )


def test_build_feeds_errors_back_until_valid(sandbox: SubprocessSandbox) -> None:
    unsafe = MARGIN.model_copy(update={"code": "import os\n" + MARGIN_CODE})
    failing = MARGIN.model_copy(update={"tests": MARGIN_TESTS + "\ndef test_bad():\n    assert False\n"})
    client = FakeClient([parsed(unsafe), parsed(failing), parsed(MARGIN)])
    builder = _builder(client, sandbox)
    builder.build("margin")
    second_prompt = client.messages.parse_calls[1]["messages"][-1]["content"]
    third_prompt = client.messages.parse_calls[2]["messages"][-1]["content"]
    assert "import of 'os' is not allowed" in second_prompt
    assert "test_bad failed" in third_prompt
    # the correction happens in the same conversation
    assert len(client.messages.parse_calls[2]["messages"]) == 5


def test_build_with_extra_imports_and_own_test_sandbox(sandbox: SubprocessSandbox) -> None:
    with_math = MARGIN.model_copy(
        update={
            "code": "import cmath\n"
            + MARGIN_CODE.replace("revenue <= 0", "revenue <= 0 or cmath.isnan(revenue)")
        }
    )
    tests_sandbox = SubprocessSandbox(timeout=10)
    client = FakeClient([parsed(with_math)])
    builder = ToolBuilder(
        CodingAgent(client=client),  # type: ignore[arg-type]
        ToolRegistry(sandbox=sandbox),
        approver=AutoApprover(),
        extra_imports={"cmath"},
        sandbox=tests_sandbox,
    )
    assert builder.sandbox is tests_sandbox and builder.registry.sandbox is sandbox
    tool = builder.build("margin")
    assert isinstance(tool, SandboxedTool) and tool.sandbox is sandbox


def test_build_rejects_imports_that_are_not_extra(sandbox: SubprocessSandbox) -> None:
    with_numpy = MARGIN.model_copy(update={"code": "import numpy\n" + MARGIN_CODE})
    client = FakeClient([parsed(with_numpy), parsed(MARGIN)])
    builder = _builder(client, sandbox)
    builder.build("margin")
    assert "import of 'numpy' is not allowed" in client.messages.parse_calls[1]["messages"][-1]["content"]


def test_build_gives_up_after_max_attempts(sandbox: SubprocessSandbox) -> None:
    bad = MARGIN.model_copy(update={"code": "import socket\n" + MARGIN_CODE})
    builder = _builder(FakeClient([parsed(bad), parsed(bad)]), sandbox, max_attempts=2)
    with pytest.raises(ToolValidationError, match="after 2 attempts"):
        builder.build("margin")


def test_build_rejects_reserved_and_existing_names(sandbox: SubprocessSandbox) -> None:
    reserved = ToolProposal(
        function_name="run_agent",
        description="d",
        code=MARGIN_CODE.replace("calculate_operating_margin", "run_agent"),
        tests=MARGIN_TESTS.replace("calculate_operating_margin", "run_agent"),
    )
    builder = _builder(FakeClient([parsed(reserved)]), sandbox, max_attempts=1)
    with pytest.raises(ToolValidationError, match="reserved"):
        builder.build("x")
    builder = _builder(FakeClient([parsed(MARGIN), parsed(MARGIN)]), sandbox, max_attempts=1)
    builder.build("x")
    with pytest.raises(ToolValidationError, match="already exists"):
        builder.build("x")


def test_build_approval_denied_marks_rejected(sandbox: SubprocessSandbox) -> None:
    builder = _builder(FakeClient([parsed(MARGIN)]), sandbox, approver=DenyApprover())
    with pytest.raises(ApprovalDeniedError):
        builder.build("margin")
    assert [r.status for r in builder.registry.versions("calculate_operating_margin")] == ["rejected"]


def test_build_respects_generated_tool_limit(sandbox: SubprocessSandbox) -> None:
    builder = _builder(FakeClient([]), sandbox, policy=DynamicPolicy(max_generated_tools=0))
    with pytest.raises(PolicyViolationError):
        builder.build("margin")


def test_investment_agent_creates_and_uses_tool_in_one_ask(sandbox: SubprocessSandbox) -> None:
    """The scenario from the plan: detect a gap, build the tool, use it, all in one ask()."""
    client = FakeClient(
        [
            tool_use("find_tools", {"query": "operating margin"}, "t1"),
            tool_use("create_tool", {"requirement": "Operating margin from revenue and income."}, "t2"),
            parsed(MARGIN),  # the Coding Agent's answer
            tool_use("calculate_operating_margin", {"revenue": 200, "operating_income": 50}, "t3"),
            text("The operating margin is 25 %."),
        ]
    )
    investment_agent = ClaudeAgent(name="Investment Agent", model="claude-opus-5-5", client=client)  # type: ignore[arg-type]
    builder = ToolBuilder(CodingAgent(client=client), ToolRegistry(sandbox=sandbox), approver=AutoApprover())  # type: ignore[arg-type]
    for tool in builder.as_tools(investment_agent):
        investment_agent.add_tool(tool)

    assert investment_agent.ask("Analyze operating profitability.") == "The operating margin is 25 %."

    calls = client.messages.create_calls
    assert json.loads(tool_result(calls[1])["content"]) == []
    assert "was created and registered" in tool_result(calls[2])["content"]
    assert "calculate_operating_margin" in [t["name"] for t in calls[2]["tools"]]
    assert "calculate_operating_margin" not in [t["name"] for t in calls[0]["tools"]]
    assert tool_result(calls[3])["content"] == "0.25"
    assert investment_agent.usage_tracker is not builder.coding_agent.agent.usage_tracker

    # revoking removes it from the agent again
    builder.revoke("calculate_operating_margin")
    assert "calculate_operating_margin" not in investment_agent.tools


def test_load_tool_and_failed_creation_are_reported_as_text(sandbox: SubprocessSandbox) -> None:
    bad = MARGIN.model_copy(update={"code": "import os\n" + MARGIN_CODE})
    client = FakeClient([parsed(bad)])
    agent = ClaudeAgent(model="claude-opus-5-5", client=client)  # type: ignore[arg-type]
    builder = _builder(client, sandbox, max_attempts=1)
    builder.registry.register_trusted(web_search)
    find_tools, load_tool, create_tool = builder.as_tools(agent)
    assert "web_search" in find_tools(query="web")
    assert "now available" in load_tool(name="web_search")
    assert "already available" in load_tool(name="web_search")
    assert load_tool(name="nope").startswith("Error:")
    assert create_tool(requirement="margin").startswith("Error: tool could not be created")
