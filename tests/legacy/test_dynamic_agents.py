from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from claude_agent import ClaudeAgent, Tool
from claude_agent.dynamic import (
    AgentArchitect,
    AgentFactory,
    AgentRegistry,
    AgentSpec,
    ApprovalDeniedError,
    AutoApprover,
    BudgetExceededError,
    BudgetTracker,
    CodingAgent,
    DenyApprover,
    DynamicPolicy,
    Orchestrator,
    OutputFormatRegistry,
    PolicyViolationError,
    RecursionLimitError,
    RegistryError,
    SubprocessSandbox,
    ToolBuilder,
    ToolRegistry,
    ToolRequirement,
)
from claude_agent.dynamic.orchestrator import _depth
from claude_agent.usage import UsageRecord

from .conftest import FakeClient, ParsedResponse, make_message
from .dynamic_helpers import MARGIN, parsed, text, tool_result, tool_use


@Tool
def web_search(query: str) -> str:
    """Search the web.

    Args:
        query: Search terms.

    Returns:
        Search results.
    """
    return f"results for {query}"


class RiskReport(BaseModel):
    """Regulatory risks of a company."""

    risks: list[str]
    severity: int


def _spec(**overrides: Any) -> AgentSpec:
    values: dict[str, Any] = {
        "name": "RegulatoryRiskAgent",
        "description": "Analyzes regulatory risks of companies.",
        "system_prompt": "You analyze regulatory risks.",
        "model": "claude-sonnet-5-5",
        "effort": "medium",
        "tools": ["web_search"],
    }
    values.update(overrides)
    return AgentSpec(**values)


def _registry() -> ToolRegistry:
    registry = ToolRegistry(sandbox=SubprocessSandbox())
    registry.register_trusted(web_search)
    return registry


# ---- policy and budget ---------------------------------------------------------------


def test_policy_defaults_and_validation() -> None:
    policy = DynamicPolicy(allowed_models=["claude-opus-5-5"], allowed_agent_tools=["a"], budget_usd=1.5)
    assert policy.allowed_models == ("claude-opus-5-5",)
    assert policy.allowed_agent_tools == frozenset({"a"})
    assert policy.budget_usd == Decimal("1.5")
    with pytest.raises(ValueError):
        DynamicPolicy(allowed_models=[])
    with pytest.raises(ValueError):
        DynamicPolicy(max_effort="extreme")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DynamicPolicy(budget_usd=-1)
    with pytest.raises(AttributeError):
        policy.max_depth = 5  # type: ignore[misc]


def _record(cost: str | None) -> UsageRecord:
    return UsageRecord(
        agent="a",
        model="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation_5m_input_tokens=0,
        cache_creation_1h_input_tokens=0,
        thinking_tokens=0,
        web_search_requests=0,
        web_fetch_requests=0,
        estimated_cost_usd=None if cost is None else Decimal(cost),
    )


def test_budget_tracker() -> None:
    tracker = BudgetTracker("0.10")
    tracker.add(_record("0.06"))
    tracker.add(_record(None))
    assert tracker.spent_usd == Decimal("0.06") and tracker.remaining_usd == Decimal("0.04")
    with pytest.raises(BudgetExceededError):
        tracker.add(_record("0.05"))
    assert len(tracker.records) == 3  # recorded before raising
    with pytest.raises(BudgetExceededError):
        tracker.check()
    unlimited = BudgetTracker()
    unlimited.add(_record("1000"))
    assert unlimited.remaining_usd is None


def test_budget_aborts_a_running_agent() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}  # $5 on Opus 5.5
    client = FakeClient([make_message([{"type": "text", "text": "x"}], usage=usage)])
    agent = ClaudeAgent(model="claude-opus-5-5", client=client, usage_tracker=BudgetTracker(1))  # type: ignore[arg-type]
    with pytest.raises(BudgetExceededError):
        agent.ask("hi")
    assert agent.history() == []


# ---- factory -------------------------------------------------------------------


def test_factory_creates_agent_and_clamps_limits() -> None:
    policy = DynamicPolicy(max_effort="medium", max_tokens=2048, max_tool_rounds=3)
    tracker = BudgetTracker()
    factory = AgentFactory(_registry(), OutputFormatRegistry([RiskReport]), policy, usage_tracker=tracker)
    spec = _spec(effort="max", max_tokens=100_000, max_tool_rounds=50, output_format="RiskReport")
    agent = factory.create(spec)
    assert agent.name == "RegulatoryRiskAgent"
    assert agent.model == "claude-sonnet-5-5"
    assert agent.effort == "medium" and agent.max_tokens == 2048 and agent.max_tool_rounds == 3
    assert agent.tools["web_search"] is web_search
    assert agent.output_format is RiskReport
    assert agent.usage_tracker is tracker
    assert agent.raise_tool_errors is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "bad name!"}, "invalid agent name"),
        ({"model": "claude-fable-5-1"}, "not allowed"),
        ({"tools": ["shell"]}, "unknown tool 'shell'"),
        ({"tools": ["web_search", "web_search"]}, "duplicate"),
        ({"tools": ["run_agent"]}, "reserved"),
        ({"output_format": "Nope"}, "unknown output format"),
        ({"system_prompt": " "}, "empty system prompt"),
        ({"missing_tools": [ToolRequirement(name="x", requirement="y")]}, "unresolved missing tools"),
    ],
)
def test_factory_rejects_policy_violations(overrides: dict[str, Any], message: str) -> None:
    factory = AgentFactory(_registry())
    with pytest.raises(PolicyViolationError, match=message):
        factory.validate(_spec(**overrides))


def test_factory_tool_allowlist_and_delegation() -> None:
    factory = AgentFactory(_registry(), policy=DynamicPolicy(allowed_agent_tools=[]))
    with pytest.raises(PolicyViolationError, match="not allowed for generated agents"):
        factory.validate(_spec())
    factory = AgentFactory(
        _registry(),
        policy=DynamicPolicy(allow_agent_delegation=True),
        delegation_tools={"run_agent": web_search},
    )
    assert factory.create(_spec(tools=["run_agent"])).tools["web_search"] is web_search


# ---- agent registry --------------------------------------------------------------


def test_agent_registry_versions_and_persistence(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path)
    registry.register(_spec(), "risks")
    registry.register(_spec(description="Second version about regulatory risks."))
    assert registry.get("RegulatoryRiskAgent").version == 2
    assert [r.name for r in registry.search("regulatory")] == ["RegulatoryRiskAgent"]
    assert registry.catalog()[0]["description"].startswith("Second")

    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "v1.json").write_text("{}")
    reloaded = AgentRegistry(tmp_path)
    assert reloaded.get("RegulatoryRiskAgent").version == 2
    assert reloaded.versions("RegulatoryRiskAgent")[0].requirement == "risks"
    reloaded.revoke("RegulatoryRiskAgent")
    assert "RegulatoryRiskAgent" not in AgentRegistry(tmp_path)
    with pytest.raises(RegistryError):
        reloaded.get("RegulatoryRiskAgent")
    with pytest.raises(RegistryError):
        reloaded.revoke("RegulatoryRiskAgent")


# ---- orchestrator -----------------------------------------------------------------


def _orchestrator(client: FakeClient, **kwargs: Any) -> Orchestrator:
    registry = kwargs.pop("tools", None) or _registry()
    builder = ToolBuilder(CodingAgent(client=client), registry, approver=AutoApprover())  # type: ignore[arg-type]
    return Orchestrator(
        architect=AgentArchitect(client=client),  # type: ignore[arg-type]
        approver=kwargs.pop("approver", AutoApprover()),
        tool_builder=builder,
        output_formats=OutputFormatRegistry([RiskReport]),
        client=client,  # type: ignore[arg-type]
        **kwargs,
    )


def test_orchestrator_creates_and_runs_agent_end_to_end() -> None:
    report = RiskReport(risks=["EU AI Act"], severity=3)
    client = FakeClient(
        [
            tool_use("list_agents", {"query": "regulatory"}, "t1"),
            tool_use("create_agent", {"requirement": "Analyze regulatory risks."}, "t2"),
            parsed(_spec(output_format="RiskReport")),  # architect
            tool_use("run_agent", {"agent_name": "RegulatoryRiskAgent", "task": "ACME"}, "t3"),
            tool_use("web_search", {"query": "ACME regulation"}, "s1"),  # specialist
            parsed(report),  # specialist's final answer
            text("ACME faces moderate regulatory risk."),
        ]
    )
    orchestrator = _orchestrator(client)
    assert orchestrator.ask("Analyze ACME's regulatory risks.") == "ACME faces moderate regulatory risk."

    creates = client.messages.create_calls
    assert json.loads(tool_result(creates[1])["content"]) == []
    assert json.loads(tool_result(creates[2])["content"])["created"] == "RegulatoryRiskAgent"
    assert json.loads(tool_result(creates[3])["content"]) == report.model_dump()
    architect_prompt = client.messages.parse_calls[0]["messages"][0]["content"]
    assert '"web_search"' in architect_prompt and "RiskReport" in architect_prompt
    assert "claude-sonnet-5-5" in architect_prompt
    assert set(orchestrator.usage_tracker.usage_by_agent()) == {
        "Orchestrator",
        "Agent Architect",
        "RegulatoryRiskAgent",
    }


def test_create_agent_builds_missing_tools() -> None:
    spec = _spec(tools=[], missing_tools=[ToolRequirement(name="margin", requirement="Operating margin.")])
    client = FakeClient([parsed(spec), parsed(MARGIN)])
    orchestrator = _orchestrator(client)
    record = orchestrator.create_agent("A margin analyst.")
    assert record.spec.tools == ["calculate_operating_margin"]
    assert record.spec.missing_tools == []
    assert "Suggested name: margin" in client.messages.parse_calls[1]["messages"][0]["content"]
    assert set(orchestrator.usage_tracker.usage_by_agent()) == {"Agent Architect", "Coding Agent"}


def test_create_agent_without_tool_builder_rejects_missing_tools() -> None:
    spec = _spec(missing_tools=[ToolRequirement(name="x", requirement="y")])
    client = FakeClient([parsed(spec)])
    orchestrator = Orchestrator(
        architect=AgentArchitect(client=client),  # type: ignore[arg-type]
        approver=AutoApprover(),
        tools=_registry(),
        client=client,  # type: ignore[arg-type]
    )
    with pytest.raises(PolicyViolationError, match="tool generation is disabled"):
        orchestrator.create_agent("x")


def test_create_agent_limits_and_approval() -> None:
    client = FakeClient([parsed(_spec())])
    with pytest.raises(ApprovalDeniedError):
        _orchestrator(client, approver=DenyApprover()).create_agent("x")

    orchestrator = _orchestrator(
        FakeClient([parsed(_spec()), parsed(_spec())]), policy=DynamicPolicy(max_agents=5)
    )
    orchestrator.create_agent("x")
    with pytest.raises(PolicyViolationError, match="already exists"):
        orchestrator.create_agent("x")

    with pytest.raises(PolicyViolationError, match="Limit of 0 agents"):
        _orchestrator(FakeClient([]), policy=DynamicPolicy(max_agents=0)).create_agent("x")
    with pytest.raises(PolicyViolationError, match="not allowed"):
        _orchestrator(FakeClient([parsed(_spec(model="claude-fable-5-1"))])).create_agent("x")


def test_tool_errors_are_returned_to_the_orchestrator_model() -> None:
    client = FakeClient(
        [
            tool_use("run_agent", {"agent_name": "Nobody", "task": "x"}),
            text("No such agent."),
        ]
    )
    orchestrator = _orchestrator(client)
    assert orchestrator.ask("x") == "No such agent."
    assert "No approved agent named 'Nobody'" in tool_result(client.messages.create_calls[1])["content"]


def test_recursion_limit() -> None:
    policy = DynamicPolicy(allow_agent_delegation=True, max_depth=1)
    client = FakeClient(
        [
            tool_use("run_agent", {"agent_name": "Delegator", "task": "go"}, "o1"),
            tool_use("run_agent", {"agent_name": "Delegator", "task": "deeper"}, "d1"),  # specialist tries
            text("could not delegate"),
            text("done"),
        ]
    )
    orchestrator = _orchestrator(client, policy=policy)
    orchestrator.agents.register(_spec(name="Delegator", tools=["run_agent"]))
    assert orchestrator.ask("go") == "done"
    nested = tool_result(client.messages.create_calls[2])["content"]
    assert "Maximum delegation depth 1 reached" in nested
    with pytest.raises(RecursionLimitError):
        _deep_call(orchestrator)


def _deep_call(orchestrator: Orchestrator) -> None:
    """Call run_agent as if from inside a specialist (depth 1)."""
    token = _depth.set(1)
    try:
        orchestrator.run_agent("Delegator", "x")
    finally:
        _depth.reset(token)


def test_orchestrator_budget_is_shared_and_enforced() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    client = FakeClient([ParsedResponse(make_message([], usage=usage), _spec())])
    orchestrator = _orchestrator(client, policy=DynamicPolicy(budget_usd=1))
    assert orchestrator.architect.agent.usage_tracker is orchestrator.usage_tracker
    assert orchestrator.tool_builder is not None
    assert orchestrator.tool_builder.coding_agent.agent.usage_tracker is orchestrator.usage_tracker
    with pytest.raises(BudgetExceededError):
        orchestrator.create_agent("x")
    with pytest.raises(BudgetExceededError):
        orchestrator.ask("anything")


def test_orchestrator_requires_one_registry() -> None:
    client = FakeClient([])
    with pytest.raises(ValueError):
        Orchestrator(architect=AgentArchitect(client=client), approver=AutoApprover())  # type: ignore[arg-type]
    builder = ToolBuilder(CodingAgent(client=client), _registry(), approver=AutoApprover())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Orchestrator(
            architect=AgentArchitect(client=client),  # type: ignore[arg-type]
            approver=AutoApprover(),
            tool_builder=builder,
            tools=_registry(),
        )
