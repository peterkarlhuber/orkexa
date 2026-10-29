"""Mixed-provider systems: delegation, shared tools, judges, orchestration and budgets.

Every agent talks to a scripted SDK client of its own provider (see
``tests/contract/harness.py``); no network access, no API key.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
import warnings

from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    ClaudeAgent,
    Criterion,
    Evaluator,
    ModelPricing,
    SQLiteIndex,
    Tool,
    UsageTracker,
    citations,
    refine,
    retrieval_sources,
)
from orkexa.dynamic import (
    AgentArchitect,
    AgentFactory,
    AgentSpec,
    AutoApprover,
    BudgetTracker,
    CodingAgent,
    DynamicPolicy,
    Orchestrator,
    OutputFormatRegistry,
    PolicyViolationError,
    ProviderSettings,
    SubprocessSandbox,
    ToolBuilder,
    ToolRegistry,
    UnpricedModelError,
    UnpricedUsageWarning,
    UnsafeOptInWarning,
)

from .contract.harness import AnthropicHarness, GeminiHarness, OpenAIHarness, Script
from .legacy.dynamic_helpers import MARGIN

pytestmark = pytest.mark.filterwarnings("ignore::orkexa.dynamic.UnpricedUsageWarning")


class Report(BaseModel):
    risks: list[str]
    severity: int


def _harnesses() -> tuple[AnthropicHarness, OpenAIHarness, GeminiHarness]:
    return AnthropicHarness(), OpenAIHarness(), GeminiHarness()


def _settings(an: AnthropicHarness, oa: OpenAIHarness, ge: GeminiHarness) -> dict[str, ProviderSettings]:
    return {
        "anthropic": ProviderSettings(client=an.client(False)),
        "openai": ProviderSettings(client=oa.client(False)),
        "gemini": ProviderSettings(client=ge.client(False)),
    }


def _approver() -> AutoApprover:
    return AutoApprover(acknowledge_unsafe=True)


# ---- 1. a Claude agent delegates to OpenAI and Gemini agents ------------------------------


def test_claude_delegates_to_openai_and_gemini_with_shared_usage() -> None:
    an, oa, ge = _harnesses()
    usage = UsageTracker()
    researcher = oa.agent([oa.text("OpenAI research")], usage_tracker=usage, name="researcher")
    reviewer = ge.agent([ge.text("Gemini review")], usage_tracker=usage, name="reviewer")

    @Tool
    def research(question: str) -> str:
        """Ask the research agent.

        Args:
            question: The research question.

        Returns:
            The research agent's answer.
        """
        return researcher.ask(question)

    @Tool
    def review(text: str) -> str:
        """Ask the review agent.

        Args:
            text: The text to review.

        Returns:
            The review agent's answer.
        """
        return reviewer.ask(text)

    lead = ClaudeAgent(
        model="claude-opus-5-5", client=an.client(False), tools=[research, review], usage_tracker=usage
    )
    an.script = Script(
        [
            an.tool_calls([("t1", "research", {"question": "q"}), ("t2", "review", {"text": "draft"})]),
            an.text("combined"),
        ]
    )
    assert lead.ask("analyze") == "combined"
    results = an.tool_results(an.script.calls[1])
    assert [r[:2] for r in results] == [("t1", "OpenAI research"), ("t2", "Gemini review")]
    assert set(usage.usage_by_provider()) == {"anthropic", "openai", "gemini"}
    assert set(usage.usage_by_agent()) == {"claude-opus-5-5", "researcher", "reviewer"}
    assert usage.usage.input_tokens == 4 * 80


# ---- 2. a tool built with provider A is used by an agent of provider B --------------------


def test_tool_built_with_openai_is_used_by_gemini() -> None:
    an, oa, ge = _harnesses()
    oa.script = Script([oa.text(MARGIN.model_dump_json())])
    coding = CodingAgent(model="openai:gpt-5.5", credentials=_settings(an, oa, ge))
    assert coding.agent.adapter.name == "openai"
    builder = ToolBuilder(
        coding, ToolRegistry(sandbox=SubprocessSandbox(acknowledge_unsafe=True)), approver=_approver()
    )
    tool = builder.build("operating margin from revenue and operating income")
    assert tool.name == "calculate_operating_margin"

    analyst = ge.agent(
        [
            ge.tool_calls(
                [("g1", "calculate_operating_margin", {"revenue": 200.0, "operating_income": 50.0})]
            ),
            ge.text("25%"),
        ],
        tools=[tool],
    )
    assert analyst.ask("margin?") == "25%"
    assert ge.tool_results(ge.script.calls[1])[0][:2] == ("g1", "0.25")


# ---- 3. the judge uses another provider than the judged agent -----------------------------


def _verdict(**passed: bool) -> str:
    return (
        '{"criteria": ['
        + ", ".join(
            f'{{"name": "{n}", "reasoning": "r", "passed": {str(p).lower()}}}' for n, p in passed.items()
        )
        + '], "summary": "s", "suggestions": ["be specific"]}'
    )


def test_gemini_judge_refines_an_openai_agent() -> None:
    _, oa, ge = _harnesses()
    agent = oa.agent([oa.text("draft"), oa.text("better")])
    ge.script = Script([ge.text(_verdict(specific=False)), ge.text(_verdict(specific=True))])
    judge = Evaluator(
        criteria=[Criterion(name="specific", description="Names concrete numbers.")],
        model="gemini:gemini-3-pro",
        client=ge.client(False),
    )
    assert judge.agent.adapter.name == "gemini"
    assert not isinstance(judge.agent, ClaudeAgent)
    result = refine(agent, "write", judge)
    assert result.output == "better"
    assert result.evaluation.passed
    assert "be specific" in oa.user_texts(oa.script.calls[1])[-1]


# ---- 4. the orchestrator creates specialists of other providers ---------------------------


def _spec(model: str, **overrides: Any) -> AgentSpec:
    values: dict[str, Any] = {
        "name": "RiskAgent",
        "description": "Analyzes risks.",
        "system_prompt": "You analyze risks.",
        "model": model,
        "effort": "medium",
        "tools": [],
        "output_format": "Report",
    }
    values.update(overrides)
    return AgentSpec(**values)


def test_orchestrator_with_three_providers_policy_and_budget() -> None:
    an, oa, ge = _harnesses()
    settings = _settings(an, oa, ge)
    budget = BudgetTracker(Decimal("5"), on_unpriced="warn")
    oa.script = Script(
        [
            oa.tool_calls([("o1", "create_agent", {"requirement": "risk analysis"})]),
            oa.tool_calls([("o2", "run_agent", {"agent_name": "RiskAgent", "task": "ACME"})]),
            oa.text("ACME has moderate risk."),
        ]
    )
    an.script = Script([an.text(_spec("gemini:gemini-3-pro").model_dump_json())])
    ge.script = Script([ge.text(Report(risks=["regulation"], severity=3).model_dump_json())])
    orchestrator = Orchestrator(
        architect=AgentArchitect(model="claude-opus-5-5", credentials=settings),
        approver=_approver(),
        tools=ToolRegistry(),
        output_formats=OutputFormatRegistry([Report]),
        policy=DynamicPolicy(allowed_models=["gemini:gemini-3-pro", "anthropic:claude-sonnet-5-5"]),
        model="openai:gpt-5.5",
        credentials=settings,
        usage_tracker=budget,
    )
    with pytest.warns(UnpricedUsageWarning):
        assert orchestrator.ask("Analyze ACME") == "ACME has moderate risk."
    assert orchestrator.agent.adapter.name == "openai"
    run_result = oa.tool_results(oa.script.calls[2])[0][1]
    assert Report.model_validate_json(run_result) == Report(risks=["regulation"], severity=3)
    assert set(budget.usage_by_provider()) == {"openai", "anthropic", "gemini"}
    assert len(budget.unpriced_records) == 4  # 3 OpenAI + 1 Gemini request: no shipped prices
    assert budget.spent_usd > 0  # the Anthropic architect request has a known price
    # One workflow label covers the orchestrator, the architect and the delegated specialist.
    label = orchestrator.last_workflow
    assert label is not None and label.startswith("Orchestrator:")
    assert {r.workflow for r in budget.records} == {label}
    assert budget.usage_by_workflow()[label].requests == len(budget.records) == 5
    assert len(budget.usage_by_turn()) == 3  # orchestrator, architect, specialist


def test_policy_accepts_bare_and_qualified_names_and_rejects_others() -> None:
    an, oa, ge = _harnesses()
    factory = AgentFactory(
        ToolRegistry(),
        OutputFormatRegistry([Report]),
        DynamicPolicy(allowed_models=["claude-opus-5-5", "openai:gpt-5.5"]),
        credentials=_settings(an, oa, ge),
    )
    assert factory.create(_spec("anthropic:claude-opus-5-5")).adapter.name == "anthropic"
    assert isinstance(factory.create(_spec("claude-opus-5-5")), ClaudeAgent)
    created = factory.create(_spec("openai:gpt-5.5"))
    assert type(created) is Agent and created.adapter.name == "openai"
    with pytest.raises(PolicyViolationError, match="not allowed"):
        factory.validate(_spec("gemini:gemini-3-pro"))


def test_factory_never_shares_credentials_between_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = AgentFactory(
        ToolRegistry(),
        OutputFormatRegistry([Report]),
        DynamicPolicy(allowed_models=["openai:gpt-5.5", "claude-opus-5-5"]),
        api_key="sk-ant-legacy-anthropic-key",
        credentials={"openai": ProviderSettings(api_key="sk-openai-key")},
    )
    openai_agent = factory.create(_spec("openai:gpt-5.5"))
    assert openai_agent.adapter.config.secret() == "sk-openai-key"
    claude = factory.create(_spec("claude-opus-5-5"))
    assert claude.adapter.config.secret() == "sk-ant-legacy-anthropic-key"
    bare = AgentFactory(
        ToolRegistry(),
        OutputFormatRegistry([Report]),
        DynamicPolicy(allowed_models=["openai:gpt-5.5"]),
        api_key="sk-ant-legacy-anthropic-key",
    ).create(_spec("openai:gpt-5.5"))
    assert bare.adapter.config.secret() is None  # the Anthropic key is not reused
    assert "sk-openai-key" not in repr(ProviderSettings(api_key="sk-openai-key"))


def test_effort_is_dropped_for_models_without_reasoning() -> None:
    factory = AgentFactory(
        ToolRegistry(),
        OutputFormatRegistry([Report]),
        DynamicPolicy(allowed_models=["openai:gpt-4.1"]),
        credentials={"openai": ProviderSettings(api_key="k")},
    )
    assert factory.create(_spec("openai:gpt-4.1")).effort is None


# ---- 5. budgets with unpriced models --------------------------------------------------------


def test_budget_rejects_unpriced_models_before_the_request() -> None:
    _, oa, _ = _harnesses()
    budget = BudgetTracker(1, on_unpriced="reject")
    agent = oa.agent([oa.text("never sent")], usage_tracker=budget)
    with pytest.raises(UnpricedModelError, match="register_pricing"):
        agent.ask("hi")
    assert oa.script.calls == []


def test_budget_counts_custom_pricing_for_any_provider() -> None:
    _, oa, _ = _harnesses()
    budget = BudgetTracker("0.0001", on_unpriced="reject")
    pricing = ModelPricing.of("1", "2", "1", "1", "0.1")
    agent = oa.agent([oa.text("a"), oa.text("b")], usage_tracker=budget, pricing=pricing)
    from orkexa.dynamic import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        agent.ask("one")  # recorded first, then over the limit
    assert budget.spent_usd == Decimal("0.000182")  # 80 x $1 + 50 x $2 + 20 x $0.1 per million
    with pytest.raises(BudgetExceededError):
        agent.ask("two")
    assert len(oa.script.calls) == 1  # the second request was refused before it was sent


def test_budget_unpriced_policies() -> None:
    _, oa, _ = _harnesses()
    ignore = BudgetTracker()
    assert ignore.on_unpriced == "ignore"
    oa.agent([oa.text("x")], usage_tracker=ignore).ask("hi")
    assert len(ignore.unpriced_records) == 1 and ignore.spent_usd == 0

    warn = BudgetTracker(10)
    assert warn.on_unpriced == "warn"
    agent = oa.agent([oa.text("x"), oa.text("y")], usage_tracker=warn)
    with pytest.warns(UnpricedUsageWarning, match="openai:gpt-5.5"):
        agent.ask("hi")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent.ask("again")  # one warning per provider and model
    with pytest.raises(ValueError):
        BudgetTracker(on_unpriced="maybe")  # type: ignore[arg-type]


def test_auto_approver_is_an_explicit_unsafe_opt_in() -> None:
    with pytest.warns(UnsafeOptInWarning):
        AutoApprover()
    assert AutoApprover.unsafe
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AutoApprover(acknowledge_unsafe=True)


# ---- 6. recursion limits and delegation control are provider-independent -------------------


def test_recursion_limit_applies_to_every_provider() -> None:
    an, oa, ge = _harnesses()
    settings = _settings(an, oa, ge)
    oa.script = Script(
        [
            oa.tool_calls([("o1", "run_agent", {"agent_name": "Deep", "task": "x"})]),
            oa.text("done"),
        ]
    )
    ge.script = Script(
        [
            ge.tool_calls([("g1", "run_agent", {"agent_name": "Deep", "task": "deeper"})]),
            ge.text("stopped"),
        ]
    )
    orchestrator = Orchestrator(
        architect=AgentArchitect(credentials=settings),
        approver=_approver(),
        tools=ToolRegistry(),
        policy=DynamicPolicy(
            allowed_models=["gemini:gemini-3-pro"], allow_agent_delegation=True, max_depth=1
        ),
        model="openai:gpt-5.5",
        credentials=settings,
    )
    orchestrator.agents.register(
        _spec("gemini:gemini-3-pro", name="Deep", tools=["run_agent"], output_format=None)
    )
    assert orchestrator.ask("go") == "done"
    nested = ge.tool_results(ge.script.calls[1])[0]
    assert nested[1] == "Error: Maximum delegation depth 1 reached"  # as in claude-agent 0.8.0


# ---- RAG across providers -------------------------------------------------------------------


def test_search_tool_results_are_verified_sources_for_gemini(tmp_path: Any) -> None:
    _, _, ge = _harnesses()
    index = SQLiteIndex(tmp_path / "rag.db")
    index.add("The warranty lasts 24 months.", source="handbook.md", title="Handbook")
    agent = ge.agent(
        [ge.tool_calls([("g1", "search", {"query": "warranty"})]), ge.text("24 months")], index=index
    )
    assert agent.ask("How long is the warranty?") == "24 months"
    sent = ge.tool_results(ge.script.calls[1])[0][1]
    assert "[Search result | source: handbook.md | title: Handbook]" in sent
    (source,) = retrieval_sources(agent)
    assert (source.kind, source.source, source.title) == ("retrieval", "handbook.md", "Handbook")
    assert "24 months" in source.cited_text
    assert citations(agent.last_response) == []  # Gemini returned no API-native citations
    index.close()


# ---- memory and files across providers -------------------------------------------------------


def test_memory_is_a_function_tool_for_openai() -> None:
    from orkexa import EphemeralMemory

    _, oa, _ = _harnesses()
    memory = EphemeralMemory()
    agent = oa.agent(
        [
            oa.tool_calls(
                [("m1", "memory", {"command": "create", "path": "/memories/notes.md", "file_text": "hi"})]
            ),
            oa.text("saved"),
        ],
        memory=memory,
    )
    assert agent.ask("remember hi") == "saved"
    names = [t["name"] for t in oa.script.calls[0]["tools"]]
    assert names == ["memory"]
    assert memory.view("/memories/notes.md").endswith("hi")
    assert oa.tool_results(oa.script.calls[1])[0][2] is False


def test_files_tool_loads_an_image_for_openai_and_a_document_for_gemini(tmp_path: Any) -> None:
    import base64

    from orkexa import LocalFileStore

    _, oa, ge = _harnesses()
    store = LocalFileStore(tmp_path / "files")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    image = store.add(png, name="chart.png", description="A chart")
    note = store.add(b"Revenue grew 12%.", name="note.txt", description="A note")

    agent = oa.agent(
        [
            oa.tool_calls([("f1", "files", {"command": "load", "file_id": image.id, "query": None})]),
            oa.text("a chart"),
        ],
        files=store,
    )
    assert agent.ask("what is in the chart?") == "a chart"
    output = next(i for i in oa.script.calls[1]["input"] if i.get("type") == "function_call_output")["output"]
    assert [o["type"] for o in output] == ["input_text", "input_image"]

    agent = ge.agent(
        [ge.tool_calls([("g1", "files", {"command": "load", "file_id": note.id})]), ge.text("12%")],
        files=store,
    )
    assert agent.ask("growth?") == "12%"
    assert "Revenue grew 12%." in ge.tool_results(ge.script.calls[1])[0][1]


# ---- evaluation batches -----------------------------------------------------------------------


def test_evaluate_many_without_native_batch_is_explicit() -> None:
    from orkexa import EvaluationCase, UnsupportedCapabilityError

    _, oa, _ = _harnesses()
    oa.script = Script([oa.text(_verdict(specific=True)), oa.text(_verdict(specific=False))])
    judge = Evaluator(
        criteria=[Criterion(name="specific", description="Names concrete numbers.")],
        model="openai:gpt-5.5",
        client=oa.client(False),
    )
    cases = [EvaluationCase(task="t", output="a"), EvaluationCase(task="t", output="b")]
    with pytest.raises(UnsupportedCapabilityError, match=r"batch=False"):
        judge.evaluate_many(cases)
    assert oa.script.calls == []
    results = judge.evaluate_many(cases, batch=False)
    assert [r.passed for r in results] == [True, False]
    with pytest.raises(TimeoutError, match="0 of 2"):
        judge.evaluate_many(cases, batch=False, timeout=0)
