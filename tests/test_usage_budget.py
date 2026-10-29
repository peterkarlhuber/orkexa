"""Budget limits before requests, and usage aggregation per turn and workflow."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import threading

import pytest

from orkexa import Agent, AsyncAgent, ModelPricing, Tool, UsageTracker
from orkexa.dynamic import BudgetExceededError, BudgetTracker
from orkexa.providers.fake import FakeModel, text, tool_calls
from orkexa.usage import current_workflow, workflow

# The fake reports 10 input and 5 output tokens per request: $15 at $1M/1M tokens.
PRICE = ModelPricing.of("1000000", "1000000", "0", "0", "0")
COST = Decimal("15")


def _agent(budget: BudgetTracker, script: list[object], name: str = "a") -> tuple[Agent[str], FakeModel]:
    fake = FakeModel(script)  # type: ignore[arg-type]
    return Agent(model="fake:m", client=fake, pricing=PRICE, usage_tracker=budget, name=name), fake


# ---- budget --------------------------------------------------------------------------------


@pytest.mark.parametrize("on_unpriced", ["ignore", "warn", "reject"])
def test_zero_budget_sends_no_request(on_unpriced: str) -> None:
    budget = BudgetTracker(0, on_unpriced=on_unpriced)  # type: ignore[arg-type]
    agent, fake = _agent(budget, [text("never")])
    with pytest.raises(BudgetExceededError, match="not sent"):
        agent.ask("hi")
    assert fake.requests == []
    assert budget.records == ()


def test_request_that_reaches_the_limit_exactly_is_billed_and_blocks_the_next() -> None:
    budget = BudgetTracker(COST)
    agent, fake = _agent(budget, [text("one"), text("two")])
    assert agent.ask("first") == "one"  # spent == limit is allowed after the fact
    assert budget.spent_usd == COST and budget.remaining_usd == 0
    with pytest.raises(BudgetExceededError, match="used up"):
        agent.ask("second")
    assert len(fake.requests) == 1
    assert len(budget.records) == 1


def test_budget_just_below_the_spent_amount_admits_the_next_request() -> None:
    budget = BudgetTracker(COST + Decimal("0.0001"))
    agent, fake = _agent(budget, [text("one"), text("two")])
    agent.ask("first")
    with pytest.raises(BudgetExceededError, match="Budget exceeded"):
        agent.ask("second")  # admitted, then over the limit: the cost is still recorded
    assert len(fake.requests) == 2
    assert budget.spent_usd == 2 * COST


def test_shared_budget_stops_every_agent() -> None:
    budget = BudgetTracker(COST)
    first, _ = _agent(budget, [text("one")], name="first")
    second, fake = _agent(budget, [text("never")], name="second")
    first.ask("go")
    with pytest.raises(BudgetExceededError):
        second.ask("go")
    assert fake.requests == []


def test_concurrent_agents_overshoot_a_shared_budget_by_one_request_each() -> None:
    """The documented soft limit: requests admitted together are all billed."""
    budget = BudgetTracker(COST)
    barrier = threading.Barrier(3, timeout=5)

    def reply(request: object) -> object:
        barrier.wait()  # all three requests are in flight before any cost is known
        return text("x")

    agents = [_agent(budget, [reply], name=f"a{i}")[0] for i in range(3)]
    errors: list[BaseException] = []

    def run(agent: Agent[str]) -> None:
        try:
            agent.ask("go")
        except BudgetExceededError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(agent,)) for agent in agents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(budget.records) == 3
    assert budget.spent_usd == 3 * COST
    assert len(errors) >= 1  # whoever records past the limit learns about it
    with pytest.raises(BudgetExceededError):
        agents[0].ask("again")


def test_plain_usage_tracker_check_is_a_no_op() -> None:
    UsageTracker().check()


# ---- aggregation ---------------------------------------------------------------------------


def test_usage_by_turn_and_workflow() -> None:
    tracker = UsageTracker()
    fake = FakeModel([tool_calls(("add", {"a": 1, "b": 2})), text("3"), text("ok"), text("later")])

    @Tool
    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: First.
            b: Second.

        Returns:
            The sum.
        """
        return a + b

    agent = Agent(model="fake:m", client=fake, tools=[add], usage_tracker=tracker)
    with workflow("report"):
        agent.ask("1+2")  # two requests, one turn
        agent.ask("thanks")
    agent.ask("outside")
    turns = tracker.usage_by_turn()
    assert [summary.requests for summary in turns.values()] == [2, 1, 1]
    assert None not in turns
    by_workflow = tracker.usage_by_workflow()
    assert by_workflow["report"].requests == 3
    assert by_workflow[None].requests == 1
    assert sum(s.requests for s in by_workflow.values()) == tracker.usage.requests
    assert tracker.usage_by_agent()["m"].requests == 4


def test_workflow_label_follows_delegation_into_tool_threads_and_tasks() -> None:
    tracker = UsageTracker()
    child = Agent(
        model="fake:child", client=FakeModel([text("child answer")]), usage_tracker=tracker, name="child"
    )

    @Tool
    def delegate(task: str) -> str:
        """Ask the child agent.

        Args:
            task: The task.

        Returns:
            The child's answer.
        """
        return child.ask(task)

    parent = Agent(
        model="fake:parent",
        client=FakeModel([tool_calls(("delegate", {"task": "t"})), text("done")]),
        tools=[delegate],
        usage_tracker=tracker,
        name="parent",
    )
    with workflow("outer"):
        assert current_workflow() == "outer"
        parent.ask("go")
        with workflow("inner"):
            asyncio.run(
                AsyncAgent(
                    model="fake:x", client=FakeModel([text("y")]), usage_tracker=tracker, name="x"
                ).ask("z")
            )
    assert current_workflow() is None
    assert {(r.agent, r.workflow) for r in tracker.records} == {
        ("parent", "outer"),
        ("child", "outer"),
        ("x", "inner"),
    }
    assert set(tracker.usage_by_agent()) == {"parent", "child", "x"}
    assert tracker.usage_by_workflow()["outer"].requests == 3


def test_unpriced_requests_make_aggregated_cost_unknown() -> None:
    tracker = UsageTracker()
    priced = Agent(model="fake:p", client=FakeModel([text("a")]), pricing=PRICE, usage_tracker=tracker)
    unpriced = Agent(model="fake:u", client=FakeModel([text("b")]), usage_tracker=tracker)
    with workflow("mixed"):
        priced.ask("x")
        unpriced.ask("y")
    summary = tracker.usage_by_workflow()["mixed"]
    assert summary.estimated_cost_usd is None and summary.unpriced_requests == 1
    by_model = tracker.usage_by_model()
    assert by_model["fake:p"].estimated_cost_usd == COST
    assert by_model["fake:u"].estimated_cost_usd is None
    assert tracker.usage_by_provider()["fake"].requests == 2


def test_workflow_label_must_be_nonempty() -> None:
    with pytest.raises(ValueError), workflow(""):
        pass


# ---- batches -------------------------------------------------------------------------------


def _batch_agent(budget: UsageTracker, calls: list[int]) -> Agent[str]:
    agent = Agent(model="fake:m", client=FakeModel(), pricing=PRICE, usage_tracker=budget)

    def run_batch(requests: dict[str, object], *, poll_interval: float, timeout: float | None) -> object:
        calls.append(len(requests))
        return "batch-1", {key: text("ok") for key in requests}

    agent.adapter.run_batch = run_batch  # type: ignore[method-assign,assignment]
    return agent


def test_used_up_budget_submits_no_batch() -> None:
    calls: list[int] = []
    agent = _batch_agent(BudgetTracker(0), calls)
    with pytest.raises(BudgetExceededError, match="not sent"):
        agent._run_batch({"a": "x", "b": "y"}, poll_interval=1, timeout=None)
    assert calls == []


def test_unpriced_batch_is_rejected_before_submission() -> None:
    from orkexa.dynamic import UnpricedModelError

    calls: list[int] = []
    agent = _batch_agent(BudgetTracker(100, on_unpriced="reject"), calls)
    agent.pricing = None
    with pytest.raises(UnpricedModelError):
        agent._run_batch({"a": "x"}, poll_interval=1, timeout=None)
    assert calls == []


def test_batch_over_budget_records_every_result_before_raising() -> None:
    calls: list[int] = []
    budget = BudgetTracker(1)
    agent = _batch_agent(budget, calls)
    with pytest.raises(BudgetExceededError):
        agent._run_batch({"one": "a", "two": "b", "three": "c"}, poll_interval=1, timeout=None)
    assert calls == [3]
    assert len(budget.records) == 3 and all(r.turn_id is None for r in budget.records)
    assert budget.spent_usd == 3 * COST / 2  # batch prices are half
    assert agent.usage.requests == 3


def test_evaluator_batch_respects_the_budget() -> None:
    from orkexa.evaluation import Criterion, EvaluationCase, Evaluator

    calls: list[int] = []
    budget = BudgetTracker(0)
    evaluator = Evaluator(
        criteria=[Criterion(name="ok", description="Output is acceptable")],
        model="fake:m",
        effort=None,
        client=FakeModel(),
        capabilities={"batch": "supported"},
        usage_tracker=budget,
    )
    evaluator.agent.adapter.run_batch = lambda requests, **_: calls.append(len(requests))  # type: ignore[method-assign,assignment,func-returns-value,misc]
    with pytest.raises(BudgetExceededError):
        evaluator.evaluate_many([EvaluationCase(task="Task", output="Result")], batch=True)
    assert calls == []


def test_batch_usage_is_not_part_of_the_last_turn() -> None:
    calls: list[int] = []
    tracker = UsageTracker()
    agent = _batch_agent(tracker, calls)
    agent._run_batch({"a": "x"}, poll_interval=1, timeout=None)
    assert agent.last_turn_usage.requests == 0  # no ask() yet
    agent.client.script.append(text("answer"))
    agent.ask("hi")
    agent._run_batch({"a": "x", "b": "y"}, poll_interval=1, timeout=None)
    assert agent.last_turn_usage.requests == 1
    assert agent.usage.requests == tracker.usage.requests == 4
