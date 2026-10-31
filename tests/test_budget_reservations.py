"""Budget reservations: requests in flight hold back their most expensive outcome."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import threading
import time

import pytest

from orkexa import Agent, AsyncAgent, ModelPricing, UsageTracker
from orkexa.core import runtime as core_runtime
from orkexa.core.cost_bound import FRAMING_TOKENS, input_token_bound, request_cost_bound
from orkexa.core.types import ImagePart, MediaSource, Message, ModelRequest, TextPart, ToolDefinition
from orkexa.dynamic import BudgetExceededError, BudgetTracker
from orkexa.pricing import FeeRate, LongContextTier, ModelPrice, Tariff, TokenRates
from orkexa.providers.fake import FakeModel, text

PRICE = ModelPricing.of("1000000", "1000000", "0", "0", "0")  # the fake's request: $15
COST = Decimal("15")
CHEAP = ModelPricing.of("1", "1", "0", "0", "0")


def _agent(budget: UsageTracker, script: list[object], name: str = "a") -> Agent[str]:
    return Agent(model="fake:m", client=FakeModel(script), pricing=PRICE, usage_tracker=budget, name=name)  # type: ignore[arg-type]


@pytest.fixture
def bound_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request can cost at most $1 (whatever its size), so limits are easy to read."""
    monkeypatch.setattr(core_runtime, "request_cost_bound", lambda request, price: Decimal(1))


def _run_all(agents: list[Agent[str]]) -> list[BaseException]:
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
        thread.join(timeout=10)
    return errors


# ---- the tracker -----------------------------------------------------------------------------


def test_requests_in_flight_never_exceed_the_budget_together(bound_of_one: None) -> None:
    cheap = ModelPricing.of("1", "1", "0", "0", "0")
    budget = BudgetTracker("2.5")  # room for two reservations of $1, not three
    running, most = 0, 0
    lock = threading.Lock()

    def reply(request: object) -> object:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.2)
        with lock:
            running -= 1
        return text("x")

    agents = [
        Agent(model="fake:m", client=FakeModel([reply]), pricing=cheap, usage_tracker=budget, name=f"a{i}")  # type: ignore[list-item]
        for i in range(4)
    ]
    assert _run_all(agents) == []
    assert most == 2  # the others waited for a release instead of overshooting
    assert len(budget.records) == 4
    assert budget.reserved_usd == 0 and budget.hard_maximum


def test_concurrent_agents_no_longer_overshoot_a_shared_budget() -> None:
    """Before reservations, three requests admitted together were all billed."""
    budget = BudgetTracker(COST)
    agents = [_agent(budget, [text("x")], name=f"a{i}") for i in range(3)]
    errors = _run_all(agents)
    assert len(budget.records) == 1  # only one request was sent
    assert budget.spent_usd == COST
    assert len(errors) == 2 and all("not sent" in str(error) for error in errors)


def test_an_oversized_request_runs_alone_or_is_refused() -> None:
    # The bound of the fake's request (16000 output tokens at $1 each) far exceeds $20.
    alone = BudgetTracker(20)
    _agent(alone, [text("x")]).ask("hi")
    assert alone.oversized_requests == 1 and not alone.hard_maximum

    strict = BudgetTracker(20, on_oversized="reject")
    fake = FakeModel([text("x")])
    agent = Agent(model="fake:m", client=fake, pricing=PRICE, usage_tracker=strict)
    with pytest.raises(BudgetExceededError, match=r"can cost up to .* not sent"):
        agent.ask("hi")
    assert fake.requests == [] and strict.reserved_usd == 0


def test_a_failed_request_releases_its_reservation(bound_of_one: None) -> None:
    budget = BudgetTracker(5)

    def broken(request: object) -> object:
        raise RuntimeError("provider down")

    agent = Agent(model="fake:m", client=FakeModel([broken, text("ok")]), pricing=CHEAP, usage_tracker=budget)  # type: ignore[list-item]
    with pytest.raises(Exception, match="provider down"):
        agent.ask("hi")
    assert budget.reserved_usd == 0
    assert agent.ask("again") == "ok"  # nothing is held back any more


def test_unknown_prices_are_never_reserved_as_zero() -> None:
    budget = BudgetTracker(100)
    agent = Agent(model="fake:m", client=FakeModel([text("x")]), usage_tracker=budget)  # no price
    with pytest.warns(Warning):
        agent.ask("hi")
    assert budget.unbounded_requests == 1
    assert not budget.hard_maximum  # the limit did not cap this request


def test_plain_trackers_admit_everything() -> None:
    tracker = UsageTracker()
    reservation = tracker.try_reserve(agent="a", provider="p", model="m", amount=Decimal(10**9))
    assert reservation is not None
    tracker.release(reservation)
    tracker.release(reservation)  # twice is harmless


# ---- parent and child scopes ------------------------------------------------------------------


def test_a_child_cannot_spend_past_its_parent_and_vice_versa(bound_of_one: None) -> None:
    parent = BudgetTracker("1.5")  # one request of $1 at a time
    child = BudgetTracker(100, parent=parent, prefix="team")
    running, most = 0, 0
    lock = threading.Lock()

    def reply(request: object) -> object:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.15)
        with lock:
            running -= 1
        return text("x")

    cheap = ModelPricing.of("1", "1", "0", "0", "0")
    inner = Agent(model="fake:m", client=FakeModel([reply]), pricing=cheap, usage_tracker=child, name="c")  # type: ignore[list-item]
    outer = Agent(model="fake:m", client=FakeModel([reply]), pricing=cheap, usage_tracker=parent, name="p")  # type: ignore[list-item]
    assert _run_all([inner, outer]) == []
    assert most == 1  # the parent's limit made the child wait (or the other way round)
    assert [record.agent for record in child.records] == ["c"]
    assert sorted(record.agent for record in parent.records) == ["p", "team/c"]
    assert parent.reserved_usd == 0 and child.reserved_usd == 0


def test_a_child_limit_holds_while_the_parent_has_money_left() -> None:
    parent = BudgetTracker(1000)
    child = BudgetTracker(COST, parent=parent, prefix="team")
    agent = _agent(child, [text("x"), text("y")])
    agent.ask("first")  # oversized, admitted alone, spends the child's whole budget
    with pytest.raises(BudgetExceededError, match="used up"):
        agent.ask("second")
    assert parent.spent_usd == COST


def test_a_refused_parent_reservation_leaves_nothing_held_in_the_child() -> None:
    parent = BudgetTracker(1, on_oversized="reject")
    child = BudgetTracker(100, parent=parent)
    with pytest.raises(BudgetExceededError):
        child.try_reserve(agent="a", provider="p", model="m", amount=Decimal(5))
    assert child.reserved_usd == 0 and parent.reserved_usd == 0


# ---- async ------------------------------------------------------------------------------------


def test_async_agents_wait_without_blocking_the_event_loop(bound_of_one: None) -> None:
    budget = BudgetTracker("1.5")
    cheap = ModelPricing.of("1", "1", "0", "0", "0")

    async def main() -> None:
        agents = [
            AsyncAgent(
                model="fake:m",
                client=FakeModel([text(str(i))]),
                pricing=cheap,
                usage_tracker=budget,
                name=f"a{i}",
            )  # type: ignore[list-item]
            for i in range(3)
        ]
        answers = await asyncio.wait_for(asyncio.gather(*(agent.ask("go") for agent in agents)), timeout=10)
        assert sorted(answers) == ["0", "1", "2"]

    asyncio.run(main())
    assert len(budget.records) == 3 and budget.reserved_usd == 0


# ---- the bound --------------------------------------------------------------------------------


def _request(*parts: object, **options: object) -> ModelRequest:
    message = Message(role="user", content=list(parts) or [TextPart(text="hello")])  # type: ignore[arg-type]
    return ModelRequest(model="m", messages=[message], **options)  # type: ignore[arg-type]


def test_the_bound_covers_input_size_and_max_tokens() -> None:
    price = ModelPrice("p", "m", TokenRates.of("2", "10"))
    request = _request(max_tokens=1000)
    tokens = input_token_bound(request)
    assert tokens is not None and tokens > FRAMING_TOKENS
    assert request_cost_bound(request, price) == (Decimal(tokens) * 2 + 1000 * 10) / Decimal(1_000_000)


def test_the_bound_takes_the_highest_rates_of_tiers_classes_and_tariffs() -> None:
    price = ModelPrice(
        "p",
        "m",
        TokenRates.of("1", "5", cache_read="0.1", cache_write_1h="3"),
        long_context=(LongContextTier(200_000, TokenRates.of("2", "8")),),
        tariffs=(Tariff.of("speed", "standard", standard="1", fast="1.5"),),
    )
    request = _request(max_tokens=100)
    tokens = input_token_bound(request)
    assert tokens is not None
    assert request_cost_bound(request, price) == (Decimal(tokens) * 3 + 100 * 8) * Decimal("1.5") / Decimal(
        10**6
    )


def test_requests_without_a_bound() -> None:
    price = ModelPrice("p", "m", TokenRates.of("1", "1"))
    assert request_cost_bound(_request(), None) is None  # no price
    assert (
        request_cost_bound(_request(), ModelPrice("p", "m", TokenRates.of("1", "1"), currency="EUR")) is None
    )
    image = ImagePart(source=MediaSource(type="url", url="https://example.com/a.png"))
    assert request_cost_bound(_request(image), price) is None
    assert request_cost_bound(_request(native_tools=[{"type": "web_search"}]), price) is None
    fees = ModelPrice(
        "p", "m", TokenRates.of("1", "1"), fees=(FeeRate("web_search_request", Decimal("0.01")),)
    )
    assert request_cost_bound(_request(), fees) is not None  # fees only matter with native tools


def test_tools_and_schemas_count_as_input() -> None:
    small = input_token_bound(_request())
    tool = ToolDefinition("lookup", "x" * 5000, {"type": "object"})
    large = input_token_bound(_request(tools=[tool]))
    assert small is not None and large is not None and large >= small + 5000
