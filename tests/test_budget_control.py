"""Structured budget errors and changing a BudgetTracker's limit while agents run."""

from __future__ import annotations

from decimal import Decimal
import random
import threading
import time

import pytest

from orkexa import Agent, ModelPricing
from orkexa import agent as agent_module
from orkexa.core import runtime as core_runtime
from orkexa.dynamic import BudgetExceededError, BudgetTracker, UnpricedModelError
from orkexa.providers.fake import FakeModel, text
from orkexa.usage import _RELEASED, UsageRecord

CHEAP = ModelPricing.of("1", "1", "0", "0", "0")


def _record(cost: str, agent: str = "a") -> UsageRecord:
    return UsageRecord(
        agent=agent,
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
        estimated_cost_usd=Decimal(cost),
        provider="p",
    )


def _reserve(tracker: BudgetTracker, amount: str | None) -> object:
    return tracker.try_reserve(
        agent="a", provider="p", model="m", amount=None if amount is None else Decimal(amount)
    )


# ---- structured errors ------------------------------------------------------------------------


def test_errors_raised_without_details_keep_working() -> None:
    plain = BudgetExceededError("over")
    assert str(plain) == "over" and plain.args == ("over",)
    assert plain.reason is None and plain.tracker is None and plain.limit_usd is None
    assert plain.spent_usd is None and plain.reserved_usd is None and plain.requested_usd is None
    unpriced = UnpricedModelError("no price")
    assert unpriced.reason is None and unpriced.tracker is None
    detailed = BudgetExceededError("x", limit_usd=Decimal(1), reason="used_up")
    assert detailed.limit_usd == 1 and detailed.reason == "used_up"


def test_before_request_reports_a_used_up_budget() -> None:
    tracker = BudgetTracker(0)
    with pytest.raises(BudgetExceededError, match="used up") as caught:
        tracker.before_request(agent="a", provider="p", model="m", priced=True)
    exc = caught.value
    assert exc.reason == "used_up" and exc.tracker is tracker
    assert exc.limit_usd == 0 and exc.spent_usd == 0 and exc.reserved_usd == 0
    assert exc.requested_usd is None


def test_try_reserve_reports_a_used_up_budget_with_what_is_in_flight() -> None:
    tracker = BudgetTracker(1)
    held = _reserve(tracker, "0.5")
    assert held is not None
    tracker.add(_record("1"))  # spent == limit: used up, not yet exceeded
    with pytest.raises(BudgetExceededError, match="used up") as caught:
        _reserve(tracker, "0.25")
    exc = caught.value
    assert exc.reason == "used_up" and exc.tracker is tracker
    assert (exc.limit_usd, exc.spent_usd, exc.reserved_usd, exc.requested_usd) == (
        Decimal(1),
        Decimal(1),
        Decimal("0.5"),
        Decimal("0.25"),
    )


def test_try_reserve_reports_a_request_too_large_for_the_budget() -> None:
    tracker = BudgetTracker("2", on_oversized="reject")
    tracker.add(_record("0.5"))
    with pytest.raises(BudgetExceededError, match="too small") as caught:
        _reserve(tracker, "5")
    exc = caught.value
    assert exc.reason == "request_too_large" and exc.tracker is tracker
    assert (exc.limit_usd, exc.spent_usd, exc.reserved_usd, exc.requested_usd) == (
        Decimal(2),
        Decimal("0.5"),
        Decimal(0),
        Decimal(5),
    )
    assert tracker.reserved_usd == 0


def test_check_reports_an_exceeded_budget() -> None:
    tracker = BudgetTracker(1)
    with pytest.raises(BudgetExceededError, match="exceeded") as caught:
        tracker.add(_record("1.5"))
    exc = caught.value
    assert exc.reason == "exceeded" and exc.tracker is tracker
    assert (exc.limit_usd, exc.spent_usd, exc.requested_usd) == (Decimal(1), Decimal("1.5"), None)
    with pytest.raises(BudgetExceededError) as again:
        tracker.check()
    assert again.value.reason == "exceeded"


def test_the_error_names_the_parent_whose_limit_blocked() -> None:
    parent = BudgetTracker(1, on_oversized="reject")
    child = BudgetTracker(100, parent=parent, prefix="team")
    with pytest.raises(BudgetExceededError) as too_large:
        _reserve(child, "5")
    assert too_large.value.tracker is parent and too_large.value.reason == "request_too_large"
    assert too_large.value.limit_usd == 1 and too_large.value.requested_usd == 5
    assert child.reserved_usd == 0 and parent.reserved_usd == 0

    with pytest.raises(BudgetExceededError) as exceeded:
        child.add(_record("2"))
    assert exceeded.value.tracker is parent and exceeded.value.reason == "exceeded"
    assert exceeded.value.spent_usd == 2

    for blocked in (
        lambda: child.before_request(agent="a", provider="p", model="m", priced=True),
        lambda: _reserve(child, "0.1"),
    ):
        with pytest.raises(BudgetExceededError) as used_up:
            blocked()
        assert used_up.value.tracker is parent and used_up.value.reason == "used_up"


def test_the_error_names_the_child_when_its_own_limit_blocks() -> None:
    parent = BudgetTracker(100)
    child = BudgetTracker(1, parent=parent, on_oversized="reject")
    with pytest.raises(BudgetExceededError) as caught:
        _reserve(child, "5")
    assert caught.value.tracker is child and caught.value.limit_usd == 1


def test_an_agent_sees_the_structured_error() -> None:
    budget = BudgetTracker(0)
    agent = Agent(model="fake:m", client=FakeModel([text("x")]), pricing=CHEAP, usage_tracker=budget)
    with pytest.raises(BudgetExceededError) as caught:
        agent.ask("hi")
    assert caught.value.reason == "used_up" and caught.value.tracker is budget


# ---- set_limit --------------------------------------------------------------------------------


def test_set_limit_returns_the_previous_limit_and_validates() -> None:
    tracker = BudgetTracker("1.5")
    assert tracker.set_limit(3) == Decimal("1.5")
    assert tracker.limit_usd == Decimal(3)
    assert tracker.set_limit("0.25") == 3 and tracker.limit_usd == Decimal("0.25")
    assert tracker.set_limit(None) == Decimal("0.25") and tracker.limit_usd is None
    assert tracker.remaining_usd is None
    with pytest.raises(ValueError, match="nonnegative"):
        tracker.set_limit(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        tracker.set_limit("NaN")
    assert tracker.limit_usd is None  # a refused change leaves the limit alone
    assert tracker.set_limit(0.5) is None and tracker.limit_usd == Decimal("0.5")


def test_raising_the_limit_admits_a_request_that_was_used_up() -> None:
    tracker = BudgetTracker(0)
    with pytest.raises(BudgetExceededError):
        _reserve(tracker, "1")
    tracker.set_limit(2)
    assert _reserve(tracker, "1") is not None
    assert tracker.reserved_usd == 1


def test_lowering_the_limit_keeps_requests_in_flight() -> None:
    tracker = BudgetTracker(10)
    held = _reserve(tracker, "4")
    assert held is not None
    tracker.set_limit(1)
    assert tracker.reserved_usd == 4  # not cancelled
    assert _reserve(tracker, "1") is None  # the next one waits for the one in flight
    tracker.release(held)  # type: ignore[arg-type]
    assert tracker.reserved_usd == 0
    assert _reserve(tracker, "1") is not None


def test_set_limit_wakes_a_request_that_waits_for_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every request can cost at most $1; without a wake-up the waiter would sleep for 30 s.
    monkeypatch.setattr(core_runtime, "request_cost_bound", lambda request, price: Decimal(1))
    monkeypatch.setattr(agent_module, "_BUDGET_WAIT", 30.0)
    budget = BudgetTracker("1.5")
    held = _reserve(budget, "1")  # a request in flight holds $1; another $1 does not fit
    assert held is not None
    waiting = threading.Event()
    original = budget.wait_for_release

    def wait_for_release(timeout: float) -> None:
        with _RELEASED:  # set the flag while holding the condition: no lost wake-up
            waiting.set()
            original(timeout)

    monkeypatch.setattr(budget, "wait_for_release", wait_for_release)
    model = FakeModel([text("done")])
    agent = Agent(model="fake:m", client=model, pricing=CHEAP, usage_tracker=budget)
    answers: list[str] = []
    thread = threading.Thread(target=lambda: answers.append(agent.ask("go")))
    thread.start()
    try:
        assert waiting.wait(5), "the request did not wait for budget"
        assert model.requests == []
        started = time.monotonic()
        assert budget.set_limit(3) == Decimal("1.5")
        thread.join(5)
        assert not thread.is_alive(), "set_limit did not wake the waiting request"
        assert time.monotonic() - started < 5
        assert answers == ["done"] and len(model.requests) == 1
    finally:
        budget.release(held)  # type: ignore[arg-type]
        thread.join(5)
    assert budget.reserved_usd == 0


def test_set_limit_is_thread_safe_under_concurrent_reservations() -> None:
    tracker = BudgetTracker(2)
    highest = Decimal(0)
    stop = threading.Event()
    failures: list[BaseException] = []
    lock = threading.Lock()

    def reserve(seed: int) -> None:
        nonlocal highest
        rng = random.Random(seed)
        try:
            for _ in range(300):
                reservation = _reserve(tracker, rng.choice(["0.25", "0.5", "1"]))
                if reservation is None:
                    continue
                with lock:
                    highest = max(highest, tracker.reserved_usd)
                tracker.release(reservation)  # type: ignore[arg-type]
        except BaseException as exc:
            failures.append(exc)

    def toggle() -> None:
        rng = random.Random(99)
        while not stop.is_set():
            tracker.set_limit(rng.choice(["2", "3", "4"]))

    toggler = threading.Thread(target=toggle)
    workers = [threading.Thread(target=reserve, args=(seed,)) for seed in range(8)]
    toggler.start()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
    stop.set()
    toggler.join(5)
    assert failures == []
    assert highest <= 4  # never more held back than the highest limit allowed
    assert tracker.reserved_usd == 0 and tracker._in_flight == 0
    assert tracker.oversized_requests == 0 and tracker.limit_usd in (2, 3, 4)
