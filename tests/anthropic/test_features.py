"""Tests for streaming, multimodal input, hooks, request options and related 0.4.0 features."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import threading
from typing import Any

from pydantic import BaseModel
import pytest

from orkexa import (
    ClaudeAgent,
    MaxToolRoundsError,
    RefusalError,
    Tool,
    ToolCall,
    ToolDefinitionError,
    document,
    image,
    uploaded_file,
)
from orkexa.content import to_content
from orkexa.evaluation import tool_results_of_last_turn

from .conftest import FakeClient, ParsedResponse, make_beta_message, make_message


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First summand.
        b: Second summand.

    Returns:
        The sum.
    """
    return a + b


def _text(text: str, **kwargs: Any) -> Any:
    return make_message([{"type": "text", "text": text}], **kwargs)


def _tool_uses(*calls: tuple[str, dict[str, Any]]) -> Any:
    return make_message(
        [
            {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": payload}
            for i, (name, payload) in enumerate(calls)
        ],
        stop_reason="tool_use",
    )


def _agent(client: FakeClient, **kwargs: Any) -> ClaudeAgent[Any]:
    return ClaudeAgent(client=client, **kwargs)  # type: ignore[arg-type]


# ---- request options, betas, validation ---------------------------------------


def test_request_options_are_merged_and_betas_use_the_beta_namespace() -> None:
    client = FakeClient([_text("ok")])
    agent = _agent(
        client,
        model="claude-opus-5-5",
        effort="high",
        request_options={
            "betas": ["task-budgets-2026-03-13"],
            "output_config": {"task_budget": {"type": "tokens", "total": 64000}},
            "inference_geo": "eu",
        },
    )
    agent.ask("Hi")
    call = client.messages.create_calls[0]
    assert call["output_config"] == {"effort": "high", "task_budget": {"type": "tokens", "total": 64000}}
    assert call["inference_geo"] == "eu"
    assert call["betas"] == ["task-budgets-2026-03-13"]


def test_reserved_request_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        _agent(FakeClient([]), request_options={"model": "x"})


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-sonnet-5-5-20260901", "anthropic.claude-fable-5-1"]
)
def test_forced_tool_choice_is_rejected_early(model: str) -> None:
    with pytest.raises(ValueError, match="forced tool_choice"):
        _agent(FakeClient([]), model=model, tool_choice={"type": "any"})


def test_disabled_thinking_is_rejected_early_but_allowed_where_supported() -> None:
    with pytest.raises(ValueError, match="cannot disable thinking"):
        _agent(FakeClient([]), model="claude-opus-5-5", thinking="disabled")
    agent = _agent(FakeClient([]), model="claude-opus-5", thinking="disabled")
    agent.model = "claude-fable-5"
    with pytest.raises(ValueError, match="cannot disable thinking"):
        agent.ask("Hi")


def test_pause_turn_resumes_without_a_new_user_message() -> None:
    paused = make_message(
        [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x"}},
        ],
        stop_reason="pause_turn",
    )
    client = FakeClient([paused, _text("Found it.")])
    agent = _agent(client, server_tools=[{"type": "web_search_20260209", "name": "web_search"}])
    assert agent.ask("Search") == "Found it."
    second = client.messages.create_calls[1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant"]
    assert second["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]
    assert second["tool_choice"] == {"type": "auto"}


def test_pause_turn_counts_as_a_round() -> None:
    paused = make_message([{"type": "text", "text": "..."}], stop_reason="pause_turn")
    with pytest.raises(MaxToolRoundsError):
        _agent(FakeClient([paused, paused]), max_tool_rounds=1).ask("Hi")


def test_server_tools_are_listed_after_local_tools_and_removable() -> None:
    agent = _agent(
        FakeClient([]), tools=[add], server_tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}]
    )
    assert [t["name"] for t in agent._tool_definitions()] == ["add", "web_fetch"]
    with pytest.raises(ValueError, match="Duplicate"):
        agent.add_server_tool({"type": "x", "name": "add"})
    with pytest.raises(TypeError):
        agent.add_server_tool({"name": "without_type"})
    with pytest.raises(TypeError):
        agent.add_server_tool({"type": "without_name"})
    agent.add_server_tool({"type": "mcp_toolset", "mcp_server_name": "crm"})
    assert "crm" in repr(agent)
    agent.remove_tool("crm")
    assert agent.remove_tool("web_fetch") == {"type": "web_fetch_20260209", "name": "web_fetch"}
    assert "web_fetch" not in repr(agent)


# ---- streaming ------------------------------------------------------------------


def test_on_text_streams_and_returns_the_final_message() -> None:
    client = FakeClient([make_message([{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}])])
    chunks: list[str] = []
    agent = _agent(client, on_text=chunks.append)
    assert agent.ask("Hi") == "Hel\nlo"
    assert chunks == ["Hel", "lo"]
    assert client.messages.stream_calls and not client.messages.create_calls


def test_large_max_tokens_streams_automatically_and_stream_false_overrides() -> None:
    client = FakeClient([_text("a"), _text("b")])
    _agent(client, max_tokens=64_000).ask("Hi")
    _agent(client, max_tokens=64_000, stream=False).ask("Hi")
    assert len(client.messages.stream_calls) == 1 and len(client.messages.create_calls) == 1


def test_structured_output_can_be_streamed() -> None:
    class Answer(BaseModel):
        value: int

    client = FakeClient([ParsedResponse(_text('{"value": 3}'), Answer(value=3))])
    agent = ClaudeAgent(client=client, output_format=Answer, stream=True)  # type: ignore[arg-type]
    assert agent.ask("Three?") == Answer(value=3)
    call = client.messages.stream_calls[0]
    assert call["output_format"] is Answer
    assert call["extra_body"] == {"cache_control": {"type": "ephemeral"}}


# ---- multimodal input -----------------------------------------------------------


def test_content_helpers(tmp_path: Path) -> None:
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    notes = tmp_path / "notes.txt"
    notes.write_text("Plain notes", encoding="utf-8")

    assert image(png)["source"] == {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}
    assert image("https://example.com/a.png")["source"] == {"type": "url", "url": "https://example.com/a.png"}
    pdf_block = document(pdf, title="Report", citations=True)
    assert pdf_block["source"]["media_type"] == "application/pdf"
    assert pdf_block["title"] == "Report" and pdf_block["citations"] == {"enabled": True}
    assert document(notes)["source"] == {"type": "text", "media_type": "text/plain", "data": "Plain notes"}
    assert uploaded_file("file_1") == {"type": "document", "source": {"type": "file", "file_id": "file_1"}}
    assert uploaded_file("file_2", kind="image")["type"] == "image"
    with pytest.raises(ValueError, match="image media type"):
        image(b"raw")
    with pytest.raises(ValueError, match="documents only"):
        uploaded_file("file_3", kind="image", citations=True)


def test_to_content_validation() -> None:
    assert to_content(["Hello", {"type": "text", "text": "World"}]) == [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ]
    for bad in ("  ", [], ["  "], [{"no": "type"}], 5):
        with pytest.raises(ValueError):
            to_content(bad)  # type: ignore[arg-type]


def test_multimodal_prompt_is_sent_and_committed() -> None:
    client = FakeClient([_tool_uses(("add", {"a": 1, "b": 2})), _text("3")])
    agent = _agent(client, tools=[add])
    block = image("https://example.com/sum.png")
    assert agent.ask([block, "What is the sum?"]) == "3"
    assert agent.messages[0]["content"] == [block, {"type": "text", "text": "What is the sum?"}]
    assert tool_results_of_last_turn(agent) == 'Call add({"a": 1, "b": 2})\nadd result: 3'


# ---- hooks and parallel tools -----------------------------------------------------


def test_hooks_see_requests_responses_and_tool_calls() -> None:
    client = FakeClient([_tool_uses(("add", {"a": 1, "b": 2}), ("add", {"a": 5, "b": 5})), _text("done")])
    seen: list[Any] = []

    def before_tool(call: ToolCall) -> str | None:
        seen.append(("before", call.name, call.input, call.agent))
        return "too large" if call.input["a"] > 4 else None

    agent = _agent(
        client,
        name="calc",
        tools=[add],
        on_request=lambda args: seen.append(("request", len(args["messages"]))),
        on_response=lambda response: seen.append(("response", response.stop_reason)),
        before_tool=before_tool,
        after_tool=lambda call, content, is_error: seen.append(("after", content, is_error)),
    )
    agent.ask("Sums")
    assert seen == [
        ("request", 1),
        ("response", "tool_use"),
        ("before", "add", {"a": 1, "b": 2}, "calc"),
        ("after", "3", False),
        ("before", "add", {"a": 5, "b": 5}, "calc"),
        ("after", "Tool call denied: too large", True),
        ("request", 3),
        ("response", "end_turn"),
    ]
    results = agent.messages[2]["content"]
    assert results[1] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "Tool call denied: too large",
        "is_error": True,
    }


def test_before_tool_false_denies_with_raise_tool_errors() -> None:
    client = FakeClient([_tool_uses(("add", {"a": 1, "b": 2})), _text("ok")])
    agent = _agent(client, tools=[add], before_tool=lambda call: False)
    agent.ask("Sum")
    assert agent.messages[2]["content"][0]["content"] == "Tool call denied."


def test_parallel_tools_run_concurrently_in_order() -> None:
    barrier = threading.Barrier(2, timeout=5)

    @Tool
    def wait(label: str) -> str:
        """Wait until another call runs at the same time.

        Args:
            label: Echoed back.

        Returns:
            The label.
        """
        barrier.wait()  # Deadlocks (times out) unless both calls run concurrently.
        return label

    client = FakeClient([_tool_uses(("wait", {"label": "a"}), ("wait", {"label": "b"})), _text("ok")])
    agent = _agent(client, tools=[wait], max_parallel_tools=4)
    agent.ask("Go")
    assert [r["content"] for r in agent.messages[2]["content"]] == ["a", "b"]


def test_parallel_tools_see_the_callers_context_variables() -> None:
    from contextvars import ContextVar

    depth: ContextVar[int] = ContextVar("depth", default=0)

    @Tool
    def read_depth(label: str) -> int:
        """Return the current depth.

        Args:
            label: Ignored.

        Returns:
            The depth.
        """
        return depth.get()

    client = FakeClient(
        [_tool_uses(("read_depth", {"label": "a"}), ("read_depth", {"label": "b"})), _text("ok")]
    )
    agent = _agent(client, tools=[read_depth], max_parallel_tools=2)
    token = depth.set(3)
    try:
        agent.ask("Go")
    finally:
        depth.reset(token)
    assert [r["content"] for r in agent.messages[2]["content"]] == ["3", "3"]


def test_sync_agent_rejects_async_tools() -> None:
    @Tool
    async def slow(a: int) -> int:
        """Slow identity.

        Args:
            a: A number.

        Returns:
            The number.
        """
        return a

    with pytest.raises(ToolDefinitionError, match="AsyncClaudeAgent"):
        _agent(FakeClient([]), tools=[slow])


def test_request_id_is_recorded() -> None:
    response = _text("ok")
    response._request_id = "req_123"
    agent = _agent(FakeClient([response]))
    agent.ask("Hi")
    assert agent.last_usage is not None and agent.last_usage.request_id == "req_123"


# ---- refusals and fallbacks ---------------------------------------------------------


def test_fallbacks_use_the_matching_beta() -> None:
    client = FakeClient([_text("a"), _text("b")])
    _agent(client, model="claude-opus-5-5", fallbacks="default").ask("Hi")
    _agent(client, model="claude-opus-5-5", fallbacks=[{"model": "claude-opus-4-8"}]).ask("Hi")
    first, second = client.messages.create_calls
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert second["fallbacks"] == [{"model": "claude-opus-4-8"}]
    assert second["betas"] == ["server-side-fallback-2026-06-01"]
    with pytest.raises(ValueError, match="fallbacks"):
        _agent(client, fallbacks="sometimes")  # type: ignore[arg-type]


def test_refusal_error_carries_stop_details() -> None:
    class Answer(BaseModel):
        value: int

    refused = make_beta_message(
        [{"type": "text", "text": ""}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
    )
    agent = ClaudeAgent(client=FakeClient([ParsedResponse(refused, None)]), output_format=Answer)  # type: ignore[arg-type]
    with pytest.raises(RefusalError) as info:
        agent.ask("Hi")
    assert info.value.category == "cyber"
    assert info.value.stop_details.explanation == "Declined."


def test_fallback_turn_is_priced_per_model() -> None:
    served = make_beta_message(
        [{"type": "text", "text": "ok"}],
        model="claude-opus-4-8",
        usage={
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "iterations": [
                {
                    "type": "message",
                    "model": "claude-opus-5-5",
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            ],
        },
    )
    agent = _agent(FakeClient([served]), model="claude-opus-5-5", fallbacks="default")
    agent.ask("Hi")
    record = agent.last_usage
    assert record is not None
    assert record.model == "claude-opus-4-8"
    assert record.input_tokens == 2_000_000
    assert record.estimated_cost_usd == Decimal("9")  # 4 USD (Opus 5.5) + 5 USD (Opus 4.8)


# ---- compaction, history, token counting ----------------------------------------------


def test_compaction_blocks_stay_in_history_and_are_priced() -> None:
    compacted = make_beta_message(
        [
            {"type": "compaction", "content": "Summary of earlier turns"},
            {"type": "text", "text": "Next step"},
        ],
        usage={
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                {
                    "type": "message",
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            ],
        },
    )
    client = FakeClient([compacted])
    agent = _agent(
        client,
        model="claude-opus-5-5",
        request_options={
            "betas": ["compact-2026-01-12"],
            "context_management": {"edits": [{"type": "compact_20260112"}]},
        },
    )
    assert agent.ask("Continue") == "Next step"
    assert agent.messages[1]["content"][0] == {"type": "compaction", "content": "Summary of earlier turns"}
    assert client.messages.create_calls[0]["context_management"] == {"edits": [{"type": "compact_20260112"}]}
    assert agent.last_usage is not None and agent.last_usage.estimated_cost_usd == Decimal("8")


def test_batch_usage_is_half_price() -> None:
    from orkexa.usage import record_api_usage

    response = _text("ok", usage={"input_tokens": 1_000_000, "output_tokens": 0, "service_tier": "batch"})
    record = record_api_usage(response, agent_name="a", default_model="claude-opus-5-5")
    assert record is not None and record.estimated_cost_usd == Decimal("2")


def test_save_and_load_history(tmp_path: Path) -> None:
    agent = _agent(FakeClient([_text("Hallo")]))
    agent.ask("Hi")
    path = tmp_path / "history.json"
    agent.save_history(path)
    other = _agent(FakeClient([]))
    other.load_history(path)
    assert other.history() == agent.history()
    path.write_text('{"not": "a history"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a conversation history"):
        other.load_history(path)


def test_count_tokens_uses_the_next_request() -> None:
    client = FakeClient([])
    agent = _agent(client, tools=[add], effort="low")
    assert agent.count_tokens("How much?") == 1234
    call = client.messages.count_calls[0]
    assert call["messages"] == [{"role": "user", "content": "How much?"}]
    assert call["tools"][0]["name"] == "add" and call["output_config"] == {"effort": "low"}
    assert "max_tokens" not in call and "cache_control" not in call
    assert agent.messages == []
    with pytest.raises(ValueError, match="Nothing to count"):
        agent.count_tokens()
