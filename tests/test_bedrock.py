"""The Amazon Bedrock provider (Converse/ConverseStream through boto3), without network access.

Non-streamed requests go through a real ``bedrock-runtime`` client behind
``botocore.stub.Stubber``, which validates every request against the service model;
streamed requests use a patched ``converse_stream`` that validates the same way
(:class:`tests.contract.harness.BedrockHarness`). Every client has fake credentials.
Data residency is tested in ``test_bedrock_residency.py``.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import threading
import traceback
from typing import Any

from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    NoRegionError,
    ReadTimeoutError,
)
from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    AuthenticationError,
    BedrockOptions,
    ConfigurationError,
    HistoryConversionError,
    InvalidRequestError,
    MissingDependencyError,
    OrkexaError,
    ProviderConnectionError,
    ProviderServerError,
    RateLimitError,
    Tool,
    ToolCall,
    ToolCallSuspended,
    TurnSuspended,
    UnknownCapabilityError,
    UnsupportedCapabilityError,
    UsageTracker,
)
from orkexa.content import ToolContent, document, image
from orkexa.core.conversion import dump_history
from orkexa.core.types import (
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ReasoningPart,
    ResponseCompleted,
    TextDelta,
    ToolCallDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
)
from orkexa.dynamic import BudgetExceededError, BudgetTracker
from orkexa.pricing import ModelPrice, PriceResolver, Tariff, TokenRates
from orkexa.providers import bedrock as bedrock_module
from orkexa.providers.base import Provider
from orkexa.providers.bedrock import BedrockProvider, bedrock_capabilities, bedrock_usage
from orkexa.providers.bedrock_residency import parse_model_id
from orkexa.providers.registry import create_provider, provider_names, resolve_model

from .contract.harness import (
    FAKE_AWS,
    SECRET,
    AnthropicHarness,
    BedrockHarness,
    FakeEventStream,
    OpenAIHarness,
    Script,
    converse_events,
    converse_response,
)
from .helpers import add, lookup, multiply

pytestmark = pytest.mark.filterwarnings("ignore::orkexa.HistoryConversionWarning")

CLAUDE_45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
CLAUDE_5 = "anthropic.claude-sonnet-5-5"
NOVA = "amazon.nova-pro-v1:0"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"0" * 32


def make(
    responses: list[Any],
    *,
    model: str = CLAUDE_45,
    region: str = "eu-central-1",
    cls: type[Any] = Agent,
    **kwargs: Any,
) -> tuple[Any, BedrockHarness]:
    """An agent on a scripted, stubbed ``bedrock-runtime`` client."""
    harness = BedrockHarness()
    harness.script = Script(responses)
    agent = cls(model=f"bedrock:{model}", client=harness.scripted_client(region), **kwargs)
    return agent, harness


def tool_use(*calls: tuple[str, str, dict[str, Any]], reasoning: bool = True) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if reasoning:
        blocks.append({"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}})
    blocks += [{"toolUse": {"toolUseId": i, "name": n, "input": a}} for i, n, a in calls]
    return converse_response(blocks, "tool_use")


def text(value: str, **kwargs: Any) -> dict[str, Any]:
    return converse_response([{"text": value}], **kwargs)


def patch_stream(client: Any, *streams: FakeEventStream) -> list[dict[str, Any]]:
    """Answer the next ConverseStream calls with ``streams``; returns the recorded calls."""
    calls: list[dict[str, Any]] = []
    queue = list(streams)

    def converse_stream(**params: Any) -> Any:
        calls.append(params)
        return {"stream": queue.pop(0), "ResponseMetadata": {"RequestId": "req-stream"}}

    client.converse_stream = converse_stream
    return calls


# ---- registry and model ids ------------------------------------------------------------------


def test_bedrock_is_registered_and_model_ids_keep_colons_and_arns() -> None:
    assert "bedrock" in provider_names()
    assert resolve_model(f"bedrock:{CLAUDE_45}") == ("bedrock", CLAUDE_45)
    arn = "arn:aws:bedrock:eu-central-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-5-5"
    assert resolve_model(f"bedrock:{arn}") == ("bedrock", arn)
    assert resolve_model(arn, "bedrock") == ("bedrock", arn)
    assert isinstance(create_provider("bedrock"), BedrockProvider)


@pytest.mark.parametrize(
    ("model_id", "kind", "base"),
    [
        (CLAUDE_45, "foundation-model", CLAUDE_45),
        ("eu.anthropic.claude-sonnet-5-5", "inference-profile", CLAUDE_5),
        ("global.anthropic.claude-sonnet-5-5", "inference-profile", CLAUDE_5),
        ("us.amazon.nova-pro-v1:0", "inference-profile", NOVA),
        (f"arn:aws:bedrock:eu-central-1::foundation-model/{CLAUDE_45}", "foundation-model", CLAUDE_45),
        (
            "arn:aws:bedrock:eu-central-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-5-5",
            "inference-profile",
            CLAUDE_5,
        ),
        (
            "arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/a1b2",
            "application-inference-profile",
            None,
        ),
        ("arn:aws:bedrock:eu-central-1:123456789012:provisioned-model/x1", "in-region-arn", None),
        ("arn:aws:bedrock:eu-central-1:123456789012:prompt/P1", "unverifiable", None),
        ("arn:aws:sagemaker:eu-central-1:123456789012:endpoint/e", "unverifiable", None),
        ("mymodel", "unverifiable", None),
    ],
)
def test_model_ids_are_classified_without_network(model_id: str, kind: str, base: str | None) -> None:
    ref = parse_model_id(model_id)
    assert (ref.kind, ref.base_model) == (kind, base)


def test_requests_carry_the_full_model_id() -> None:
    arn = "arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/a1b2"
    agent, harness = make([text("ok")], model=arn)
    assert agent.ask("hi") == "ok"
    assert harness.script.calls[0]["modelId"] == arn
    agent, harness = make([text("ok")])
    agent.ask("hi")
    assert harness.script.calls[0]["modelId"] == CLAUDE_45


def test_capabilities_come_from_the_base_model() -> None:
    sonnet45 = bedrock_capabilities(CLAUDE_45)
    assert sonnet45.supports("structured_output") and sonnet45.supports("token_counting")
    assert sonnet45.supports("tools") and sonnet45.supports("caching")
    claude5 = bedrock_capabilities(CLAUDE_5)
    assert claude5.support("token_counting") == "unsupported"
    assert claude5.support("forced_tool_choice") == "unsupported"
    assert claude5.supports("reasoning")
    nova = bedrock_capabilities(NOVA)
    assert nova.support("structured_output") == "unsupported" and nova.supports("image_input")
    assert bedrock_capabilities("amazon.nova-micro-v1:0").support("image_input") == "unsupported"
    unknown = bedrock_capabilities("meta.llama3-70b-instruct-v1:0")
    assert unknown.support("tools") == "unknown"
    assert unknown.support("system_messages") == "unsupported"
    provider = create_provider("bedrock")
    assert provider.capabilities("eu.anthropic.claude-sonnet-5-5") == claude5


# ---- optional dependency ---------------------------------------------------------------------


def test_missing_boto3_raises_only_when_the_client_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    agent = Agent(model=f"bedrock:{CLAUDE_45}")
    assert agent.adapter._client is None
    with pytest.raises(MissingDependencyError, match=r"orkexa\[bedrock\]"):
        agent.client  # noqa: B018
    with pytest.raises(MissingDependencyError, match=r"orkexa\[bedrock\]"):
        agent.ask("hi")


def test_import_orkexa_works_without_boto3() -> None:
    code = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("boto3", "botocore"):
            raise ImportError("blocked " + name)
        return None

sys.meta_path.insert(0, Block())
import orkexa
from orkexa import Agent, BedrockOptions, MissingDependencyError
agent = Agent(model="bedrock:anthropic.claude-sonnet-5-5", options=BedrockOptions(region_name="eu-central-1"))
assert "boto3" not in sys.modules and "botocore" not in sys.modules
try:
    agent.client
except MissingDependencyError as exc:
    assert "orkexa[bedrock]" in str(exc), exc
    print("ok")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ---- credentials, region and endpoints ----------------------------------------------------------


class RecordingSession:
    instances: list[RecordingSession] = []  # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.clients: list[tuple[str, dict[str, Any]]] = []
        RecordingSession.instances.append(self)

    def client(self, service: str, **kwargs: Any) -> Any:
        self.clients.append((service, kwargs))
        return object()


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[RecordingSession]:
    import boto3

    RecordingSession.instances = []
    monkeypatch.setattr(boto3, "Session", RecordingSession)
    return RecordingSession.instances


def test_session_gets_profile_and_region(sessions: list[RecordingSession]) -> None:
    agent = Agent(
        model=f"bedrock:{CLAUDE_45}", options=BedrockOptions(region_name="eu-west-1", profile_name="prod")
    )
    agent.client  # noqa: B018
    (session,) = sessions
    assert session.kwargs == {"profile_name": "prod", "region_name": "eu-west-1"}
    ((service, kwargs),) = session.clients
    assert service == "bedrock-runtime" and kwargs["region_name"] == "eu-west-1"
    assert "endpoint_url" not in kwargs
    config = kwargs["config"]
    assert config.ignore_configured_endpoint_urls is True
    assert config.read_timeout == 900
    assert config.retries == {"mode": "standard"}


def test_timeout_and_retries_reach_the_botocore_config(sessions: list[RecordingSession]) -> None:
    agent = Agent(model=f"bedrock:{CLAUDE_45}", timeout=30, max_retries=2)
    agent.client  # noqa: B018
    config = sessions[0].clients[0][1]["config"]
    assert (config.read_timeout, config.connect_timeout) == (30, 10)
    assert config.retries == {"mode": "standard", "total_max_attempts": 3}


def test_an_injected_session_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    def no_session(**kwargs: Any) -> Any:
        raise AssertionError("boto3.Session must not be called")

    monkeypatch.setattr(boto3, "Session", no_session)
    session = RecordingSession(region_name="eu-west-3")
    agent = Agent(model=f"bedrock:{CLAUDE_45}", options=BedrockOptions(session=session))
    agent.client  # noqa: B018
    assert session.clients[0][0] == "bedrock-runtime"
    with pytest.raises(ConfigurationError, match="session= or profile_name="):
        BedrockOptions(session=session, profile_name="prod")


def test_api_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="AWS credentials"):
        Agent(model=f"bedrock:{CLAUDE_45}", api_key="sk-something-1234567890")


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fake AWS credentials and region in the environment; no shared config files."""
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_AWS["aws_access_key_id"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_AWS["aws_secret_access_key"])
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def test_other_providers_keys_are_never_read(aws_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    read: list[str] = []
    monkeypatch.setattr(Provider, "env_key", staticmethod(lambda variable: read.append(variable)))
    agent = Agent(model=f"bedrock:{CLAUDE_45}")
    client = agent.client
    credentials = client._request_signer._credentials
    assert credentials.access_key == FAKE_AWS["aws_access_key_id"]
    assert SECRET not in (credentials.access_key, credentials.secret_key, credentials.token)
    assert agent.adapter.config.api_key is None
    assert read == []


def test_configured_endpoint_urls_are_ignored(aws_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "https://attacker.example")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://attacker.example")
    client = Agent(model=f"bedrock:{CLAUDE_45}").client
    assert client.meta.endpoint_url == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    assert client.meta.region_name == "eu-central-1"


@pytest.mark.parametrize("url", ["http://bedrock.example.com", "ftp://x.example", "http://10.0.0.1:8080"])
def test_base_url_must_use_https(url: str) -> None:
    with pytest.raises(ConfigurationError, match="https"):
        Agent(model=f"bedrock:{CLAUDE_45}", base_url=url)


def test_https_and_localhost_base_urls_are_used(aws_env: None) -> None:
    vpce = "https://vpce-0abc-1234.bedrock-runtime.eu-central-1.vpce.amazonaws.com"
    assert Agent(model=f"bedrock:{CLAUDE_45}", base_url=vpce).client.meta.endpoint_url == vpce
    local = "http://localhost:4566"
    assert Agent(model=f"bedrock:{CLAUDE_45}", base_url=local).client.meta.endpoint_url == local


def test_a_missing_region_is_a_configuration_error(aws_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.delenv("AWS_DEFAULT_REGION")
    agent = Agent(model=f"bedrock:{CLAUDE_45}")
    with pytest.raises(ConfigurationError, match="region"):
        agent.ask("hi")


def test_changed_options_rebuild_the_client(sessions: list[RecordingSession]) -> None:
    agent = Agent(model=f"bedrock:{CLAUDE_45}", options=BedrockOptions(region_name="eu-west-1"))
    first = agent.adapter._runtime(agent.options)
    assert agent.adapter._runtime(agent.options) is first
    agent.options = BedrockOptions(region_name="eu-north-1")
    agent.adapter._runtime(agent.options)
    assert [s.kwargs["region_name"] for s in sessions] == ["eu-west-1", "eu-north-1"]


# ---- options ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"additional_model_request_fields": {"tools": []}},
        {"additional_model_request_fields": {"anthropic_beta": ["x"]}},
        {"allowed_regions": "eu-central-1"},
        {"allowed_regions": ()},
        {"allowed_regions": ("EU-CENTRAL-1",)},
        {"data_residency": "eu_only", "allowed_regions": ("eu-central-1", "us-east-1")},
        {"data_residency": "eu_only", "allowed_regions": ("eu-west-2",)},
        {"data_residency": "us_only"},
        {"cache_ttl": "10m"},
        {"service_tier": "cheap"},
        {"performance_latency": "fast"},
        {"stop_sequences": "END"},
        {"guardrail": {"guardrailVersion": "1"}},
        {"guardrail": {"guardrailIdentifier": "g", "bogus": "x"}},
        {"request_metadata": {"k": 1}},
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        BedrockOptions(**kwargs)


def test_other_provider_options_are_rejected() -> None:
    from orkexa import OpenAIOptions

    with pytest.raises(ConfigurationError, match="BedrockOptions"):
        Agent(model=f"bedrock:{CLAUDE_45}", options=OpenAIOptions())


def test_options_reach_the_converse_request() -> None:
    options = BedrockOptions(
        temperature=0.2,
        top_p=0.9,
        stop_sequences=("END",),
        service_tier="flex",
        performance_latency="optimized",
        guardrail={"guardrailIdentifier": "gr-1", "guardrailVersion": "2"},
        additional_model_request_fields={"top_k": 40},
        request_metadata={"team": "search"},
    )
    agent, harness = make([text("ok")], options=options, system_prompt="Be brief.", max_tokens=300)
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call["inferenceConfig"] == {
        "maxTokens": 300,
        "temperature": 0.2,
        "topP": 0.9,
        "stopSequences": ["END"],
    }
    assert call["system"] == [{"text": "Be brief."}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert call["serviceTier"] == {"type": "flex"}
    assert call["performanceConfig"] == {"latency": "optimized"}
    assert call["guardrailConfig"] == {"guardrailIdentifier": "gr-1", "guardrailVersion": "2"}
    assert call["additionalModelRequestFields"] == {"top_k": 40}
    assert call["requestMetadata"] == {"team": "search"}


def test_tool_choice_mapping() -> None:
    agent, harness = make([text("ok")], tools=[add], tool_choice="required")
    agent.ask("go")
    assert harness.script.calls[0]["toolConfig"]["toolChoice"] == {"any": {}}
    agent, harness = make([text("ok")], tools=[add], tool_choice={"name": "add"})
    agent.ask("go")
    assert harness.script.calls[0]["toolConfig"]["toolChoice"] == {"tool": {"name": "add"}}
    spec = harness.script.calls[0]["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "add" and spec["inputSchema"]["json"]["required"] == ["a", "b"]
    with pytest.raises(ConfigurationError, match="'none'"):
        make([text("ok")], tools=[add], tool_choice="none")


# ---- asking ----------------------------------------------------------------------------------


def test_sync_ask() -> None:
    agent, _harness = make([text("Hello!")])
    assert agent.ask("Hi") == "Hello!"
    response = agent.last_response
    assert response is not None
    assert (response.provider, response.model, response.stop_reason) == ("bedrock", CLAUDE_45, "end_turn")
    assert response.request_id == "req-bedrock-1"
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].provider == "bedrock"


def test_async_ask_runs_boto3_in_a_worker_thread() -> None:
    agent, _harness = make([text("Hello!")], cls=AsyncAgent)
    threads: list[str] = []
    agent.client.meta.events.register(
        "provide-client-params.bedrock-runtime.Converse",
        lambda **kw: threads.append(threading.current_thread().name),
    )

    async def main() -> str:
        result: str = await agent.ask("Hi")
        return result

    assert asyncio.run(main()) == "Hello!"
    assert threads == ["orkexa-bedrock"]


# ---- streaming -------------------------------------------------------------------------------


def _mixed_response() -> dict[str, Any]:
    return converse_response(
        [
            {"reasoningContent": {"reasoningText": {"text": "think", "signature": "sig-1"}}},
            {"text": "Let me add."},
            {"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 1, "b": 2}}},
            {"toolUse": {"toolUseId": "t2", "name": "multiply", "input": {"a": 3, "b": 4}}},
        ],
        "tool_use",
    )


def test_stream_deltas_and_history_equal_the_non_streamed_turn() -> None:
    plain, _ = make([_mixed_response(), text("3 and 12")], tools=[add, multiply], stream=False)
    assert plain.ask("go") == "3 and 12"
    events: list[Any] = []
    streamed, harness = make(
        [_mixed_response(), text("3 and 12")], tools=[add, multiply], stream=True, on_event=events.append
    )
    assert streamed.ask("go") == "3 and 12"
    assert dump_history(plain.messages) == dump_history(streamed.messages)
    starts = [(e.call_id, e.name) for e in events if isinstance(e, ToolCallStarted)]
    assert starts == [("t1", "add"), ("t2", "multiply")]
    arguments = "".join(
        e.arguments_delta for e in events if isinstance(e, ToolCallDelta) and e.call_id == "t1"
    )
    assert json.loads(arguments) == {"a": 1, "b": 2}
    assert "".join(e.text for e in events if isinstance(e, ReasoningDelta)) == "think"
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Let me add.3 and 12"
    completed = [e.response for e in events if isinstance(e, ResponseCompleted)]
    assert [r.request_id for r in completed] == ["req-bedrock-1", "req-bedrock-1"]
    assert all(stream.closed for stream in harness.streams)


def test_async_stream() -> None:
    deltas: list[str] = []
    agent, harness = make([text("streamed answer")], cls=AsyncAgent, on_text=deltas.append)
    assert asyncio.run(agent.ask("hi")) == "streamed answer"
    assert "".join(deltas) == "streamed answer"
    assert harness.streams[0].closed


@pytest.mark.parametrize(
    ("event", "error"),
    [
        ({"throttlingException": {"message": "slow down AKIAIOSFODNN7EXAMPLE"}}, RateLimitError),
        ({"modelStreamErrorException": {"message": "model failed"}}, ProviderServerError),
        ({"internalServerException": {"message": "oops"}}, ProviderServerError),
        ({"validationException": {"message": "bad"}}, InvalidRequestError),
        ({"serviceUnavailableException": {"message": "busy"}}, ProviderServerError),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_exception_events_in_the_stream_raise(
    event: dict[str, Any], error: type[Exception], asynchronous: bool
) -> None:
    agent, _ = make([], cls=AsyncAgent if asynchronous else Agent, stream=True)
    events = [*converse_events(text("partial"))[:2], event]
    stream = FakeEventStream(events)
    patch_stream(agent.client, stream)
    with pytest.raises(error) as info:
        asyncio.run(agent.ask("hi")) if asynchronous else agent.ask("hi")
    assert "AKIAIOSFODNN7EXAMPLE" not in str(info.value)
    assert stream.closed and agent.messages == []


@pytest.mark.parametrize("drop", ["messageStop", "metadata"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_incomplete_streams_raise(drop: str, asynchronous: bool) -> None:
    tracker = UsageTracker()
    agent, _ = make([], cls=AsyncAgent if asynchronous else Agent, stream=True, usage_tracker=tracker)
    stream = FakeEventStream([e for e in converse_events(text("partial")) if drop not in e])
    patch_stream(agent.client, stream)
    with pytest.raises(ProviderConnectionError, match=drop if drop == "messageStop" else "usage metadata"):
        asyncio.run(agent.ask("hi")) if asynchronous else agent.ask("hi")
    assert stream.closed and agent.messages == []


def test_stream_without_event_stream_is_an_error() -> None:
    agent, _ = make([], stream=True)
    agent.client.converse_stream = lambda **kw: {"ResponseMetadata": {"RequestId": "r"}}
    with pytest.raises(OrkexaError, match="no event stream"):
        agent.ask("hi")


def test_sync_early_exit_closes_the_stream() -> None:
    def stop(delta: str) -> None:
        raise KeyboardInterrupt

    agent, _ = make([], stream=True, on_text=stop)
    events = converse_events(text("a long answer in many pieces"))
    stream = FakeEventStream(events)
    patch_stream(agent.client, stream)
    with pytest.raises(KeyboardInterrupt):
        agent.ask("hi")
    assert stream.closed and stream.consumed < len(events)


def test_closing_the_provider_stream_closes_the_event_stream() -> None:
    agent, _ = make([], stream=True)
    stream = FakeEventStream(converse_events(text("one two")))
    patch_stream(agent.client, stream)
    request = ModelRequest(model=CLAUDE_45, messages=[Message(role="user", content="hi")], max_tokens=10)
    provider = agent.adapter
    events = provider.stream(provider.prepare(request), request)
    assert isinstance(next(events), TextDelta)
    events.close()
    assert stream.closed


class BlockingStream(FakeEventStream):
    """Yields its first event, then blocks until it is closed (like a slow network read)."""

    def __init__(self, events: list[Any]) -> None:
        super().__init__(events)
        self.release = threading.Event()

    def __iter__(self) -> Any:
        yield from self.events[:2]  # messageStart and the first text delta
        self.release.wait(10)
        if not self.closed:
            yield from self.events[2:]

    def close(self) -> None:
        super().close()
        self.release.set()


def test_async_cancellation_closes_the_stream() -> None:
    first = asyncio.Event()
    agent, _ = make([], cls=AsyncAgent, on_text=lambda delta: first.set())
    stream = BlockingStream(converse_events(text("first, then never more")))
    patch_stream(agent.client, stream)

    async def main() -> None:
        task = asyncio.create_task(agent.ask("hi"))
        await asyncio.wait_for(first.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert stream.closed
    assert agent.messages == []


# ---- tools and reasoning ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "status"),
    [(CLAUDE_45, True), ("anthropic.claude-3-5-sonnet-20240620-v1:0", True), (NOVA, True), (CLAUDE_5, False)],
)
def test_parallel_tool_results(model: str, status: bool) -> None:
    capabilities = {"tools": "supported"} if model.startswith("anthropic.claude-3") else None
    agent, harness = make(
        [
            tool_use(("t1", "add", {"a": 1, "b": 2}), ("t2", "lookup", {"key": "x"}), reasoning=False),
            text("done"),
        ],
        model=model,
        tools=[add, lookup],
        raise_tool_errors=False,
        capabilities=capabilities,
    )
    assert agent.ask("go") == "done"
    second = harness.script.calls[1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    first, failed = (b["toolResult"] for b in second["messages"][-1]["content"])
    assert first == {"toolUseId": "t1", "content": [{"text": "3"}]}
    assert failed["toolUseId"] == "t2"
    if status:
        assert failed["status"] == "error"
        assert "unknown key" in failed["content"][0]["text"]
    else:
        assert "status" not in failed
        assert failed["content"][0] == {"text": "Error:"}
        assert "unknown key" in failed["content"][1]["text"]


def test_multi_round_agent() -> None:
    agent, harness = make(
        [
            tool_use(("t1", "add", {"a": 1, "b": 2})),
            tool_use(("t2", "multiply", {"a": 3, "b": 4})),
            text("12"),
        ],
        tools=[add, multiply],
    )
    assert agent.ask("compute") == "12"
    last = harness.script.calls[2]
    assert [m["role"] for m in last["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    ids = [b["toolResult"]["toolUseId"] for m in last["messages"] for b in m["content"] if "toolResult" in b]
    assert ids == ["t1", "t2"]
    replayed = [
        b
        for m in last["messages"]
        if m["role"] == "assistant"
        for b in m["content"]
        if "reasoningContent" in b
    ]
    assert len(replayed) == 2


def test_reasoning_blocks_are_replayed_with_signature_and_redacted_bytes(tmp_path: Path) -> None:
    secret_bytes = b"\x00\x01opaque-redacted"
    response = converse_response(
        [
            {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}},
            {"reasoningContent": {"redactedContent": secret_bytes}},
            {"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 1, "b": 1}}},
        ],
        "tool_use",
    )
    agent, harness = make([response, text("two")], tools=[add])
    agent.ask("1+1")
    reasoning = [p for m in agent.messages for p in m.parts() if isinstance(p, ReasoningPart)]
    assert [(p.text, p.redacted) for p in reasoning] == [("plan", False), (None, True)]
    path = tmp_path / "history.json"
    agent.save_history(path)
    resumed, harness = make([text("again")], tools=[add])
    resumed.load_history(path)
    resumed.ask("more")
    assistant = harness.script.calls[0]["messages"][1]["content"]
    assert assistant[0] == {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}}
    assert assistant[1] == {"reasoningContent": {"redactedContent": secret_bytes}}


def test_foreign_tool_ids_are_mapped_consistently() -> None:
    long_id = "call/" + "x" * 80
    history = [
        Message(role="user", content="add"),
        Message(role="assistant", content=[ToolCallPart(id=long_id, name="add", arguments={"a": 1, "b": 1})]),
        Message(role="user", content=[ToolResultPart(call_id=long_id, name="add", content="2")]),
        Message(role="assistant", content="two"),
    ]
    agent, harness = make([text("ok")], tools=[add], messages=history)
    agent.ask("again")
    messages = harness.script.calls[0]["messages"]
    use_id = messages[1]["content"][0]["toolUse"]["toolUseId"]
    assert use_id == messages[2]["content"][0]["toolResult"]["toolUseId"]
    assert use_id.startswith("orkexa_") and len(use_id) <= 64


# ---- images and documents ----------------------------------------------------------------------


def test_image_and_pdf_are_sent_as_bytes() -> None:
    agent, harness = make([text("ok")])
    agent.ask(
        [
            image(PNG, media_type="image/png"),
            document(PDF, media_type="application/pdf", title="Q3 report: final!", context="from finance"),
            "Summarize.",
        ]
    )
    content = harness.script.calls[0]["messages"][0]["content"]
    assert content[0] == {"image": {"format": "png", "source": {"bytes": PNG}}}
    assert content[1] == {
        "document": {
            "format": "pdf",
            "name": "Q3 report final",
            "source": {"bytes": PDF},
            "context": "from finance",
        }
    }
    assert content[2] == {"text": "Summarize."}


def test_webp_and_jpeg_images() -> None:
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
    agent, harness = make([text("ok")])
    agent.ask([image(webp, media_type="image/webp"), image(JPEG, media_type="image/jpeg"), "Compare."])
    formats = [
        b["image"]["format"] for b in harness.script.calls[0]["messages"][0]["content"] if "image" in b
    ]
    assert formats == ["webp", "jpeg"]


def test_text_documents_become_text() -> None:
    agent, harness = make([text("ok")])
    agent.ask([document(b"plain notes", media_type="text/plain", title="notes"), "Summarize."])
    content = harness.script.calls[0]["messages"][0]["content"]
    assert all("text" in block for block in content)
    assert "plain notes" in content[0]["text"]


def test_duplicate_document_names_are_made_unique() -> None:
    agent, harness = make([text("ok")])
    agent.ask(
        [document(PDF, media_type="application/pdf"), document(PDF, media_type="application/pdf"), "Diff."]
    )
    names = [
        b["document"]["name"] for b in harness.script.calls[0]["messages"][0]["content"] if "document" in b
    ]
    assert names == ["Document 1", "Document 2"]


def _rejected(prompt: Any, error: type[Exception], match: str, model: str = CLAUDE_45) -> None:
    agent, harness = make([text("unused")], model=model)
    with pytest.raises(error, match=match):
        agent.ask(prompt)
    assert harness.script.calls == []


def test_a_document_needs_a_text_block() -> None:
    _rejected([document(PDF, media_type="application/pdf")], InvalidRequestError, "text block")


def test_media_limits() -> None:
    _rejected([image(PNG + b"\x00" * 3_750_000, media_type="image/png"), "x"], InvalidRequestError, "at most")
    _rejected([image(PNG, media_type="image/png")] * 21 + ["x"], InvalidRequestError, "at most 20 images")
    _rejected(
        [document(PDF, media_type="application/pdf")] * 6 + ["x"], InvalidRequestError, "at most 5 documents"
    )
    big_pdf = PDF + b"0" * 4_500_000
    _rejected([document(big_pdf, media_type="application/pdf"), "x"], InvalidRequestError, "at most")


def test_content_must_match_the_declared_format() -> None:
    _rejected([image(JPEG, media_type="image/png"), "x"], InvalidRequestError, "declared format 'png'")
    _rejected(
        [document(PNG, media_type="application/pdf"), "x"], InvalidRequestError, "declared format 'pdf'"
    )


def test_urls_are_not_fetched() -> None:
    _rejected(
        [image("https://example.com/cat.png"), "x"], UnsupportedCapabilityError, "cannot fetch image URLs"
    )
    _rejected(
        [document("https://example.com/a.pdf"), "x"], UnsupportedCapabilityError, "cannot fetch document"
    )


S3_IMAGE = {
    "type": "image",
    "source": {"type": "url", "url": "s3://bucket/cat.png", "media_type": "image/png"},
}


def test_s3_sources_only_for_nova() -> None:
    _rejected([S3_IMAGE, "x"], UnsupportedCapabilityError, "Only Amazon Nova")
    agent, harness = make([text("ok")], model=NOVA)
    agent.ask([S3_IMAGE, "Describe."])
    block = harness.script.calls[0]["messages"][0]["content"][0]
    assert block == {"image": {"format": "png", "source": {"s3Location": {"uri": "s3://bucket/cat.png"}}}}


@Tool
def screenshot() -> ToolContent:
    """Take a screenshot.

    Returns:
        The image.
    """
    return [image(PNG, media_type="image/png")]


def test_images_in_tool_results() -> None:
    agent, harness = make([tool_use(("t1", "screenshot", {})), text("seen")], tools=[screenshot])
    assert agent.ask("look") == "seen"
    result = harness.script.calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert result["content"] == [{"image": {"format": "png", "source": {"bytes": PNG}}}]


# ---- structured output, effort, caching ----------------------------------------------------------


class City(BaseModel):
    name: str
    population: int


def test_structured_output() -> None:
    agent, harness = make([text('{"name": "Vienna", "population": 2000000}')], output_format=City)
    assert agent.ask("Capital of Austria?") == City(name="Vienna", population=2000000)
    text_format = harness.script.calls[0]["outputConfig"]["textFormat"]
    assert text_format["type"] == "json_schema"
    structure = text_format["structure"]["jsonSchema"]
    assert structure["name"] == "City"
    schema = json.loads(structure["schema"])
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "population"}


def test_structured_output_is_rejected_for_nova() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="structured outputs"):
        make([text("{}")], model=NOVA, output_format=City)


def test_effort_goes_to_output_config() -> None:
    agent, harness = make([text("ok"), text("ok")], model=CLAUDE_5, effort="high")
    agent.ask("think")
    assert harness.script.calls[0]["outputConfig"] == {"effort": "high"}
    agent.ask("again", effort="low")
    assert harness.script.calls[1]["outputConfig"] == {"effort": "low"}


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_prompt_caching_adds_cache_points(ttl: str) -> None:
    point = {"type": "default"} if ttl == "5m" else {"type": "default", "ttl": "1h"}
    agent, harness = make(
        [text("ok")],
        tools=[add],
        options=BedrockOptions(prompt_caching=True, cache_ttl=ttl),  # type: ignore[arg-type]
    )
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call["system"][-1] == {"cachePoint": point}
    assert call["toolConfig"]["tools"][-1] == {"cachePoint": point}
    assert call["messages"][-1]["content"][-1] == {"cachePoint": point}


def test_prompt_caching_on_nova_and_unknown_models() -> None:
    agent, harness = make([text("ok")], model=NOVA, tools=[add], options=BedrockOptions(prompt_caching=True))
    agent.ask("hi")
    assert all("cachePoint" not in t for t in harness.script.calls[0]["toolConfig"]["tools"])
    with pytest.raises(UnsupportedCapabilityError, match="5 minutes"):
        make([text("ok")], model=NOVA, options=BedrockOptions(prompt_caching=True, cache_ttl="1h"))
    with pytest.raises(UnknownCapabilityError):
        make([text("ok")], model="meta.llama3-70b-instruct-v1:0", options=BedrockOptions(prompt_caching=True))


# ---- usage and pricing -------------------------------------------------------------------------


def test_usage_normalization() -> None:
    usage = bedrock_usage(
        {
            "inputTokens": 100,
            "outputTokens": 40,
            "totalTokens": 190,
            "cacheReadInputTokens": 30,
            "cacheWriteInputTokens": 20,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 15}, {"ttl": "1h", "inputTokens": 5}],
        },
        {"type": "priority"},
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (100, 40)
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (30, 20)
    assert (usage.cache_creation_5m_input_tokens, usage.cache_creation_1h_input_tokens) == (15, 5)
    assert usage.service_tier == "priority"
    assert bedrock_usage(None) is None


def _bedrock_price(model: str = CLAUDE_45, tariffs: tuple[Tariff, ...] = ()) -> ModelPrice:
    return ModelPrice(
        provider="bedrock",
        model=model,
        rates=TokenRates.of("3", "15", "0.3", "3.75", "6"),
        tariffs=tariffs,
        source="test",
        version="2026-11",
    )


def test_records_carry_provider_and_region() -> None:
    tracker = UsageTracker()
    price = _bedrock_price(tariffs=(Tariff.of("aws_region", "eu-west-3"),))
    agent, _ = make(
        [text("ok")], usage_tracker=tracker, region="eu-west-3", pricing=PriceResolver([price], version="t")
    )
    agent.ask("hi")
    (record,) = tracker.records
    assert record.provider == "bedrock" and record.model == CLAUDE_45
    assert (record.input_tokens, record.cache_read_input_tokens, record.output_tokens) == (80, 20, 50)
    assert record.request_id == "req-bedrock-1"
    assert record.pricing_conditions == (("aws_region", "eu-west-3"),)


def test_unknown_prices_stay_unknown() -> None:
    agent, _ = make([text("ok")])
    agent.ask("hi")
    assert agent.last_usage is not None and agent.last_usage.estimated_cost_usd is None


def test_a_region_tariff_prices_the_request() -> None:
    price = _bedrock_price(tariffs=(Tariff.of("aws_region", "eu-central-1", **{"eu-west-1": "1.1"}),))
    agent, _ = make([text("ok"), text("ok")], pricing=PriceResolver([price], version="t"))
    agent.ask("hi")
    record = agent.last_usage
    # 80 x $3 + 50 x $15 + 20 x $0.3 per million tokens
    assert record is not None and record.estimated_cost_usd == Decimal("0.000996")
    agent, _ = make([text("ok")], pricing=PriceResolver([price], version="t"), region="eu-west-1")
    agent.ask("hi")
    assert agent.last_usage is not None and agent.last_usage.estimated_cost_usd == Decimal("0.0010956")


def test_a_price_without_the_region_tariff_is_not_used() -> None:
    for price in (
        _bedrock_price(),
        _bedrock_price(tariffs=(Tariff.of("aws_region", "us-east-1"),)),
    ):
        agent, _ = make([text("ok")], pricing=PriceResolver([price], version="t"))
        agent.ask("hi")
        record = agent.last_usage
        assert record is not None and record.estimated_cost_usd is None
        assert record.missing_cost_components == ("tariff:aws_region=eu-central-1",)


@pytest.mark.parametrize("model", ["anthropic.claude-opus-5-5", "eu.anthropic.claude-opus-5-5"])
def test_anthropic_first_party_prices_never_apply(model: str) -> None:
    agent, _ = make([text("ok")], model=model)
    agent.ask("hi")
    assert agent.last_usage is not None and agent.last_usage.estimated_cost_usd is None
    assert agent.adapter.has_pricing(model) is False
    assert agent.adapter.has_pricing("claude-opus-5-5") is False


def test_budget_reservations_are_enforced_before_the_request() -> None:
    price = _bedrock_price(tariffs=(Tariff.of("aws_region", "eu-central-1"),))
    budget = BudgetTracker("0.0001", on_oversized="reject")
    agent, harness = make([text("ok")], pricing=PriceResolver([price], version="t"), usage_tracker=budget)
    with pytest.raises(BudgetExceededError, match="not sent"):
        agent.ask("hi")
    assert harness.script.calls == []
    assert budget.reserved_usd == 0


def test_budgets_count_recorded_bedrock_costs() -> None:
    price = _bedrock_price(tariffs=(Tariff.of("aws_region", "eu-central-1"),))
    budget = BudgetTracker("0.0015")
    agent, harness = make(
        [text("one"), text("two")],
        pricing=PriceResolver([price], version="t"),
        usage_tracker=budget,
        max_tokens=10,
    )
    agent.ask("hi")
    assert budget.spent_usd == Decimal("0.000996") and budget.unbounded_requests == 0
    with pytest.raises(BudgetExceededError):
        agent.ask("again")  # sent (the bound fits), then the recorded cost exceeds the limit
    with pytest.raises(BudgetExceededError, match="used up"):
        agent.ask("once more")
    assert len(harness.script.calls) == 2


# ---- errors ------------------------------------------------------------------------------------


def _client_error(code: str, status: int, message: str = "failed") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": "req-err"},
        },
        "Converse",
    )


@pytest.mark.parametrize(
    ("code", "status", "error"),
    [
        ("AccessDeniedException", 403, AuthenticationError),
        ("ExpiredTokenException", 403, AuthenticationError),
        ("UnrecognizedClientException", 403, AuthenticationError),
        ("ThrottlingException", 429, RateLimitError),
        ("ServiceQuotaExceededException", 400, RateLimitError),
        ("InternalServerException", 500, ProviderServerError),
        ("ModelTimeoutException", 408, ProviderServerError),
        ("ServiceUnavailableException", 503, ProviderServerError),
        ("ValidationException", 400, InvalidRequestError),
        ("ResourceNotFoundException", 404, InvalidRequestError),
        ("SomethingNew", 418, InvalidRequestError),
        ("SomethingElse", 502, ProviderServerError),
    ],
)
def test_client_errors_are_mapped(code: str, status: int, error: type[Exception]) -> None:
    agent, _ = make([_client_error(code, status)])
    with pytest.raises(error) as info:
        agent.ask("hi")
    assert type(info.value) is error
    assert info.value.provider == "bedrock"  # type: ignore[attr-defined]
    assert info.value.status_code == status  # type: ignore[attr-defined]
    assert agent.messages == []
    if code == "AccessDeniedException":
        assert "model access" in str(info.value)


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (NoCredentialsError(), AuthenticationError),
        (
            EndpointConnectionError(endpoint_url="https://bedrock-runtime.eu-central-1.amazonaws.com"),
            ProviderConnectionError,
        ),
        (
            ReadTimeoutError(endpoint_url="https://bedrock-runtime.eu-central-1.amazonaws.com"),
            ProviderConnectionError,
        ),
        (NoRegionError(), ConfigurationError),
    ],
)
def test_botocore_errors_are_mapped(exc: Exception, error: type[Exception]) -> None:
    agent, _ = make([exc])
    with pytest.raises(error):
        agent.ask("hi")


AWS_SECRETS = (
    "AKIAIOSFODNN7EXAMPLE",
    "ASIAZZZZEXAMPLE12345",
    "IQoJb3JpZ2luX2VjEXAMPLEtokenEXAMPLEtokenEXAMPLE",
    "ABSKQmVkcm9ja0FQSUtleS1EXAMPLEexample",
    "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29tEXAMPLE",
    "0123456789abcdef0123456789abcdef",
)


@pytest.mark.parametrize("stream", [False, True])
def test_errors_are_redacted(stream: bool) -> None:
    message = (
        "denied for AKIAIOSFODNN7EXAMPLE / ASIAZZZZEXAMPLE12345 token "
        "IQoJb3JpZ2luX2VjEXAMPLEtokenEXAMPLEtokenEXAMPLE key ABSKQmVkcm9ja0FQSUtleS1EXAMPLEexample "
        "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29tEXAMPLE "
        "X-Amz-Signature=0123456789abcdef0123456789abcdef"
    )
    agent, _ = make([_client_error("AccessDeniedException", 403, message)], stream=stream)
    with pytest.raises(AuthenticationError) as info:
        agent.ask("hi")
    shown = str(info.value) + "".join(traceback.format_exception(info.value))
    for secret in AWS_SECRETS:
        assert secret not in shown, secret


# ---- history from other providers --------------------------------------------------------------


def _history(source: Any, tmp_path: Path) -> Path:
    agent = source.agent(
        [source.tool_calls([("toolu_01", "add", {"a": 1, "b": 1})]), source.text("two")], tools=[add]
    )
    agent.ask("one plus one")
    path = tmp_path / "history.json"
    agent.save_history(path)
    return path


@pytest.mark.parametrize("source", [OpenAIHarness, AnthropicHarness], ids=["openai", "anthropic"])
def test_history_of_another_provider(source: type[Any], tmp_path: Path) -> None:
    path = _history(source(), tmp_path)
    strict, harness = make([text("unused")], tools=[add])
    strict.load_history(path)
    with pytest.raises(HistoryConversionError):
        strict.ask("continue")
    assert harness.script.calls == []

    converted, harness = make([text("continued")], tools=[add], history_policy="convert")
    converted.load_history(path)
    assert converted.ask("continue") == "continued"
    messages = harness.script.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    blocks = [b for m in messages for b in m["content"]]
    assert not any("reasoningContent" in b for b in blocks)  # another model's reasoning is dropped
    use = next(b["toolUse"] for b in blocks if "toolUse" in b)
    result = next(b["toolResult"] for b in blocks if "toolResult" in b)
    assert use["toolUseId"] == result["toolUseId"] == "toolu_01"
    assert result["content"] == [{"text": "2"}]


def test_bedrock_reasoning_is_not_replayed_to_another_bedrock_model() -> None:
    agent, harness = make(
        [tool_use(("t1", "add", {"a": 1, "b": 1})), text("two"), text("ok")],
        tools=[add],
        history_policy="convert",
    )
    agent.ask("1+1")
    agent.model = f"bedrock:{CLAUDE_5}"
    agent.ask("again")
    blocks = [b for m in harness.script.calls[2]["messages"] for b in m["content"]]
    assert harness.script.calls[2]["modelId"] == CLAUDE_5
    assert not any("reasoningContent" in b for b in blocks)


# ---- token counting ----------------------------------------------------------------------------


def test_count_tokens_where_supported() -> None:
    agent, _harness = make([], tools=[add])
    agent.client.stubber.add_response(
        "count_tokens",
        {"inputTokens": 42},
        expected_params={
            "modelId": CLAUDE_45,
            "input": {
                "converse": {
                    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                    "system": [{"text": "You are a helpful assistant."}],
                    "toolConfig": harness_tool_config(agent),
                }
            },
        },
    )
    assert agent.count_tokens("hi") == 42
    agent.client.stubber.assert_no_pending_responses()


def harness_tool_config(agent: Any) -> dict[str, Any]:
    request = agent._count_request("hi")
    return agent.adapter.prepare(request)["toolConfig"]  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("model", "error"), [(CLAUDE_5, UnsupportedCapabilityError), (NOVA, UnknownCapabilityError)]
)
def test_count_tokens_is_gated_by_capability(model: str, error: type[Exception]) -> None:
    agent, _ = make([], model=model)
    with pytest.raises(error):
        agent.count_tokens("hi")
    agent.client.stubber.assert_no_pending_responses()


def test_count_tokens_async() -> None:
    agent, _ = make([], cls=AsyncAgent)
    agent.client.stubber.add_response("count_tokens", {"inputTokens": 7})
    assert asyncio.run(agent.count_tokens("hi")) == 7


# ---- hooks, budgets and checkpoints -------------------------------------------------------------


def test_hooks_see_requests_responses_and_tool_calls() -> None:
    requests: list[ModelRequest] = []
    responses: list[ModelResponse] = []
    calls: list[ToolCall] = []

    def gate(call: ToolCall) -> Any:
        calls.append(call)
        return "not allowed" if call.name == "multiply" else None

    agent, harness = make(
        [tool_use(("t1", "add", {"a": 1, "b": 2}), ("t2", "multiply", {"a": 2, "b": 2})), text("done")],
        tools=[add, multiply],
        on_request=requests.append,
        on_response=responses.append,
        before_tool=gate,
    )
    assert agent.ask("go") == "done"
    assert [r.model for r in requests] == [CLAUDE_45, CLAUDE_45]
    assert [r.provider for r in responses] == ["bedrock", "bedrock"]
    assert [c.name for c in calls] == ["add", "multiply"]
    denied = harness.script.calls[1]["messages"][-1]["content"][1]["toolResult"]
    assert denied["status"] == "error" and "not allowed" in denied["content"][0]["text"]


def test_suspend_and_resume_with_bedrock() -> None:
    def gate(call: ToolCall) -> Any:
        if call.name == "multiply" and not call.resumed:
            raise ToolCallSuspended("approval needed", ref="approval-1")
        return None

    first = tool_use(("t1", "add", {"a": 1, "b": 2}), ("t2", "multiply", {"a": 3, "b": 4}))
    agent, harness = make([first], tools=[add, multiply], before_tool=gate)
    with pytest.raises(TurnSuspended) as caught:
        agent.ask("compute")
    assert agent.messages == []
    saved = json.loads(json.dumps(caught.value.checkpoint.model_dump(mode="json")))

    resumed, harness = make([text("3 and 12")], tools=[add, multiply], before_tool=gate)
    assert resumed.resume(saved) == "3 and 12"
    (call,) = harness.script.calls
    results = [b["toolResult"] for b in call["messages"][-1]["content"]]
    assert [(r["toolUseId"], r["content"]) for r in results] == [
        ("t1", [{"text": "3"}]),
        ("t2", [{"text": "12"}]),
    ]
    assert {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig-abc"}}} in call[
        "messages"
    ][1]["content"]


def test_refusals_and_guardrails() -> None:
    agent, _ = make([converse_response([{"text": "Blocked."}], "content_filtered")], on_refusal="return")
    assert agent.ask("bad") == "Blocked."
    response = agent.last_response
    assert response is not None and response.stop_reason == "refusal"
    assert response.stop_details is not None and response.stop_details.category == "content_filtered"


def test_context_window_exceeded_is_reported() -> None:
    agent, _ = make([converse_response([{"text": "part"}], "model_context_window_exceeded")])
    with pytest.raises(OrkexaError):
        agent.ask("long")
    assert agent.messages == []


def test_unknown_blocks_are_kept_as_native_parts() -> None:
    citations = {"citationsContent": {"content": [{"text": "cited"}], "citations": []}}
    agent, harness = make([converse_response([citations, {"text": "answer"}]), text("next")])
    assert agent.ask("q") == "answer"
    agent.ask("again")
    assert harness.script.calls[1]["messages"][1]["content"][0] == citations


def test_module_constants() -> None:
    assert bedrock_module.DEFAULT_READ_TIMEOUT == 900
    assert bedrock_module.RESIDENCY_CACHE_SECONDS == 300


# ---- regressions from the review ---------------------------------------------------------------


def test_options_are_frozen_and_keep_private_copies() -> None:
    import dataclasses

    fields = {"top_k": 50}
    options = BedrockOptions(region_name="eu-central-1", additional_model_request_fields=fields)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.region_name = "us-east-1"  # type: ignore[misc]
    fields["tools"] = []  # the caller's dict, not the options' copy
    assert options.additional_model_request_fields == {"top_k": 50}
    assert dataclasses.replace(options, region_name="eu-west-1").region_name == "eu-west-1"


def test_reserved_fields_are_checked_again_when_the_request_is_built() -> None:
    options = BedrockOptions(additional_model_request_fields={"top_k": 50})
    agent, harness = make([text("unused")], options=options)
    options.additional_model_request_fields["anthropic_beta"] = ["x"]
    with pytest.raises(ConfigurationError, match="anthropic_beta"):
        agent.ask("hi")
    assert harness.script.calls == []


@pytest.mark.parametrize(("ttl", "split"), [("5m", (12, 0)), ("1h", (0, 12))])
def test_cache_writes_without_details_count_for_the_requested_ttl(ttl: str, split: tuple[int, int]) -> None:
    usage = bedrock_usage({"inputTokens": 1, "outputTokens": 1, "cacheWriteInputTokens": 12}, None, ttl)
    assert usage is not None
    assert (usage.cache_creation_5m_input_tokens, usage.cache_creation_1h_input_tokens) == split
    tracker = UsageTracker()
    response = converse_response([{"text": "ok"}])
    response["usage"]["cacheWriteInputTokens"] = 12
    response["usage"].pop("cacheDetails", None)
    agent, _ = make(
        [response], usage_tracker=tracker, options=BedrockOptions(prompt_caching=True, cache_ttl=ttl)
    )
    agent.ask("hi")
    (record,) = tracker.records
    assert (record.cache_creation_5m_input_tokens, record.cache_creation_1h_input_tokens) == split


def _prepared(*messages: Message) -> dict[str, Any]:
    provider = create_provider("bedrock")
    request = ModelRequest(model=CLAUDE_45, messages=list(messages), max_tokens=10)
    return provider.prepare(request)


def test_media_limits_apply_to_joined_messages() -> None:
    from orkexa.core.types import TextPart

    picture = image(PNG, media_type="image/png")
    half = Message(role="user", content=[picture] * 10 + [TextPart(text="x")])
    assert len(_prepared(half)["messages"]) == 1
    with pytest.raises(InvalidRequestError, match="at most 20 images"):
        _prepared(half, half, Message(role="user", content=[picture]))


def test_a_document_and_its_question_may_come_in_adjacent_messages() -> None:
    pdf = Message(role="user", content=[document(PDF, media_type="application/pdf")])
    native = _prepared(pdf, Message(role="user", content="Summarize it."))
    assert [list(b) for b in native["messages"][0]["content"]] == [["document"], ["text"]]
    with pytest.raises(InvalidRequestError, match="text block"):
        _prepared(pdf)


@pytest.mark.parametrize(
    "data",
    [
        {"image": {"format": "png", "source": {"s3Location": {"uri": "s3://bucket/a.png"}}}},
        {"document": {"format": "pdf", "name": "a", "source": {"bytes": {"$bytes": "JVBERg=="}}}},
        {"video": {"format": "mp4", "source": {"s3Location": {"uri": "s3://bucket/a.mp4"}}}},
    ],
)
def test_native_media_blocks_are_not_replayed(data: dict[str, Any]) -> None:
    from orkexa.core.types import NativePart, TextPart

    message = Message(role="user", content=[NativePart(provider="bedrock", data=data), TextPart(text="x")])
    with pytest.raises(HistoryConversionError, match="media"):
        _prepared(message)


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-instant-v1",
        "anthropic.claude-v2:1",
        "amazon.nova-reel-v1:0",
        "amazon.nova-canvas-v1:0",
        "amazon.nova-sonic-v1:0",
        "amazon.titan-text-express-v1",
    ],
)
def test_lookalike_models_get_no_claude_or_nova_capabilities(model: str) -> None:
    capabilities = bedrock_capabilities(model)
    assert capabilities.support("tools") == "unknown"
    assert capabilities.support("structured_output") == "unknown"


def test_a_result_after_cancellation_goes_to_the_orphan_handler() -> None:
    release, orphaned, started = threading.Event(), threading.Event(), threading.Event()
    seen: list[str] = []

    def opened(cancelled: threading.Event) -> str:
        started.set()
        release.wait(5)
        return "stream"

    def orphan(result: str) -> None:
        seen.append(result)
        orphaned.set()

    async def main() -> None:
        task = asyncio.create_task(bedrock_module._in_thread(opened, on_orphan=orphan))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    release.set()
    assert orphaned.wait(5) and seen == ["stream"]


def test_a_cancelled_call_does_not_start_the_paid_request() -> None:
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        bedrock_module._not_cancelled(cancelled)
