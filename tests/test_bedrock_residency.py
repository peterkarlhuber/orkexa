"""Bedrock data residency (``BedrockOptions(data_residency="eu_only")``), without network access.

The ``bedrock`` control-plane client is a real boto3 client behind ``Stubber``
(``GetInferenceProfile``/``GetGuardrail`` requests are validated against the service
model); every call to it is recorded. Every rejection must happen before Converse or
ConverseStream is called.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from botocore.stub import Stubber
import pytest

from orkexa import Agent, AsyncAgent, BedrockOptions, ConfigurationError, DataResidencyError
from orkexa.content import image
from orkexa.providers import bedrock as bedrock_module
from orkexa.providers.bedrock_residency import (
    EU_REGIONS,
    check_endpoint,
    parse_model_id,
    profile_destinations,
    static_destinations,
)

from .contract.harness import (
    BedrockHarness,
    FakeEventStream,
    Script,
    bedrock_client,
    converse_events,
    converse_response,
)

ACCOUNT = "123456789012"
EU_PROFILE = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
FOUNDATION = "anthropic.claude-sonnet-4-5-20250929-v1:0"
APP_PROFILE = f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:application-inference-profile/a1b2c3d4"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def model_arn(region: str, model: str = FOUNDATION) -> str:
    return f"arn:aws:bedrock:{region}::foundation-model/{model}"


def profile(
    identifier: str, *regions: str, status: str = "ACTIVE", kind: str = "SYSTEM_DEFINED"
) -> dict[str, Any]:
    """A GetInferenceProfile response whose destinations are ``regions``."""
    return {
        "inferenceProfileName": "profile",
        "inferenceProfileArn": f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:inference-profile/{identifier}",
        "inferenceProfileId": identifier.rsplit("/", 1)[-1],
        "models": [{"modelArn": model_arn(r) if r.count("-") >= 2 else r} for r in regions],
        "status": status,
        "type": kind,
    }


def guardrail(cross_region: bool = False) -> dict[str, Any]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    response: dict[str, Any] = {
        "name": "g",
        "guardrailId": "gr-1",
        "guardrailArn": f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:guardrail/gr-1",
        "version": "1",
        "status": "READY",
        "createdAt": now,
        "updatedAt": now,
        "blockedInputMessaging": "blocked",
        "blockedOutputsMessaging": "blocked",
    }
    if cross_region:
        response["crossRegionDetails"] = {
            "guardrailProfileId": "eu.guardrail.v1:0",
            "guardrailProfileArn": (
                f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:guardrail-profile/eu.guardrail.v1:0"
            ),
        }
    return response


class Control:
    """A stubbed ``bedrock`` control-plane client that records the operations called."""

    def __init__(self, region: str = "eu-central-1") -> None:
        self.client = bedrock_client("bedrock", region)
        self.stubber = Stubber(self.client)
        self.stubber.activate()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.client.meta.events.register(
            "provide-client-params.bedrock.*",
            lambda params, model, **_: self.calls.append((model.name, params)),
        )

    def profile(self, identifier: str, *regions: str, **kwargs: Any) -> None:
        self.stubber.add_response(
            "get_inference_profile",
            profile(identifier, *regions, **kwargs),
            expected_params={"inferenceProfileIdentifier": identifier},
        )


def make(
    model: str,
    *,
    region: str = "eu-central-1",
    control: Control | None = None,
    responses: int = 1,
    cls: type[Any] = Agent,
    client_kwargs: dict[str, Any] | None = None,
    **options: Any,
) -> tuple[Any, BedrockHarness]:
    harness = BedrockHarness()
    harness.script = Script([converse_response([{"text": "ok"}]) for _ in range(responses)])
    client = harness.scripted_client(region, **(client_kwargs or {}))
    options.setdefault("data_residency", "eu_only")
    if control is not None:
        options["control_client"] = control.client
    agent = cls(model=f"bedrock:{model}", client=client, options=BedrockOptions(**options))
    return agent, harness


def rejected(agent: Any, harness: BedrockHarness, match: str | None = None) -> DataResidencyError:
    """The request is refused with DataResidencyError and nothing reached Converse/ConverseStream."""
    with pytest.raises(DataResidencyError, match=match) as info:
        agent.ask("hi")
    assert harness.script.calls == []
    assert harness.streams == []
    assert agent.messages == []
    return info.value


# ---- allowed ------------------------------------------------------------------------------------


def test_eu_profile_from_frankfurt_is_allowed_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1", "eu-west-3", "eu-north-1")
    agent, harness = make(EU_PROFILE, control=control, responses=3)
    assert agent.ask("hi") == "ok"
    assert agent.ask("again") == "ok"  # verified once, then cached
    assert [name for name, _ in control.calls] == ["GetInferenceProfile"]
    assert len(harness.script.calls) == 2

    now = bedrock_module.time.monotonic()
    monkeypatch.setattr(
        bedrock_module.time, "monotonic", lambda: now + bedrock_module.RESIDENCY_CACHE_SECONDS + 1
    )
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1")
    assert agent.ask("later") == "ok"  # checked again after the cache expired
    assert len(control.calls) == 2
    control.stubber.assert_no_pending_responses()


def test_foundation_model_in_an_eu_region_needs_no_lookup() -> None:
    control = Control()
    agent, _ = make(FOUNDATION, control=control)
    assert agent.ask("hi") == "ok"
    assert control.calls == []
    agent, _ = make(model_arn("eu-west-1"), region="eu-west-1", control=control)
    assert agent.ask("hi") == "ok"
    assert control.calls == []


def test_injected_client_without_control_client_works_for_foundation_models() -> None:
    agent, _harness = make(FOUNDATION)
    assert agent.ask("hi") == "ok"


def test_application_inference_profile_is_verified_by_lookup() -> None:
    control = Control()
    control.profile(APP_PROFILE, "eu-central-1", kind="APPLICATION")
    agent, harness = make(APP_PROFILE, control=control)
    assert agent.ask("hi") == "ok"
    assert control.calls == [("GetInferenceProfile", {"inferenceProfileIdentifier": APP_PROFILE})]
    assert harness.script.calls[0]["modelId"] == APP_PROFILE


def test_provisioned_throughput_in_the_client_region_is_allowed() -> None:
    arn = f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:provisioned-model/pt-1"
    agent, _harness = make(arn, control=Control())
    assert agent.ask("hi") == "ok"


def test_vpc_endpoint_of_the_region_is_allowed() -> None:
    vpce = "https://vpce-0abc-1234.bedrock-runtime.eu-central-1.vpce.amazonaws.com"
    agent, _harness = make(FOUNDATION, client_kwargs={"endpoint_url": vpce})
    assert agent.ask("hi") == "ok"


def test_explicit_allowlist_without_eu_only() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1")
    agent, harness = make(EU_PROFILE, control=control, data_residency=None, allowed_regions=("eu-central-1",))
    rejected(agent, harness, "eu-west-1")
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1")
    agent, harness = make(
        EU_PROFILE, control=control, data_residency=None, allowed_regions=("eu-central-1", "eu-west-1")
    )
    assert agent.ask("hi") == "ok"


def test_streamed_and_async_requests_are_checked_too() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-2")
    agent, harness = make(EU_PROFILE, control=control, cls=AsyncAgent)
    for stream in (False, True):
        with pytest.raises(DataResidencyError):
            asyncio.run(agent.ask("hi", max_tokens=10) if not stream else _streamed(agent))
    assert harness.script.calls == [] and harness.streams == []


async def _streamed(agent: Any) -> Any:
    agent.stream = True
    return await agent.ask("hi")


def test_streamed_request_from_an_allowed_profile() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1")
    agent, harness = make(EU_PROFILE, control=control)
    agent.stream = True
    assert agent.ask("hi") == "ok"
    assert len(harness.streams) == 1 and harness.streams[0].closed


# ---- rejected -----------------------------------------------------------------------------------


def test_eu_profile_listing_london_is_rejected() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1", "eu-west-2")
    agent, harness = make(EU_PROFILE, control=control)
    error = rejected(agent, harness, "eu-west-2")
    assert "not sent" in str(error)


def test_us_profile_is_rejected() -> None:
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    control = Control()
    control.profile(model, "us-east-1", "us-east-2", "us-west-2")
    agent, harness = make(model, control=control)
    rejected(agent, harness, "us-east-1")


@pytest.mark.parametrize(
    "model",
    [
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:inference-profile/global.anthropic.claude-sonnet-5-5",
    ],
)
def test_global_profiles_are_rejected_without_a_lookup(model: str) -> None:
    control = Control()
    harness = BedrockHarness()
    with pytest.raises(DataResidencyError, match="global"):
        agent = Agent(
            model=f"bedrock:{model}",
            client=harness.scripted_client(),
            options=BedrockOptions(data_residency="eu_only", control_client=control.client),
        )
        agent.ask("hi")
    assert control.calls == [] and harness.script.calls == []


@pytest.mark.parametrize("region", ["us-east-1", "eu-west-2", "eu-central-2"])
def test_client_outside_the_eu_is_rejected(region: str) -> None:
    control = Control(region)
    agent, harness = make(FOUNDATION, region=region, control=control)
    rejected(agent, harness, region)
    assert control.calls == []


def test_foundation_model_arn_of_another_region_is_rejected() -> None:
    agent, harness = make(model_arn("us-east-1"), control=Control())
    rejected(agent, harness, "us-east-1")


def test_lookup_failures_are_rejections() -> None:
    control = Control()
    control.stubber.add_client_error(
        "get_inference_profile", service_error_code="AccessDeniedException", http_status_code=403
    )
    agent, harness = make(EU_PROFILE, control=control)
    error = rejected(agent, harness, "GetInferenceProfile")
    assert "AccessDeniedException" in str(error)


@pytest.mark.parametrize(
    "response",
    [
        profile(EU_PROFILE, "eu-central-1", status="INACTIVE"),
        profile(EU_PROFILE, "not-an-arn"),
        profile(EU_PROFILE, "arn:aws:bedrock:moon-base:foundation-model/x"),
    ],
    ids=["inactive", "garbage-arn", "bad-region"],
)
def test_unverifiable_profiles_are_rejected(response: dict[str, Any]) -> None:
    control = Control()
    control.stubber.add_response("get_inference_profile", response)
    agent, harness = make(EU_PROFILE, control=control)
    rejected(agent, harness)


@pytest.mark.parametrize(
    "model",
    [
        f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:prompt/P1",
        f"arn:aws:bedrock:eu-central-1:{ACCOUNT}:default-prompt-router/anthropic.claude:1",
        f"arn:aws:sagemaker:eu-central-1:{ACCOUNT}:endpoint/my-endpoint",
        "arn:aws:bedrock",
        "mymodel",
    ],
)
def test_unparsable_or_unverifiable_ids_are_rejected(model: str) -> None:
    control = Control()
    agent, harness = make(model, control=control)
    rejected(agent, harness)
    assert control.calls == []


def test_application_profile_routing_outside_the_eu_is_rejected() -> None:
    control = Control()
    control.profile(APP_PROFILE, "eu-central-1", "us-east-1", kind="APPLICATION")
    agent, harness = make(APP_PROFILE, control=control)
    rejected(agent, harness, "us-east-1")


def test_application_profile_of_another_region_is_rejected() -> None:
    arn = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:application-inference-profile/x"
    control = Control()
    agent, harness = make(arn, control=control)
    rejected(agent, harness, "us-east-1")
    assert control.calls == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://bedrock.example.com",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        "https://bedrock-runtime.eu-central-1.amazonaws.com.attacker.example",
        "http://localhost:4566",
    ],
)
def test_non_aws_or_foreign_endpoints_are_rejected(endpoint: str) -> None:
    agent, harness = make(FOUNDATION, client_kwargs={"endpoint_url": endpoint})
    rejected(agent, harness, "endpoint")


def test_base_url_is_checked_when_the_provider_builds_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "AWS_CONFIG_FILE": "/nonexistent/config",
        "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent/credentials",
    }.items():
        monkeypatch.setenv(name, value)
    agent = Agent(
        model=f"bedrock:{FOUNDATION}",
        base_url="https://gateway.example.com",
        options=BedrockOptions(region_name="eu-central-1", data_residency="eu_only"),
    )
    called: list[Any] = []
    agent.client.converse = lambda **kw: called.append(kw)
    with pytest.raises(DataResidencyError, match="endpoint"):
        agent.ask("hi")
    assert called == []


def test_injected_client_needs_a_control_client_for_profiles() -> None:
    agent, harness = make(EU_PROFILE)
    rejected(agent, harness, "control_client")


def test_guardrails_are_checked() -> None:
    config = {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}
    control = Control()
    control.stubber.add_response("get_guardrail", guardrail(), expected_params=config)
    agent, harness = make(FOUNDATION, control=control, guardrail=config)
    assert agent.ask("hi") == "ok"
    assert harness.script.calls[0]["guardrailConfig"] == config

    control = Control()
    control.stubber.add_response("get_guardrail", guardrail(cross_region=True))
    agent, harness = make(FOUNDATION, control=control, guardrail=config)
    rejected(agent, harness, "cross-Region")

    control = Control()
    control.stubber.add_client_error("get_guardrail", service_error_code="ResourceNotFoundException")
    agent, harness = make(FOUNDATION, control=control, guardrail=config)
    rejected(agent, harness, "GetGuardrail")

    foreign = {
        "guardrailIdentifier": f"arn:aws:bedrock:us-east-1:{ACCOUNT}:guardrail/gr-1",
        "guardrailVersion": "1",
    }
    control = Control()
    agent, harness = make(FOUNDATION, control=control, guardrail=foreign)
    rejected(agent, harness, "not in eu-central-1")
    assert control.calls == []


def test_url_media_sources_are_rejected() -> None:
    s3_image = {"type": "image", "source": {"type": "url", "url": "s3://b/a.png", "media_type": "image/png"}}
    agent, harness = make("amazon.nova-pro-v1:0")
    with pytest.raises(DataResidencyError, match="S3 and URL"):
        agent.ask([s3_image, "Describe."])
    assert harness.script.calls == []
    agent, harness = make(FOUNDATION)
    assert agent.ask([image(PNG, media_type="image/png"), "Describe."]) == "ok"


def test_count_tokens_is_checked_too() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-west-2")
    agent, _harness = make(EU_PROFILE, control=control)
    with pytest.raises(DataResidencyError):
        agent.count_tokens("hi")
    agent.client.stubber.assert_no_pending_responses()


def test_residency_error_is_a_configuration_error() -> None:
    assert issubclass(DataResidencyError, ConfigurationError)


# ---- the residency module on its own ----------------------------------------------------------------


def test_eu_regions() -> None:
    assert {"eu-central-1", "eu-north-1", "eu-south-1", "eu-south-2", "eu-west-1", "eu-west-3"} == EU_REGIONS
    assert "eu-west-2" not in EU_REGIONS and "eu-central-2" not in EU_REGIONS


def test_profile_destinations_fail_closed() -> None:
    assert profile_destinations(
        profile(EU_PROFILE, "eu-west-1", "eu-central-1", "eu-west-1"), EU_PROFILE
    ) == (
        "eu-central-1",
        "eu-west-1",
    )
    for response in (None, [], {"status": "ACTIVE"}, {"status": "ACTIVE", "models": [{"modelArn": 3}]}):
        with pytest.raises(DataResidencyError):
            profile_destinations(response, EU_PROFILE)


def test_static_destinations() -> None:
    assert static_destinations(parse_model_id(FOUNDATION), "eu-west-1") == ("eu-west-1",)
    assert static_destinations(parse_model_id(EU_PROFILE), "eu-west-1") is None
    with pytest.raises(DataResidencyError, match="global"):
        static_destinations(parse_model_id("global.amazon.nova-pro-v1:0"), "eu-west-1")


def test_check_endpoint() -> None:
    check_endpoint("eu-west-1", None, EU_REGIONS)
    check_endpoint("eu-west-1", "https://bedrock-runtime.eu-west-1.amazonaws.com", EU_REGIONS)
    check_endpoint("eu-west-1", "https://bedrock-runtime-fips.eu-west-1.amazonaws.com", EU_REGIONS)
    for region, url in (
        (None, None),
        ("eu-west-1", "http://bedrock-runtime.eu-west-1.amazonaws.com"),
        ("eu-west-1", "https://bedrock-runtime.eu-west-3.amazonaws.com"),
    ):
        with pytest.raises(DataResidencyError):
            check_endpoint(region, url, EU_REGIONS)


def test_converse_events_helper_round_trip() -> None:
    """The scripted stream used above produces exactly one messageStop and one metadata event."""
    events = converse_events(converse_response([{"text": "ok"}]))
    stream = FakeEventStream(events)
    assert [next(iter(e)) for e in stream][-2:] == ["messageStop", "metadata"]


# ---- regressions from the review ---------------------------------------------------------------


def test_a_narrower_allowlist_applies_to_cached_destinations() -> None:
    control = Control()
    control.profile(EU_PROFILE, "eu-central-1", "eu-west-1")
    agent, harness = make(EU_PROFILE, control=control, responses=2)
    assert agent.ask("hi") == "ok"
    agent.options = BedrockOptions(
        data_residency="eu_only", allowed_regions=("eu-central-1",), control_client=control.client
    )
    with pytest.raises(DataResidencyError, match="eu-west-1"):
        agent.ask("again")
    assert len(harness.script.calls) == 1
    assert [name for name, _ in control.calls] == ["GetInferenceProfile"]  # cached, checked again


def test_a_control_client_in_another_region_is_rejected() -> None:
    agent, harness = make(EU_PROFILE, control=Control("eu-west-1"))
    rejected(agent, harness, "control_client uses region eu-west-1")


def test_an_assigned_client_counts_as_injected() -> None:
    harness = BedrockHarness()
    harness.script = Script([converse_response([{"text": "ok"}])])
    agent = Agent(
        model=f"bedrock:{EU_PROFILE}",
        options=BedrockOptions(region_name="eu-central-1", data_residency="eu_only"),
    )
    agent.client = harness.scripted_client("eu-central-1")
    rejected(agent, harness, "injected client")
    assert agent.client is not None and agent.adapter._own is None
