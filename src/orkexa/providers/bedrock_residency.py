"""Region allowlists and data-residency checks for the Bedrock provider.

Nothing here imports boto3; the checks call the ``bedrock`` control-plane client they
are given. Every check fails closed: a model, profile, endpoint or guardrail whose
processing regions cannot be determined is rejected with
:class:`~orkexa.errors.DataResidencyError`, never assumed to be compliant.

How Bedrock decides where a request is processed:

* **Foundation model id or ARN** (``anthropic.claude-...``,
  ``arn:aws:bedrock:eu-central-1::foundation-model/...``): in-region inference, the
  request is processed in the region of the ``bedrock-runtime`` endpoint.
* **System-defined (cross-Region) inference profile** (``eu.anthropic...``,
  ``global.anthropic...`` or its ARN): routed to one of the profile's destination
  regions, which depend on the source region. ``GetInferenceProfile`` called in the
  source region lists them (the region of every ``models[].modelArn``). Global profiles
  route to all commercial regions and are always rejected.
* **Application inference profile** (``...:application-inference-profile/...``):
  wraps a foundation model or a system-defined profile; ``GetInferenceProfile``
  lists its destination regions the same way.
* **Provisioned throughput, custom model deployments and imported models** (ARNs):
  processed in the region of the ARN.
* **Prompt resources, prompt routers and SageMaker endpoints**: their routing cannot
  be verified here, so they are rejected.

The geographic prefix of a profile id (``eu.``) is never trusted on its own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from ..errors import DataResidencyError

__all__ = [
    "EU_REGIONS",
    "ModelRef",
    "check_endpoint",
    "parse_model_id",
    "profile_destinations",
    "verify_guardrail",
]

EU_REGIONS: frozenset[str] = frozenset(
    {
        "eu-central-1",  # Frankfurt, Germany
        "eu-north-1",  # Stockholm, Sweden
        "eu-south-1",  # Milan, Italy
        "eu-south-2",  # Spain
        "eu-west-1",  # Ireland
        "eu-west-3",  # Paris, France
    }
)
"""AWS commercial regions located in EU member states (the default ``eu_only`` allowlist).

Deliberately not included: ``eu-central-2`` (Zurich, Switzerland) and ``eu-west-2``
(London, United Kingdom) are in Europe but not in the EU. EU cross-Region inference
profiles called from those source regions add them as destinations, so such calls
are rejected in ``eu_only`` mode. The AWS European Sovereign Cloud is a separate
partition and is not covered.
"""

ModelKind = Literal[
    "foundation-model",
    "inference-profile",
    "application-inference-profile",
    "in-region-arn",
    "unverifiable",
]

#: Bedrock model providers whose ids are plain foundation model ids (``<provider>.<model>``).
_MODEL_PROVIDERS = frozenset(
    {
        "ai21",
        "amazon",
        "anthropic",
        "cohere",
        "deepseek",
        "google",
        "luma",
        "meta",
        "minimax",
        "mistral",
        "moonshot",
        "moonshotai",
        "nvidia",
        "openai",
        "qwen",
        "stability",
        "twelvelabs",
        "writer",
        "xai",
        "zai",
    }
)
_IN_REGION_RESOURCES = frozenset(
    {"provisioned-model", "custom-model-deployment", "imported-model", "custom-model"}
)
_REGION = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d+$")
_RUNTIME_HOST = re.compile(
    r"^(?:[a-z0-9-]+\.)*bedrock-runtime(?:-fips)?\.(?P<region>[a-z0-9-]+)\.(?:vpce\.)?amazonaws\.com$"
)


@dataclass(frozen=True)
class ModelRef:
    """What a ``modelId`` refers to.

    Attributes:
        model_id: The id or ARN as given.
        kind: How its processing regions are determined (see the module docs).
        base_model: The foundation model id behind it when it can be read from the id
            (for capability lookups), else ``None``.
        arn_region: The region of an ARN, else ``None``.
    """

    model_id: str
    kind: ModelKind
    base_model: str | None
    arn_region: str | None = None


def _strip_geo(identifier: str) -> str | None:
    """``eu.anthropic.claude-x`` -> ``anthropic.claude-x``; ``None`` if it is no profile id."""
    prefix, sep, rest = identifier.partition(".")
    if sep and prefix not in _MODEL_PROVIDERS and rest.partition(".")[0] in _MODEL_PROVIDERS:
        return rest
    return None


def parse_model_id(model_id: str) -> ModelRef:
    """Classify a Converse ``modelId`` (no network access)."""
    if model_id.startswith("arn:"):
        pieces = model_id.split(":", 5)
        if len(pieces) != 6 or pieces[2] != "bedrock":
            return ModelRef(model_id, "unverifiable", None, pieces[3] if len(pieces) > 3 else None)
        region = pieces[3] or None
        resource_type, _, resource = pieces[5].partition("/")
        if resource_type == "foundation-model":
            return ModelRef(model_id, "foundation-model", resource or None, region)
        if resource_type == "inference-profile":
            return ModelRef(model_id, "inference-profile", _strip_geo(resource), region)
        if resource_type == "application-inference-profile":
            return ModelRef(model_id, "application-inference-profile", None, region)
        if resource_type in _IN_REGION_RESOURCES:
            return ModelRef(model_id, "in-region-arn", None, region)
        return ModelRef(model_id, "unverifiable", None, region)
    if model_id.partition(".")[0] in _MODEL_PROVIDERS:
        return ModelRef(model_id, "foundation-model", model_id)
    if "." in model_id:
        return ModelRef(model_id, "inference-profile", _strip_geo(model_id))
    return ModelRef(model_id, "unverifiable", None)


def _check_region(region: str | None, allowed: Iterable[str], what: str) -> None:
    allowed = frozenset(allowed)
    if not region or region not in allowed:
        raise DataResidencyError(
            f"{what} is {region or 'unknown'}, which is not in the allowed regions "
            f"({', '.join(sorted(allowed))}); the request was not sent"
        )


def check_endpoint(region: str | None, endpoint_url: str | None, allowed: Iterable[str]) -> None:
    """The ``bedrock-runtime`` client's region and endpoint must be on the allowlist.

    Only AWS endpoints of the client's own region are accepted (the public regional
    endpoint or an interface VPC endpoint, ``*.bedrock-runtime.<region>.vpce.amazonaws.com``).
    """
    _check_region(region, allowed, "The Bedrock client's region")
    if endpoint_url is None:
        return
    parts = urlsplit(endpoint_url)
    match = _RUNTIME_HOST.match(parts.hostname or "")
    if parts.scheme != "https" or match is None or match.group("region") != region:
        raise DataResidencyError(
            f"The endpoint {parts.scheme}://{parts.hostname} is not an AWS bedrock-runtime endpoint "
            f"of {region}; data residency cannot be verified for it"
        )


def static_destinations(ref: ModelRef, region: str) -> tuple[str, ...] | None:
    """Destination regions known without a network call, ``None`` if a lookup is needed.

    Raises:
        DataResidencyError: the model id cannot be verified at all, or names a global
            profile or another region.
    """
    if ref.arn_region is not None and ref.arn_region != region:
        raise DataResidencyError(
            f"{ref.model_id} belongs to region {ref.arn_region}, but the client uses {region}"
        )
    if ref.kind == "unverifiable":
        raise DataResidencyError(
            f"The processing regions of {ref.model_id!r} cannot be verified (only foundation models, "
            "inference profiles, provisioned throughput, custom model deployments and imported "
            "models are supported with data residency)"
        )
    resource = ref.model_id.rsplit("/", 1)[-1] if ref.model_id.startswith("arn:") else ref.model_id
    if ref.kind == "inference-profile" and resource.startswith("global."):
        raise DataResidencyError(
            f"{ref.model_id} is a global cross-Region inference profile; it routes to all "
            "commercial AWS regions"
        )
    if ref.kind in ("foundation-model", "in-region-arn"):
        return (region,)
    return None


def _arn_region(arn: Any) -> str | None:
    if not isinstance(arn, str) or not arn.startswith("arn:"):
        return None
    pieces = arn.split(":", 5)
    if len(pieces) != 6 or pieces[2] != "bedrock" or not _REGION.match(pieces[3]):
        return None
    return pieces[3]


def profile_destinations(response: Any, model_id: str) -> tuple[str, ...]:
    """Destination regions from a ``GetInferenceProfile`` response (fails closed)."""
    if not isinstance(response, dict):
        raise DataResidencyError(f"GetInferenceProfile returned no data for {model_id}")
    status = response.get("status")
    if status != "ACTIVE":
        raise DataResidencyError(f"The inference profile {model_id} is not ACTIVE (status={status!r})")
    models = response.get("models")
    if not isinstance(models, list) or not models:
        raise DataResidencyError(f"GetInferenceProfile lists no destination models for {model_id}")
    regions: list[str] = []
    for entry in models:
        region = _arn_region(entry.get("modelArn") if isinstance(entry, dict) else None)
        if region is None:
            raise DataResidencyError(
                f"A destination of the inference profile {model_id} has no recognizable region "
                f"({entry!r}); data residency cannot be verified"
            )
        regions.append(region)
    return tuple(sorted(set(regions)))


def check_destinations(destinations: Iterable[str], allowed: Iterable[str], model_id: str) -> None:
    allowed = frozenset(allowed)
    outside = sorted(set(destinations) - allowed)
    if outside:
        raise DataResidencyError(
            f"{model_id} can route requests to {', '.join(outside)}, which is not in the allowed "
            f"regions ({', '.join(sorted(allowed))}); the request was not sent"
        )


def verify_guardrail(response: Any, identifier: str) -> None:
    """A guardrail with a cross-Region guardrail profile may evaluate content elsewhere."""
    if not isinstance(response, dict):
        raise DataResidencyError(f"GetGuardrail returned no data for {identifier}")
    if response.get("crossRegionDetails"):
        raise DataResidencyError(
            f"The guardrail {identifier} uses cross-Region inference "
            f"({response['crossRegionDetails'].get('guardrailProfileId', 'a guardrail profile')}); "
            "its processing regions cannot be verified"
        )
